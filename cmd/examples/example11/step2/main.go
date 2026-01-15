// This example provides a chat interface for the video that was processed
// in step1.
//
// # Running the example:
//
//	$ make example11-step2
//
// # This requires running the following commands:
//
//	$ make kronk-up  // This starts the Kronk service.

package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"github.com/ardanlabs/ai-training/foundation/tiktoken"
	"go.mongodb.org/mongo-driver/mongo"
)

var (
	urlTextEmbed   = "http://localhost:8080/v1/embeddings"
	modelTextEmbed = "embeddinggemma-300m-qat-Q8_0"
	urlChat        = "http://localhost:8080/v1/chat/completions"
	modelChat      = "cerebras_Qwen3-Coder-REAP-25B-A3B-Q8_0"
	contextWindow  = 32 * 1024
)

func init() {
	if v := os.Getenv("LLM_CHATSERVER"); v != "" {
		urlChat = v
	}

	if v := os.Getenv("LLM_CHATMODEL"); v != "" {
		modelChat = v
	}
}

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	// -------------------------------------------------------------------------
	// Declare a function that can accept user input which the agent will use
	// when it's the users turn.

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	// -------------------------------------------------------------------------
	// Construct the agent and get it started.

	agent, err := NewAgent(getUserMessage)
	if err != nil {
		return fmt.Errorf("failed to create agent: %w", err)
	}

	return agent.Run(context.TODO())
}

// =============================================================================

// Tool describes the features which all tools must implement.
type Tool interface {
	Call(ctx context.Context, toolCall client.ToolCall) client.D
}

// =============================================================================

// Agent represents the chat agent that can use tools to perform tasks.
type Agent struct {
	chatClient      *client.LLM
	textEmbedClient *client.LLM
	sseClient       *client.SSEClient[client.ChatSSE]
	col             *mongo.Collection
	getUserMessage  func() (string, bool)
	tke             *tiktoken.Tiktoken
	tools           map[string]Tool
	toolDocuments   []client.D
}

// NewAgent creates a new instance of Agent.
func NewAgent(getUserMessage func() (string, bool)) (*Agent, error) {
	// -------------------------------------------------------------------------
	// Init access to the DB.

	ctx := context.Background()
	dbClient, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return nil, fmt.Errorf("mongodb.Connect: %w", err)
	}

	col, err := initDB(ctx, dbClient)
	if err != nil {
		return nil, fmt.Errorf("initDB: %w", err)
	}

	// -------------------------------------------------------------------------
	// Construct the tokenizer.

	tke, err := tiktoken.NewTiktoken()
	if err != nil {
		return nil, fmt.Errorf("failed to create tiktoken: %w", err)
	}

	// -------------------------------------------------------------------------
	// Construct the agent.

	tools := map[string]Tool{}

	agent := Agent{
		chatClient:      client.NewLLM(urlChat, modelChat),
		textEmbedClient: client.NewLLM(urlTextEmbed, modelTextEmbed),
		sseClient:       client.NewSSE[client.ChatSSE](client.StdoutLogger),
		col:             col,
		getUserMessage:  getUserMessage,
		tke:             tke,
		tools:           tools,
		toolDocuments:   []client.D{},
	}

	return &agent, nil
}

// The system prompt for the model so it behaves as expected.
const systemPrompt = `
You are a helpful coding assistant that has tools to assist you in coding.

After you request a tool call, you will receive a JSON document with two fields,
"status" and "data". Always check the "status" field to know if the call "SUCCEED"
or "FAILED". The information you need to respond will be provided under the "data"
field. If the called "FAILED", just inform the user and don't try using the tool
again for the current response.

When reading Go source code always start counting lines of code from the top of
the source code file.

If you get back results from a tool call, do not verify the results.

Reasoning: high
`

// Run starts the agent and runs the chat loop.
func (a *Agent) Run(ctx context.Context) error {
	var conversation []client.D // History of the conversation
	var reasonContent []string  // Reasoning content per model call
	var inToolCall bool         // Need to know we are inside a tool call request

	conversation = append(conversation, client.D{
		"role":    "system",
		"content": systemPrompt,
	})

	fmt.Printf("\nChat with %s (use 'ctrl-c' to quit)\n", modelChat)

	timeForResult := time.NewTicker(100 * time.Millisecond)

	for {
		// ---------------------------------------------------------------------
		// If we are not in a tool call then we can ask the user
		// to provide their next question or request.

		if !inToolCall {
			fmt.Print("\u001b[94m\nYou\u001b[0m: ")
			userInput, ok := a.getUserMessage()
			if !ok {
				break
			}

			userInput, err := a.injectContext(ctx, conversation, userInput)
			if err != nil {
				fmt.Printf("\n\n\u001b[91mERROR:%s\u001b[0m\n\n", err)
				continue
			}

			conversation = append(conversation, client.D{
				"role":    "user",
				"content": userInput,
			})
		}

		inToolCall = false

		// ---------------------------------------------------------------------
		// Let's show how long we are waiting for the model response.

		wctx, cancelTimer := context.WithCancel(ctx)
		timeForResult.Reset(100 * time.Millisecond)
		start := time.Now()

		var wg sync.WaitGroup
		wg.Go(func() {
			for {
				select {
				case <-timeForResult.C:
					m := time.Since(start).Milliseconds()
					fmt.Printf("\r\u001b[93m%s %d.%03d\u001b[0m: ", modelChat, m/1000, m%1000)

				case <-wctx.Done():
					fmt.Print("\n")
					timeForResult.Stop()
					cancelTimer()
					return
				}
			}
		})

		// ---------------------------------------------------------------------
		// Now we will make a call to the model, we could be responding to a
		// tool call or providing a user request.

		d := client.D{
			"model":          modelChat,
			"messages":       conversation,
			"max_tokens":     contextWindow,
			"temperature":    0.0,
			"top_p":          0.1,
			"top_k":          1,
			"stream":         true,
			"tools":          a.toolDocuments,
			"tool_selection": "auto",
		}

		fmt.Printf("\u001b[93m\n%s\u001b[0m: 0.000", modelChat)

		ch := make(chan client.ChatSSE, 100)
		ctx, cancelDoCall := context.WithTimeout(ctx, time.Minute*5)

		if err := a.sseClient.Do(ctx, http.MethodPost, urlChat, d, ch); err != nil {
			fmt.Printf("\n\n\u001b[91mERROR:%s\u001b[0m\n\n", err)
			inToolCall = false
			cancelDoCall()
			continue
		}

		// ---------------------------------------------------------------------
		// Now we will make a call to the model.

		var chunks []string      // Store the response chunks since we are streaming.
		reasonThinking := false  // GPT models provide a Reasoning field.
		contentThinking := false // Other reasoning models use <think> tags.
		reasonContent = nil      // Reset the reasoning content for this next call.

		// ---------------------------------------------------------------------
		// Process the response which comes in as chunks. So we need to process
		// and save each chunk.

		waitingForResponse := true

		for resp := range ch {
			if len(resp.Choices) == 0 {
				continue
			}

			// Check if this is the first response. If it is, we will shutdown
			// the G displaying the latency.
			if waitingForResponse {
				waitingForResponse = false
				cancelTimer()
				wg.Wait()
			}

			switch {

			// Did the model ask us to execute a tool call?
			case len(resp.Choices[0].Delta.ToolCalls) > 0:
				fmt.Print("\n\n")

				toolCall := resp.Choices[0].Delta.ToolCalls[0]

				conversation = a.addToConversation(reasonContent, conversation, client.D{
					"role": "assistant",
					"content": fmt.Sprintf("Tool call %s: %s(%v)",
						toolCall.ID,
						toolCall.Function.Name,
						toolCall.Function.Arguments),
				})

				results := a.callTools(ctx, resp.Choices[0].Delta.ToolCalls)
				if len(results) > 0 {
					conversation = a.addToConversation(reasonContent, conversation, results...)
					inToolCall = true
				}

			// Did we get content? With some models a <think> tag could exist to
			// indicate reasoning. We need to filter that out and display it as
			// a different color.
			case resp.Choices[0].Delta.Content != "":
				if reasonThinking {
					reasonThinking = false
					fmt.Print("\n\n")
				}

				switch resp.Choices[0].Delta.Content {
				case "<think>":
					contentThinking = true
					continue
				case "</think>":
					contentThinking = false
					continue
				}

				switch {
				case !contentThinking:
					fmt.Print(resp.Choices[0].Delta.Content)
					chunks = append(chunks, resp.Choices[0].Delta.Content)

				case contentThinking:
					reasonContent = append(reasonContent, resp.Choices[0].Delta.Content)
					fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Content)
				}

			// Did we get reasoning content? ChatGPT models provide reasoning in
			// the Delta.Reasoning field. Display it as a different color.
			case resp.Choices[0].Delta.Reasoning != "":
				reasonThinking = true

				if len(reasonContent) == 0 {
					fmt.Print("\n")
				}

				reasonContent = append(reasonContent, resp.Choices[0].Delta.Reasoning)
				fmt.Printf("\u001b[91m%s\u001b[0m", resp.Choices[0].Delta.Reasoning)
			}
		}

		cancelDoCall()

		// ---------------------------------------------------------------------
		// We processed all the chunks from the response so we need to add
		// this to the conversation history.

		if !inToolCall && len(chunks) > 0 {
			fmt.Print("\n")

			content := strings.Join(chunks, " ")
			content = strings.TrimLeft(content, "\n")

			if content != "" {
				conversation = a.addToConversation(reasonContent, conversation, client.D{
					"role":    "assistant",
					"content": content,
				})
			}
		}
	}

	return nil
}

// injectContext will add context to the user input based on the search results.
func (a *Agent) injectContext(ctx context.Context, conversation []client.D, userInput string) (string, error) {
	yes, err := a.isQuestionRelevant(ctx, conversation, userInput)
	if err != nil {
		return "", fmt.Errorf("failed to check if search is needed: %w", err)
	}

	if !yes {
		return userInput, nil
	}

	results, err := textVectorSearch(ctx, a.textEmbedClient, a.col, userInput)
	if err != nil {
		return "", fmt.Errorf("failed to search for context: %w", err)
	}

	var extraContext string
	for _, result := range results {
		fmt.Printf("\u001b[95m\nScore: %.2f\u001b[0m:\n", result.Score)
		fmt.Printf("\u001b[95m%v\u001b[0m:\n", result.Text)
		if result.Score >= .70 {
			extraContext += result.Text + "\n"
		}
	}

	if extraContext != "" {
		const prompt = "please answer the following question only using the context provided."
		userInput = fmt.Sprintf("%s\nCONTEXT: %s\nQUESTION:%s", prompt, extraContext, userInput)
	}

	return userInput, nil
}

// isQuestionRelevant will check if the user input is relevant to the Go API
// service development class.
func (a *Agent) isQuestionRelevant(ctx context.Context, conversation []client.D, userInput string) (bool, error) {
	const prompt = `
	You are a relevance filter for Bill's Go API service development class. 
	Determine if the following question with the current message
	history is relevant to learning how to write API services in
	the Go programming language presented by Bill.
	
	Relevant topics include: Go syntax, HTTP handlers, REST APIs,
	JSON handling, middleware, routing, database connections,
	authentication, testing, error handling, logging, configuration,
	deployment, and Go best practices for web services.
	
	Irrelevant topics include: other programming languages,
	unrelated Go topics (like GUI development), general programming
	theory without Go context, or completely off-topic questions.

	If the user is asking a followup question to a previous question,
	consider it not relevant.

	History: %s
	
	Question: %s

	Respond with only "RELEVANT" or "NOT RELEVANT" followed by a brief reason.
`

	var history string
	start := max(len(conversation)-2, 0)
	for _, msg := range conversation[start:] {
		history += msg["content"].(string) + "\n"
	}

	text := fmt.Sprintf(prompt, history, userInput)

	result, err := a.chatClient.ChatCompletions(ctx, text)
	if err != nil {
		return false, fmt.Errorf("failed to check if search is needed: %w", err)
	}

	fmt.Printf("\u001b[95m\n%v\u001b[0m:\n", result)

	if strings.Contains(result, "NOT RELEVANT") {
		return false, nil
	}

	return true, nil
}

// addToConversation will add new messages to the conversation history and
// calculate the different tokens used in the conversation and display it to the
// user. It will also check the amount of input tokens currently in history
// and remove the oldest messages if we are over.
func (a *Agent) addToConversation(reasoning []string, conversation []client.D, newMessages ...client.D) []client.D {
	conversation = append(conversation, newMessages...)

	fmt.Print("\n")

	for {
		var currentWindow int
		for _, msg := range conversation {
			currentWindow += a.tke.TokenCount(msg["content"].(string))
		}

		r := strings.Join(reasoning, " ")
		reasonTokens := a.tke.TokenCount(r)

		totalTokens := currentWindow + reasonTokens
		percentage := (float64(currentWindow) / float64(contextWindow)) * 100
		of := float32(contextWindow) / float32(1024)

		fmt.Printf("\u001b[90mTokens Total[%d] Reason[%d] Window[%d] (%.0f%% of %.0fK)\u001b[0m\n", totalTokens, reasonTokens, currentWindow, percentage, of)

		// ---------------------------------------------------------------------
		// Check if we have too many input tokens and start removing messages.

		if currentWindow > contextWindow {
			fmt.Print("\u001b[90mRemoving conversation history\u001b[0m\n")
			conversation = slices.Delete(conversation, 1, 2)
			continue
		}

		break
	}

	return conversation
}

// callTools will lookup a requested tool by name and call it.
func (a *Agent) callTools(ctx context.Context, toolCalls []client.ToolCall) []client.D {
	var resps []client.D

	for _, toolCall := range toolCalls {
		tool, exists := a.tools[toolCall.Function.Name]
		if !exists {
			continue
		}

		fmt.Printf("\n\u001b[92m%s(%v)\u001b[0m:\n\n", toolCall.Function.Name, toolCall.Function.Arguments)

		resp := tool.Call(ctx, toolCall)
		resps = append(resps, resp)

		fmt.Printf("%#v\n", resps)
	}

	return resps
}
