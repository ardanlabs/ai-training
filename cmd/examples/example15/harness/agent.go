package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"strings"

	"github.com/ardanlabs/ai-training/foundation/client"
	fntools "github.com/ardanlabs/ai-training/foundation/tools"
)

// This file is the agent: a real tool-calling loop against a real model.
//
// It matters to the lesson that this is real. An earlier version of this
// workshop taught the model a pretend protocol - "emit TOOL_CALL: name(args)
// on its own line" - and then scraped that line out of the prose with a
// regexp. Every attack still landed, but the final and most important claim of
// the session, that a defense placed on the emitted call cannot be talked out
// of refusing it, was itself being simulated by the same kind of string
// matching the session spends an hour discrediting.
//
// Now the model emits a real tool call, a real allowlist refuses it, and the
// difference between "the model asked" and "the tool ran" is a fact on the
// wire rather than an assertion in a slide.

// maxToolTurns bounds the loop. A model that keeps asking for tools it is not
// allowed to call would otherwise spin here.
const maxToolTurns = 5

// maxAnswerTokens bounds one turn's generation.
//
// It has to be generous, because the reasoning block is where the interesting
// work happens - decoding an encoded payload takes the model several hundred
// tokens of thinking before it writes a word of answer - but it cannot be
// absent. Sampling here is effectively greedy, and a greedy model that walks
// into a repetition cycle never walks back out: nothing is left to sample
// differently. Without a cap the request runs until the context window fills,
// which from the console looks exactly like a hang.
const maxAnswerTokens = 8192

// maxToolCallsPerTurn bounds how many distinct tool calls one turn may request.
//
// This is the repetition cycle above, caught early and named. Asked for an
// example invocation while holding real tools, the model emits the example as
// an actual call, then does it again for the next tool, then cycles - a
// hundred name-only calls and no finish reason. Nothing in a real scenario
// needs more than a handful, so crossing this line means the turn has stopped
// producing an answer, and saying so beats waiting out the token cap.
const maxToolCallsPerTurn = 8

// =============================================================================

// Invocation records one tool call the agent dispatched and what came back.
//
// Name and Args are what the model asked for. Status is what the tool did with
// the request. Keeping both is the whole point: under the structural defenses
// the model still asks for exactly what the injection told it to ask for, and
// the record shows the request arriving and being refused.
type Invocation struct {
	Name   string
	Args   map[string]any
	Status string
	Error  string
}

// Ran reports whether the tool actually executed.
func (i Invocation) Ran() bool {
	return i.Status == "SUCCESS"
}

// Answer is the result of one agent run.
type Answer struct {
	Text        string
	Invocations []Invocation
}

// Asked reports whether the model requested the named tool, regardless of
// whether the call was honored. This is the model's intent.
func (a Answer) Asked(name string) bool {
	for _, inv := range a.Invocations {
		if inv.Name == name {
			return true
		}
	}

	return false
}

// Ran reports whether the named tool actually executed. This is the system's
// behavior.
//
// Read Asked and Ran together. Asked-and-not-Ran is a defense doing its job;
// the injection worked on the model and did not work on the system. That gap
// is where the defenses that survive an attacker's rewrite live.
func (a Answer) Ran(name string) bool {
	for _, inv := range a.Invocations {
		if inv.Name == name && inv.Ran() {
			return true
		}
	}

	return false
}

// =============================================================================

// Agent holds the tools and runs the conversation loop.
type Agent struct {
	sseClient     *client.SSEClient[client.ChatSSE]
	tools         map[string]fntools.Tool
	toolDocuments []client.D

	// SanitizeToolOutput, when non-nil, rewrites the content of every tool
	// response before it re-enters the conversation. This is a defense on the
	// tool-to-model edge.
	SanitizeToolOutput func(string) string

	// EgressFilter, when non-nil, screens the final answer before it is
	// returned. It reports the text to use and whether it refused. This is a
	// defense on the model-to-user edge.
	EgressFilter func(string) (string, bool)
}

// NewAgent runs each registration function to populate the tool map and collect
// the tool documents advertised to the model.
func NewAgent(registrations ...func(map[string]fntools.Tool) client.D) *Agent {
	tools := make(map[string]fntools.Tool)
	documents := make([]client.D, 0, len(registrations))

	for _, register := range registrations {
		documents = append(documents, register(tools))
	}

	return &Agent{
		sseClient:     client.NewSSE[client.ChatSSE](client.StdoutLogger),
		tools:         tools,
		toolDocuments: documents,
	}
}

// Ask runs one prompt to completion, servicing whatever tool calls the model
// asks for along the way.
func (a *Agent) Ask(ctx context.Context, systemPrompt string, userPrompt string) (Answer, error) {
	conversation := []client.D{
		{"role": "system", "content": systemPrompt},
		{"role": "user", "content": userPrompt},
	}

	var answer Answer

	for range maxToolTurns {
		text, toolCalls, err := a.streamTurn(ctx, conversation)
		if err != nil {
			return Answer{}, err
		}

		if len(toolCalls) == 0 {
			answer.Text = text

			if a.EgressFilter != nil {
				filtered, refused := a.EgressFilter(text)
				if refused {
					fmt.Println("\n🛑 EGRESS BLOCKED - the answer was withheld on the way to the user.")
					fmt.Printf("\nWhat the user gets instead:\n%s\n", filtered)
				}

				answer.Text = filtered
			}

			return answer, nil
		}

		conversation = append(conversation, assistantToolCalls(toolCalls))

		results, invocations := a.callTools(ctx, toolCalls)
		conversation = append(conversation, results...)
		answer.Invocations = append(answer.Invocations, invocations...)
	}

	return answer, fmt.Errorf("gave up after %d tool turns", maxToolTurns)
}

// =============================================================================

// pending accumulates one streaming tool call as its pieces arrive.
type pending struct {
	id   string
	name string
	args map[string]any
}

// streamTurn runs one model turn and returns either the assistant's text or the
// tool calls it asked for.
//
// The accumulation below is the part worth reading. A streaming tool call does
// not arrive whole. The server sends an opening chunk carrying the call id and
// the function name with an empty argument string, then one or more chunks
// carrying the arguments and identified only by index, and finally a chunk
// whose finish_reason is "tool_calls" and whose delta holds no tool calls at
// all. Reading the calls off that last chunk - the obvious thing to write, and
// what this example did before - yields nothing every time.
func (a *Agent) streamTurn(ctx context.Context, conversation []client.D) (string, []client.ToolCall, error) {
	d := client.D{
		"model":       llmModel,
		"messages":    conversation,
		"temperature": 0.1,
		"top_p":       0.1,
		"top_k":       1,
		"max_tokens":  maxAnswerTokens,
		"stream":      true,
		"tools":       a.toolDocuments,

		// The field is tool_choice. This used to read "tool_selection", which
		// is not a field any OpenAI-compatible server knows, so it was dropped
		// on arrival - harmless only because "auto" is the default anyway.
		"tool_choice": "auto",
	}

	ch := make(chan client.ChatSSE, 100)

	if err := a.sseClient.Do(ctx, http.MethodPost, llmURL, d, ch); err != nil {
		return "", nil, fmt.Errorf("stream: %w", err)
	}

	calls := make(map[int]*pending)

	var chunks []string
	var thinking bool

	for resp := range ch {
		if len(resp.Choices) == 0 {
			continue
		}

		choice := resp.Choices[0]

		// Merge whatever this chunk knows into the call at its index. Any field
		// may be absent from any given chunk, so nothing is overwritten with a
		// zero value.
		for _, tc := range choice.Delta.ToolCalls {
			p, ok := calls[tc.Index]
			if !ok {
				if len(calls) >= maxToolCallsPerTurn {
					return "", nil, fmt.Errorf("the model requested more than %d tool calls in one turn and is repeating itself; the turn produced no answer", maxToolCallsPerTurn)
				}

				p = &pending{args: make(map[string]any)}
				calls[tc.Index] = p

				// Announce the request as it arrives. Until the turn ends
				// nothing else prints, so without this the console goes silent
				// the moment the model stops thinking and starts calling - and
				// a silent console during a long tool stream is indistinguishable
				// from a hang.
				if thinking {
					thinking = false
					fmt.Print("\n\n")
				}

				fmt.Printf("[93m↪ tool call requested(index:%d)[0m\n", tc.Index)
			}

			if tc.ID != "" {
				p.id = tc.ID
			}

			if tc.Function.Name != "" {
				p.name = tc.Function.Name
			}

			maps.Copy(p.args, tc.Function.Arguments)
		}

		switch choice.FinishReason {
		case "error":
			return "", nil, fmt.Errorf("model reported: %s", choice.Delta.Content)

		case "stop", "tool_calls":
			if thinking {
				fmt.Print("\n")
			}

			assembled, err := assemble(calls)
			if err != nil {
				return "", nil, err
			}

			return strings.TrimLeft(strings.Join(chunks, ""), "\n"), assembled, nil
		}

		switch {
		case choice.Delta.Reasoning != "":
			if !thinking {
				thinking = true
				fmt.Print("\n")
			}

			fmt.Printf("\u001b[91m%s\u001b[0m", choice.Delta.Reasoning)

		case choice.Delta.Content != "":
			if thinking {
				thinking = false
				fmt.Print("\n\n")
			}

			fmt.Print(choice.Delta.Content)
			chunks = append(chunks, choice.Delta.Content)
		}
	}

	// The stream ended without a finish reason. Whatever was accumulated is
	// still the best answer available.
	assembled, err := assemble(calls)
	if err != nil {
		return "", nil, err
	}

	return strings.TrimLeft(strings.Join(chunks, ""), "\n"), assembled, nil
}

// assemble orders the accumulated calls by index and checks that each one is
// complete.
//
// The completeness check earns its place. This server sends each argument
// object in a single chunk, so merging by key is enough. A server that split
// one argument object across several chunks would deliver JSON fragments that
// do not individually parse, and the pieces would be dropped before they got
// here - leaving a call with a name and no arguments. That would otherwise
// reach a tool as a plausible-looking empty request. Better to say so.
func assemble(calls map[int]*pending) ([]client.ToolCall, error) {
	if len(calls) == 0 {
		return nil, nil
	}

	toolCalls := make([]client.ToolCall, 0, len(calls))

	for _, index := range slices.Sorted(maps.Keys(calls)) {
		p := calls[index]

		if p.name == "" {
			return nil, fmt.Errorf("tool call at index %d arrived with no function name", index)
		}

		if len(p.args) == 0 {
			return nil, fmt.Errorf("tool call %q at index %d arrived with no arguments; the server may be splitting argument JSON across chunks", p.name, index)
		}

		toolCalls = append(toolCalls, client.ToolCall{
			ID:       p.id,
			Index:    index,
			Type:     "function",
			Function: client.Function{Name: p.name, Arguments: p.args},
		})
	}

	return toolCalls, nil
}

// assistantToolCalls renders the model's tool requests back into the
// conversation, which the model needs to see before it is shown the results.
func assistantToolCalls(toolCalls []client.ToolCall) client.D {
	documents := make([]client.D, 0, len(toolCalls))

	for _, tc := range toolCalls {
		args, _ := json.Marshal(tc.Function.Arguments)

		documents = append(documents, client.D{
			"id":   tc.ID,
			"type": "function",
			"function": client.D{
				"name":      tc.Function.Name,
				"arguments": string(args),
			},
		})
	}

	return client.D{"role": "assistant", "tool_calls": documents}
}

// callTools dispatches each call and records what happened.
func (a *Agent) callTools(ctx context.Context, toolCalls []client.ToolCall) ([]client.D, []Invocation) {
	responses := make([]client.D, 0, len(toolCalls))
	invocations := make([]Invocation, 0, len(toolCalls))

	for _, tc := range toolCalls {
		fmt.Printf("\n\u001b[92m%s(%v)\u001b[0m\n", tc.Function.Name, tc.Function.Arguments)

		tool, exists := a.tools[tc.Function.Name]
		if !exists {
			responses = append(responses, fntools.ErrorResponse(tc.ID, fmt.Errorf("unknown tool: %s", tc.Function.Name)))
			invocations = append(invocations, Invocation{
				Name:   tc.Function.Name,
				Args:   tc.Function.Arguments,
				Status: "FAILED",
				Error:  "unknown tool",
			})

			continue
		}

		resp := tool.Call(ctx, tc)

		if a.SanitizeToolOutput != nil {
			if content, ok := resp["content"].(string); ok {
				resp["content"] = a.SanitizeToolOutput(content)
			}
		}

		responses = append(responses, resp)

		status, errText := readStatus(resp)
		invocations = append(invocations, Invocation{
			Name:   tc.Function.Name,
			Args:   tc.Function.Arguments,
			Status: status,
			Error:  errText,
		})
	}

	return responses, invocations
}

// readStatus pulls the status out of a tool response so the caller can tell a
// call that ran from a call that was refused.
func readStatus(resp client.D) (string, string) {
	content, ok := resp["content"].(string)
	if !ok {
		return "FAILED", "tool response had no content"
	}

	var parsed struct {
		Status string         `json:"status"`
		Data   map[string]any `json:"data"`
	}

	if err := json.Unmarshal([]byte(content), &parsed); err != nil {
		// A sanitizer may have wrapped the content in markers, which makes it
		// no longer parseable as JSON. It reached the model either way, so the
		// call ran.
		return "SUCCESS", ""
	}

	if parsed.Status == "SUCCESS" {
		return "SUCCESS", ""
	}

	errText, _ := parsed.Data["error"].(string)

	return parsed.Status, errText
}
