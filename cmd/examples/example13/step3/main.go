// This example shows you how to create a simple chat application against an
// inference model using llamacpp directly via a native Go application.
//
// # Running the example:
//
//	$ make example13-step3
//
// # This requires running the following command:
//
//	$ make yzma-models // This downloads the needed models

package main

import (
	"bufio"
	"fmt"
	"os"

	"github.com/hybridgroup/yzma/pkg/llama"
)

var (
	modelURL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-fp16.gguf?download=true"
	libPath  = os.Getenv("YZMA_LIB")
)

func main() {
	if err := handleFlags(); err != nil {
		showUsage()
		os.Exit(0)
	}

	if err := installLlamaCPP(); err != nil {
		fmt.Println("unable to install llamacpp", err)
		os.Exit(0)
	}

	modelFile, err := installModel(modelURL)
	if err != nil {
		fmt.Println("unable to install model", err)
		os.Exit(0)
	}

	// -------------------------------------------------------------------------

	if err := llama.Load(libPath); err != nil {
		fmt.Println("unable to load library", err.Error())
		os.Exit(1)
	}

	llama.Init()
	defer llama.BackendFree()

	if !*verbose {
		llama.LogSet(llama.LogSilent())
	}

	// -------------------------------------------------------------------------

	fmt.Println("\n- Loading Model", modelFile)

	model := llama.ModelLoadFromFile(modelFile, llama.ModelDefaultParams())
	defer llama.ModelFree(model)

	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = uint32(*contextSize)

	lctx := llama.InitFromModel(model, ctxParams)
	defer llama.Free(lctx)

	vocab := llama.ModelGetVocab(model)

	sampler := llama.SamplerChainInit(llama.SamplerChainDefaultParams())

	if *topK != 0 {
		llama.SamplerChainAdd(sampler, llama.SamplerInitTopK(int32(*topK)))
	}

	if *topP < 1.0 {
		llama.SamplerChainAdd(sampler, llama.SamplerInitTopP(float32(*topP), 1))
	}

	if *minP > 0 {
		llama.SamplerChainAdd(sampler, llama.SamplerInitMinP(float32(*minP), 1))
	}

	llama.SamplerChainAdd(sampler, llama.SamplerInitTempExt(float32(*temperature), 0, 1.0))
	llama.SamplerChainAdd(sampler, llama.SamplerInitDist(llama.DefaultSeed))

	// -------------------------------------------------------------------------

	template := llama.ModelChatTemplate(model, "")
	if template == "" {
		v, _ := llama.ModelMetaValStr(model, "tokenizer.chat_template")
		template = v
	}

	if template == "" {
		template = "chatml"
	}

	// -------------------------------------------------------------------------

	var messages []llama.ChatMessage
	if *systemPrompt != "" {
		messages = append(messages, llama.NewChatMessage("system", *systemPrompt))
	}

	// -------------------------------------------------------------------------

	fmt.Println()

	first := true

	for {
		fmt.Print("USER> ")

		reader := bufio.NewReader(os.Stdin)

		pmpt, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println("unable to read user input", err.Error())
			os.Exit(1)
		}

		messages = append(messages, llama.NewChatMessage("user", pmpt))

		chat(lctx, model, vocab, sampler, chatTemplate(true, template, messages), first)

		first = false
	}
}

func chat(lctx llama.Context, model llama.Model, vocab llama.Vocab, sampler llama.Sampler, text string, first bool) {
	count := llama.Tokenize(vocab, text, nil, first, true)

	tokens := make([]llama.Token, count)
	llama.Tokenize(vocab, text, tokens, first, true)

	batch := llama.BatchGetOne(tokens)

	if llama.ModelHasEncoder(model) {
		llama.Encode(lctx, batch)

		start := llama.ModelDecoderStartToken(model)
		if start == llama.TokenNull {
			start = llama.VocabBOS(vocab)
		}

		batch = llama.BatchGetOne([]llama.Token{start})
	}

	fmt.Println()

	for range llama.MaxToken {
		llama.Decode(lctx, batch)
		token := llama.SamplerSample(sampler, lctx, -1)

		if llama.VocabIsEOG(vocab, token) {
			fmt.Println()
			break
		}

		buf := make([]byte, 256)
		l := llama.TokenToPiece(vocab, token, buf, 0, false)
		next := string(buf[:l])

		batch = llama.BatchGetOne([]llama.Token{token})

		fmt.Print(next)
	}

	fmt.Println()
}

func chatTemplate(add bool, template string, messages []llama.ChatMessage) string {
	buf := make([]byte, 1024)
	len := llama.ChatApplyTemplate(template, messages, add, buf)

	return string(buf[:len])
}
