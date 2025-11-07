// This example shows you how to use yzma to execute a simple prompt
// against a model using llamacpp directly via a native Go application.
//
// # Running the example:
//
//	$ make example13-step1

// # This requires running the following command:
//
//	$ make yzma-models // This downloads the needed models

package main

import (
	"fmt"
	"os"

	"github.com/hybridgroup/yzma/pkg/llama"
)

var (
	modelFile = "zarf/models/qwen2.5-0.5b-instruct-fp16.gguf"
	libPath   = os.Getenv("YZMA_LIB")
)

func main() {
	if err := installLlamaCPP(); err != nil {
		fmt.Println("unable to install llamacpp", err)
		os.Exit(0)
	}

	// -------------------------------------------------------------------------

	if err := llama.Load(libPath); err != nil {
		fmt.Println("unable to load library", err.Error())
		os.Exit(1)
	}

	llama.Init()
	defer llama.BackendFree()

	// -------------------------------------------------------------------------

	fmt.Println("\n- Loading Model", modelFile)

	model := llama.ModelLoadFromFile(modelFile, llama.ModelDefaultParams())
	defer llama.ModelFree(model)

	showModelInfo(model)

	lctx := llama.InitFromModel(model, llama.ContextDefaultParams())
	defer llama.Free(lctx)

	// -------------------------------------------------------------------------

	sampler := llama.SamplerChainInit(llama.SamplerChainDefaultParams())
	llama.SamplerChainAdd(sampler, llama.SamplerInitTopK(int32(1.0)))
	llama.SamplerChainAdd(sampler, llama.SamplerInitTempExt(float32(1.0), 0, 1.0))
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

	question := "Write a hello world program in Go?"

	messages := []llama.ChatMessage{
		llama.NewChatMessage("user", question),
	}

	// -------------------------------------------------------------------------

	vocab := llama.ModelGetVocab(model)

	buf := make([]byte, 8196)
	len := llama.ChatApplyTemplate(template, messages, true, buf)

	text := string(buf[:len])

	count := llama.Tokenize(vocab, text, nil, true, true)

	tokens := make([]llama.Token, count)
	llama.Tokenize(vocab, text, tokens, true, true)

	// -------------------------------------------------------------------------

	batch := llama.BatchGetOne(tokens)

	if llama.ModelHasEncoder(model) {
		llama.Encode(lctx, batch)

		start := llama.ModelDecoderStartToken(model)
		if start == llama.TokenNull {
			start = llama.VocabBOS(vocab)
		}

		batch = llama.BatchGetOne([]llama.Token{start})
	}

	// -------------------------------------------------------------------------

	fmt.Printf("Question: %s\n\n", question)

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
}

func showModelInfo(model llama.Model) {
	fmt.Println()

	desc := llama.ModelDesc(model)
	fmt.Printf("Model Description: %s\n", desc)

	size := llama.ModelSize(model)
	fmt.Printf("Model Size: %d tensors\n", size)

	encoder := llama.ModelHasEncoder(model)
	fmt.Printf("Model Has Encoder: %v\n", encoder)

	decoder := llama.ModelHasDecoder(model)
	fmt.Printf("Model Has Decoder: %v\n", decoder)

	recurrent := llama.ModelIsRecurrent(model)
	fmt.Printf("Model Is Recurrent: %v\n", recurrent)

	hybrid := llama.ModelIsHybrid(model)
	fmt.Printf("Model Is Hybrid: %v\n", hybrid)

	count := llama.ModelMetaCount(model)
	fmt.Printf("Model Metadata (%d entries):\n", count)
	for i := range count {
		key, ok := llama.ModelMetaKeyByIndex(model, i)
		if !ok {
			fmt.Printf("Error getting key for index %d\n", i)
			continue
		}

		value, ok := llama.ModelMetaValStrByIndex(model, i)
		if !ok {
			fmt.Printf("Error getting value for index %d\n", i)
			continue
		}

		fmt.Printf("  %s: %s\n", key, value)
	}

	fmt.Println()
}
