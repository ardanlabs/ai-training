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

/*
	This link will provide access to the llamacpp libraries. Thanks to ymza,
	we have an installer process that is called when the program starts. The
	right libraries for your OS and ARCH are loaded into the zarf/llamacpp folder.
	https://github.com/ggml-org/llama.cpp/releases

	This is the model to use for this example. Once downloaded, move the
	model to the `zarf/models/` folder.
	https://huggingface.co/QuantFactory/SmolLM-135M-GGUF/resolve/main/SmolLM-135M.Q2_K.gguf?download=true

	You can use `make yzma-models` to download all the models for these examples.

	I had to tell the MacOS Gatekeeper to allow these libraries to be loaded. I
	am not sure how to automate this process.
*/

var (
	modelFile      = "zarf/models/SmolLM-135M.Q2_K.gguf"
	prompt         = "Are you ready to go?"
	libPath        = os.Getenv("YZMA_LIB")
	responseLength = int32(52)
)

func main() {
	if err := installLlamaCPP(); err != nil {
		fmt.Println("unable to install llamacpp", err)
		os.Exit(0)
	}

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

	modelInfo(model)

	lctx := llama.InitFromModel(model, llama.ContextDefaultParams())
	defer llama.Free(lctx)

	vocab := llama.ModelGetVocab(model)

	// -------------------------------------------------------------------------

	// Call once to get the size of the tokens from the prompt.
	count := llama.Tokenize(vocab, prompt, nil, true, false)

	// Now get the actual tokens.
	tokens := make([]llama.Token, count)
	llama.Tokenize(vocab, prompt, tokens, true, false)

	// -------------------------------------------------------------------------

	sampler := llama.SamplerChainInit(llama.SamplerChainDefaultParams())
	llama.SamplerChainAdd(sampler, llama.SamplerInitGreedy())

	// -------------------------------------------------------------------------

	fmt.Println("- Extract Response")

	batch := llama.BatchGetOne(tokens)

	for pos := int32(0); pos+batch.NTokens < count+responseLength; pos += batch.NTokens {
		llama.Decode(lctx, batch)
		token := llama.SamplerSample(sampler, lctx, -1)

		if llama.VocabIsEOG(vocab, token) {
			fmt.Println()
			break
		}

		buf := make([]byte, 128)
		len := llama.TokenToPiece(vocab, token, buf, 0, true)

		fmt.Print(string(buf[:len]))

		batch = llama.BatchGetOne([]llama.Token{token})
	}

	// -------------------------------------------------------------------------

	fmt.Println()
}

func modelInfo(model llama.Model) {
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
