package main

import (
	"fmt"
	"os"

	"github.com/hybridgroup/yzma/pkg/llama"
)

/*
	This link will provide access to the llamacpp libraries. We can
	add the OS-ARCH support as needed.
	https://github.com/ggml-org/llama.cpp/releases

	This is the model to use for this example. Once downloaded, move the
	model to the `zarf/models/` folder.
	https://huggingface.co/QuantFactory/SmolLM-135M-GGUF/resolve/main/SmolLM-135M.Q2_K.gguf?download=true

	I had to tell the MacOS Gatekeeper to allow these libraries to be loaded. I
	am not sure how to automate this process.
*/

var (
	modelFile            = "zarf/models/SmolLM-135M.Q2_K.gguf"
	prompt               = "Are you ready to go?"
	libPath              = os.Getenv("YZMA_LIB")
	responseLength int32 = 12
)

func main() {
	llama.Load(libPath)
	llama.Init()

	model := llama.ModelLoadFromFile(modelFile, llama.ModelDefaultParams())
	lctx := llama.InitFromModel(model, llama.ContextDefaultParams())

	vocab := llama.ModelGetVocab(model)

	// Call once to get the size of the tokens from the prompt.
	count := llama.Tokenize(vocab, prompt, nil, true, false)

	// Now get the actual tokens.
	tokens := make([]llama.Token, count)
	llama.Tokenize(vocab, prompt, tokens, true, false)

	batch := llama.BatchGetOne(tokens)

	sampler := llama.SamplerChainInit(llama.SamplerChainDefaultParams())
	llama.SamplerChainAdd(sampler, llama.SamplerInitGreedy())

	for pos := int32(0); pos+batch.NTokens < count+responseLength; pos += batch.NTokens {
		llama.Decode(lctx, batch)
		token := llama.SamplerSample(sampler, lctx, -1)

		if llama.VocabIsEOG(vocab, token) {
			fmt.Println()
			break
		}

		buf := make([]byte, 36)
		len := llama.TokenToPiece(vocab, token, buf, 0, true)

		fmt.Print(string(buf[:len]))

		batch = llama.BatchGetOne([]llama.Token{token})
	}

	fmt.Println()
}
