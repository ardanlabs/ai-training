package main

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/mtmd"
)

/*
	This is the model and projection file to use for this example. Once downloaded,
	move the model files to the `zarf/models/` folder.
	https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q8_0.gguf?download=true
	https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf?download=true

	You can use `make yzma-models` to download all the models for these examples.
*/

func main() {
	if err := handleFlags(); err != nil {
		showUsage()
		os.Exit(0)
	}

	// -------------------------------------------------------------------------

	if err := llama.Load(*libPath); err != nil {
		fmt.Println("unable to load library", err.Error())
		os.Exit(1)
	}

	llama.Init()
	defer llama.BackendFree()

	// -------------------------------------------------------------------------

	fmt.Println("***> Loading Model", *modelFile)

	model := llama.ModelLoadFromFile(*modelFile, llama.ModelDefaultParams())
	defer llama.ModelFree(model)

	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = 4096
	ctxParams.NBatch = 2048

	lctx := llama.InitFromModel(model, ctxParams)
	defer llama.Free(lctx)

	vocab := llama.ModelGetVocab(model)

	sampler := llama.NewSampler(model, llama.DefaultSamplers)

	// -------------------------------------------------------------------------

	fmt.Println("***> Init mtmd")

	if err := mtmd.Load(*libPath); err != nil {
		fmt.Println("unable to load library", err.Error())
		os.Exit(1)
	}

	mctxParams := mtmd.ContextParamsDefault()
	if !*verbose {
		llama.LogSet(llama.LogSilent())
		mctxParams.Verbosity = llama.LogLevelContinue
	}

	mtmdCtx := mtmd.InitFromFile(*projFile, model, mctxParams)
	defer mtmd.Free(mtmdCtx)

	// -------------------------------------------------------------------------

	fmt.Println("***> Tokenize")

	if *template == "" {
		*template = llama.ModelChatTemplate(model, "")
	}

	var messages []llama.ChatMessage
	if *systemPrompt != "" {
		messages = append(messages, llama.NewChatMessage("system", *systemPrompt))
	}
	messages = append(messages, llama.NewChatMessage("user", *prompt+mtmd.DefaultMarker()))

	output := mtmd.InputChunksInit()
	input := mtmd.NewInputText(chatTemplate(true, *template, messages), true, true)
	bitmap := mtmd.BitmapInitFromFile(mtmdCtx, *imageFile)
	defer mtmd.BitmapFree(bitmap)

	mtmd.Tokenize(mtmdCtx, output, input, []mtmd.Bitmap{bitmap})

	// -------------------------------------------------------------------------

	fmt.Println("***> HelperEvalChunks")

	var n llama.Pos
	mtmd.HelperEvalChunks(mtmdCtx, lctx, output, 0, 0, int32(ctxParams.NBatch), true, &n)

	// -------------------------------------------------------------------------

	var sz int32 = 1
	batch := llama.BatchInit(1, 0, 1)
	batch.NSeqId = &sz
	batch.NTokens = 1
	seqs := unsafe.SliceData([]llama.SeqId{0})
	batch.SeqId = &seqs

	// -------------------------------------------------------------------------

	fmt.Println("***> Extract Response")

	for range llama.MaxToken {
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
}

func chatTemplate(add bool, template string, messages []llama.ChatMessage) string {
	buf := make([]byte, 1024)
	len := llama.ChatApplyTemplate(template, messages, add, buf)
	result := string(buf[:len])
	return result
}
