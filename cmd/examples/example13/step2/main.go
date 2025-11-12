// This example shows you how to use yzma to execute a simple prompt
// against a vision model using llamacpp directly via a native Go application.
//
// # Running the example:
//
//	$ make example13-step2
//
// # This requires running the following command:
//
//	$ make yzma-models // This downloads the needed models

package main

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/mtmd"
)

var (
	modelURL  = "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q8_0.gguf?download=true"
	projURL   = "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf?download=true"
	imageFile = "zarf/samples/gallery/giraffe.jpg"
	libPath   = os.Getenv("YZMA_LIB")
)

func main() {
	if err := installLlamaCPP(); err != nil {
		fmt.Println("unable to install llamacpp", err)
		os.Exit(0)
	}

	modelFile, err := installModel(modelURL)
	if err != nil {
		fmt.Println("unable to install model", err)
		os.Exit(0)
	}

	projFile, err := installModel(projURL)
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

	llama.LogSet(llama.LogSilent())

	// -------------------------------------------------------------------------

	fmt.Println("\n- Loading Model", modelFile)

	model := llama.ModelLoadFromFile(modelFile, llama.ModelDefaultParams())
	defer llama.ModelFree(model)

	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = uint32(4096)

	lctx := llama.InitFromModel(model, ctxParams)
	defer llama.Free(lctx)

	vocab := llama.ModelGetVocab(model)

	// -------------------------------------------------------------------------

	sampler := llama.SamplerChainInit(llama.SamplerChainDefaultParams())
	llama.SamplerChainAdd(sampler, llama.SamplerInitTopK(int32(1.0)))
	llama.SamplerChainAdd(sampler, llama.SamplerInitTempExt(float32(1.0), 0, 1.0))
	llama.SamplerChainAdd(sampler, llama.SamplerInitDist(llama.DefaultSeed))

	// -------------------------------------------------------------------------

	fmt.Println("- Init mtmd")

	if err := mtmd.Load(libPath); err != nil {
		fmt.Println("unable to load library", err.Error())
		os.Exit(1)
	}

	mctxParams := mtmd.ContextParamsDefault()
	llama.LogSet(llama.LogSilent())
	mctxParams.Verbosity = llama.LogLevelContinue

	mtmdCtx := mtmd.InitFromFile(projFile, model, mctxParams)
	defer mtmd.Free(mtmdCtx)

	// -------------------------------------------------------------------------

	fmt.Println("- Tokenize")

	text := "What is in this picture?\n\n"

	template := llama.ModelChatTemplate(model, "")
	if template == "" {
		v, _ := llama.ModelMetaValStr(model, "tokenizer.chat_template")
		template = v
	}

	if template == "" {
		template = "chatml"
	}

	messages := []llama.ChatMessage{
		llama.NewChatMessage("user", text+mtmd.DefaultMarker()),
	}

	output := mtmd.InputChunksInit()
	input := mtmd.NewInputText(chatTemplate(true, template, messages), true, true)

	bitmap := mtmd.BitmapInitFromFile(mtmdCtx, imageFile)
	defer mtmd.BitmapFree(bitmap)

	mtmd.Tokenize(mtmdCtx, output, input, []mtmd.Bitmap{bitmap})

	// -------------------------------------------------------------------------

	fmt.Print("- HelperEvalChunks\n\n")

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

	fmt.Print("\n- Extract Response\n\n")

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

	return string(buf[:len])
}
