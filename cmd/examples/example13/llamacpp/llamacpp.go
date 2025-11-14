// Package llamacpp provides support for working with models using llamacpp.
package llamacpp

import (
	"fmt"
	"math"
	"sort"
	"unsafe"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/mtmd"
)

type Params struct {
	TopK float32
	TopP float32
	Temp float32
}

func (p Params) sampler() llama.Sampler {
	sampler := llama.SamplerChainInit(llama.SamplerChainDefaultParams())

	if p.TopK > 0 {
		llama.SamplerChainAdd(sampler, llama.SamplerInitTopK(int32(p.TopK)))
	}
	if p.TopP > 0 {
		llama.SamplerChainAdd(sampler, llama.SamplerInitTopP(p.TopP, 0))
	}
	if p.Temp > 0 {
		llama.SamplerChainAdd(sampler, llama.SamplerInitTempExt(p.Temp, 0, 1.0))
	}

	llama.SamplerChainAdd(sampler, llama.SamplerInitDist(llama.DefaultSeed))

	return sampler
}

// =============================================================================

type Config struct {
	ContextWindow uint32
	Embeddings    bool
}

func (cfg Config) ctxParams() llama.ContextParams {
	ctxParams := llama.ContextDefaultParams()

	if cfg.Embeddings {
		ctxParams.Embeddings = 1
	}

	if cfg.ContextWindow > 0 {
		ctxParams.NBatch = cfg.ContextWindow
		ctxParams.NUbatch = cfg.ContextWindow
		ctxParams.NCtx = cfg.ContextWindow
	}

	return ctxParams
}

// =============================================================================

type RankingDocument struct {
	Document  string
	Embedding []float64
}

type Ranking struct {
	Document string
	Score    float64
}

type ChatMessage struct {
	Role    string
	Content string
}

type ChatResponse struct {
	Response string
	Err      error
}

// =============================================================================

type Llama struct {
	libPath   string
	model     llama.Model
	vocab     llama.Vocab
	ctxParams llama.ContextParams
	template  string
	projFile  string
}

func WithProjection(projFile string) func(llm *Llama) error {
	return func(llm *Llama) error {
		if err := mtmd.Load(llm.libPath); err != nil {
			return fmt.Errorf("unable to load mtmd library: %w", err)
		}

		llm.projFile = projFile

		return nil
	}
}

func New(libPath string, modelFile string, cfg Config, options ...func(llm *Llama) error) (*Llama, error) {
	if err := llama.Load(libPath); err != nil {
		return nil, fmt.Errorf("unable to load library: %w", err)
	}

	// -------------------------------------------------------------------------

	llama.Init()
	llama.LogSet(llama.LogSilent())

	// -------------------------------------------------------------------------

	model, err := llama.ModelLoadFromFile(modelFile, llama.ModelDefaultParams())
	if err != nil {
		return nil, fmt.Errorf("unable to load model: %w", err)
	}

	vocab := llama.ModelGetVocab(model)

	// -------------------------------------------------------------------------

	template := llama.ModelChatTemplate(model, "")
	if template == "" {
		template, _ = llama.ModelMetaValStr(model, "tokenizer.chat_template")
	}

	if template == "" {
		template = "chatml"
	}

	// -------------------------------------------------------------------------

	llm := Llama{
		libPath:   libPath,
		model:     model,
		vocab:     vocab,
		ctxParams: cfg.ctxParams(),
		template:  template,
	}

	for _, option := range options {
		if err := option(&llm); err != nil {
			return nil, err
		}
	}

	return &llm, nil
}

func (llm *Llama) Unload() {
	llama.ModelFree(llm.model)
	llama.BackendFree()
}

func (llm *Llama) ChatCompletions(messages []ChatMessage, params Params) <-chan ChatResponse {
	ch := make(chan ChatResponse)

	go func() {
		lctx, err := llama.InitFromModel(llm.model, llm.ctxParams)
		if err != nil {
			ch <- ChatResponse{Err: fmt.Errorf("unable to init from model: %w", err)}
			close(ch)
			return
		}
		defer func() {
			llama.Synchronize(lctx)
			llama.Free(lctx)
		}()

		// ---------------------------------------------------------------------

		msgs := make([]llama.ChatMessage, len(messages))
		for i, msg := range messages {
			msgs[i] = llama.NewChatMessage(msg.Role, msg.Content)
		}

		buf := make([]byte, 1024*32)
		l := llama.ChatApplyTemplate(llm.template, msgs, true, buf)
		text := string(buf[:l])

		// ---------------------------------------------------------------------

		tokens := llama.Tokenize(llm.vocab, text, true, true)
		batch := llama.BatchGetOne(tokens)
		sampler := params.sampler()

		// ---------------------------------------------------------------------

		for range llama.MaxToken {
			llama.Decode(lctx, batch)
			token := llama.SamplerSample(sampler, lctx, -1)

			if llama.VocabIsEOG(llm.vocab, token) {
				close(ch)
				break
			}

			buf := make([]byte, 1024*32)
			l := llama.TokenToPiece(llm.vocab, token, buf, 0, false)

			resp := string(buf[:l])
			if resp == "" {
				close(ch)
				break
			}

			ch <- ChatResponse{Response: resp}

			batch = llama.BatchGetOne([]llama.Token{token})
		}
	}()

	return ch
}

func (llm *Llama) ChatVision(message ChatMessage, imageFile string, params Params) (<-chan ChatResponse, error) {
	if llm.projFile == "" {
		return nil, fmt.Errorf("projection file not set")
	}

	ch := make(chan ChatResponse)

	go func() {
		lctx, err := llama.InitFromModel(llm.model, llm.ctxParams)
		if err != nil {
			ch <- ChatResponse{Err: fmt.Errorf("unable to init from model: %v", err)}
			close(ch)
			return
		}
		defer func() {
			llama.Synchronize(lctx)
			llama.Free(lctx)
		}()

		// ---------------------------------------------------------------------

		msgs := []llama.ChatMessage{
			llama.NewChatMessage(message.Role, message.Content),
			llama.NewChatMessage("user", mtmd.DefaultMarker()),
		}

		buf := make([]byte, 1024*32)
		len := llama.ChatApplyTemplate(llm.template, msgs, true, buf)
		template := string(buf[:len])

		// ---------------------------------------------------------------------

		output := mtmd.InputChunksInit()
		input := mtmd.NewInputText(template, true, true)

		mctxParams := mtmd.ContextParamsDefault()

		mtmdCtx, err := mtmd.InitFromFile(llm.projFile, llm.model, mctxParams)
		if err != nil {
			ch <- ChatResponse{Err: fmt.Errorf("unable to init from model: %v", err)}
			close(ch)
			return
		}
		defer mtmd.Free(mtmdCtx)

		bitmap := mtmd.BitmapInitFromFile(mtmdCtx, imageFile)
		defer mtmd.BitmapFree(bitmap)

		mtmd.Tokenize(mtmdCtx, output, input, []mtmd.Bitmap{bitmap})

		var n llama.Pos
		mtmd.HelperEvalChunks(mtmdCtx, lctx, output, 0, 0, int32(llm.ctxParams.NBatch), true, &n)

		// ---------------------------------------------------------------------

		var sz int32 = 1
		batch := llama.BatchInit(1, 0, 1)
		batch.NSeqId = &sz
		batch.NTokens = 1
		seqs := unsafe.SliceData([]llama.SeqId{0})
		batch.SeqId = &seqs

		// ---------------------------------------------------------------------

		sampler := params.sampler()

		for range llama.MaxToken {
			llama.Decode(lctx, batch)
			token := llama.SamplerSample(sampler, lctx, -1)

			if llama.VocabIsEOG(llm.vocab, token) {
				close(ch)
				break
			}

			buf := make([]byte, 1024*32)
			l := llama.TokenToPiece(llm.vocab, token, buf, 0, false)

			resp := string(buf[:l])
			if resp == "" {
				close(ch)
				break
			}

			ch <- ChatResponse{
				Response: resp,
			}

			batch = llama.BatchGetOne([]llama.Token{token})
		}
	}()

	return ch, nil
}

func (llm *Llama) Embed(text string) ([]float32, error) {
	lctx, err := llama.InitFromModel(llm.model, llm.ctxParams)
	if err != nil {
		return nil, fmt.Errorf("unable to init from model: %v", err)
	}
	defer func() {
		llama.Synchronize(lctx)
		llama.Free(lctx)
	}()

	// -------------------------------------------------------------------------

	tokens := llama.Tokenize(llm.vocab, text, true, true)
	batch := llama.BatchGetOne(tokens)
	llama.Decode(lctx, batch)
	nEmbd := llama.ModelNEmbd(llm.model)
	vec, err := llama.GetEmbeddingsSeq(lctx, 0, nEmbd)
	if err != nil {
		return nil, fmt.Errorf("unable to get embeddings: %v", err)
	}

	// -------------------------------------------------------------------------

	var sum float64
	for _, v := range vec {
		sum += float64(v * v)
	}

	sum = math.Sqrt(sum)
	norm := float32(1.0 / sum)

	for i, v := range vec {
		vec[i] = v * norm
	}

	return vec, nil
}

func (llm *Llama) Rerank(rankingDocs []RankingDocument) ([]Ranking, error) {
	rerankedDocs := make([]Ranking, len(rankingDocs))

	// Simple scoring based on embedding magnitude and positive values.
	for i, doc := range rankingDocs {
		var sumPositive, sumTotal float64
		for _, val := range doc.Embedding {
			sumTotal += val * val
			if val > 0 {
				sumPositive += val
			}
		}

		if sumTotal == 0 {
			rerankedDocs[i] = Ranking{Document: doc.Document, Score: 0}
		}

		// Normalize and combine magnitude with positive bias
		magnitude := math.Sqrt(sumTotal) / float64(len(doc.Embedding))
		positiveRatio := sumPositive / float64(len(doc.Embedding))
		score := (magnitude + positiveRatio) / 2

		rerankedDocs[i] = Ranking{Document: doc.Document, Score: score}
	}

	sort.Slice(rerankedDocs, func(i, j int) bool {
		return rerankedDocs[i].Score > rerankedDocs[j].Score
	})

	return rerankedDocs, nil
}

func (llm *Llama) ShowModelInfo() {
	fmt.Println()

	desc := llama.ModelDesc(llm.model)
	fmt.Printf("Model Description: %s\n", desc)

	size := llama.ModelSize(llm.model)
	fmt.Printf("Model Size: %d tensors\n", size)

	encoder := llama.ModelHasEncoder(llm.model)
	fmt.Printf("Model Has Encoder: %v\n", encoder)

	decoder := llama.ModelHasDecoder(llm.model)
	fmt.Printf("Model Has Decoder: %v\n", decoder)

	recurrent := llama.ModelIsRecurrent(llm.model)
	fmt.Printf("Model Is Recurrent: %v\n", recurrent)

	hybrid := llama.ModelIsHybrid(llm.model)
	fmt.Printf("Model Is Hybrid: %v\n", hybrid)

	count := llama.ModelMetaCount(llm.model)
	fmt.Printf("Model Metadata (%d entries):\n", count)
	for i := range count {
		key, ok := llama.ModelMetaKeyByIndex(llm.model, i)
		if !ok {
			fmt.Printf("Error getting key for index %d\n", i)
			continue
		}

		value, ok := llama.ModelMetaValStrByIndex(llm.model, i)
		if !ok {
			fmt.Printf("Error getting value for index %d\n", i)
			continue
		}

		fmt.Printf("  %s: %s\n", key, value)
	}

	fmt.Println()
}
