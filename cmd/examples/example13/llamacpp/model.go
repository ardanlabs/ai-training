package llamacpp

import (
	"fmt"
	"math"
	"sync"
	"unsafe"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/mtmd"
)

type model struct {
	libPath   string
	model     llama.Model
	vocab     llama.Vocab
	ctxParams llama.ContextParams
	template  string
	projFile  string
	muHEC     sync.Mutex
}

func newModel(libPath string, modelFile string, cfg Config, options ...func(
	m *model) error) (*model, error) {
	if err := llama.Load(libPath); err != nil {
		return nil, fmt.Errorf("unable to load library: %w", err)
	}

	// -------------------------------------------------------------------------

	llama.Init()
	llama.LogSet(llama.LogSilent())

	// -------------------------------------------------------------------------

	mdl, err := llama.ModelLoadFromFile(modelFile, llama.ModelDefaultParams())
	if err != nil {
		return nil, fmt.Errorf("unable to load model: %w", err)
	}

	vocab := llama.ModelGetVocab(mdl)

	// -------------------------------------------------------------------------

	template := llama.ModelChatTemplate(mdl, "")
	if template == "" {
		template, _ = llama.ModelMetaValStr(mdl, "tokenizer.chat_template")
	}

	if template == "" {
		template = "chatml"
	}

	// -------------------------------------------------------------------------

	m := model{
		libPath:   libPath,
		model:     mdl,
		vocab:     vocab,
		ctxParams: cfg.ctxParams(),
		template:  template,
	}

	for _, option := range options {
		if err := option(&m); err != nil {
			return nil, err
		}
	}

	return &m, nil
}

func (m *model) unload() {
	llama.ModelFree(m.model)
	llama.BackendFree()
}

func (m *model) modelInfo() ModelInfo {
	desc := llama.ModelDesc(m.model)
	size := llama.ModelSize(m.model)
	encoder := llama.ModelHasEncoder(m.model)
	decoder := llama.ModelHasDecoder(m.model)
	recurrent := llama.ModelIsRecurrent(m.model)
	hybrid := llama.ModelIsHybrid(m.model)
	count := llama.ModelMetaCount(m.model)
	metadata := make(map[string]string)

	for i := range count {
		key, ok := llama.ModelMetaKeyByIndex(m.model, i)
		if !ok {
			continue
		}

		value, ok := llama.ModelMetaValStrByIndex(m.model, i)
		if !ok {
			continue
		}

		metadata[key] = value
	}

	return ModelInfo{
		Desc:        desc,
		Size:        size,
		HasEncoder:  encoder,
		HasDecoder:  decoder,
		IsRecurrent: recurrent,
		IsHybrid:    hybrid,
		Metadata:    metadata,
	}
}

func (m *model) chatCompletions(messages []ChatMessage, params Params) <-chan ChatResponse {
	ch := make(chan ChatResponse)

	go func() {
		lctx, err := llama.InitFromModel(m.model, m.ctxParams)
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
		l := llama.ChatApplyTemplate(m.template, msgs, true, buf)
		text := string(buf[:l])

		// ---------------------------------------------------------------------

		tokens := llama.Tokenize(m.vocab, text, true, true)
		batch := llama.BatchGetOne(tokens)
		sampler := params.sampler()

		// ---------------------------------------------------------------------

		for range llama.MaxToken {
			llama.Decode(lctx, batch)
			token := llama.SamplerSample(sampler, lctx, -1)

			if llama.VocabIsEOG(m.vocab, token) {
				close(ch)
				break
			}

			buf := make([]byte, 1024*32)
			l := llama.TokenToPiece(m.vocab, token, buf, 0, false)

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

func (m *model) chatVision(message ChatMessage, imageFile string, params Params) <-chan ChatResponse {
	ch := make(chan ChatResponse)

	go func() {
		if m.projFile == "" {
			ch <- ChatResponse{Err: fmt.Errorf("projection file not set")}
			close(ch)
			return
		}

		lctx, err := llama.InitFromModel(m.model, m.ctxParams)
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
		len := llama.ChatApplyTemplate(m.template, msgs, true, buf)
		template := string(buf[:len])

		// ---------------------------------------------------------------------

		output := mtmd.InputChunksInit()
		input := mtmd.NewInputText(template, true, true)

		mctxParams := mtmd.ContextParamsDefault()

		mtmdCtx, err := mtmd.InitFromFile(m.projFile, m.model, mctxParams)
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

		func() {
			// Docs indicate this function is NOT thread-safe.
			m.muHEC.Lock()
			defer m.muHEC.Unlock()
			mtmd.HelperEvalChunks(mtmdCtx, lctx, output, 0, 0, int32(m.ctxParams.NBatch), true, &n)
		}()

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

			if llama.VocabIsEOG(m.vocab, token) {
				close(ch)
				break
			}

			buf := make([]byte, 1024*32)
			l := llama.TokenToPiece(m.vocab, token, buf, 0, false)

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

	return ch
}

func (m *model) embed(text string) ([]float32, error) {
	lctx, err := llama.InitFromModel(m.model, m.ctxParams)
	if err != nil {
		return nil, fmt.Errorf("unable to init from model: %v", err)
	}
	defer func() {
		llama.Synchronize(lctx)
		llama.Free(lctx)
	}()

	// -------------------------------------------------------------------------

	tokens := llama.Tokenize(m.vocab, text, true, true)
	batch := llama.BatchGetOne(tokens)
	llama.Decode(lctx, batch)
	nEmbd := llama.ModelNEmbd(m.model)
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
