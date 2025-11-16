package llamacpp

import (
	"context"
)

// Group represents a concurrency group of a specified model.
type Group struct {
	llama chan *Llama
}

// NewGroup creates a new concurrency group of the specified model. The model
// is loaded multiple times and is shared. If all the models are being used,
// the call will block until a model becomes available or the context times out.
func NewGroup(concurrency int, libPath string, modelFile string, cfg Config, options ...func(llg *Llama) error) (*Group, error) {
	llama := make(chan *Llama, concurrency)

	for range concurrency {
		l, err := New(libPath, modelFile, cfg, options...)
		if err != nil {
			return nil, err
		}

		llama <- l
	}

	mgr := Group{
		llama: llama,
	}

	return &mgr, nil
}

// Unload will close down all loaded models. You should call this only when you
// are completely done using the group.
func (g *Group) Unload() {
	close(g.llama)
	for llama := range g.llama {
		llama.Unload()
	}
}

// ChatCompletions is a wrapper around the ChatCompletions method of the Llama
// api. It will block until a model becomes available or the context times out.
func (g *Group) ChatCompletions(ctx context.Context, messages []ChatMessage, params Params) (<-chan ChatResponse, error) {
	ch := make(chan ChatResponse)

	select {
	case <-ctx.Done():
		return nil, ctx.Err()

	case llama := <-g.llama:
		go func() {
			defer func() {
				close(ch)
				g.llama <- llama
			}()

			lch := llama.ChatCompletions(messages, params)
			for msg := range lch {
				ch <- msg
			}
		}()
	}

	return ch, nil
}

// ChatVision is a wrapper around the ChatVision method of the Llama api. It
// will block until a model becomes available or the context times out.
func (g *Group) ChatVision(ctx context.Context, message ChatMessage, imageFile string, params Params) (<-chan ChatResponse, error) {
	ch := make(chan ChatResponse)

	select {
	case <-ctx.Done():
		return nil, ctx.Err()

	case llama := <-g.llama:
		go func() {
			defer func() {
				close(ch)
				g.llama <- llama
			}()

			lch := llama.ChatVision(message, imageFile, params)
			for msg := range lch {
				ch <- msg
			}
		}()
	}

	return ch, nil
}

// Embed is a wrapper around the Embed method of the Llama api. It will block
// until a model becomes available or the context times out.
func (g *Group) Embed(ctx context.Context, text string) ([]float32, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()

	case llama := <-g.llama:
		defer func() {
			g.llama <- llama
		}()

		vec, err := llama.Embed(text)
		if err != nil {
			return nil, err
		}

		return vec, nil
	}
}

// Rerank is a wrapper around the Rerank method of the Llama api. It will block
// until a model becomes available or the context times out.
func (g *Group) Rerank(ctx context.Context, rankingDocs []RankingDocument) ([]Ranking, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()

	case llama := <-g.llama:
		defer func() {
			g.llama <- llama
		}()

		rerankedDocs, err := llama.Rerank(rankingDocs)
		if err != nil {
			return nil, err
		}

		return rerankedDocs, nil
	}
}
