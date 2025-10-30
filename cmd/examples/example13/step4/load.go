package main

import (
	"database/sql"
	"fmt"
	"math"
	"os"
	"regexp"
	"strings"

	"github.com/hybridgroup/yzma/pkg/llama"
)

// modelFile := "zarf/models/bge-m3-q8_0.gguf"

type EmbeddingModel struct {
	model llama.Model
	lctx  llama.Context
	vocab llama.Vocab
}

func NewEmbeddingModel(modelFile string) (*EmbeddingModel, error) {
	libPath := os.Getenv("YZMA_LIB")

	if err := llama.Load(libPath); err != nil {
		return nil, fmt.Errorf("unable to load library: %w", err)
	}

	llama.Init()

	model := llama.ModelLoadFromFile(modelFile, llama.ModelDefaultParams())

	ctxParams := llama.ContextDefaultParams()
	ctxParams.Embeddings = 1

	em := EmbeddingModel{
		model: model,
		lctx:  llama.InitFromModel(model, ctxParams),
		vocab: llama.ModelGetVocab(model),
	}

	return &em, nil
}

func (em *EmbeddingModel) Unload() {
	llama.Free(em.lctx)
	llama.ModelFree(em.model)
	llama.BackendFree()
}

func (em *EmbeddingModel) Embed(text string) ([]float32, error) {
	count := llama.Tokenize(em.vocab, text, nil, true, true)
	tokens := make([]llama.Token, count)
	llama.Tokenize(em.vocab, text, tokens, true, true)

	batch := llama.BatchGetOne(tokens)
	llama.Decode(em.lctx, batch)

	nEmbd := llama.ModelNEmbd(em.model)
	vec := llama.GetEmbeddingsSeq(em.lctx, 0, nEmbd)

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

func loadData(db *sql.DB, em *EmbeddingModel) error {
	type document struct {
		ID        int       `bson:"id"`
		Text      string    `bson:"text"`
		Embedding []float64 `bson:"embedding"`
	}

	data, err := os.ReadFile("zarf/data/book.chunks")
	if err != nil {
		return fmt.Errorf("read file: %w", err)
	}

	fmt.Print("\n")
	fmt.Print("\033[s")

	r := regexp.MustCompile(`<CHUNK>[\w\W]*?<\/CHUNK>`)
	chunks := r.FindAllString(string(data), -1)

	for counter, chunk := range chunks {
		fmt.Print("\033[u\033[K")
		fmt.Printf("Vectorizing Data: %d of %d", counter, len(chunks))

		chunk = strings.Trim(chunk, "<CHUNK>")
		chunk = strings.Trim(chunk, "</CHUNK>")

		// YOU WILL WANT TO KNOW HOW MANY TOKENS ARE CURRENTLY IN THE CHUNK
		// SO YOU DON'T EXCEED THE NUMBER OF TOKENS THE MODEL WILL USE TO
		// CREATE THE VECTOR EMBEDDING. THE MODEL WILL TRUNCATE YOUR CHUNK IF IT
		// EXCEEDS THE NUMBER OF TOKENS IT CAN USE TO CREATE THE VECTOR
		// EMBEDDING. THERE ARE MODELS THAT ONLY VECTORIZE AS LITTLE AS 512
		// TOKENS. THERE IS A TIKTOKEN PACKAGE IN FOUNDATION TO HELP YOU WITH
		// THIS.

		contextLimit := 1024

		vec, err := em.Embed(chunk[:min(len(chunk), contextLimit)])
		if err != nil {
			return err
		}

		chunk = strings.ReplaceAll(chunk, "'", "''")
		vecStr := strings.ReplaceAll(fmt.Sprintf("%v", vec), " ", ",")

		sql := fmt.Sprintf("INSERT INTO items (id, name, embedding) VALUES(%d, '%s', %v);", counter, chunk, vecStr)

		if _, err := db.Exec(sql); err != nil {
			return err
		}
	}

	fmt.Print("\n")

	return nil
}
