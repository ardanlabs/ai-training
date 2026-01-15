package tiktoken

import (
	"bytes"
	_ "embed"
	"encoding/gob"
	"fmt"
)

//go:embed cl100k.gob
var cl100k []byte

// -----------------------------------------------------------------------------

type encoding struct {
	Name           string
	PatStr         string
	MergeableRanks map[string]int
	SpecialTokens  map[string]int
}

func cl100kBaseEncoding() (*encoding, error) {
	const (
		endOfText   string = "<|endoftext|>"
		fimPrefix   string = "<|fim_prefix|>"
		fimMiddle   string = "<|fim_middle|>"
		fimSuffix   string = "<|fim_suffix|>"
		endOfPrompt string = "<|endofprompt|>"
	)

	const modelCl100KBase string = "cl100k_base"

	specialTokens := map[string]int{
		endOfText:   100257,
		fimPrefix:   100258,
		fimMiddle:   100259,
		fimSuffix:   100260,
		endOfPrompt: 100276,
	}

	var vocabCL100K map[string]int
	if err := gob.NewDecoder(bytes.NewReader(cl100k)).Decode(&vocabCL100K); err != nil {
		return nil, fmt.Errorf("decoding: %w", err)
	}

	enc := encoding{
		Name:           modelCl100KBase,
		PatStr:         `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
		MergeableRanks: vocabCL100K,
		SpecialTokens:  specialTokens,
	}

	return &enc, nil
}
