/*
	This code was taken from: https://github.com/shapor/tiktoken-go
	I have removed any code that we didn't need and tried to clean up the
	syntax where possible.
*/

// Package tiktoken provides support for GPT-3+ token counting.
package tiktoken

import (
	_ "embed"
	"fmt"
)

type Tiktoken struct {
	bpe *coreBPE
}

func NewTiktoken() (*Tiktoken, error) {
	bpe, err := newCoreBPE()
	if err != nil {
		return nil, fmt.Errorf("new core bpe: %w", err)
	}

	tt := Tiktoken{
		bpe: bpe,
	}

	return &tt, nil
}

func (t *Tiktoken) TokenCount(text string) int {
	tokens, _ := t.bpe.encodeNative(text)
	return len(tokens)
}
