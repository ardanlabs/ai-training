package tiktoken

import (
	"fmt"

	"github.com/dlclark/regexp2"
)

type coreBPE struct {
	encoder map[string]int
	tlRegex *regexp2.Regexp
}

func newCoreBPE() (*coreBPE, error) {
	enc, err := cl100kBaseEncoding()
	if err != nil {
		return nil, fmt.Errorf("error loading base encoding model: %w", err)
	}

	regex, err := regexp2.Compile(enc.PatStr, regexp2.None)
	if err != nil {
		return nil, fmt.Errorf("error compiling regex: %w", err)
	}

	bp := coreBPE{
		encoder: enc.MergeableRanks,
		tlRegex: regex,
	}

	return &bp, nil
}

func (bp *coreBPE) encodeNative(text string) ([]int, int) {
	regex := bp.tlRegex
	ret := []int{}
	lastPieceTokenLen := 0
	textRunes := []rune(text)

	start := 0
	end := len([]rune(text))

	for _, mat := range findRegex2AllStringMatchIndex(cutRunes(textRunes, start, end), regex) {
		piece := cutRunes(textRunes, start+mat[0], start+mat[1])
		if token, ok := bp.encoder[piece]; ok {
			lastPieceTokenLen = 1
			ret = append(ret, token)
			continue
		}

		tokens := bytePairEncode([]byte(piece), bp.encoder)
		lastPieceTokenLen = len(tokens)
		ret = append(ret, tokens...)
	}

	return ret, lastPieceTokenLen
}

func findRegex2AllStringMatchIndex(text string, reg *regexp2.Regexp) [][]int {
	var matches [][]int

	m, _ := reg.FindStringMatch(text)

	for m != nil {
		result := make([]int, 2)
		result[0] = m.Index
		result[1] = m.Index + m.Length
		matches = append(matches, result)
		m, _ = reg.FindNextMatch(m)
	}

	return matches
}

func cutRunes(runes []rune, start, end int) string {
	if start < 0 {
		start = 0
	}

	if end > len(runes) {
		end = len(runes)
	}

	return string(runes[start:end])
}
