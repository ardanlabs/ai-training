package ai

import (
	"fmt"
	"strings"
)

// ParseBoardText takes a similar board and breaks it up in a key/value pair.
func ParseBoardText(board SimilarBoard) map[string]string {
	m := make(map[string]string)
	var prefix string

	parts := strings.Split(board.Text, "\n")
	for _, part := range parts {
		keyValue := strings.Split(part, ":")
		if len(keyValue) == 1 {
			prefix = strings.Trim(keyValue[0], "- ") + "-"
			continue
		}
		key := fmt.Sprintf("%s%s", prefix, strings.Trim(keyValue[0], " "))
		m[key] = strings.Trim(keyValue[1], " ()")
	}

	return m
}
