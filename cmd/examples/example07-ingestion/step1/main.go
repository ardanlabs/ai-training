// This example shows you how to load the pre-chunked Ultimate Go Notebook into
// memory. The program reads `zarf/data/book.chunks`, extracts each
// `<CHUNK>...</CHUNK>` block, prints how many chunks were found, and shows a
// preview of the first chunk. No LLM and no database are involved yet.
//
// # Running the example
//
//	$ make example07-step1
//
//	First step — establishes the input corpus we'll embed and store in later steps.

// Example 07 — Step 1 — Load Chunks
package main

import (
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
)

const (
	chunksPath = "zarf/data/book.chunks"
)

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	// -------------------------------------------------------------------------
	// Load the pre-chunked book from disk.
	// region Load the pre-chunked book from disk.

	fmt.Println("\nLoading Chunks")

	chunks, err := loadChunks(chunksPath)
	if err != nil {
		return fmt.Errorf("loadChunks: %w", err)
	}

	fmt.Printf("Loaded %d chunks from %s\n", len(chunks), chunksPath)

	// endregion

	// -------------------------------------------------------------------------
	// Preview the first chunk so we can see what we're working with.
	// region Preview the first chunk so we can see what we're working with.

	if len(chunks) > 0 {
		preview := chunks[0]
		if len(preview) > 200 {
			preview = preview[:200] + "..."
		}

		fmt.Println("\nFirst chunk preview:")
		fmt.Println(preview)
	}

	// endregion

	return nil
}

func loadChunks(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	r := regexp.MustCompile(`<CHUNK>[\w\W]*?<\/CHUNK>`)
	raw := r.FindAllString(string(data), -1)

	chunks := make([]string, len(raw))
	for i, c := range raw {
		c = strings.TrimPrefix(c, "<CHUNK>")
		c = strings.TrimSuffix(c, "</CHUNK>")
		chunks[i] = c
	}

	return chunks, nil
}
