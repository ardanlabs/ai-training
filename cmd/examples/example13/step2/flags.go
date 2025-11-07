package main

import (
	"flag"
	"fmt"
)

var (
	systemPrompt *string
	verbose      *bool
	temperature  *float64
	topK         *int
	topP         *float64
	minP         *float64
	contextSize  *int
)

func showUsage() {
	fmt.Println(`
Usage:
vlm -model [model file path] -mmproj [projector file path] -lib [llama.cpp .so file path] -p [what you want to ask] -image [image file path]`)
}

func handleFlags() error {
	systemPrompt = flag.String("sys", "", "system prompt")
	verbose = flag.Bool("v", false, "verbose logging")
	temperature = flag.Float64("temp", 0.8, "temperature for model")
	topK = flag.Int("top-k", 40, "top-k for model")
	minP = flag.Float64("min-p", 0.1, "min-p for model")
	topP = flag.Float64("top-p", 0.9, "top-p for model")
	contextSize = flag.Int("c", 4096, "context size for model")

	flag.Parse()

	return nil
}
