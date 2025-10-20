package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/hybridgroup/yzma/pkg/download"
)

func installLlamaCPP() error {
	libPath := os.Getenv("YZMA_LIB")

	if _, err := os.Stat(filepath.Join(libPath, download.LibraryName(runtime.GOOS))); !os.IsNotExist(err) {
		fmt.Println("- llama.cpp already installed at", libPath)
		return nil
	}

	version, err := download.LlamaLatestVersion()
	if err != nil {
		return fmt.Errorf("error install: %w", err)
	}

	fmt.Println("installing llama.cpp version", version, "to", libPath)
	download.Get(runtime.GOOS, "cpu", version, libPath)

	return nil
}
