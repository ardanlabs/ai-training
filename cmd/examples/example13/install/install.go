// Package install provides helper functions to install llamacpp libraries
// and models.
package install

import (
	"fmt"
	"net/url"
	"os"
	"path"
	"path/filepath"

	"github.com/hybridgroup/yzma/pkg/download"
)

func LlamaCPP(libPath string, processor download.Processor, allowUpgrade bool) error {
	fmt.Print("- check llamacpp installation: ")

	if err := download.InstallLibraries(libPath, processor, allowUpgrade); err != nil {
		file := filepath.Join(libPath, "libmtmd.dylib")
		if _, err := os.Stat(file); !os.IsNotExist(err) {
			fmt.Println("✓")
			return nil
		}

		fmt.Println("X")
		return fmt.Errorf("unable to install llamacpp: %w", err)
	}

	fmt.Println("✓")

	return nil
}

func Model(modelURL string, modelPath string) (string, error) {
	u, err := url.Parse(modelURL)
	if err != nil {
		return "", fmt.Errorf("unable to parse modelURL: %w", err)
	}

	localPath := filepath.Join(modelPath, path.Base(u.Path))

	fmt.Printf("- check %q installation: ", localPath)
	if _, err := os.Stat(localPath); !os.IsNotExist(err) {
		fmt.Println("✓")
		return localPath, nil
	}

	if err := download.GetModel(modelURL, modelPath); err != nil {
		fmt.Println("X")
		return "", fmt.Errorf("unable to download model: %w", err)
	}

	fmt.Println("✓")

	return localPath, nil
}
