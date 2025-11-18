// Package install provides helper functions to install llamacpp libraries
// and models.
package install

import (
	"fmt"
	"net/url"
	"path"
	"strings"

	"github.com/ardanlabs/llamacpp"
	"github.com/hybridgroup/yzma/pkg/download"
)

func LlamaCPP(libPath string, processor download.Processor, allowUpgrade bool) error {
	fmt.Print("- check llamacpp installation: ")

	if err := llamacpp.InstallLlama(libPath, processor, allowUpgrade); err != nil {
		fmt.Println("X")
		return fmt.Errorf("unable to install llamacpp: %w", err)
	}

	fmt.Println("✓")

	return nil
}

func Model(modelURL string, modelPath string) (string, error) {
	u, _ := url.Parse(modelURL)
	filename := path.Base(u.Path)
	name := strings.TrimSuffix(filename, path.Ext(filename))
	fmt.Printf("- check %q installation: ", name)

	localPath, err := llamacpp.InstallModel(modelURL, modelPath)
	if err != nil {
		fmt.Println("X")
		return "", fmt.Errorf("unable to download model: %w", err)
	}

	fmt.Println("✓")

	return localPath, nil
}
