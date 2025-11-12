package main

import (
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
)

func installModel(modelURL string) (string, error) {
	u, _ := url.Parse(modelURL)
	localPath := fmt.Sprintf("zarf/models/%s", path.Base(u.Path))

	if _, err := os.Stat(localPath); !os.IsNotExist(err) {
		fmt.Println("- model file already installed at", localPath)
		return localPath, nil
	}

	r, err := http.Get(modelURL)
	if err != nil {
		return "", fmt.Errorf("error requesting model file: %w", err)
	}
	defer r.Body.Close()

	f, err := os.Create(localPath)
	if err != nil {
		return "", fmt.Errorf("error creating model file: %w", err)
	}
	defer f.Close()

	_, err = io.Copy(f, r.Body)
	if err != nil {
		return "", fmt.Errorf("error downloading model file: %w", err)
	}

	return localPath, nil
}
