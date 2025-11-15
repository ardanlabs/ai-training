package llamacpp

import (
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
)

func InstallModel(modelURL string, modelPath string) (string, error) {
	u, err := url.Parse(modelURL)
	if err != nil {
		return "", fmt.Errorf("unable to parse modelURL: %w", err)
	}

	localPath := filepath.Join(modelPath, path.Base(u.Path))

	if _, err := os.Stat(localPath); !os.IsNotExist(err) {
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
