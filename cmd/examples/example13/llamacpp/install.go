package llamacpp

import (
	"fmt"
	"net/url"
	"path"
	"path/filepath"

	"github.com/hybridgroup/yzma/pkg/download"
)

func InstallModel(modelURL string, modelPath string) (string, error) {
	u, err := url.Parse(modelURL)
	if err != nil {
		return "", fmt.Errorf("unable to parse modelURL: %w", err)
	}

	if err := download.GetModel(modelURL, modelPath); err != nil {
		return "", fmt.Errorf("unable to download model: %w", err)
	}

	localPath := filepath.Join(modelPath, path.Base(u.Path))

	return localPath, nil
}
