package llamacpp

import (
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/hybridgroup/yzma/pkg/download"
)

func InstallLibraries(libPath string) error {
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

func InstallModel(modelURL string, modelPath string) (string, error) {
	u, err := url.Parse(modelURL)
	if err != nil {
		return "", fmt.Errorf("unable to parse modelURL: %w", err)
	}

	modelPath = strings.TrimSuffix(modelPath, "/")
	localPath := fmt.Sprintf("%s/%s", modelPath, path.Base(u.Path))

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
