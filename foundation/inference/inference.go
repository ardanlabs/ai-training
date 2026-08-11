// Package inference boots a model that runs inside this process.
//
// Every example that uses the Kronk SDK directly, instead of talking to a
// Kronk service over HTTP, has to do the same two things first: download the
// llama.cpp libraries and the model weights, then initialize the runtime and
// load the model. That is machine setup, not a lesson, so it lives here and
// each example opens on the thing it actually teaches.
package inference

import (
	"context"
	"fmt"

	"github.com/ardanlabs/kronk/sdk/kronk"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
	"github.com/ardanlabs/kronk/sdk/tools/defaults"
	"github.com/ardanlabs/kronk/sdk/tools/libs"
	"github.com/ardanlabs/kronk/sdk/tools/models"
)

// Download installs the llama.cpp libraries and then downloads each model
// source, returning one path per source in the order they were given. The
// first run of a given model is slow because it fetches the weights; after
// that everything is already on disk and this returns immediately.
func Download(ctx context.Context, sources ...string) ([]models.Path, error) {
	lbs, err := libs.New(
		libs.WithVersion(defaults.LibVersion("")),
	)
	if err != nil {
		return nil, fmt.Errorf("unable to create libs api: %w", err)
	}

	if _, err := lbs.Download(ctx, kronk.FmtLogger); err != nil {
		return nil, fmt.Errorf("unable to install llama.cpp: %w", err)
	}

	mdls, err := models.New()
	if err != nil {
		return nil, fmt.Errorf("unable to create models api: %w", err)
	}

	paths := make([]models.Path, 0, len(sources))

	for _, source := range sources {
		mp, err := mdls.Download(ctx, kronk.FmtLogger, source)
		if err != nil {
			return nil, fmt.Errorf("unable to install model %q: %w", source, err)
		}

		paths = append(paths, mp)
	}

	return paths, nil
}

// Open initializes the runtime, loads the model at mp, and prints what the
// loaded model can do. A vision model carries a projector file alongside its
// weights; when mp has one it is loaded too, which is the only difference
// between opening a vision model and a text model.
func Open(mp models.Path) (*kronk.Kronk, error) {
	fmt.Println("loading model...")

	if err := kronk.Init(); err != nil {
		return nil, fmt.Errorf("unable to init kronk: %w", err)
	}

	opts := []model.Option{
		model.WithModelFiles(mp.ModelFiles),
	}

	if mp.ProjFile != "" {
		opts = append(opts, model.WithProjFile(mp.ProjFile))
	}

	krn, err := kronk.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("unable to create inference model: %w", err)
	}

	PrintInfo(krn)

	return krn, nil
}

// PrintInfo reports what the loaded model is and what it supports. Open calls
// it, so most callers never need to.
func PrintInfo(krn *kronk.Kronk) {
	fmt.Print("- system info:\n\t")
	for k, v := range krn.SystemInfo() {
		fmt.Printf("%s:%v, ", k, v)
	}
	fmt.Println()

	cfg := krn.ModelConfig()
	info := krn.ModelInfo()

	fmt.Println("- contextWindow:", cfg.ContextWindow())
	fmt.Printf("- k/v          : %s/%s\n", cfg.CacheTypeK, cfg.CacheTypeV)
	fmt.Println("- nBatch       :", cfg.NBatch())
	fmt.Println("- nuBatch      :", cfg.NUBatch())
	fmt.Println("- embeddings   :", info.IsEmbedModel)
	fmt.Println("- template     :", info.Template.FileName)
	fmt.Println("- grammar      :", cfg.DefaultParams.Grammar != "")
}
