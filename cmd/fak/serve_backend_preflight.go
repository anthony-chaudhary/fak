package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// serveBackendForwardPreflight is non-empty only when startup admitted a
// device-specific model forward that is useful to record in the serve log.
type serveBackendForwardPreflight struct {
	Backend string
	Forward fakmodel.ForwardPathKind
	Path    string
}

// serveGGUFHeaderOpener is injectable so the startup boundary can be witnessed
// against a header whose tensor payload reader is deliberately trapped.
type serveGGUFHeaderOpener func(string) (*ggufload.WeightSource, error)

func preflightServeBackendForward(path string, be compute.Backend) (serveBackendForwardPreflight, error) {
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		raw, err := os.ReadFile(filepath.Join(path, "config.json"))
		if err != nil {
			return serveBackendForwardPreflight{}, err
		}
		var cfg fakmodel.Config
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return serveBackendForwardPreflight{}, fmt.Errorf("safetensors config: %w", err)
		}
		if err := fakmodel.ValidateBackendForwardConfig(cfg, be); err != nil {
			return serveBackendForwardPreflight{}, err
		}
		if be == nil || !cfg.IsQwen35Hybrid() {
			return serveBackendForwardPreflight{}, nil
		}
		gdn := be.(fakmodel.Qwen35GDNBackend)
		return serveBackendForwardPreflight{Backend: be.Name(), Forward: fakmodel.ForwardQwen35GDN, Path: gdn.Qwen35GDNPath()}, nil
	}
	return preflightServeBackendForwardWith(path, be, ggufload.OpenWeights)
}

// preflightServeBackendForwardWith reads only the GGUF header and tensor directory,
// derives the architecture config, closes the source, and then validates the selected
// backend. It deliberately does not call BuildModelPreflight: memory planning belongs
// to the later load stage and must not precede this forward-path refusal.
func preflightServeBackendForwardWith(
	path string,
	be compute.Backend,
	open serveGGUFHeaderOpener,
) (serveBackendForwardPreflight, error) {
	if path == "" {
		return serveBackendForwardPreflight{}, nil
	}
	ws, err := open(path)
	if err != nil {
		return serveBackendForwardPreflight{}, err
	}
	cfg, configErr := ws.File.Config()
	closeErr := ws.Close()
	if configErr != nil {
		return serveBackendForwardPreflight{}, configErr
	}
	if closeErr != nil {
		return serveBackendForwardPreflight{}, fmt.Errorf("gguf: close header-only weight source: %w", closeErr)
	}

	if err := fakmodel.ValidateBackendForwardConfig(cfg, be); err != nil {
		return serveBackendForwardPreflight{}, err
	}
	if be == nil || !cfg.IsQwen35Hybrid() {
		return serveBackendForwardPreflight{}, nil
	}
	gdn := be.(fakmodel.Qwen35GDNBackend) // validated structurally, including the exact path above
	return serveBackendForwardPreflight{
		Backend: be.Name(),
		Forward: fakmodel.ForwardQwen35GDN,
		Path:    gdn.Qwen35GDNPath(),
	}, nil
}

func serveBackendForwardPreflightMessage(result serveBackendForwardPreflight) gateway.StartupMessage {
	if result.Backend == "" || result.Forward == "" || result.Path == "" {
		return gateway.StartupMessage{}
	}
	return newServeStartupMessage("model-load", "backend-forward", "info", fmt.Sprintf(
		"backend=%s forward=%s path=%s", result.Backend, result.Forward, result.Path))
}
