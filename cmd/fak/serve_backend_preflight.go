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
	"github.com/anthony-chaudhary/fak/internal/modelreg"
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

// serveLocalRuntimeLauncher is the narrow handoff from pure admission to the
// later harness-owned runtime lifecycle. It is invoked only for an admitted,
// fully evidenced resource plan.
type serveLocalRuntimeLauncher func(modelreg.LocalLaunchResourceReservation) error

// preflightServeLocalRuntime is the declaration-aware admission seam adjacent
// to the existing backend-forward preflight. Legacy fak-native serve startup
// continues to use preflightServeBackendForward unchanged; a harness-owned
// local-runtime launcher enters here and therefore cannot run before artifact,
// runtime, device, and resource facts have admitted it.
func preflightServeLocalRuntime(req modelreg.LocalAdmissionRequest, launch serveLocalRuntimeLauncher) (modelreg.LocalAdmissionDecision, error) {
	decision := modelreg.EvaluateLocalAdmission(req)
	if err := decision.RefusalError(); err != nil {
		return decision, err
	}
	if launch == nil {
		return decision, nil
	}
	if err := launch(*decision.Plan); err != nil {
		return decision, fmt.Errorf("serve local runtime launch: %w", err)
	}
	return decision, nil
}

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
		return validatedBackendForwardPreflight(cfg, be)
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

	return validatedBackendForwardPreflight(cfg, be)
}

func validatedBackendForwardPreflight(cfg fakmodel.Config, be compute.Backend) (serveBackendForwardPreflight, error) {
	if err := fakmodel.ValidateBackendForwardConfig(cfg, be); err != nil {
		return serveBackendForwardPreflight{}, err
	}
	return qwen35BackendForwardPreflight(cfg, be), nil
}

func qwen35BackendForwardPreflight(cfg fakmodel.Config, be compute.Backend) serveBackendForwardPreflight {
	if be == nil || !cfg.IsQwen35Hybrid() {
		return serveBackendForwardPreflight{}
	}
	gdn := be.(fakmodel.Qwen35GDNBackend) // validated structurally before this helper is called
	return serveBackendForwardPreflight{
		Backend: be.Name(),
		Forward: fakmodel.ForwardQwen35GDN,
		Path:    gdn.Qwen35GDNPath(),
	}
}

func serveBackendForwardPreflightMessage(result serveBackendForwardPreflight) gateway.StartupMessage {
	if result.Backend == "" || result.Forward == "" || result.Path == "" {
		return gateway.StartupMessage{}
	}
	return newServeStartupMessage("model-load", "backend-forward", "info", fmt.Sprintf(
		"backend=%s forward=%s path=%s", result.Backend, result.Forward, result.Path))
}
