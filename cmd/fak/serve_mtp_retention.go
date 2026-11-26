package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// serveMTPConsumingModelTypes are the model_type values whose MTP route consumes the nextn head.
var serveMTPConsumingModelTypes = map[string]bool{"qwen35": true, "qwen35moe": true}

// serveMTPArchConsumes reports whether a checkpoint's model_type has an MTP route that reads the head.
// HF directories spell the family qwen3_5 / qwen3_5_moe (and *_text for the text config).
func serveMTPArchConsumes(modelType string) bool {
	modelType = strings.ToLower(strings.TrimSpace(modelType))
	return serveMTPConsumingModelTypes[modelType] || strings.HasPrefix(modelType, "qwen3_5")
}

// serveMTPPeekModelType reads a checkpoint's model_type without loading weights: the GGUF header
// for a file, config.json for a directory. An unreadable or unknown path yields "", which never
// retains the head.
func serveMTPPeekModelType(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	info, err := os.Stat(path)
	if err != nil {
		return ""
	}
	if info.IsDir() {
		raw, err := os.ReadFile(filepath.Join(path, "config.json"))
		if err != nil {
			return ""
		}
		var cfg struct {
			ModelType  string `json:"model_type"`
			TextConfig struct {
				ModelType string `json:"model_type"`
			} `json:"text_config"`
		}
		if json.Unmarshal(raw, &cfg) != nil {
			return ""
		}
		if cfg.ModelType == "" || (!serveMTPArchConsumes(cfg.ModelType) && serveMTPArchConsumes(cfg.TextConfig.ModelType)) {
			return cfg.TextConfig.ModelType
		}
		return cfg.ModelType
	}
	gg, err := ggufload.Open(path)
	if err != nil {
		return ""
	}
	cfg, err := gg.Config()
	if err != nil {
		return ""
	}
	if cfg.IsQwen35Hybrid() {
		return "qwen35"
	}
	return cfg.ModelType
}

// serveRetainMTPForModel is the default MTP retention decision: the resolved route and backend
// must consume the head, and the checkpoint must be a consuming family unless the operator forced
// the route through FAK_SPECULATIVE/SPECULATIVE.
func serveRetainMTPForModel(resolvedMode string, metal bool, backendName, modelType string, envForced bool) bool {
	if !serveRetainMTPHead(resolvedMode, metal, backendName) {
		return false
	}
	return envForced || serveMTPArchConsumes(modelType)
}

// serveMTPEnvForced reports whether the operator named the speculative route in the environment.
func serveMTPEnvForced(getenv func(string) string) bool {
	return strings.TrimSpace(getenv("FAK_SPECULATIVE")) != "" || strings.TrimSpace(getenv("SPECULATIVE")) != ""
}

// serveMTPRetentionError reports whether a load error was caused by the retained head. Other
// failures (fit, OOM, I/O) must not drop the head: a smaller fit-commit layout may still carry it.
func serveMTPRetentionError(err error) bool {
	if err == nil {
		return false
	}
	var incomplete *fakmodel.Qwen35MTPIncompleteError
	if errors.As(err, &incomplete) {
		return true
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "mtp") || strings.Contains(msg, "nextn")
}

// serveMTPRetentionFallback reports whether a failed load may be retried once without the MTP
// head. Only a default-retained load refused by the head qualifies; an operator-forced route
// stays fail-loud.
func serveMTPRetentionFallback(err error, retained, envForced bool) bool {
	return retained && !envForced && serveMTPRetentionError(err)
}

// serveNoteMTPRetentionFallback arms the one-shot retry when the load is refused under default
// retention. It reports the refusal, drops retention for the rest of the process, and returns
// true so the caller reloads once.
func serveNoteMTPRetentionFallback(err error, getenv func(string) string) bool {
	if !serveMTPRetentionFallback(err, fakmodel.RetainMTP, serveMTPEnvForced(getenv)) {
		return false
	}
	fmt.Fprintf(os.Stderr, "fak serve: load with the MTP head retained failed (%v); retrying once without the head (MTP speculation unavailable for this model)\n", err)
	fakmodel.SetRetainMTP(false)
	return true
}
