// Package benchcli holds the small, identical helpers the benchmark-CLI mains
// (cmd/*bench and the demo/cert commands beside them) had each copy-pasted into
// their own file. Two functions covered the bulk of the duplication: reading a
// Hugging Face config.json into a model.Config, and writing a byte slice to a
// path while creating its parent directory. Both are pure, resource-free, and
// behaviour-identical across the callers, so one shared copy retires the clone
// family without changing any caller's observable output (#983, child of #775).
package benchcli

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// ReadHFConfig loads the Hugging Face config.json from dir into a model.Config.
// HF Llama/Qwen2 configs omit head_dim (it is hidden_size/num_attention_heads),
// so it is derived when absent — matching what every bench main did by hand.
func ReadHFConfig(dir string) (model.Config, error) {
	var cfg model.Config
	cb, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return cfg, fmt.Errorf("config.json: %w", err)
	}
	if err := json.Unmarshal(cb, &cfg); err != nil {
		return cfg, fmt.Errorf("config.json parse: %w", err)
	}
	if cfg.HeadDim == 0 && cfg.NumHeads != 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	return cfg, nil
}

// WriteFile writes b to path, first creating path's parent directory tree. A
// bare filename (Dir is "." or "") needs no mkdir, so that no-op is skipped.
func WriteFile(path string, b []byte) error {
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(path, b, 0o644)
}
