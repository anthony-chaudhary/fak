package projectassets

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// pi_settings.go — write Pi's SAFE auto-compaction threshold into settings.json. Pi decides
// when to compact with the predicate `contextTokens > contextWindow - reserveTokens`
// (docs/compaction.md), reading reserveTokens/keepRecentTokens from its settings file. Pi's
// own defaults (reserve 16384, keep 20000) are tuned for a ~128k window; on a small local
// window they leave the session no usable working set after a fire. fak writes an explicit
// compaction block DERIVED from the same PiContextBudget it writes into models.json, so the
// tripwire (contextWindow) and the post-fire floor (keepRecentTokens) agree.
//
// This file only ever touches the `compaction` object and preserves every other user key in
// settings.json, the same non-clobbering discipline EnsurePiProviderConfig uses for models.

// DefaultPiSettingsPath returns the canonical path to Pi's settings.json:
// $PI_CODING_AGENT_DIR/settings.json if set, else ~/.pi/agent/settings.json.
func DefaultPiSettingsPath() string {
	if envDir := os.Getenv("PI_CODING_AGENT_DIR"); envDir != "" {
		return filepath.Join(envDir, "settings.json")
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		home = "."
	}
	return filepath.Join(home, ".pi", "agent", "settings.json")
}

// ResolvePiSettingsPath resolves a destination path or directory for Pi's settings.json.
// If target is empty, it returns DefaultPiSettingsPath().
// If target ends with .json, it is treated as the file; otherwise it is a directory.
func ResolvePiSettingsPath(target string) string {
	target = strings.TrimSpace(target)
	if target == "" {
		return DefaultPiSettingsPath()
	}
	if strings.HasSuffix(strings.ToLower(target), ".json") {
		return target
	}
	return filepath.Join(target, "settings.json")
}

// EnsurePiSafeCompaction ensures the Pi settings file carries a compaction block consistent
// with budget: auto-compaction enabled, reserveTokens = the derived output reserve, and
// keepRecentTokens = the derived recent floor. Every unrelated user key is preserved. It
// returns (resolvedPath, modified, error). A missing file is created with just the block.
func EnsurePiSafeCompaction(target string, budget PiContextBudget) (string, bool, error) {
	path := ResolvePiSettingsPath(target)

	data, err := os.ReadFile(path)
	raw := map[string]interface{}{}
	switch {
	case err == nil:
		if unmarshalErr := json.Unmarshal(stripUTF8BOM(data), &raw); unmarshalErr != nil {
			return path, false, fmt.Errorf("parse existing %s: %w", path, unmarshalErr)
		}
		if raw == nil {
			raw = map[string]interface{}{}
		}
	case os.IsNotExist(err):
		if mkErr := os.MkdirAll(filepath.Dir(path), 0755); mkErr != nil {
			return path, false, fmt.Errorf("mkdir %s: %w", filepath.Dir(path), mkErr)
		}
	default:
		return path, false, fmt.Errorf("read %s: %w", path, err)
	}

	block, _ := raw["compaction"].(map[string]interface{})
	if block == nil {
		block = map[string]interface{}{}
	}

	modified := false
	if v, ok := block["enabled"].(bool); !ok || !v {
		block["enabled"] = true
		modified = true
	}
	// Only RAISE safety, never fight a stricter operator budget: write the derived reserve
	// when it is missing or when the existing reserve leaves less headroom than the derived
	// one (a smaller reserve means compaction fires later, closer to the ceiling).
	if cur, ok := numericField(block["reserveTokens"]); !ok || cur < budget.OutputReserve {
		block["reserveTokens"] = budget.OutputReserve
		modified = true
	}
	if cur, ok := numericField(block["keepRecentTokens"]); !ok || cur <= 0 {
		block["keepRecentTokens"] = budget.KeepRecentTokens
		modified = true
	}
	raw["compaction"] = block

	if !modified {
		return path, false, nil
	}
	out, marshalErr := json.MarshalIndent(raw, "", "  ")
	if marshalErr != nil {
		return path, false, fmt.Errorf("serialize %s: %w", path, marshalErr)
	}
	if writeErr := os.WriteFile(path, append(out, '\n'), 0644); writeErr != nil {
		return path, false, fmt.Errorf("write %s: %w", path, writeErr)
	}
	return path, true, nil
}
