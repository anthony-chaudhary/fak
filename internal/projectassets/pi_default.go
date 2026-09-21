package projectassets

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// pi_default.go — pin the Pi harness DEFAULT onto the fak router.
//
// pi_config.go writes the `fak` PROVIDER (models.json: endpoint + model catalog) and
// pi_settings.go writes the SAFE COMPACTION envelope (settings.json). Neither decides
// what a bare `pi` invocation actually uses: Pi resolves its default provider/model from
// the top-level `defaultProvider` / `defaultModel` keys in settings.json. Without this
// file those keys stay wherever the machine was last pointed (on the operator box,
// `hive-ai` + a hosted DeepSeek id), so `fak pi`'s models.json work is invisible to a
// plain `pi` launch — the router is configured but NOT the default.
//
// This file is the third, smallest leg: it sets exactly two keys and preserves every
// other user key, the same non-clobbering discipline EnsurePiProviderConfig (models) and
// EnsurePiSafeCompaction (compaction) already use.

// EnsurePiDefaultProviderModel points Pi's default provider/model at the fak router.
//
// It sets settings.json `defaultProvider` = providerID and `defaultModel` = modelID,
// creating the file when missing and preserving every unrelated key. A value that is
// already correct is left untouched (and reports modified=false), so `fak pi` is
// idempotent and never rewrites a file it does not need to.
//
// providerID/modelID are normalized the same way models.json is (NormalizePiModelID),
// so the default cannot drift from the id the `fak` provider actually advertises.
// Returns (resolvedPath, modified, error).
func EnsurePiDefaultProviderModel(target, providerID, modelID string) (string, bool, error) {
	path := ResolvePiSettingsPath(target)

	providerID = strings.TrimSpace(providerID)
	if providerID == "" {
		providerID = DefaultPiProviderID
	}
	modelID = NormalizePiModelID(modelID)

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

	modified := false
	if cur, _ := raw["defaultProvider"].(string); strings.TrimSpace(cur) != providerID {
		raw["defaultProvider"] = providerID
		modified = true
	}
	if cur, _ := raw["defaultModel"].(string); strings.TrimSpace(cur) != modelID {
		raw["defaultModel"] = modelID
		modified = true
	}

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
