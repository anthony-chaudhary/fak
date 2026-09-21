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

// PiSettingsDefaultModel reads the top-level `defaultModel` from Pi's settings.json and
// returns it trimmed. An absent file, absent key, non-string value, or blank value all
// report "" (no deliberate default configured) with a nil error; only a read or parse
// failure other than not-exist is an error.
//
// This is the read half of the auto-detect guard: a launcher MUST NOT overwrite a
// deliberately configured defaultModel with a `/healthz`-detected id, so the caller asks
// whether a default is already configured before adopting a detected model.
func PiSettingsDefaultModel(target string) (string, error) {
	path := ResolvePiSettingsPath(target)
	data, err := os.ReadFile(path)
	switch {
	case os.IsNotExist(err):
		return "", nil
	case err != nil:
		return "", fmt.Errorf("read %s: %w", path, err)
	}
	raw := map[string]interface{}{}
	if unmarshalErr := json.Unmarshal(stripUTF8BOM(data), &raw); unmarshalErr != nil {
		return "", fmt.Errorf("parse existing %s: %w", path, unmarshalErr)
	}
	cur, _ := raw["defaultModel"].(string)
	return strings.TrimSpace(cur), nil
}

// ShouldAdoptDetectedPiModel reports whether a `/healthz`-detected model id may become the
// launch model. It is a pure predicate over PiSettingsDefaultModel, so precedence is
// unit-testable without a live backend. It returns:
//
//   - false when detectedModel is blank or the non-model sentinel "mock";
//   - false when explicitModel is non-blank (an operator --model always wins; the caller
//     applies the flag directly, so this predicate refuses to override it);
//   - false when settings.json already carries a deliberate, non-blank defaultModel;
//   - true otherwise (no default configured: auto-detect seeds the first-ever value).
//
// A read/parse error degrades conservatively to false: a settings file we cannot read is
// not license to clobber whatever default it may hold.
func ShouldAdoptDetectedPiModel(target, explicitModel, detectedModel string) bool {
	if strings.TrimSpace(explicitModel) != "" {
		return false
	}
	detectedModel = strings.TrimSpace(detectedModel)
	if detectedModel == "" || detectedModel == "mock" {
		return false
	}
	existing, err := PiSettingsDefaultModel(target)
	if err != nil {
		return false
	}
	return existing == ""
}

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
