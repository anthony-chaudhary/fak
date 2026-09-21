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

// resolvePiWindowRow is lookupPiModelWindow without the value: it reports whether the
// registry names this model id (exact id, alias, or family glob).
func resolvePiWindowRow(modelID string) bool {
	_, ok := lookupPiModelWindow(modelID)
	return ok
}

// IsPiKnownModel reports whether fak has a per-model entry for this id in the served-window
// registry (see pi_model_windows.go). A registered id is a model fak knows how to route and
// size, so it is a plausible deliberate default even when the backend currently answering the
// liveness probe does not advertise it (a routing-mode router only lists the models whose
// backends are up, and /healthz names just the local engine).
//
// This is the companion to IsPiModelAdvertised: advertised-OR-known is the test for
// "a deliberate default", while neither is the test for "a stale placeholder" that the
// launcher is allowed to replace. Matching tolerates case, surrounding space, and a
// provider-qualified form, exactly like the registry itself.
func IsPiKnownModel(modelID string) bool {
	return resolvePiWindowRow(modelID)
}

// PiDefaultModelIsDeliberate reports whether an existing settings.json defaultModel must be
// honored rather than replaced by a backend-detected id. It is true when the id is one fak's
// per-model registry knows OR one the backend actually advertises; false only for an id that
// is neither AND that a live catalog positively omits (a stale placeholder such as
// `custom-model`, which would otherwise be re-confirmed forever).
//
// An EMPTY advertised set returns true for an unregistered id: no catalog observed is not
// evidence against the id, and the launcher must never clobber a default on absence of
// evidence. The replacement path therefore requires a live catalog that omits the id, so a
// backend that is merely down cannot rewrite the operator's pin.
func PiDefaultModelIsDeliberate(modelID string, advertised []string) bool {
	if IsPiKnownModel(modelID) {
		return true
	}
	return IsPiModelAdvertised(modelID, advertised)
}

// IsPiModelAdvertised reports whether a configured Pi defaultModel is one of the ids a
// backend actually advertises on its /v1/models catalog.
//
// This is the guard against a SELF-PERPETUATING bad default. Once a placeholder id (the
// `custom-model` a test or a stale tool wrote) lands in settings.json, the launcher's
// "honor an existing deliberate defaultModel" rule would re-confirm it on every run and
// even write it into models.json, so the wrong model is never corrected — the harness
// resolves provider `fak` (the router) with a model the router does not serve, which
// reads as "the default keeps slipping back to the router". An advertised catalog is the
// external witness that distinguishes a real operator pin from a stale placeholder: a
// model the backend does not list cannot be a deliberate choice for THAT backend.
//
// IsPiModelAdvertised treats an empty advertised set as true (nothing to check against —
// never clobber from absence of evidence). Matching tolerates case, surrounding space, and a
// provider-qualified form on either side, so `provider/model` and `model` agree.
func IsPiModelAdvertised(modelID string, advertised []string) bool {
	modelID = strings.TrimSpace(modelID)
	if modelID == "" {
		return false
	}
	if len(advertised) == 0 {
		return true
	}
	want := modelKeyCandidates(modelID)
	for _, id := range advertised {
		id = strings.TrimSpace(id)
		if id == "" {
			continue
		}
		for have := range modelKeyCandidates(id) {
			if want[have] {
				return true
			}
		}
	}
	return false
}

// modelKeyCandidates returns the comparison keys for a model id: the normalized whole id
// and its last path segment (the bare model name of a provider-qualified id), so
// `deepseek-ai/DeepSeek-V4.1-Flash` and `DeepSeek-V4.1-Flash` compare equal.
func modelKeyCandidates(id string) map[string]bool {
	keys := map[string]bool{}
	full := normalizePiModelKey(id)
	if full == "" {
		return keys
	}
	keys[full] = true
	if leaf := lastPathSegment(full); leaf != "" {
		keys[leaf] = true
	}
	return keys
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
//
// Callers that sourced modelID from a backend's /healthz label rather than from
// deliberate operator intent must use EnsurePiDefaultProviderModelIfAbsent instead: on
// a routing-mode router /healthz names the local planner engine, not the routed model
// set, so adopting it as the default would clobber the operator's chosen route.
//
// Returns (resolvedPath, modified, error).
func EnsurePiDefaultProviderModel(target, providerID, modelID string) (string, bool, error) {
	return writePiDefaultProviderModel(target, providerID, modelID, false)
}

// EnsurePiDefaultProviderModelIfAbsent is the seed-only variant of
// EnsurePiDefaultProviderModel: it writes `defaultProvider`/`defaultModel` only when
// settings.json has no non-empty `defaultModel` yet, and leaves a deliberate default
// untouched otherwise.
//
// This is the correct writer for a launcher whose model id was auto-detected from a
// backend liveness surface. /healthz describes the engine the backend happens to be
// running; it is NOT a statement about which model the operator wants a bare `pi` to
// default to. A routing-mode router fronts N models (with ordered fallbacks), so
// pinning its local engine label as the harness default makes the configured routing
// ladder unreachable in practice. Seeding when absent keeps a first-ever config
// working without ever overwriting operator intent.
//
// When a default already exists (or when --model was passed explicitly, which flows
// through EnsurePiDefaultProviderModel), this reports modified=false.
// Returns (resolvedPath, modified, error).
func EnsurePiDefaultProviderModelIfAbsent(target, providerID, modelID string) (string, bool, error) {
	return writePiDefaultProviderModel(target, providerID, modelID, true)
}

// writePiDefaultProviderModel is the shared body for the authoritative and seed-only
// default writers. When seedOnly is true an existing non-empty `defaultModel` is left
// in place (provider is still set, so a bare launch reaches the fak router) and the
// function reports modified=false.
func writePiDefaultProviderModel(target, providerID, modelID string, seedOnly bool) (string, bool, error) {
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

	curModel, _ := raw["defaultModel"].(string)
	if seedOnly && strings.TrimSpace(curModel) != "" {
		// A deliberate default is already pinned; never overwrite it with a detected id.
		if !modified {
			return path, false, nil
		}
		return writePiSettings(path, raw)
	}
	if strings.TrimSpace(curModel) != modelID {
		raw["defaultModel"] = modelID
		modified = true
	}

	if !modified {
		return path, false, nil
	}
	return writePiSettings(path, raw)
}

// writePiSettings marshals raw and writes it back to path with a trailing newline.
func writePiSettings(path string, raw map[string]interface{}) (string, bool, error) {
	out, marshalErr := json.MarshalIndent(raw, "", "  ")
	if marshalErr != nil {
		return path, false, fmt.Errorf("serialize %s: %w", path, marshalErr)
	}
	if writeErr := os.WriteFile(path, append(out, '\n'), 0644); writeErr != nil {
		return path, false, fmt.Errorf("write %s: %w", path, writeErr)
	}
	return path, true, nil
}
