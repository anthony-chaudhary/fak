package engine

// This file adds the THREE-VERDICT config validate() — issue #3399, under epic
// #3366 (clean-room INSPIRE borrow from LMCache's config gate, borrow id
// D3-config-three-outcome-gate; technique only, no vendored bytes).
//
// The gap it closes: fak's config validators are two-outcome — a config either
// hard-refuses (ggufload bad magic, TPPlan.Validate) or silently passes, so a
// safe, unambiguous drift (a trailing slash, stray whitespace, an omitted
// worker id) either sneaks through unnamed or forces an operator round-trip a
// machine could have absorbed. Validate() instead names one of THREE verdicts:
//
//   - refuse           — illegal, no safe correction exists; the config is unusable;
//   - auto-correct     — safe drift with exactly one unambiguous fix; the corrected
//     config is returned WITH a named warning per applied fix (never silently);
//   - accept           — well-formed as-is.
//
// It reuses AdmitActiveCache's closed-vocabulary, fail-closed pattern: verdicts,
// refusal reasons and correction warnings are all CLOSED enums, the verdict zero
// value is NOT admitted (an unset or out-of-vocabulary verdict read off a wire
// fails closed), and a refused validation yields no usable config at all.
//
// The concrete surface is VLLMConfig — the adapter config whose drift the
// runtime already resolves implicitly (joinEndpoint trims whitespace and
// trailing slashes; NewVLLMEngine defaults an empty WorkerID), so every
// auto-correction here mirrors a witnessed runtime behavior rather than
// inventing policy, and every refusal names a config the runtime would fail on.

import (
	"net/url"
	"strings"
)

// ConfigVerdict is the CLOSED vocabulary of config-validation outcomes. The
// zero value is deliberately NOT a member: an unset verdict is invalid and not
// admitted, so a caller that forgets to validate fails closed by construction.
type ConfigVerdict string

const (
	// ConfigVerdictRefuse: the config is illegal and no safe correction exists.
	// The validation carries named refusal reasons and NO usable config.
	ConfigVerdictRefuse ConfigVerdict = "refuse"
	// ConfigVerdictAutoCorrect: the config drifted in a safe, unambiguously
	// correctable way. The validation carries the corrected config plus a named
	// warning for every applied correction — accepted, but never silently.
	ConfigVerdictAutoCorrect ConfigVerdict = "auto-correct"
	// ConfigVerdictAccept: the config is well-formed exactly as given.
	ConfigVerdictAccept ConfigVerdict = "accept"
)

// ConfigVerdicts returns the closed verdict vocabulary in a stable order, so a
// conformance test can prove the enum admits ONLY these members.
func ConfigVerdicts() []ConfigVerdict {
	return []ConfigVerdict{ConfigVerdictRefuse, ConfigVerdictAutoCorrect, ConfigVerdictAccept}
}

// Valid reports whether v is a member of the closed verdict vocabulary.
func (v ConfigVerdict) Valid() bool {
	switch v {
	case ConfigVerdictRefuse, ConfigVerdictAutoCorrect, ConfigVerdictAccept:
		return true
	default:
		return false
	}
}

// Admitted reports whether the validated config may be used. Only auto-correct
// and accept admit; refuse, the zero value, and anything out-of-vocabulary all
// fail closed.
func (v ConfigVerdict) Admitted() bool {
	return v == ConfigVerdictAutoCorrect || v == ConfigVerdictAccept
}

// ConfigRefusal is the CLOSED vocabulary of named reasons a config was refused
// outright — conditions with no safe correction.
type ConfigRefusal string

const (
	// ConfigRefusalBaseURLMissing: BaseURL is empty (or whitespace-only). There
	// is no correct guess for where the engine lives; Admit would error.
	ConfigRefusalBaseURLMissing ConfigRefusal = "base-url-missing"
	// ConfigRefusalBaseURLMalformed: BaseURL does not parse as a URL with both
	// a scheme and a host — the same shape joinEndpoint fails on at call time.
	ConfigRefusalBaseURLMalformed ConfigRefusal = "base-url-malformed"
	// ConfigRefusalBaseURLSchemeNotHTTP: BaseURL parses but its scheme is
	// not http/https, which the OpenAI-compatible HTTP transport cannot speak.
	ConfigRefusalBaseURLSchemeNotHTTP ConfigRefusal = "base-url-scheme-not-http"
)

// ConfigWarning is the CLOSED vocabulary of named auto-corrections — safe drift
// the validator fixed and must disclose. Every member mirrors a correction the
// runtime already performs implicitly; validation makes it explicit and named.
type ConfigWarning string

const (
	// ConfigWarningWhitespaceTrimmed: one or more string fields carried leading
	// or trailing whitespace, which was trimmed (joinEndpoint's TrimSpace).
	ConfigWarningWhitespaceTrimmed ConfigWarning = "field-whitespace-trimmed"
	// ConfigWarningTrailingSlashTrimmed: BaseURL carried trailing slash(es),
	// which were trimmed (joinEndpoint's path TrimRight).
	ConfigWarningTrailingSlashTrimmed ConfigWarning = "base-url-trailing-slash-trimmed"
	// ConfigWarningWorkerIDDefaulted: WorkerID was empty and was defaulted to
	// the adapter id, exactly as NewVLLMEngine's defaultWorkerID would.
	ConfigWarningWorkerIDDefaulted ConfigWarning = "worker-id-defaulted"
)

// VLLMConfigValidation is the outcome of ValidateVLLMConfig. Exactly one shape
// per verdict: refuse carries Refusals and a ZERO Config (fail closed — a
// refused config is unusable, so no partially-corrected copy is offered);
// auto-correct carries the corrected Config plus one Warning per applied fix;
// accept carries the config unchanged with neither.
type VLLMConfigValidation struct {
	Verdict  ConfigVerdict
	Config   VLLMConfig
	Refusals []ConfigRefusal
	Warnings []ConfigWarning
}

// ValidateVLLMConfig validates cfg and returns one of the three verdicts.
// It never panics, applies only corrections with exactly one unambiguous fix
// (each disclosed by name), and refuses everything else with a named reason
// from the closed vocabulary.
func ValidateVLLMConfig(cfg VLLMConfig) VLLMConfigValidation {
	var refusals []ConfigRefusal
	var warnings []ConfigWarning
	corrected := cfg

	// Safe drift 1: surrounding whitespace on string fields. The runtime trims
	// BaseURL at call time; validation trims every string field and says so.
	trimmed := false
	trim := func(s string) string {
		t := strings.TrimSpace(s)
		if t != s {
			trimmed = true
		}
		return t
	}
	corrected.BaseURL = trim(cfg.BaseURL)
	corrected.Model = trim(cfg.Model)
	corrected.APIKey = trim(cfg.APIKey)
	corrected.WorkerID = trim(cfg.WorkerID)
	corrected.MetricsURL = trim(cfg.MetricsURL)
	if trimmed {
		warnings = append(warnings, ConfigWarningWhitespaceTrimmed)
	}

	// BaseURL: required, parseable, scheme+host, http(s) only.
	if corrected.BaseURL == "" {
		refusals = append(refusals, ConfigRefusalBaseURLMissing)
	} else {
		// Safe drift 2: trailing slash(es). joinEndpoint trims these from the
		// path at call time; trimming the whole URL here is the same fix made
		// explicit. A bare "/" is left alone — it is malformed, not drift.
		if t := strings.TrimRight(corrected.BaseURL, "/"); t != corrected.BaseURL && t != "" {
			corrected.BaseURL = t
			warnings = append(warnings, ConfigWarningTrailingSlashTrimmed)
		}
		u, err := url.Parse(corrected.BaseURL)
		switch {
		case err != nil || u.Scheme == "" || u.Host == "":
			refusals = append(refusals, ConfigRefusalBaseURLMalformed)
		case u.Scheme != "http" && u.Scheme != "https":
			refusals = append(refusals, ConfigRefusalBaseURLSchemeNotHTTP)
		}
	}

	// Safe drift 3: empty WorkerID defaults to the adapter id, mirroring
	// NewVLLMEngine's defaultWorkerID(cfg.WorkerID, "vllm").
	if corrected.WorkerID == "" {
		corrected.WorkerID = VLLMEngineID
		warnings = append(warnings, ConfigWarningWorkerIDDefaulted)
	}

	if len(refusals) > 0 {
		// Fail closed: name every reason, return no usable config, and drop
		// the cosmetic corrections — a refused config is not partially usable.
		return VLLMConfigValidation{Verdict: ConfigVerdictRefuse, Refusals: refusals}
	}
	if len(warnings) > 0 {
		return VLLMConfigValidation{Verdict: ConfigVerdictAutoCorrect, Config: corrected, Warnings: warnings}
	}
	return VLLMConfigValidation{Verdict: ConfigVerdictAccept, Config: corrected}
}
