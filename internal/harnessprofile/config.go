package harnessprofile

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// config.go is the C6 (#1957) payoff: a user can teach guard a NEW harness — or repin a
// built-in field — by editing config, not Go. It parses a small JSON document of
// HarnessProfile overrides, validates them against the CLOSED vocabularies (Wire /
// RepointMechanism / IdentityKind / CredentialKind — an unknown value fails loud, never a
// silent fallback), and MERGES them over the built-ins (a user entry sharing a detect-name
// overrides field-by-field; a novel name is added). The merged set becomes the process-wide
// active registry via SetActive, so every existing Lookup call (guard detection, repoint
// gating) consults built-ins + config with no threading change.
//
// The config is JSON, decoded with DisallowUnknownFields — the same closed-set discipline the
// policy loader uses (internal/policy) — so a typo'd field is rejected, not ignored. A user
// may REFERENCE a shipped wire, never declare a brand-new upstream protocol here (that needs
// an internal/agent adapter + code review); the closed Wire set is what enforces that fence.

// activeProfiles is the process-wide registry the package-level Lookup/Profiles read. nil
// means "use the built-ins"; guard replaces it once at startup via SetActive(merged config).
// It is deliberately simple process-global state: guard loads config once before it detects
// the harness, and nothing mutates it mid-run.
var activeProfiles []HarnessProfile

func active() []HarnessProfile {
	if activeProfiles == nil {
		return builtins
	}
	return activeProfiles
}

// SetActive replaces the active registry (built-ins + resolved config). guard calls it once
// at startup. Passing nil restores the built-ins (see ResetActive).
func SetActive(profiles []HarnessProfile) { activeProfiles = profiles }

// ResetActive restores the built-in-only registry — for tests, and for a guard teardown that
// wants the default set back.
func ResetActive() { activeProfiles = nil }

// configDoc is the JSON config surface: a `harnesses` array of profile overrides.
type configDoc struct {
	Harnesses []HarnessProfile `json:"harnesses"`
}

// ParseConfig decodes a harnesses JSON document into the list of user overrides, rejecting
// an unknown field (DisallowUnknownFields) and validating each override's provided closed-
// vocabulary values. It does NOT merge — Merge does — so a caller can inspect the raw
// overrides. An empty/whitespace body is no overrides (nil), not an error.
func ParseConfig(raw []byte) ([]HarnessProfile, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, nil
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var doc configDoc
	if err := dec.Decode(&doc); err != nil {
		return nil, fmt.Errorf("harnessprofile: parse config: %w", err)
	}
	for i, p := range doc.Harnesses {
		if err := p.validateProvided(); err != nil {
			return nil, fmt.Errorf("harnessprofile: harness[%d] (%q): %w", i, p.Name, err)
		}
	}
	return doc.Harnesses, nil
}

// Resolve is ParseConfig followed by Merge over the built-ins — the one call guard makes to
// turn a config body into the active registry. A nil/empty body yields the built-ins
// unchanged.
func Resolve(raw []byte) ([]HarnessProfile, error) {
	overrides, err := ParseConfig(raw)
	if err != nil {
		return nil, err
	}
	return Merge(Builtins(), overrides)
}

// Merge overlays user overrides onto a base registry. An override that shares a detect-name
// with a base profile REPLACES that base profile field-by-field (a zero field leaves the
// base value intact — so "pin just the default_base_url" keeps codex's wire/repoint/
// credential); an override with a novel detect-name is APPENDED. The merged profile is
// validated for completeness, so a bad override fails loud at load. Overrides are applied in
// order, so two overrides touching the same base compose left-to-right.
func Merge(base, overrides []HarnessProfile) ([]HarnessProfile, error) {
	out := cloneProfiles(base)
	for _, ov := range overrides {
		if idx := indexByDetectOverlap(out, ov); idx >= 0 {
			out[idx] = overlayProfile(out[idx], ov)
			if err := out[idx].validateComplete(); err != nil {
				return nil, fmt.Errorf("harnessprofile: override of %q invalid: %w", out[idx].Name, err)
			}
		} else {
			if err := ov.validateComplete(); err != nil {
				return nil, fmt.Errorf("harnessprofile: new harness %q invalid: %w", ov.Name, err)
			}
			out = append(out, ov)
		}
	}
	return out, nil
}

// indexByDetectOverlap returns the index of the first profile in set that shares any detect
// name with ov, or -1. It is how an override finds the built-in it repins.
func indexByDetectOverlap(set []HarnessProfile, ov HarnessProfile) int {
	for i, p := range set {
		for _, a := range p.Names {
			for _, b := range ov.Names {
				if a == b {
					return i
				}
			}
		}
	}
	return -1
}

// overlayProfile applies ov's NON-ZERO fields onto base, returning the merged profile. A zero
// field in ov leaves base's value — so a partial override (only default_base_url set) repins
// exactly that field.
func overlayProfile(base, ov HarnessProfile) HarnessProfile {
	out := base
	if ov.Name != "" {
		out.Name = ov.Name
	}
	if len(ov.Names) > 0 {
		out.Names = ov.Names
	}
	if ov.Wire != "" {
		out.Wire = ov.Wire
	}
	if ov.DefaultBaseURL != "" {
		out.DefaultBaseURL = ov.DefaultBaseURL
	}
	if len(ov.Repoint) > 0 {
		out.Repoint = ov.Repoint
	}
	if ov.Credential.Kind != "" {
		out.Credential = ov.Credential
	}
	if ov.ConfigHomeGlob != "" {
		out.ConfigHomeGlob = ov.ConfigHomeGlob
	}
	if ov.Identity != "" {
		out.Identity = ov.Identity
	}
	return out
}

// validateProvided rejects any closed-vocabulary value the override actually SET that is not
// in its closed set (an unknown wire / repoint mechanism / identity / credential kind). It
// does not require completeness — a partial override (repin one field) is legal; completeness
// is checked on the MERGED profile by validateComplete.
func (p HarnessProfile) validateProvided() error {
	if p.Wire != "" && !p.Wire.Valid() {
		return fmt.Errorf("unknown wire %q (want anthropic|openai|openai-responses)", p.Wire)
	}
	for _, m := range p.Repoint {
		if !m.Valid() {
			return fmt.Errorf("unknown repoint mechanism %q (want env|cli-config|settings-file)", m)
		}
	}
	if p.Identity != "" && !p.Identity.validKind() {
		return fmt.Errorf("unknown identity kind %q (want claude|codex|env-key)", p.Identity)
	}
	if p.Credential.Kind != "" && !p.Credential.Kind.valid() {
		return fmt.Errorf("unknown credential kind %q (want env-key|oauth-file)", p.Credential.Kind)
	}
	return nil
}

// validateComplete asserts a fully-resolved profile is usable: it names at least one detect
// name, a valid wire, and only valid closed-vocabulary values. Applied to a merged profile
// (a repin or a new harness) so an incomplete declaration fails loud at load.
func (p HarnessProfile) validateComplete() error {
	if err := p.validateProvided(); err != nil {
		return err
	}
	if len(p.Names) == 0 {
		return fmt.Errorf("no detect names (needs at least one `names` entry)")
	}
	if !p.Wire.Valid() {
		return fmt.Errorf("missing/unknown wire (want anthropic|openai|openai-responses)")
	}
	return nil
}

func (k IdentityKind) validKind() bool {
	switch k {
	case IdentityNone, IdentityClaude, IdentityCodex, IdentityEnvKey:
		return true
	default:
		return false
	}
}

func (k CredentialKind) valid() bool {
	switch k {
	case CredentialEnvKey, CredentialOAuthFile:
		return true
	default:
		return false
	}
}

// DefaultBaseURLForCommand resolves a wrapped-agent command to the base URL guard should
// default to: the matched profile's DefaultBaseURL when set (so a config override of the base
// URL wins — C6), else the profile's wire default (BaseURLForWire), else "" for an
// unrecognized agent (the caller requires an explicit --base-url or applies its anthropic
// fallback). It is the profile-aware twin of guardDefaultBaseURL(wire), which cannot see a
// per-profile override because it is keyed on the wire alone.
func DefaultBaseURLForCommand(command string) string {
	p, ok := Lookup(command)
	if !ok {
		return ""
	}
	if p.DefaultBaseURL != "" {
		return p.DefaultBaseURL
	}
	return BaseURLForWire(p.Wire)
}

// DumpJSON renders the active registry as pretty JSON — the body behind guard's
// --dump-harness-profiles, so the effective (built-in + config) set is inspectable.
func DumpJSON() ([]byte, error) {
	return json.MarshalIndent(configDoc{Harnesses: Profiles()}, "", "  ")
}
