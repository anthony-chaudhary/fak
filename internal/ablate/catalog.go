package ablate

import (
	"fmt"
	"sort"
	"strings"
)

// FeatureCard is the STATIC identity of one sweepable cache lever — everything a user needs
// to decide whether to try it, WITHOUT running a replay. It is the single source of truth for
// a feature's owner/plane/fidelity classification: cacheEffectForFeature seeds its per-arm
// CacheEffect from the card and then layers the dynamic Status/Reason a run witnesses on top,
// so the "what is this lever" story a `fak ablate --list` prints is byte-for-byte the same
// classification a live run reports. A new lever is one row in featureCards (plus, if it wants
// an env gate, one row in envFeatureVars).
type FeatureCard struct {
	Token           string `json:"token"`             // the --sweep token (e.g. "bp_plan")
	Owner           string `json:"owner"`             // fak | provider | external
	Plane           string `json:"plane"`             // kernel_tool_cache | local_kv | provider_prompt_cache_control | context_compression | context_view
	Component       string `json:"component"`         // vdso | radix | headroom_compressor | ...
	Dependency      string `json:"dependency"`        // in_process | subprocess_env | subprocess_env_and_provider | ...
	Fidelity        string `json:"fidelity"`          // lossless | lossy | recoverable | passive
	EnvVar          string `json:"env_var,omitempty"` // FAK_* the rung-2 child carries, "" for the in-process vdso knob
	RuntimeSettable bool   `json:"runtime_settable"`  // true only for vdso (flippable in a live process, rung 1)
	Summary         string `json:"summary"`           // one line: what the lever does to the cache
}

// featureCards is the CLOSED catalog: one card per token in KnownFeatures. The static fields
// mirror the on-branch of cacheEffectForFeature exactly (that function reads them back through
// cardFor), so the two never drift. Summary is the plain-language "what it does" a user reads
// when picking what to try; for the wire levers it is the same sentence the per-arm CacheEffect
// reason carries.
var featureCards = map[string]FeatureCard{
	FeatureVDSO: {
		Token:           FeatureVDSO,
		Owner:           "fak",
		Plane:           "kernel_tool_cache",
		Component:       "vdso",
		Dependency:      "in_process",
		Fidelity:        "lossless",
		RuntimeSettable: true,
		Summary:         "serves a repeated tool call from the in-kernel vDSO fast path without re-adjudicating or re-calling the engine",
	},
	FeatureRadix: {
		Token:      FeatureRadix,
		Owner:      "fak",
		Plane:      "local_kv",
		Component:  "inkernel_radix",
		Dependency: "subprocess_env",
		Fidelity:   "lossless",
		Summary:    "reuses a shared KV prefix across turns via in-kernel RadixAttention (requires engine=inkernel)",
	},
	FeatureCtxplanSeam: {
		Token:      FeatureCtxplanSeam,
		Owner:      "fak",
		Plane:      "context_view",
		Component:  "ctxplan_seam",
		Dependency: "subprocess_env",
		Fidelity:   "recoverable",
		Summary:    "serves cache-safe materialized context views through the ctxplan seam (requires engine=inkernel)",
	},
	FeatureCompressor: {
		Token:      FeatureCompressor,
		Owner:      "fak",
		Plane:      "context_compression",
		Component:  "headroom_compressor",
		Dependency: "subprocess_env_or_external_sidecar",
		Fidelity:   "recoverable",
		Summary:    "sheds recoverable context through the headroom compressor to hold the cacheable prefix under budget",
	},
	FeatureBreakpointPlan: {
		Token:      FeatureBreakpointPlan,
		Owner:      "fak",
		Plane:      "provider_prompt_cache_control",
		Component:  "breakpoint_planner",
		Dependency: "subprocess_env_and_provider",
		Fidelity:   "lossless",
		Summary:    "places provider cache breakpoints without changing model-visible prefix bytes",
	},
	FeatureTTL1H: {
		Token:      FeatureTTL1H,
		Owner:      "fak",
		Plane:      "provider_prompt_cache_control",
		Component:  "ttl_1h",
		Dependency: "subprocess_env_and_provider",
		Fidelity:   "passive",
		Summary:    "changes provider cache retention only; model-visible prefix bytes are unchanged",
	},
	FeaturePrefixGuard: {
		Token:      FeaturePrefixGuard,
		Owner:      "fak",
		Plane:      "provider_prompt_cache_control",
		Component:  "prefix_guard",
		Dependency: "subprocess_env_and_provider",
		Fidelity:   "lossless",
		Summary:    "guards prefix stability before relying on provider-cache economics",
	},
	FeatureUncachedTrim: {
		Token:      FeatureUncachedTrim,
		Owner:      "fak",
		Plane:      "provider_prompt_cache_control",
		Component:  "uncached_trim",
		Dependency: "subprocess_env_and_provider",
		Fidelity:   "lossy",
		Summary:    "sheds or rewrites uncached context; prefix integrity covers only the guarded cacheable prefix",
	},
	FeatureNegframeReframe: {
		Token:      FeatureNegframeReframe,
		Owner:      "fak",
		Plane:      "context_view",
		Component:  "negframe_reframe",
		Dependency: "subprocess_env",
		Fidelity:   "lossless",
		EnvVar:     "FAK_ABLATE_NEGFRAME_REFRAME",
		Summary:    "routes fak-authored injected prose through the emit-time positive-voice reframe (default ON); the OFF arm restores raw negative-framed injection for the #3546 steerability control",
	},
}

func init() {
	// The catalog is the CACHE-lever menu. Not every sweepable feature is a cache lever
	// (normgate/ifc/gitgate/wire_screen/wire_redact are guard/wire knobs, not cache), so a
	// KnownFeatures token may legitimately have no card. But every card MUST name a sweepable
	// token and agree with the closed env map — else --list would advertise a lever the user
	// cannot actually sweep, or misname its child env. Assert that direction; leave the other
	// open so a new non-cache feature never has to fabricate a cache card.
	for token := range featureCards {
		if !knownFeature(token) {
			panic(fmt.Sprintf("ablate: FeatureCard %q names a token that is not sweepable (add it to KnownFeatures or drop the card)", token))
		}
	}
}

// cardFor returns the static card for a token. The second result is false for an unknown
// token, so callers stay honest about the closed set. cacheEffectForFeature uses it to seed
// the static half of a per-arm CacheEffect, keeping one classification source of truth.
func cardFor(token string) (FeatureCard, bool) {
	c, ok := featureCards[token]
	return c, ok
}

// FeatureCatalog returns every sweepable lever's static card, sorted by token, so a CLI
// caller can print the full menu (human table or JSON) WITHOUT running a replay. This is the
// "see what I can try" surface behind `fak ablate --list`.
func FeatureCatalog() []FeatureCard {
	out := make([]FeatureCard, 0, len(featureCards))
	for _, c := range featureCards {
		// Seed the EnvVar from the closed map so the card and the actual child env can never
		// disagree (init already asserts equality; this keeps the exported view canonical).
		if concept, ok := registeredConcept(c.Token); ok {
			c.EnvVar = concept.EnvVar
			c.RuntimeSettable = concept.Runtime != nil
		}
		out = append(out, c)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Token < out[j].Token })
	return out
}

// Presets are named groups of tokens a user can sweep in one flag: `--sweep @wire-cache`
// expands to every provider-prompt-cache lever, so trying a whole cache DIMENSION at once is
// one token instead of four. The "@" prefix keeps a preset unambiguous against a real token.
// The lists are derived from the catalog's Plane, so a new lever on an existing plane joins
// its preset automatically (see presetTokens).
const PresetPrefix = "@"

// presetPlanes maps a preset name to the cache Plane whose levers it groups. A preset expands
// to every catalog token on that plane (in sorted order), so adding a new provider-cache lever
// puts it in @wire-cache without editing the preset.
var presetPlanes = map[string]string{
	"wire-cache": "provider_prompt_cache_control",
	"local":      "local_kv",
	"context":    "context_compression",
}

// PresetNames returns the sorted preset names (without the "@") for help text and --list.
func PresetNames() []string {
	names := make([]string, 0, len(presetPlanes))
	for n := range presetPlanes {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// PresetExpansion returns the sorted tokens a preset name expands to (without the "@"), or nil
// for an unknown name. It is the exported view for help text / --list, mirroring what
// ExpandPresets substitutes at sweep time.
func PresetExpansion(name string) []string {
	toks, _ := presetTokens(name)
	return toks
}

// presetTokens returns the sorted catalog tokens a preset expands to, or (nil,false) if the
// name is not a known preset. A preset that matched no token is still "known" but empty — the
// caller treats an empty expansion as an error so a typo'd plane can never silently sweep zero.
func presetTokens(name string) ([]string, bool) {
	plane, ok := presetPlanes[name]
	if !ok {
		return nil, false
	}
	var toks []string
	for token, card := range featureCards {
		if card.Plane == plane {
			toks = append(toks, token)
		}
	}
	sort.Strings(toks)
	return toks, true
}

// ExpandPresets turns a raw sweep list into a flat token list, replacing any "@name" entry
// with the preset's tokens. Non-preset entries pass through untouched (BuildSweep still
// validates them). An unknown preset, or a preset that expands to nothing, is a hard error so
// a mistyped group fails loud instead of silently sweeping fewer arms than the user asked for.
// Order is preserved and duplicates are collapsed, so `--sweep vdso,@wire-cache,bp_plan` sweeps
// each lever once.
func ExpandPresets(raw []string) ([]string, error) {
	seen := map[string]bool{}
	var out []string
	add := func(tok string) {
		if !seen[tok] {
			seen[tok] = true
			out = append(out, tok)
		}
	}
	for _, entry := range raw {
		e := strings.TrimSpace(entry)
		if e == "" {
			continue
		}
		if !strings.HasPrefix(e, PresetPrefix) {
			add(e)
			continue
		}
		name := strings.TrimPrefix(e, PresetPrefix)
		toks, ok := presetTokens(name)
		if !ok {
			return nil, fmt.Errorf("ablate: unknown sweep preset %q (known presets: %s)",
				e, strings.Join(prefixed(PresetNames()), ", "))
		}
		if len(toks) == 0 {
			return nil, fmt.Errorf("ablate: sweep preset %q currently expands to no lever", e)
		}
		for _, tok := range toks {
			add(tok)
		}
	}
	return out, nil
}

// prefixed re-attaches the "@" to preset names for user-facing lists.
func prefixed(names []string) []string {
	out := make([]string, len(names))
	for i, n := range names {
		out[i] = PresetPrefix + n
	}
	return out
}
