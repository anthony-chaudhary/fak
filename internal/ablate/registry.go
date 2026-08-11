package ablate

import (
	"fmt"
	"sort"
	"strings"
	"sync"
)

// GateResult is the machine-readable verdict returned by a concept correctness gate.
type GateResult struct {
	Verdict string `json:"verdict"`
	Detail  string `json:"detail,omitempty"`
}

// Concept describes one independently sweepable fak concept. Owning packages may
// register concepts from init; duplicate or malformed registrations panic so a typo
// cannot silently alter an experiment.
// EnvArmContract declares the exact values an env-gated production reader
// accepts for the two ablation arms. Enabled mirrors that reader's value dialect;
// Register uses it to reject contracts whose values do not actually flip state.
type EnvArmContract struct {
	On      string
	Off     string
	Enabled func(string) bool
}

type Concept struct {
	Token        string
	EnvVar       string
	EnvArms      *EnvArmContract
	Runtime      func(bool)
	StreamXform  bool
	PrefixStable bool
	Owner        string
	Reversible   bool
	Correctness  func() GateResult
	// ChildAOwnServeEstimate optionally supplies the CHILD-A own-serve dollar estimate.
	ChildAOwnServeEstimate func() *float64
}

var conceptRegistry = struct {
	sync.RWMutex
	concepts map[string]Concept
}{concepts: make(map[string]Concept)}

// Register adds a concept to the process registry. It is intended for package init functions.
func Register(c Concept) {
	c.Token = strings.TrimSpace(c.Token)
	if c.Token == "" || strings.ContainsAny(c.Token, ",@ ") {
		panic(fmt.Sprintf("ablate: invalid concept token %q", c.Token))
	}
	if c.EnvVar == "" && c.Runtime == nil {
		panic(fmt.Sprintf("ablate: concept %q has neither EnvVar nor Runtime", c.Token))
	}
	if c.EnvVar != "" && c.Runtime != nil {
		panic(fmt.Sprintf("ablate: concept %q has both EnvVar and Runtime", c.Token))
	}
	if c.EnvVar != "" {
		if c.EnvArms == nil || c.EnvArms.Enabled == nil {
			panic(fmt.Sprintf("ablate: env concept %q has no arm contract", c.Token))
		}
		if c.EnvArms.Enabled(c.EnvArms.Off) {
			panic(fmt.Sprintf("ablate: env concept %q OFF value %q enables its reader", c.Token, c.EnvArms.Off))
		}
		if !c.EnvArms.Enabled(c.EnvArms.On) {
			panic(fmt.Sprintf("ablate: env concept %q ON value %q disables its reader", c.Token, c.EnvArms.On))
		}
	} else if c.EnvArms != nil {
		panic(fmt.Sprintf("ablate: runtime concept %q has an env arm contract", c.Token))
	}
	if c.Owner != "fak" && c.Owner != "provider" {
		panic(fmt.Sprintf("ablate: concept %q has invalid owner %q", c.Token, c.Owner))
	}
	conceptRegistry.Lock()
	defer conceptRegistry.Unlock()
	if _, exists := conceptRegistry.concepts[c.Token]; exists {
		panic(fmt.Sprintf("ablate: concept %q already registered", c.Token))
	}
	conceptRegistry.concepts[c.Token] = c
}

// KnownFeatures returns a stable snapshot of all registered sweep tokens.
func KnownFeatures() []string {
	conceptRegistry.RLock()
	defer conceptRegistry.RUnlock()
	out := make([]string, 0, len(conceptRegistry.concepts))
	for token := range conceptRegistry.concepts {
		out = append(out, token)
	}
	sort.Strings(out)
	return out
}

func knownFeature(token string) bool { _, ok := registeredConcept(token); return ok }

// EnvGated reports whether token is registered on the subprocess environment rung.
func EnvGated(token string) bool { c, ok := registeredConcept(token); return ok && c.EnvVar != "" }

func registeredConcept(token string) (Concept, bool) {
	conceptRegistry.RLock()
	defer conceptRegistry.RUnlock()
	c, ok := conceptRegistry.concepts[token]
	return c, ok
}

func registerBuiltins() {
	defaultOn := &EnvArmContract{On: "", Off: "off", Enabled: func(v string) bool { return !strings.EqualFold(strings.TrimSpace(v), "off") }}
	presenceInverted := &EnvArmContract{On: "", Off: "off", Enabled: func(v string) bool { return strings.TrimSpace(v) == "" }}
	defaultOff := &EnvArmContract{On: "1", Off: "0", Enabled: defaultOffEnabled}
	env := map[string]struct {
		variable string
		arms     *EnvArmContract
	}{
		FeatureNormgate: {"FAK_NORMGATE", defaultOn}, FeatureRadix: {"FAK_INKERNEL_RADIX", defaultOn},
		FeatureCompressor: {"FAK_COMPRESSOR", presenceInverted}, FeatureIFC: {"FAK_IFC", defaultOn},
		FeatureGitgate: {"FAK_GITGATE", defaultOn}, FeatureCtxplanSeam: {"FAK_CTXPLAN_SEAM", &EnvArmContract{On: "on", Off: "off", Enabled: defaultOffEnabled}},
		FeatureWireScreen:     {"FAK_WIRE_SCREEN", &EnvArmContract{On: "heuristic", Off: "", Enabled: func(v string) bool { return strings.TrimSpace(v) == "heuristic" }}},
		FeatureWireRedact:     {"FAK_WIRE_REDACT", &EnvArmContract{On: "pii", Off: "", Enabled: func(v string) bool { return strings.TrimSpace(v) == "pii" }}},
		FeatureBreakpointPlan: {"FAK_ABLATE_BP_PLAN", defaultOff}, FeatureTTL1H: {"FAK_ABLATE_TTL_1H", defaultOff},
		FeaturePrefixGuard: {"FAK_ABLATE_PREFIX_GUARD", defaultOff}, FeatureUncachedTrim: {"FAK_ABLATE_UNCACHED_TRIM", defaultOff},
		FeatureNegframeReframe: {"FAK_ABLATE_NEGFRAME_REFRAME", defaultOff},
	}
	Register(Concept{Token: FeatureVDSO, Runtime: func(bool) {}, Owner: "fak", Reversible: true, PrefixStable: true})
	registerCacheLevers()
	for token, gate := range env {
		Register(Concept{Token: token, EnvVar: gate.variable, EnvArms: gate.arms, Owner: "fak", Reversible: true, PrefixStable: true})
	}
}

func defaultOffEnabled(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// builtinsRegistered is deliberately a package-level VAR initializer, not a func init():
// Go runs every var initializer before ANY init() function, and catalog.go's init()
// asserts each FeatureCard token is already registered. As a func init() here it runs
// AFTER catalog.go's (init()s fire in file order, catalog.go first) and that assertion
// panics at process start in every binary importing this package. The var is
// "unreferenced" on purpose — it exists only for this ordering guarantee; do not fold
// it back into init().
//
//slop:keep intentionally unreferenced — the var initializer IS the ordering guarantee.
var builtinsRegistered = func() bool { registerBuiltins(); return true }()
