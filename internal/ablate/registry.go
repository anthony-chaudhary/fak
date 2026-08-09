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
type Concept struct {
	Token        string
	EnvVar       string
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
	env := map[string]string{
		FeatureNormgate: "FAK_NORMGATE", FeatureRadix: "FAK_INKERNEL_RADIX", FeatureCompressor: "FAK_COMPRESSOR",
		FeatureIFC: "FAK_IFC", FeatureGitgate: "FAK_GITGATE", FeatureCtxplanSeam: "FAK_CTXPLAN_SEAM",
		FeatureWireScreen: "FAK_WIRE_SCREEN", FeatureWireRedact: "FAK_WIRE_REDACT",
		FeatureBreakpointPlan: "FAK_ABLATE_BP_PLAN", FeatureTTL1H: "FAK_ABLATE_TTL_1H",
		FeaturePrefixGuard: "FAK_ABLATE_PREFIX_GUARD", FeatureUncachedTrim: "FAK_ABLATE_UNCACHED_TRIM",
		FeatureNegframeReframe: "FAK_ABLATE_NEGFRAME_REFRAME",
	}
	Register(Concept{Token: FeatureVDSO, Runtime: func(bool) {}, Owner: "fak", Reversible: true, PrefixStable: true})
	registerCacheLevers()
	for token, variable := range env {
		Register(Concept{Token: token, EnvVar: variable, Owner: "fak", Reversible: true, PrefixStable: true})
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
