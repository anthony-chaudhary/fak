package ablate

import "bytes"

// CacheLever describes a token-stream transform that CHILD-C can sweep without
// knowing its implementation. Transforms remain default-off until their correctness
// gate passes; metadata is deliberately separate from the eventual wire codec.
type CacheLever interface {
	Name() string
	Apply(stream []byte) ([]byte, error)
	PrefixStable() bool
	Owner() string
	CorrectnessGate() GateResult
}

const (
	FeatureIdentifierDictionary = "identifier_dictionary"
	FeatureRereadDiff           = "reread_diff"
	FeatureRetrievalSubstitute  = "retrieval_substitute"
	FeatureCanonicalMinify      = "canonical_minify"
	FeatureNoopTurnElision      = "noop_turn_elision"
)

type identityCacheLever struct {
	name         string
	prefixStable bool
}

func (l identityCacheLever) Name() string { return l.name }
func (l identityCacheLever) Apply(stream []byte) ([]byte, error) {
	// The descriptor rung is intentionally identity-only. A later implementation
	// replaces this method, but cannot become default-on without this gate changing
	// from an identity witness to a transform-specific semantic witness.
	return append([]byte(nil), stream...), nil
}
func (l identityCacheLever) PrefixStable() bool { return l.prefixStable }
func (identityCacheLever) Owner() string        { return "fak" }
func (l identityCacheLever) CorrectnessGate() GateResult {
	fixture := []byte("system\x00tool_result\n{\"identifier\":\"alpha\"}")
	got, err := l.Apply(fixture)
	if err != nil {
		return GateResult{Verdict: "fail", Detail: err.Error()}
	}
	if !bytes.Equal(got, fixture) {
		return GateResult{Verdict: "fail", Detail: "identity descriptor changed the semantic fixture"}
	}
	return GateResult{Verdict: "pass", Detail: "identity descriptor preserves the semantic fixture"}
}

var builtinCacheLevers = []CacheLever{
	identityCacheLever{name: FeatureToonWire, prefixStable: true},
	identityCacheLever{name: FeatureIdentifierDictionary, prefixStable: true},
	identityCacheLever{name: FeatureRereadDiff, prefixStable: false},
	identityCacheLever{name: FeatureRetrievalSubstitute, prefixStable: false},
	identityCacheLever{name: FeatureCanonicalMinify, prefixStable: true},
	identityCacheLever{name: FeatureNoopTurnElision, prefixStable: false},
}

// CacheLevers returns the immutable built-in descriptor set in registration order.
func CacheLevers() []CacheLever { return append([]CacheLever(nil), builtinCacheLevers...) }

func registerCacheLevers() {
	for _, lever := range builtinCacheLevers {
		l := lever
		Register(Concept{
			Token:        l.Name(),
			EnvVar:       "FAK_" + envToken(l.Name()),
			EnvArms:      &EnvArmContract{On: "1", Off: "0", Enabled: defaultOffEnabled},
			Owner:        l.Owner(),
			Reversible:   true,
			StreamXform:  true,
			PrefixStable: l.PrefixStable(),
			Correctness:  l.CorrectnessGate,
		})
	}
}

func envToken(token string) string {
	b := []byte(token)
	for i := range b {
		if b[i] >= 'a' && b[i] <= 'z' {
			b[i] -= 'a' - 'A'
		}
	}
	return string(b)
}
