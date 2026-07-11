package dispatchorder

import "testing"

// TestTreeOverlapSiblingPrefixBoundary pins the load-bearing "+/" boundary in
// treeOverlap: prefix containment is DIRECTORY ancestry, not raw string prefix.
// Two paths that merely share a NAME prefix are disjoint and must NOT overlap.
// The classic trap is cmd/fak vs cmd/fakctl — "cmd/fak" IS a string prefix of
// "cmd/fakctl", so a HasPrefix check without the trailing "/" would fuse two
// unrelated directories and serialize workers that never actually contend.
func TestTreeOverlapSiblingPrefixBoundary(t *testing.T) {
	disjoint := [][2]string{
		{"cmd/fak", "cmd/fakctl"},                                   // string-prefix trap, real dirs in this repo
		{"internal/gateway", "internal/gate"},                       // shorter shares a prefix with the longer
		{"internal/gateway/**", "internal/gatekeeper/**"},           // glob spellings, shared name prefix
		{"internal/gateway/http.go", "internal/gatekeeper/http.go"}, // files under sibling dirs
	}
	for _, p := range disjoint {
		if TreesOverlap([]string{p[0]}, []string{p[1]}) {
			t.Errorf("%q and %q share only a NAME prefix, must not overlap", p[0], p[1])
		}
		// overlap is symmetric — the boundary must hold in either argument order.
		if TreesOverlap([]string{p[1]}, []string{p[0]}) {
			t.Errorf("%q and %q must not overlap (reversed order)", p[1], p[0])
		}
	}
	// True directory ancestry MUST still overlap — the boundary rejects siblings
	// without rejecting genuine containment.
	contained := [][2]string{
		{"internal/gateway", "internal/gateway/http.go"},
		{"internal/gateway/**", "internal/gateway/sub/deep.go"},
	}
	for _, p := range contained {
		if !TreesOverlap([]string{p[0]}, []string{p[1]}) {
			t.Errorf("%q contains %q, must overlap", p[0], p[1])
		}
	}
}

// TestTreeOverlapNormalizationEquivalence pins that every spelling which names
// the SAME directory normalizes to the same prefix and therefore overlaps: the
// "/**" and "/*" glob suffixes, a trailing slash, a leading "./", and — this
// fleet runs on Windows — backslash separators. A regression in normalizeTree
// would make two spellings of one directory look disjoint and let two workers
// write it concurrently.
func TestTreeOverlapNormalizationEquivalence(t *testing.T) {
	spellings := []string{
		"internal/gateway",
		"internal/gateway/",
		"internal/gateway/**",
		"internal/gateway/*",
		"./internal/gateway",
		`internal\gateway`,     // Windows separators
		`internal\gateway\**`,  // Windows separators + glob suffix
	}
	target := []string{"internal/gateway/http.go"}
	for _, s := range spellings {
		if !TreesOverlap([]string{s}, target) {
			t.Errorf("spelling %q must normalize to internal/gateway and overlap %v", s, target)
		}
		for _, s2 := range spellings {
			if !TreesOverlap([]string{s}, []string{s2}) {
				t.Errorf("%q and %q name the same region, must overlap", s, s2)
			}
		}
	}
}

// TestTreeOverlapWildcardAllMatchesEverything pins the bare "**" / "**/*"
// whole-tree wildcard branch: a candidate that claims the entire tree overlaps
// any other claim, in either argument position. This is distinct from the
// "internal/x/**" case, where the "/**" suffix is stripped to a plain prefix;
// here the claim normalizes to the un-prefixed match-all token.
func TestTreeOverlapWildcardAllMatchesEverything(t *testing.T) {
	for _, all := range []string{"**", "**/*"} {
		if !TreesOverlap([]string{all}, []string{"internal/gateway/http.go"}) {
			t.Errorf("%q must overlap any tree", all)
		}
		if !TreesOverlap([]string{"docs/README.md"}, []string{all}) {
			t.Errorf("%q must overlap any tree (reversed argument order)", all)
		}
	}
}
