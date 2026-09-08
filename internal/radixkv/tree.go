package radixkv

import "github.com/anthony-chaudhary/fak/internal/model"

// tree.go provides regime-fenced lookup, match, and insertion entry points for Tree (issue #5273).

// LookupRegime looks up the longest prefix for tokens under the specified decode regime.
// If regime is incomplete or differs from previously cached regimes, a cache miss occurs
// cleanly without attention corruption.
func (t *Tree) LookupRegime(regime Regime, tokens []int) (*node, int) {
	return t.LookupRegimeNS(regime, "", tokens)
}

// LookupRegimeNS looks up the longest prefix under the specified regime and namespace.
func (t *Tree) LookupRegimeNS(regime Regime, ns string, tokens []int) (*node, int) {
	if !regime.Complete() {
		return nil, 0
	}
	rns := regimeNamespace(ns, regime)
	boundary, matched := t.LookupNS(rns, tokens)
	if boundary != nil && boundary.parent != nil && boundary.regimeKey != "" && boundary.regimeKey != regime.RegimeKey() {
		t.Done(boundary)
		return nil, 0
	}
	return boundary, matched
}

// InsertRegime attaches suffix to boundary under the given regime.
func (t *Tree) InsertRegime(regime Regime, boundary *node, suffix []int, kv *model.KVCache) *node {
	return t.InsertRegimeWithLogits(regime, boundary, suffix, kv, nil)
}

// InsertRegimeWithLogits attaches suffix to boundary under the given regime with logits.
func (t *Tree) InsertRegimeWithLogits(regime Regime, boundary *node, suffix []int, kv *model.KVCache, logits []float32) *node {
	if !regime.Complete() || boundary == nil {
		return boundary
	}
	if boundary.regimeKey != "" && boundary.regimeKey != regime.RegimeKey() {
		return boundary
	}
	leaf := t.InsertWithLogits(boundary, suffix, kv, logits)
	if leaf != nil {
		leaf.regimeKey = regime.RegimeKey()
	}
	return leaf
}

// MatchLenRegime reports the length of the longest prefix matching tokens under regime.
func (t *Tree) MatchLenRegime(regime Regime, tokens []int) int {
	return t.MatchLenRegimeNS(regime, "", tokens)
}

// MatchLenRegimeNS reports the length of the longest prefix matching tokens under regime and ns.
func (t *Tree) MatchLenRegimeNS(regime Regime, ns string, tokens []int) int {
	if !regime.Complete() {
		return 0
	}
	return t.MatchLenNS(regimeNamespace(ns, regime), tokens)
}

// EvictPrefixRegime evicts a prefix matching tokens under the specified regime.
func (t *Tree) EvictPrefixRegime(regime Regime, tokens []int) int {
	return t.EvictPrefixRegimeNS(regime, "", tokens)
}

// EvictPrefixRegimeNS evicts a prefix matching tokens under the specified regime and ns.
func (t *Tree) EvictPrefixRegimeNS(regime Regime, ns string, tokens []int) int {
	if !regime.Complete() {
		return 0
	}
	return t.EvictPrefixNS(regimeNamespace(ns, regime), tokens)
}
