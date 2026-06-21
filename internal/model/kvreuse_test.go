package model

import "testing"

// TestKVPrefixReuseMatchesRecompute is the rung-14 witness for the vDSO payoff: once
// the model is in the kernel, the content-addressed RESULT cache becomes real KV-PREFIX
// reuse. A shared prefix's K/V is computed ONCE; a second sequence that shares it
// clones the cache and prefills only its SUFFIX, skipping the prefix's prefill FLOPs.
// The reuse is sound iff its logits and greedy continuation are IDENTICAL to a full
// recompute — proven here bit-for-bit against the (rung-2-verified) forward pass.
func TestKVPrefixReuseMatchesRecompute(t *testing.T) {
	m, doc := loadFixture(t)
	// use a real fixture's tokens: prefix = a prompt, suffix = another prompt's ids.
	if len(doc.Prompts) < 2 {
		t.Skip("need >=2 prompts")
	}
	prefix := doc.Prompts[0].Ids
	suffix := doc.Prompts[1].Ids

	// compute the shared prefix once
	base := m.NewSession()
	base.Prefill(prefix)

	// REUSE path: clone the prefix cache, prefill only the suffix
	reuse := m.SessionFromPrefix(base.Cache)
	lReuse := reuse.Prefill(suffix)

	// RECOMPUTE path: prefill the whole thing fresh
	full := m.NewSession()
	lFull := full.Prefill(append(append([]int{}, prefix...), suffix...))

	d, _ := maxAbsDiff(lReuse, lFull)
	t.Logf("KV-prefix reuse vs full recompute: last-logit max|Δ|=%.3e (prefix prefill SKIPPED: %d positions)",
		d, len(prefix))
	if argmax(lReuse) != argmax(lFull) {
		t.Errorf("reuse argmax %d != recompute %d", argmax(lReuse), argmax(lFull))
	}
	if d != 0 {
		t.Errorf("KV-prefix reuse not bit-identical to recompute (max|Δ|=%.3e)", d)
	}
	if reuse.Cache.Len() != len(prefix)+len(suffix) {
		t.Errorf("reuse cache len %d != %d", reuse.Cache.Len(), len(prefix)+len(suffix))
	}

	// greedy continuation must also be identical
	gReuse := greedyContinue(reuse, lReuse, 8)
	gFull := greedyContinue(full, lFull, 8)
	t.Logf("reuse greedy=%v", gReuse)
	t.Logf("full  greedy=%v", gFull)
	if !eq(gReuse, gFull) {
		t.Errorf("reuse continuation != recompute\n  reuse=%v\n  full =%v", gReuse, gFull)
	}
}
