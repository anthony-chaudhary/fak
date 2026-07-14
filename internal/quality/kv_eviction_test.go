package quality

import (
	"strings"
	"testing"
)

// TestKVEvictionFaithfulRecomputePasses is the parity contract on the happy path:
// an engine that evicts positions out of its sliding window but recomputes them
// faithfully on demand decodes token-identically to the never-evicted reference.
// It also proves the pass is not vacuous: the evicting path really evicted (the
// on-demand recompute counter is positive), so parity was exercised, not skipped.
func TestKVEvictionFaithfulRecomputePasses(t *testing.T) {
	c := KVEvictionCase(42)
	res, err := RunCase(c, ReferenceRunner{}, KVEvictionEngine(""), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("faithful evict+recompute engine should match the full-KV reference; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean parity run must not carry a failure bundle: %+v", res.FailureBundle)
	}

	// Non-vacuity: the case must be long enough that the window evicted and the
	// engine recomputed at least one position on demand.
	tr, recomputes := kvEvictionDecodeEvicting(c.Params.Seed, c.Params.MaxTokens, kvEvictionCell)
	if recomputes == 0 {
		t.Fatalf("eviction never fired over %d steps with window %d; parity was not exercised",
			c.Params.MaxTokens, kvEvictionWindow)
	}
	full := kvEvictionDecodeFull(c.Params.Seed, c.Params.MaxTokens)
	if strings.Join(tr.Tokens, " ") != strings.Join(full.Tokens, " ") {
		t.Fatalf("faithful evicting decode differs from full-KV decode:\n  full:     %v\n  evicting: %v",
			full.Tokens, tr.Tokens)
	}
}

// TestKVEvictionLostPositionFailsAtDependentToken is the content-loss witness: a
// recompute that loses an evicted position's content (returns an empty cell) fails
// at exactly the FIRST token whose attention fold depended on that position —
// kvEvictionFirstMissStep, mid-sequence — with the intact prefix proving the
// localization is doing work.
func TestKVEvictionLostPositionFailsAtDependentToken(t *testing.T) {
	c := KVEvictionCase(42)
	res, err := RunCase(c, ReferenceRunner{}, KVEvictionEngine("lost-position"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("lost-position engine must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing parity run must carry a failure bundle")
	}
	if fb.FailingOracle != "kv-eviction-parity" {
		t.Errorf("first failing oracle = %q, want kv-eviction-parity", fb.FailingOracle)
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != kvEvictionFirstMissStep {
		t.Fatalf("expected first divergence at the first eviction-dependent step %d, got %+v",
			kvEvictionFirstMissStep, d)
	}
	if want := c.Reference.Tokens[kvEvictionFirstMissStep]; d.Reference != want {
		t.Errorf("divergence reference token = %q, want %q", d.Reference, want)
	}
	if got := fb.Engine.Tokens[kvEvictionFirstMissStep]; d.Engine != got {
		t.Errorf("divergence engine token = %q, want engine trace token %q", d.Engine, got)
	}
	// The prefix before the first cache miss is untouched by the defect: the
	// failure is pinned to the dependent token, not smeared over the whole decode.
	for i := 0; i < kvEvictionFirstMissStep; i++ {
		if fb.Engine.Tokens[i] != c.Reference.Tokens[i] {
			t.Errorf("token %d before the first miss should match the reference: ref %q eng %q",
				i, c.Reference.Tokens[i], fb.Engine.Tokens[i])
		}
	}
}

// TestKVEvictionStaleRecomputeFails is the miscompute witness: a recompute that
// rebuilds an evicted position from the wrong offset (off by one) diverges at the
// first token that attended to it, and the reported divergence is the FIRST
// token-wise mismatch between the two traces, computed independently here.
func TestKVEvictionStaleRecomputeFails(t *testing.T) {
	c := KVEvictionCase(42)
	res, err := RunCase(c, ReferenceRunner{}, KVEvictionEngine("stale-recompute"), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("stale-recompute engine must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing parity run must carry a failure bundle")
	}
	d := fb.FirstDivergence
	if d == nil {
		t.Fatal("stale-recompute failure must localize a first divergence")
	}
	ref, eng := fb.Reference.Tokens, fb.Engine.Tokens
	first := -1
	for i := 0; i < len(ref) && i < len(eng); i++ {
		if ref[i] != eng[i] {
			first = i
			break
		}
	}
	if first < 0 {
		t.Fatalf("stale-recompute traces should disagree token-wise; ref %v eng %v", ref, eng)
	}
	if d.Index != first {
		t.Errorf("divergence index = %d, want first mismatch %d", d.Index, first)
	}
	if first != kvEvictionFirstMissStep {
		t.Errorf("first mismatch = %d, want the first eviction-dependent step %d", first, kvEvictionFirstMissStep)
	}
	if d.Reference != ref[first] || d.Engine != eng[first] {
		t.Errorf("divergence tokens = ref %q eng %q, want ref %q eng %q",
			d.Reference, d.Engine, ref[first], eng[first])
	}
}

// TestKVEvictionTruncatedDecodeFailsAtLength covers the third defect shape the
// oracle guards: an evicting engine that stops early (e.g. crashes on a cache
// miss) is a length divergence at the truncation point, never a silent pass.
func TestKVEvictionTruncatedDecodeFailsAtLength(t *testing.T) {
	c := KVEvictionCase(42)
	full := kvEvictionDecodeFull(c.Params.Seed, c.Params.MaxTokens)
	short := Trace{Tokens: full.Tokens[:kvEvictionFirstMissStep], Text: strings.Join(full.Tokens[:kvEvictionFirstMissStep], " ")}
	v := KVEvictionParity{}.Judge(c.Reference, short, c)
	if v.Pass {
		t.Fatal("truncated evicting decode must not pass parity")
	}
	if v.FirstDivergence == nil || v.FirstDivergence.Index != kvEvictionFirstMissStep {
		t.Fatalf("expected length divergence at %d, got %+v", kvEvictionFirstMissStep, v.FirstDivergence)
	}
}
