package quality

import (
	"strings"
	"testing"
)

// TestPrefixCacheFaithfulHitMatchesCacheOff is the parity contract on the happy
// path: an engine that serves the prompt from a correctly-keyed warmed cache
// produces output identical to the cache-off reference. The hit counter proves
// the green run actually exercised cache REUSE — a pass that never touched the
// cache would be a tautology, not a witness.
func TestPrefixCacheFaithfulHitMatchesCacheOff(t *testing.T) {
	c := pcCase()
	eng := pcEngine("")
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("cache-on engine with a faithful key must match cache-off; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean parity run must not carry a failure bundle: %+v", res.FailureBundle)
	}
	if got := *eng.hits; got != 1 {
		t.Fatalf("faithful engine must have served the decode from its warmed cache (hits=1), got %d", got)
	}
}

// TestPrefixCacheColdMissStillMatches covers the miss side of the seam: an
// engine with an empty cache decodes fresh and still matches cache-off — parity
// holds whether or not the cache participates. The zero hit count also guards
// the hit counter the faithful test relies on.
func TestPrefixCacheColdMissStillMatches(t *testing.T) {
	c := pcCase()
	eng := pcEngine("cold")
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("cache-on engine with a cold cache must match cache-off; got %s", Explain(res))
	}
	if got := *eng.hits; got != 0 {
		t.Fatalf("cold engine must not have served from cache; hits=%d", got)
	}
}

// TestPrefixCacheStalePrefixFailsAtFirstSuffixToken is the localized-defect
// witness: a suffix-blind cache key serves output warmed for a DIFFERENT prompt
// sharing only the first pcSharedPrefixWords words. The prefix-covered tokens
// match — which is exactly why the defect is dangerous — and the oracle pins the
// first divergence to the first suffix-influenced token, with the reference and
// engine tokens reported.
func TestPrefixCacheStalePrefixFailsAtFirstSuffixToken(t *testing.T) {
	c := pcCase()
	eng := pcEngine("stale-prefix")
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("stale cached prefix must not pass parity; got %s", Explain(res))
	}
	if got := *eng.hits; got != 1 {
		t.Fatalf("stale engine must have served the mismatched entry from cache (hits=1), got %d", got)
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing parity run must carry a failure bundle")
	}
	if fb.FailingOracle != "prefix-cache-parity" {
		t.Errorf("first failing oracle = %q, want prefix-cache-parity", fb.FailingOracle)
	}
	d := fb.FirstDivergence
	if d == nil {
		t.Fatal("stale-prefix failure must localize a first divergence")
	}

	// Model self-check, computed independently of the oracle: the first mismatch
	// between the cache-off decodes of the case and stale prompts must be the
	// first suffix-influenced token.
	ref := pcDecode(pcCasePrompt, pcDecodeSteps).Tokens
	stale := pcDecode(pcStalePrompt, pcDecodeSteps).Tokens
	first := -1
	for i := 0; i < len(ref) && i < len(stale); i++ {
		if ref[i] != stale[i] {
			first = i
			break
		}
	}
	if first != pcStaleDivergeStep {
		t.Fatalf("model self-check: first suffix-influenced mismatch = %d, want %d (ref %v, stale %v)",
			first, pcStaleDivergeStep, ref, stale)
	}
	if d.Index != pcStaleDivergeStep {
		t.Errorf("divergence index = %d, want %d", d.Index, pcStaleDivergeStep)
	}
	if d.Reference != ref[pcStaleDivergeStep] || d.Engine != stale[pcStaleDivergeStep] {
		t.Errorf("divergence tokens = ref %q eng %q, want ref %q eng %q",
			d.Reference, d.Engine, ref[pcStaleDivergeStep], stale[pcStaleDivergeStep])
	}

	// The prefix-covered tokens must have matched — the localization is doing
	// work: the failure pins to the suffix, not to "the whole decode looked wrong".
	for i := 0; i < pcStaleDivergeStep; i++ {
		if fb.Reference.Tokens[i] != fb.Engine.Tokens[i] {
			t.Errorf("token %d inside the shared prefix should match: reference %q, engine %q",
				i, fb.Reference.Tokens[i], fb.Engine.Tokens[i])
		}
	}

	// And the engine trace is the stale prompt's decode wholesale — the failure
	// came from serving the mismatched cached entry, not from a decode bug.
	if got, want := strings.Join(fb.Engine.Tokens, " "), strings.Join(stale, " "); got != want {
		t.Errorf("engine trace should be the stale cached decode:\n  got  %s\n  want %s", got, want)
	}
}

// TestPrefixCacheKeyDefectIsSuffixBlind pins the defect mechanism itself: the
// strict key distinguishes the two prompts (a correct cache can never confuse
// them), while the defective suffix-blind key collides them — that collision is
// the injected bug the parity oracle exists to catch.
func TestPrefixCacheKeyDefectIsSuffixBlind(t *testing.T) {
	if pcCacheKey(pcCasePrompt, true) == pcCacheKey(pcStalePrompt, true) {
		t.Fatal("strict keys must distinguish prompts that differ in their suffix")
	}
	if pcCacheKey(pcCasePrompt, false) != pcCacheKey(pcStalePrompt, false) {
		t.Fatal("the defective suffix-blind key must collide for the two prompts")
	}
}
