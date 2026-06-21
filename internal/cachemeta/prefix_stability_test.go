package cachemeta

import "testing"

// prefix_stability_test.go — witnesses for the §A3 prefix-stability linter.

func seg(kind SegmentKind, tokens int64, content string) PromptSegment {
	return PromptSegment{Kind: kind, Tokens: tokens, Content: []byte(content)}
}

func stablePrefix() []PromptSegment {
	return []PromptSegment{
		seg(SegStable, 100, "You are a coding agent. Follow the rules."),
		seg(SegToolSchema, 200, `{"tools":[{"name":"read"},{"name":"write"}]}`),
	}
}

func TestDivergeIdenticalPromptsAreFullyCacheable(t *testing.T) {
	p := stablePrefix()
	d := Diverge(p, append([]PromptSegment(nil), p...))
	if !d.Identical {
		t.Fatalf("identical prompts: Identical=false (%+v)", d)
	}
	if d.LostTokens != 0 {
		t.Fatalf("identical prompts lost %d tokens, want 0", d.LostTokens)
	}
	if d.StableTokens != 300 {
		t.Fatalf("identical prompts cacheable %d, want 300", d.StableTokens)
	}
	if got := d.FirstDivergeTokenOffset(); got != 0 {
		t.Fatalf("no divergence but offset = %d, want 0", got)
	}
}

func TestDivergePureAppendKeepsWholePrefixCacheable(t *testing.T) {
	prev := stablePrefix()
	next := append(append([]PromptSegment(nil), prev...), seg(SegMessage, 25, "fix the bug in foo.go"))
	d := Diverge(prev, next)
	if d.StableTokens != 300 {
		t.Fatalf("append: cacheable %d, want 300 (whole prefix)", d.StableTokens)
	}
	if d.LostTokens != 25 {
		t.Fatalf("append: lost %d, want 25 (only the new tail)", d.LostTokens)
	}
	if d.FirstDivergeSeg != len(prev) {
		t.Fatalf("append: FirstDivergeSeg %d, want %d (a clean tail append)", d.FirstDivergeSeg, len(prev))
	}
	if d.Identical {
		t.Fatalf("append: Identical=true, want false")
	}
}

func TestDivergeVolatileInMiddleBreaksPrefixEarly(t *testing.T) {
	prev := []PromptSegment{
		seg(SegStable, 100, "You are a coding agent. Follow the rules."),
		seg(SegVolatile, 8, "ts=1782020000"),
		seg(SegToolSchema, 200, `{"tools":[{"name":"read"},{"name":"write"}]}`),
	}
	next := []PromptSegment{
		seg(SegStable, 100, "You are a coding agent. Follow the rules."),
		seg(SegVolatile, 8, "ts=1782026951"), // changed
		seg(SegToolSchema, 200, `{"tools":[{"name":"read"},{"name":"write"}]}`),
	}
	d := Diverge(prev, next)
	if d.StableTokens != 100 {
		t.Fatalf("volatile-in-middle: cacheable %d, want 100 (only the system prompt)", d.StableTokens)
	}
	if d.LostTokens != 208 {
		t.Fatalf("volatile-in-middle: lost %d, want 208 (volatile 8 + schema 200)", d.LostTokens)
	}
	if d.FirstDivergeSeg != 1 {
		t.Fatalf("volatile-in-middle: FirstDivergeSeg %d, want 1", d.FirstDivergeSeg)
	}
	if got := d.FirstDivergeTokenOffset(); got != 100 {
		t.Fatalf("volatile-in-middle: FirstDivergeAt offset %d, want 100", got)
	}
}

func TestDivergeSealedSegmentStopsPrefixEvenWhenBytesMatch(t *testing.T) {
	prefix := []PromptSegment{
		seg(SegStable, 100, "system"),
		seg(SegSealed, 50, "QUARANTINED-TOOL-OUTPUT"),
		seg(SegMessage, 30, "continue"),
	}
	d := Diverge(prefix, append([]PromptSegment(nil), prefix...))
	if !d.SealedStop {
		t.Fatalf("sealed prefix: SealedStop=false, want true")
	}
	if d.StableTokens != 100 {
		t.Fatalf("sealed prefix: cacheable %d, want 100 (stop before the sealed span)", d.StableTokens)
	}
	if d.LostTokens != 80 {
		t.Fatalf("sealed prefix: lost %d, want 80 (sealed 50 + tail 30)", d.LostTokens)
	}
	if d.Identical {
		t.Fatalf("sealed prefix must never report Identical (would imply re-serve)")
	}
}

func TestLintPrefixLayoutFlagsVolatileAheadOfStable(t *testing.T) {
	turn := []PromptSegment{
		seg(SegVolatile, 6, "req=abc123"),
		seg(SegStable, 100, "system prompt"),
		seg(SegToolSchema, 200, "tool schema"),
		seg(SegMessage, 20, "user msg"),
	}
	lint := LintPrefixLayout(turn)
	if !lint.Fixable {
		t.Fatalf("volatile-ahead: Fixable=false, want true")
	}
	if lint.EarliestVolatileSeg != 0 {
		t.Fatalf("volatile-ahead: EarliestVolatileSeg %d, want 0", lint.EarliestVolatileSeg)
	}
	if lint.StrandedStableTokens != 300 {
		t.Fatalf("volatile-ahead: stranded %d, want 300 (system 100 + schema 200)", lint.StrandedStableTokens)
	}
}

func TestLintPrefixLayoutVolatileAtTailIsClean(t *testing.T) {
	turn := []PromptSegment{
		seg(SegStable, 100, "system prompt"),
		seg(SegToolSchema, 200, "tool schema"),
		seg(SegMessage, 20, "user msg"),
		seg(SegVolatile, 6, "ts=now"),
	}
	lint := LintPrefixLayout(turn)
	if lint.Fixable {
		t.Fatalf("volatile-at-tail: Fixable=true, want false (%+v)", lint)
	}
	if lint.StrandedStableTokens != 0 {
		t.Fatalf("volatile-at-tail: stranded %d, want 0", lint.StrandedStableTokens)
	}
}

func TestRecommendLayoutHoistsVolatileAndPredictsUplift(t *testing.T) {
	turn := []PromptSegment{
		seg(SegVolatile, 6, "req=abc123"),
		seg(SegStable, 100, "system prompt"),
		seg(SegToolSchema, 200, "tool schema"),
		seg(SegMessage, 20, "user msg"),
	}
	rec := RecommendLayout(turn)
	if !rec.Changed {
		t.Fatalf("volatile-ahead: Changed=false, want true")
	}
	if rec.BeforeCacheable != 0 {
		t.Fatalf("before: cacheable front run %d, want 0 (volatile leads)", rec.BeforeCacheable)
	}
	if rec.AfterCacheable != 320 {
		t.Fatalf("after: cacheable front run %d, want 320 (stable+schema+msg)", rec.AfterCacheable)
	}
	if rec.PredictedUplift != 320 {
		t.Fatalf("uplift %d, want 320", rec.PredictedUplift)
	}
	if rec.MovedVolatile != 1 {
		t.Fatalf("moved %d volatile, want 1", rec.MovedVolatile)
	}
	if rec.Reordered[len(rec.Reordered)-1].Kind != SegVolatile {
		t.Fatalf("recommended layout does not end with the volatile segment: %+v", rec.Reordered)
	}
}

func TestRecommendLayoutAlreadyOptimalIsNoChange(t *testing.T) {
	turn := []PromptSegment{
		seg(SegStable, 100, "system prompt"),
		seg(SegToolSchema, 200, "tool schema"),
		seg(SegMessage, 20, "user msg"),
		seg(SegVolatile, 6, "ts=now"),
	}
	rec := RecommendLayout(turn)
	if rec.Changed {
		t.Fatalf("already-optimal: Changed=true, want false (%+v)", rec)
	}
	if rec.PredictedUplift != 0 {
		t.Fatalf("already-optimal: uplift %d, want 0", rec.PredictedUplift)
	}
	if rec.BeforeCacheable != 320 || rec.AfterCacheable != 320 {
		t.Fatalf("already-optimal: before/after %d/%d, want 320/320", rec.BeforeCacheable, rec.AfterCacheable)
	}
}

func TestRecommendLayoutSealedStillCapsTheRunAfterReorder(t *testing.T) {
	turn := []PromptSegment{
		seg(SegVolatile, 6, "req=1"),
		seg(SegStable, 100, "pre-sealed system"),
		seg(SegSealed, 50, "QUARANTINED"),
		seg(SegStable, 100, "post-sealed"),
	}
	rec := RecommendLayout(turn)
	if rec.BeforeCacheable != 0 {
		t.Fatalf("before %d, want 0", rec.BeforeCacheable)
	}
	if rec.AfterCacheable != 100 {
		t.Fatalf("after %d, want 100 (only the pre-sealed block; sealed caps the run)", rec.AfterCacheable)
	}
	if rec.PredictedUplift != 100 {
		t.Fatalf("uplift %d, want 100 (the post-sealed stable stays uncacheable)", rec.PredictedUplift)
	}
}

func TestRecommendLayoutNoVolatileIsNoChange(t *testing.T) {
	rec := RecommendLayout(stablePrefix())
	if rec.Changed || rec.MovedVolatile != 0 || rec.PredictedUplift != 0 {
		t.Fatalf("no-volatile turn should be unchanged: %+v", rec)
	}
}

func TestAnalyzeStabilityIdentifiesTheTurnThePrefixBroke(t *testing.T) {
	base := stablePrefix()
	turn0 := append(append([]PromptSegment(nil), base...), seg(SegMessage, 10, "hello"))
	turn1 := append(append([]PromptSegment(nil), turn0...), seg(SegMessage, 12, "and more"))
	turn2 := []PromptSegment{
		seg(SegStable, 110, "You are a coding agent. Follow the NEW rules."), // changed
		seg(SegToolSchema, 200, `{"tools":[{"name":"read"},{"name":"write"}]}`),
		seg(SegMessage, 14, "next"),
	}
	r := AnalyzeStability([][]PromptSegment{turn0, turn1, turn2})
	if r.Turns != 3 {
		t.Fatalf("Turns %d, want 3", r.Turns)
	}
	if r.BrokeAtTurn != 2 {
		t.Fatalf("BrokeAtTurn %d, want 2 (the system-prompt edit)", r.BrokeAtTurn)
	}
	if r.CacheableTokens != 310 {
		t.Fatalf("CacheableTokens %d, want 310", r.CacheableTokens)
	}
	if r.RecoverableTokens != 0 {
		t.Fatalf("RecoverableTokens %d, want 0 (no volatile-ahead ordering bug here)", r.RecoverableTokens)
	}
}

func TestAnalyzeStabilityScoresRecoverableOrderingUplift(t *testing.T) {
	mk := func(reqid, msg string) []PromptSegment {
		return []PromptSegment{
			seg(SegVolatile, 6, reqid),
			seg(SegStable, 100, "system prompt"),
			seg(SegToolSchema, 200, "tool schema"),
			seg(SegMessage, 10, msg),
		}
	}
	r := AnalyzeStability([][]PromptSegment{mk("req=1", "a"), mk("req=2", "b"), mk("req=3", "c")})
	if r.RecoverableTokens != 900 {
		t.Fatalf("RecoverableTokens %d, want 900", r.RecoverableTokens)
	}
	if r.BrokeAtTurn != 1 {
		t.Fatalf("BrokeAtTurn %d, want 1", r.BrokeAtTurn)
	}
}

func TestPrefixStabilityFeedsProviderCacheFirstDivergeAt(t *testing.T) {
	prev := []PromptSegment{seg(SegStable, 100, "sys"), seg(SegVolatile, 8, "ts=old"), seg(SegMessage, 50, "m")}
	next := []PromptSegment{seg(SegStable, 100, "sys"), seg(SegVolatile, 8, "ts=new"), seg(SegMessage, 50, "m")}
	d := Diverge(prev, next)
	e := FromProviderCache(ProviderCache{
		Provider:       "zai",
		ModelID:        "glm-5.2",
		PromptTokens:   158,
		CachedTokens:   d.StableTokens,
		FirstDivergeAt: d.FirstDivergeTokenOffset(),
		Endpoint:       "coding",
		ReasoningMode:  "max",
	})
	if e.Labels["first_diverge_at"] != "100" {
		t.Fatalf("entry first_diverge_at = %q, want \"100\"", e.Labels["first_diverge_at"])
	}
	if v := ProviderCacheVerdict(e); v.Meta["provider_cache"] != "cost_latency_only" {
		t.Fatalf("provider entry not marked cost_latency_only: %+v", v.Meta)
	}
}
