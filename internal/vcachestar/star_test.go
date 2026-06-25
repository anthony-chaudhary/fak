package vcachestar

import (
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

func testScope() Identity {
	return Identity{
		ModelID:          "claude-sonnet-4.6",
		TokenizerEpoch:   "tok-2026-06",
		BreakpointLayout: "system-last-block",
		TTL:              "5m",
		ProviderSurface:  "anthropic:first-party",
	}
}

func TestPreflightAppliesRecommendLayoutAndRefusesNoncanonicalWarmCandidate(t *testing.T) {
	req := PreflightRequest{
		Scope:           testScope(),
		MinAnchorTokens: 100,
		Parts: []Part{
			{Section: SectionMessages, Kind: cachemeta.SegVolatile, Content: []byte("request_id=abc\r\n"), Tokens: 1},
			{Section: SectionSystem, Content: []byte("\xef\xbb\xbfsystem\r\n"), Tokens: 100},
			{Section: SectionTools, Name: "search", Content: []byte(`{"z":2,"a":1}`), Tokens: 50},
			{Section: SectionMessages, Content: []byte("user question\n"), Tokens: 10},
		},
	}
	got := Preflight(req)
	if got.Action != ActionRewrite {
		t.Fatalf("action = %q, want rewrite (%s)", got.Action, got.Reason)
	}
	if got.Recommendation.MovedVolatile != 1 {
		t.Fatalf("moved volatile = %d, want 1", got.Recommendation.MovedVolatile)
	}
	if got.Applied[len(got.Applied)-1].Kind != cachemeta.SegVolatile {
		t.Fatalf("last applied kind = %q, want volatile tail", got.Applied[len(got.Applied)-1].Kind)
	}
	wantPrefix := []byte(`{"a":1,"z":2}` + "system\nuser question\n")
	if !bytes.Equal(got.PrefixBytes, wantPrefix) {
		t.Fatalf("prefix bytes = %q, want %q", got.PrefixBytes, wantPrefix)
	}

	req.WarmCandidateBytes = []byte("request_id=abc\n" + string(wantPrefix))
	refused := Preflight(req)
	if refused.Action != ActionRefuse || refused.Reason != ReasonCanonicalWarmMismatch {
		t.Fatalf("noncanonical candidate = %q/%q, want refuse/%q", refused.Action, refused.Reason, ReasonCanonicalWarmMismatch)
	}
}

func TestManifestKeyHashesSerializedPrefixBytesAndScopesHardMissAxes(t *testing.T) {
	req := PreflightRequest{
		Scope:           testScope(),
		MinAnchorTokens: 1,
		Parts: []Part{
			{Section: SectionMessages, Content: []byte("tail"), Tokens: 1},
			{Section: SectionSystem, Content: []byte("sys\n"), Tokens: 10},
			{Section: SectionTools, Name: "b", Content: []byte(`{"b":2}`), Tokens: 2},
			{Section: SectionTools, Name: "a", Content: []byte(`{"z":1,"a":0}`), Tokens: 3},
		},
	}
	got := Preflight(req)
	if got.Action == ActionRefuse {
		t.Fatalf("preflight refused: %s", got.Reason)
	}
	wantBytes := []byte(`{"a":0,"z":1}` + `{"b":2}` + "sys\n" + "tail")
	if !bytes.Equal(got.PrefixBytes, wantBytes) {
		t.Fatalf("prefix bytes = %q, want tools->system->messages %q", got.PrefixBytes, wantBytes)
	}
	if got.Key.PrefixHash != Digest(wantBytes) {
		t.Fatalf("prefix hash = %s, want digest of exact serialized bytes %s", got.Key.PrefixHash, Digest(wantBytes))
	}

	same := got.Key
	if ok, reason := got.Key.Match(same); !ok || reason != ReasonNone {
		t.Fatalf("same key match = %v/%q, want true/none", ok, reason)
	}
	for _, tc := range []struct {
		name string
		mut  func(*ManifestKey)
		want Reason
	}{
		{"model", func(k *ManifestKey) { k.Scope.ModelID = "other" }, ReasonModelMismatch},
		{"tokenizer", func(k *ManifestKey) { k.Scope.TokenizerEpoch = "tok-next" }, ReasonTokenizerMismatch},
		{"toolset", func(k *ManifestKey) { k.Scope.ToolSetHash = "other-tools" }, ReasonToolSetMismatch},
		{"breakpoint", func(k *ManifestKey) { k.Scope.BreakpointLayout = "other-layout" }, ReasonBreakpointLayoutMismatch},
		{"ttl", func(k *ManifestKey) { k.Scope.TTL = "1h" }, ReasonTTLMismatch},
		{"surface", func(k *ManifestKey) { k.Scope.ProviderSurface = "bedrock" }, ReasonProviderSurfaceMismatch},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := got.Key
			tc.mut(&want)
			if ok, reason := got.Key.Match(want); ok || reason != tc.want {
				t.Fatalf("match = %v/%q, want false/%q", ok, reason, tc.want)
			}
		})
	}
}

func TestPlanModelsAnchorAndUsesFirstNaturalRequestWithoutDedicatedWarm(t *testing.T) {
	key := ManifestKey{
		PrefixHash:   "abc",
		PrefixBytes:  4096,
		PrefixTokens: 2048,
		Scope:        Identity{ModelID: "m", TokenizerEpoch: "tok", ToolSetHash: "tools", BreakpointLayout: "bp", TTL: "5m", ProviderSurface: "openai"},
	}
	dec := Plan(StarRequest{
		Key:                  key,
		MinAnchorTokens:      1024,
		UnitTokens:           10,
		ExpectedSiblingReads: 4,
	})
	if dec.Strategy != StrategyFirstNaturalWarm {
		t.Fatalf("strategy = %q/%q, want first-natural warm", dec.Strategy, dec.Reason)
	}
	if dec.CacheUnit != CacheUnitAnchor {
		t.Fatalf("cache unit = %q, want anchor", dec.CacheUnit)
	}
	if dec.DedicatedWarm {
		t.Fatalf("M2 star anchor must not spend a dedicated warm: %+v", dec)
	}
	if !dec.FirstNaturalRequestWarms {
		t.Fatalf("first natural request did not warm: %+v", dec)
	}

	cold := Plan(StarRequest{Key: key, MinAnchorTokens: 4096, UnitTokens: 10, ExpectedSiblingReads: 4})
	if cold.Strategy != StrategyNone || cold.CacheUnit != CacheUnitAnchor || cold.Reason != ReasonBelowMinimumAnchor {
		t.Fatalf("below-min anchor = %+v, want no strategy but still anchor-modeled refusal", cold)
	}
}

func TestFoldTelemetryDemotesBelievedWarmZeroReadAndReportsDivergence(t *testing.T) {
	prev := []cachemeta.PromptSegment{
		{Kind: cachemeta.SegStable, Tokens: 100, Content: []byte("system")},
		{Kind: cachemeta.SegMessage, Tokens: 20, Content: []byte("old docs")},
	}
	next := []cachemeta.PromptSegment{
		{Kind: cachemeta.SegStable, Tokens: 100, Content: []byte("system")},
		{Kind: cachemeta.SegMessage, Tokens: 20, Content: []byte("new docs")},
	}
	res := FoldTelemetry(Belief{
		Warm:            true,
		LastPrefix:      prev,
		LastPrefixBytes: []byte("systemold docs"),
	}, Telemetry{
		CacheReadInputTokens: 0,
		UncachedInputTokens:  120,
		CurrentPrefix:        next,
		CurrentPrefixBytes:   []byte("systemnew docs"),
	})
	if !res.Demoted || !res.Alarm || res.Belief.Warm {
		t.Fatalf("zero-read fold = %+v, want demoted alarm cold", res)
	}
	if res.Reason != ReasonBelievedWarmZeroRead {
		t.Fatalf("reason = %q, want %q", res.Reason, ReasonBelievedWarmZeroRead)
	}
	if res.FirstDivergeTokenOffset != 100 {
		t.Fatalf("first diverge token offset = %d, want 100", res.FirstDivergeTokenOffset)
	}
	if res.FirstDivergeByteOffset != len("system") {
		t.Fatalf("first diverge byte offset = %d, want %d", res.FirstDivergeByteOffset, len("system"))
	}
}

func TestCostBooksUncachedAndRebatesOnlyConfirmedHits(t *testing.T) {
	miss := FoldTelemetry(Belief{Warm: true}, Telemetry{CacheReadInputTokens: 0, UncachedInputTokens: 200})
	if miss.Cost.BookedUncachedTokens != 200 || miss.Cost.RebateTokens != 0 {
		t.Fatalf("miss cost = %+v, want booked 200 rebate 0", miss.Cost)
	}
	hit := FoldTelemetry(Belief{Warm: false}, Telemetry{CacheReadInputTokens: 150, UncachedInputTokens: 200})
	if !hit.ConfirmedHit || !hit.Belief.Warm {
		t.Fatalf("hit fold = %+v, want confirmed warm", hit)
	}
	if hit.Cost.BookedUncachedTokens != 200 || hit.Cost.RebateTokens != 150 {
		t.Fatalf("hit cost = %+v, want booked 200 rebate 150", hit.Cost)
	}
}
