package cachemeta

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// prefix_score_test.go — witnesses for issue #1602's live prefix-stability score
// (PrefixStabilityTracker / PrefixStabilityScore). Named so `go test -run Prefix`
// selects them alongside the §A3/§A4 prefix witnesses.

func protectedSpan() []PromptSegment {
	return []PromptSegment{
		seg(SegStable, 100, "You are a coding agent. Follow the rules."),
		seg(SegToolSchema, 200, `{"tools":[{"name":"read"},{"name":"write"}]}`),
	}
}

func TestPrefixStabilityTrackerFirstObserveIsUnknown(t *testing.T) {
	tr := NewPrefixStabilityTracker("", abi.ScopeAgent)
	score := tr.Observe(protectedSpan())
	if score.State != PrefixUnknown {
		t.Fatalf("first Observe: state = %q, want %q", score.State, PrefixUnknown)
	}
	if score.BaselineTurn != 1 || score.ObservedTurn != 1 {
		t.Fatalf("first Observe: BaselineTurn=%d ObservedTurn=%d, want 1/1", score.BaselineTurn, score.ObservedTurn)
	}
	if score.FirstDivergentSegment != -1 {
		t.Fatalf("first Observe: FirstDivergentSegment = %d, want -1 (no divergence to report)", score.FirstDivergentSegment)
	}
}

func TestPrefixStabilityTrackerIdenticalTurnIsStable(t *testing.T) {
	tr := NewPrefixStabilityTracker("", abi.ScopeAgent)
	tr.Observe(protectedSpan())
	score := tr.Observe(protectedSpan())
	if score.State != PrefixStable {
		t.Fatalf("identical second turn: state = %q, want %q (%+v)", score.State, PrefixStable, score)
	}
	if score.BaselineTurn != 1 || score.ObservedTurn != 2 {
		t.Fatalf("identical second turn: BaselineTurn=%d ObservedTurn=%d, want 1/2", score.BaselineTurn, score.ObservedTurn)
	}
	if score.FirstDivergentSegment != -1 {
		t.Fatalf("stable turn should report no divergent segment, got %d", score.FirstDivergentSegment)
	}
	if score.LostTokens != 0 {
		t.Fatalf("stable turn lost %d tokens, want 0", score.LostTokens)
	}
}

func TestPrefixStabilityTrackerPureAppendStaysStable(t *testing.T) {
	// A conversation growing by appending new messages after the protected span must
	// still read prefix-stable: the whole protected span is still cacheable, exactly
	// as Diverge/AnalyzeStability treat a tail append as "not a break".
	tr := NewPrefixStabilityTracker("", abi.ScopeAgent)
	tr.Observe(protectedSpan())
	grown := append(append([]PromptSegment(nil), protectedSpan()...), seg(SegMessage, 25, "fix the bug in foo.go"))
	score := tr.Observe(grown)
	if score.State != PrefixStable {
		t.Fatalf("pure append: state = %q, want %q (%+v)", score.State, PrefixStable, score)
	}
	if score.FirstDivergentSegment != -1 {
		t.Fatalf("pure append: FirstDivergentSegment = %d, want -1", score.FirstDivergentSegment)
	}
}

func TestPrefixStabilityTrackerMutationReportsFirstDivergentSpan(t *testing.T) {
	tr := NewPrefixStabilityTracker("", abi.ScopeAgent)
	tr.Observe(protectedSpan())
	mutated := []PromptSegment{
		seg(SegStable, 100, "You are a coding agent. Follow the rules."),                        // unchanged
		seg(SegToolSchema, 210, `{"tools":[{"name":"read"},{"name":"write"},{"name":"edit"}]}`), // changed
	}
	score := tr.Observe(mutated)
	if score.State != PrefixMutated {
		t.Fatalf("mutated turn: state = %q, want %q (%+v)", score.State, PrefixMutated, score)
	}
	if score.FirstDivergentSegment != 1 {
		t.Fatalf("mutated turn: FirstDivergentSegment = %d, want 1 (the tool schema)", score.FirstDivergentSegment)
	}
	if score.FirstDivergentTokenOffset != 100 {
		t.Fatalf("mutated turn: FirstDivergentTokenOffset = %d, want 100 (after the untouched system prompt)", score.FirstDivergentTokenOffset)
	}
	if score.FirstDivergentKind != SegToolSchema {
		t.Fatalf("mutated turn: FirstDivergentKind = %q, want %q", score.FirstDivergentKind, SegToolSchema)
	}
	if score.LostTokens != 210 {
		t.Fatalf("mutated turn: LostTokens = %d, want 210 (whole changed tool schema re-billed)", score.LostTokens)
	}
	if score.ProtectedSpanBroken {
		t.Fatalf("an ordinary content mutation must not set ProtectedSpanBroken")
	}
}

func TestPrefixStabilityTrackerMutationAtSystemPromptDivergesAtZero(t *testing.T) {
	tr := NewPrefixStabilityTracker("", abi.ScopeAgent)
	tr.Observe(protectedSpan())
	mutated := []PromptSegment{
		seg(SegStable, 110, "You are a coding agent. Follow the NEW rules."), // changed
		seg(SegToolSchema, 200, `{"tools":[{"name":"read"},{"name":"write"}]}`),
	}
	score := tr.Observe(mutated)
	if score.State != PrefixMutated {
		t.Fatalf("state = %q, want %q", score.State, PrefixMutated)
	}
	if score.FirstDivergentSegment != 0 {
		t.Fatalf("FirstDivergentSegment = %d, want 0 (the system prompt itself changed)", score.FirstDivergentSegment)
	}
	if score.FirstDivergentTokenOffset != 0 {
		t.Fatalf("FirstDivergentTokenOffset = %d, want 0", score.FirstDivergentTokenOffset)
	}
}

func TestPrefixStabilityTrackerSealedSpanIsProtectedSpanBroken(t *testing.T) {
	tr := NewPrefixStabilityTracker("", abi.ScopeAgent)
	base := []PromptSegment{
		seg(SegStable, 100, "system"),
		seg(SegMessage, 30, "continue"),
	}
	tr.Observe(base)
	sealedNext := []PromptSegment{
		seg(SegStable, 100, "system"),
		seg(SegSealed, 50, "QUARANTINED-TOOL-OUTPUT"),
	}
	score := tr.Observe(sealedNext)
	if score.State != PrefixMutated {
		t.Fatalf("sealed span: state = %q, want %q", score.State, PrefixMutated)
	}
	if !score.ProtectedSpanBroken {
		t.Fatalf("sealed span must set ProtectedSpanBroken=true")
	}
	if score.FirstDivergentSegment != 1 {
		t.Fatalf("sealed span: FirstDivergentSegment = %d, want 1", score.FirstDivergentSegment)
	}
}

func TestPrefixStabilityTrackerResetReturnsToUnknown(t *testing.T) {
	tr := NewPrefixStabilityTracker("", abi.ScopeAgent)
	tr.Observe(protectedSpan())
	tr.Observe(protectedSpan()) // stable
	tr.Reset()
	score := tr.Observe(protectedSpan())
	if score.State != PrefixUnknown {
		t.Fatalf("after Reset: state = %q, want %q", score.State, PrefixUnknown)
	}
	if score.BaselineTurn != 3 {
		t.Fatalf("after Reset: BaselineTurn = %d, want 3 (the 3rd Observe call reseeds)", score.BaselineTurn)
	}
}

func TestPrefixStabilityTrackerCarriesTTLAndShareScope(t *testing.T) {
	tr := NewPrefixStabilityTracker("1h", abi.ScopeFleet)
	score := tr.Observe(protectedSpan())
	if score.TTLMillis != 60*60*1000 {
		t.Fatalf("TTLMillis = %d, want %d (1h retention)", score.TTLMillis, 60*60*1000)
	}
	if score.ShareScope != abi.ScopeFleet {
		t.Fatalf("ShareScope = %v, want ScopeFleet", score.ShareScope)
	}
	// A default tracker (no retention, no explicit scope) stays fail-closed private
	// with an unknown TTL, mirroring cachemeta's ScopeAgent-default convention.
	def := NewPrefixStabilityTracker("", abi.ScopeAgent)
	defScore := def.Observe(protectedSpan())
	if defScore.TTLMillis != 0 {
		t.Fatalf("default tracker TTLMillis = %d, want 0 (unknown)", defScore.TTLMillis)
	}
	if defScore.ShareScope != abi.ScopeAgent {
		t.Fatalf("default tracker ShareScope = %v, want ScopeAgent (private, fail-closed)", defScore.ShareScope)
	}
}

func TestPrefixStabilityTrackerBreakEvenAbstainsWithoutCostData(t *testing.T) {
	tr := NewPrefixStabilityTracker("", abi.ScopeAgent)
	score := tr.Observe(protectedSpan())
	if score.BreakEvenTokens != -1 {
		t.Fatalf("BreakEvenTokens = %d, want -1 (no size/tokens/prefill cost supplied)", score.BreakEvenTokens)
	}
}

func TestPrefixStabilityTrackerBreakEvenFavorsRetainForLargeCheapToStagePrefix(t *testing.T) {
	tr := NewPrefixStabilityTracker("", abi.ScopeAgent)
	tr.SizeBytes = 1 << 20        // 1 MiB
	tr.Tokens = 50_000            // a large, expensive-to-recompute prefix
	tr.PerTokenPrefillNanos = 100 // 100ns/token to recompute
	score := tr.Observe(protectedSpan())
	if score.BreakEvenTokens != 1 {
		t.Fatalf("BreakEvenTokens = %d, want 1 (retaining a large prefix beats recompute)", score.BreakEvenTokens)
	}
}

func TestPrefixStabilityTrackerMultiTurnSequenceOfStates(t *testing.T) {
	tr := NewPrefixStabilityTracker("", abi.ScopeAgent)
	states := []PrefixStabilityState{}
	turns := [][]PromptSegment{
		protectedSpan(), // turn 1: unknown (seeds baseline)
		protectedSpan(), // turn 2: stable
		{
			seg(SegStable, 100, "You are a coding agent. Follow the rules."),
			seg(SegToolSchema, 250, `{"tools":[{"name":"read"},{"name":"write"},{"name":"bash"}]}`), // turn 3: mutated
		},
	}
	for _, turn := range turns {
		states = append(states, tr.Observe(turn).State)
	}
	want := []PrefixStabilityState{PrefixUnknown, PrefixStable, PrefixMutated}
	for i, w := range want {
		if states[i] != w {
			t.Fatalf("turn %d: state = %q, want %q (full sequence %v)", i+1, states[i], w, states)
		}
	}
}
