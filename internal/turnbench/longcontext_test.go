package turnbench

import (
	"math"
	"testing"
)

// prefillTokensRef is an INDEPENDENT reimplementation of cmd/sessionbench's prefillTokens
// (copied from cmd/sessionbench/main.go) — the token floor MUST match it byte-for-byte, so
// the long-context floor cross-validates against the live bench's own contention-free floor.
func prefillTokensRef(P, T, C, D, R int) (a, b, c int64) {
	for t := 0; t < T; t++ {
		a += int64(P + t*(D+R))
	}
	a *= int64(C)
	b = int64(C) * int64(P+(T-1)*R)
	c = int64(P) + int64(C)*int64((T-1)*R)
	return
}

func TestLongContextTokenFloorMatchesSessionbench(t *testing.T) {
	shapes := []SessionShape{
		{Prefix: 2048, Turns: 50, Agents: 5, Decode: 32, Result: 64},
		{Prefix: 100_000, Turns: 10, Agents: 1, Decode: 200, Result: 500},
		{Prefix: 100_000, Turns: 10, Agents: 5, Decode: 200, Result: 500},
		{Prefix: 100_000, Turns: 50, Agents: 5, Decode: 200, Result: 500},
	}
	s, _ := NamedShape("qwen25-7b")
	for _, sh := range shapes {
		cell := ProjectLongContext(s, sh)
		a, b, c := prefillTokensRef(sh.Prefix, sh.Turns, sh.Agents, sh.Decode, sh.Result)
		if cell.A.PrefillTokens != a || cell.B.PrefillTokens != b || cell.C.PrefillTokens != c {
			t.Fatalf("token floor mismatch for %+v: floor=(%d,%d,%d) sessionbench=(%d,%d,%d)",
				sh, cell.A.PrefillTokens, cell.B.PrefillTokens, cell.C.PrefillTokens, a, b, c)
		}
	}
}

// TestLongContextAntiInflation is the load-bearing honesty gate: a SINGLE agent (C=1) has NO
// cross-agent prefix to share, so the cross-agent win B/C MUST be exactly 1 on every floor —
// the search cannot manufacture a multi-agent win where there are no peers to share with.
func TestLongContextAntiInflation(t *testing.T) {
	s, _ := NamedShape("qwen25-7b")
	for _, sh := range []SessionShape{
		{Prefix: 100_000, Turns: 10, Agents: 1, Decode: 200, Result: 500},
		{Prefix: 2048, Turns: 50, Agents: 1, Decode: 32, Result: 64},
		{Prefix: 8192, Turns: 1, Agents: 1, Decode: 16, Result: 16},
	} {
		cell := ProjectLongContext(s, sh)
		if cell.TokenBOverC != 1.0 {
			t.Errorf("C=1 token B/C = %v, want exactly 1.0 (no cross-agent reuse to claim) for %+v", cell.TokenBOverC, sh)
		}
		if math.Abs(cell.FlopBOverC-1.0) > 1e-9 {
			t.Errorf("C=1 FLOP B/C = %v, want 1.0 for %+v", cell.FlopBOverC, sh)
		}
	}
}

func TestLongContextHeadlineRegimes(t *testing.T) {
	s, _ := NamedShape("qwen25-7b")

	// Single session, ultra-long: ~10× vs naive (the turn-tax of re-prefilling a 100k context).
	single := ProjectLongContext(s, SessionShape{Prefix: 100_000, Turns: 10, Agents: 1, Decode: 200, Result: 500})
	if !single.UltraLong {
		t.Fatalf("single session max context %d should be ultra-long (>= %d)", single.MaxContextTokens, UltraLongThreshold)
	}
	if single.TokenAOverC < 9 || single.TokenAOverC > 11 {
		t.Errorf("single-session token A/C = %.2f, want ~10× (in [9,11])", single.TokenAOverC)
	}
	if single.FlopAOverC < 5 || single.FlopAOverC > 15 {
		t.Errorf("single-session FLOP A/C = %.2f, want ~10× band [5,15]", single.FlopAOverC)
	}

	// Multi-agent (5), each ultra-long: ~40×+ vs naive (turn-tax × cross-agent prefix share).
	multi := ProjectLongContext(s, SessionShape{Prefix: 100_000, Turns: 10, Agents: 5, Decode: 200, Result: 500})
	if multi.TokenAOverC < 40 || multi.TokenAOverC > 45 {
		t.Errorf("multi-agent token A/C = %.2f, want ~40×+ (in [40,45])", multi.TokenAOverC)
	}
	if multi.TokenBOverC < 4 || multi.TokenBOverC > 4.5 {
		t.Errorf("multi-agent token B/C = %.2f, want ~4× vs tuned (in [4,4.5])", multi.TokenBOverC)
	}
	if multi.FlopAOverC < 25 || multi.FlopAOverC > 45 {
		t.Errorf("multi-agent FLOP A/C = %.2f, want ~30-40× band [25,45]", multi.FlopAOverC)
	}
}

// TestLongContextBOverCMonotoneInPrefix proves the honest reconciliation: the cross-agent
// win B/C rises monotonically with the shared-prefix fraction (from ~1 at tiny prefix toward
// the agent count at huge prefix), and never exceeds the agent count C. This is WHY the
// standing ~2–4× bound (measured at P≈2k) and the much larger ultra-long win are the SAME
// formula at different prefix fractions — not a contradiction.
func TestLongContextBOverCMonotoneInPrefix(t *testing.T) {
	s, _ := NamedShape("qwen25-7b")
	const C = 8
	prefixes := []int{512, 2048, 8192, 32_768, 100_000, 400_000}
	var prev float64
	for i, P := range prefixes {
		cell := ProjectLongContext(s, SessionShape{Prefix: P, Turns: 10, Agents: C, Decode: 200, Result: 500})
		boc := cell.FlopBOverC
		if boc < 1.0-1e-9 || boc > float64(C)+1e-9 {
			t.Errorf("P=%d: B/C = %.3f out of bounds [1, %d]", P, boc, C)
		}
		if i > 0 && boc < prev-1e-9 {
			t.Errorf("B/C not monotone in prefix: P=%d gave %.3f < previous %.3f", P, boc, prev)
		}
		prev = boc
	}
}

// TestPrefillWorkQuadraticDominates proves the attention quadratic is real in the floor: well
// past the model's linear/quadratic crossover, doubling the context MORE than doubles prefill
// work (the O(L^2) term overtakes the O(L) projection term) — the reason the naive re-prefill
// arm is catastrophic at ultra-long context.
func TestPrefillWorkQuadraticDominates(t *testing.T) {
	s, _ := NamedShape("qwen25-7b")
	w1 := s.PrefillWork(100_000)
	w2 := s.PrefillWork(200_000)
	if w2/w1 <= 2.0 {
		t.Errorf("PrefillWork(200k)/PrefillWork(100k) = %.3f, want > 2 (quadratic dominance)", w2/w1)
	}
	// AppendWork(0,L) is the from-scratch prefill.
	if s.AppendWork(0, 12_345) != s.PrefillWork(12_345) {
		t.Errorf("AppendWork(0,L) must equal PrefillWork(L)")
	}
	if s.AppendWork(50_000, 0) != 0 {
		t.Errorf("AppendWork(prior,0) must be 0 (no tokens appended)")
	}
	// More prior context ⇒ more attention work to append the same n tokens (the cross term).
	if s.AppendWork(100_000, 500) <= s.AppendWork(1_000, 500) {
		t.Errorf("AppendWork must grow with prior context length (the n·prior attention term)")
	}
}

// TestLongContextDecodeFLOPsInvariant: the decode FLOPs are identical across all three arms —
// the fused kernel's decode-batching is a BANDWIDTH win, not a FLOP win, so it is correctly
// absent from this work floor (and the win is never double-counted).
func TestLongContextDecodeFLOPsInvariant(t *testing.T) {
	s, _ := NamedShape("qwen25-7b")
	cell := ProjectLongContext(s, SessionShape{Prefix: 100_000, Turns: 10, Agents: 5, Decode: 200, Result: 500})
	if cell.A.DecodeFLOPs != cell.B.DecodeFLOPs || cell.B.DecodeFLOPs != cell.C.DecodeFLOPs {
		t.Errorf("decode FLOPs must be identical across arms: A=%g B=%g C=%g",
			cell.A.DecodeFLOPs, cell.B.DecodeFLOPs, cell.C.DecodeFLOPs)
	}
	// The floor only ever ELIMINATES work: every ratio is >= 1.
	for _, r := range []float64{cell.FlopAOverC, cell.FlopBOverC, cell.FlopAOverB, cell.TokenAOverC, cell.TokenBOverC} {
		if r < 1.0-1e-9 {
			t.Errorf("work-elimination ratio %.4f < 1 — the fused arm must never do MORE work", r)
		}
	}
}

func TestRunLongContextLadderDeterministicAndPicksRegimes(t *testing.T) {
	s, ok := NamedShape("qwen25-7b")
	if !ok {
		t.Fatal("qwen25-7b shape must exist")
	}
	r1 := RunLongContextLadder(s, CanonicalLadder(), DefaultCostModel())
	r2 := RunLongContextLadder(s, CanonicalLadder(), DefaultCostModel())
	if string(r1.JSON()) != string(r2.JSON()) {
		t.Fatal("ladder report must be deterministic (byte-identical across runs)")
	}
	if r1.Cost.Version != CostModelVersion {
		t.Fatalf("cost model version = %q, want %q", r1.Cost.Version, CostModelVersion)
	}
	if r1.SingleUltraLongIdx < 0 || r1.MultiUltraLongIdx < 0 {
		t.Fatalf("canonical ladder must contain a single-agent and a multi-agent ultra-long cell (got %d, %d)",
			r1.SingleUltraLongIdx, r1.MultiUltraLongIdx)
	}
	if c := r1.Cells[r1.SingleUltraLongIdx]; c.Shape.Agents != 1 || !c.UltraLong {
		t.Errorf("single ultra-long pick is wrong: %+v", c.Shape)
	}
	if c := r1.Cells[r1.MultiUltraLongIdx]; c.Shape.Agents <= 1 || !c.UltraLong {
		t.Errorf("multi ultra-long pick is wrong: %+v", c.Shape)
	}
}

func TestNamedShapeUnknown(t *testing.T) {
	if _, ok := NamedShape("gpt-5-ultra"); ok {
		t.Error("unknown shape must return ok=false")
	}
}
