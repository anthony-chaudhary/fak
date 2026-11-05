// main_test.go — the COMMAND-lane witness for longctxbench: the same public
// turnbench surface main.go drives (NamedShape, ProjectLongContext,
// CanonicalLadder, RunLongContextLadder, FmtRatio) plus intlist.Parse, pinned
// against hand-computed closed-form fixtures so a regression in the tool's
// inputs or the floor arithmetic fails here before it ever reaches the printed
// artifact. Pure arithmetic — no model, no GPU, deterministic.
package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/intlist"
	"github.com/anthony-chaudhary/fak/internal/turnbench"
)

// appendWorkRef recomputes the append-work closed form straight from raw
// geometry fields, mirroring the implementation's order of operations so the
// result is bit-exact — deliberately NOT calling ModelShape.AppendWork, because
// a shared bug would otherwise validate itself against its own helper.
func appendWorkRef(d, layers, heads, kvHeads, headDim, ff, prior, n int) float64 {
	attnDim := float64(heads * headDim)
	kvDim := float64(kvHeads * headDim)
	lin := float64(d)*attnDim + 2*float64(d)*kvDim + attnDim*float64(d) + 3*float64(d)*float64(ff)
	pairs := float64(n)*float64(prior) + float64(n)*float64(n)/2.0
	return float64(layers) * (lin*float64(n) + 2.0*pairs*attnDim)
}

// TestHandPinnedSmollm2Fixture pins the closed contracts by hand on the
// smallest named shape. Every constant was derived on paper from the published
// closed form (smollm2-135m: d=576, L=30, heads=9, kv=3, headDim=64, ff=1536 ⇒
// lin/tok/layer = 576·576 + 2·576·192 + 576·576 + 3·576·1536 = 3,538,944), so a
// drift in ANY factor — GQA kv-dim, the ×2 attention count, the (T−1) result
// spans — breaks a specific pin. If a pin fails, recompute the hand math before
// suspecting the implementation.
func TestHandPinnedSmollm2Fixture(t *testing.T) {
	t.Parallel()
	s, ok := turnbench.NamedShape("smollm2-135m")
	if !ok {
		t.Fatal("smollm2-135m shape must resolve")
	}
	cell := turnbench.ProjectLongContext(s, turnbench.SessionShape{Prefix: 100, Turns: 3, Agents: 2, Decode: 10, Result: 20})

	if cell.MaxContextTokens != 170 { // 100 + 2·(10+20) + 10
		t.Errorf("MaxContextTokens = %d, want 170", cell.MaxContextTokens)
	}
	// Token floors: A = 2·(100+130+160), B = 2·(100+2·20), C = 100+2·(2·20).
	if cell.A.PrefillTokens != 780 || cell.B.PrefillTokens != 280 || cell.C.PrefillTokens != 180 {
		t.Errorf("token floors = (%d,%d,%d), want (780,280,180)",
			cell.A.PrefillTokens, cell.B.PrefillTokens, cell.C.PrefillTokens)
	}
	// The same division the floor performs — exact float equality, not a band.
	if cell.TokenAOverC != float64(780)/float64(180) || cell.TokenBOverC != float64(280)/float64(180) {
		t.Errorf("token ratios = (%v,%v), want exactly (780/180, 280/180)", cell.TokenAOverC, cell.TokenBOverC)
	}
	// PrefillWork(100): per layer 3,538,944·100 + 2·(100²/2)·576 = 359,654,400; ×30 layers.
	if got := s.PrefillWork(100); got != 10_789_632_000 {
		t.Errorf("PrefillWork(100) = %g, want 10789632000", got)
	}
	// Incremental ingests: AppendWork(110,20) = 2,206,310,400; AppendWork(140,20) = 2,227,046,400.
	if got := s.AppendWork(110, 20); got != 2_206_310_400 {
		t.Errorf("AppendWork(110,20) = %g, want 2206310400", got)
	}
	if got := s.AppendWork(140, 20); got != 2_227_046_400 {
		t.Errorf("AppendWork(140,20) = %g, want 2227046400", got)
	}
	// Arm C pays the prefix ONCE: 10,789,632,000 + 2·4,433,356,800.
	if got := cell.C.PrefillFLOPs; got != 19_656_345_600 {
		t.Errorf("arm C PrefillFLOPs = %g, want 19656345600", got)
	}
}

// TestFloorsMatchIndependentClosedForm re-derives every arm's FLOP floor from
// the raw ModelShape fields with the formulas written inline (appendWorkRef),
// never through the library's own helpers — so it catches a sign or factor bug
// that a test going through AppendWork would inherit. Exact ==, no tolerance:
// the floor is integer-valued arithmetic rendered in float64.
func TestFloorsMatchIndependentClosedForm(t *testing.T) {
	t.Parallel()
	cases := []struct {
		model string
		sh    turnbench.SessionShape
	}{
		{"qwen25-7b", turnbench.SessionShape{Prefix: 100_000, Turns: 10, Agents: 5, Decode: 200, Result: 500}}, // multi-agent ultra-long
		{"smollm2-135m", turnbench.SessionShape{Prefix: 100, Turns: 3, Agents: 2, Decode: 10, Result: 20}},
		{"qwen25-1.5b", turnbench.SessionShape{Prefix: 2048, Turns: 50, Agents: 5, Decode: 32, Result: 64}},
	}
	for _, tc := range cases {
		s, ok := turnbench.NamedShape(tc.model)
		if !ok {
			t.Fatalf("%s shape must resolve", tc.model)
		}
		P, T, C, D, R := tc.sh.Prefix, tc.sh.Turns, tc.sh.Agents, tc.sh.Decode, tc.sh.Result
		g := func(prior, n int) float64 {
			return appendWorkRef(s.HiddenSize, s.NumLayers, s.NumHeads, s.NumKVHeads, s.HeadDim, s.IntermediateSize, prior, n)
		}
		var aPrefill, decodePerAgent, ingest float64
		for tt := 0; tt < T; tt++ {
			Lt := P + tt*(D+R)
			aPrefill += g(0, Lt) // naive: re-prefill the whole growing context
			decodePerAgent += g(Lt, D)
			if tt < T-1 {
				ingest += g(Lt+D, R) // between-turn result ingestion
			}
		}
		prefix := g(0, P)
		cf := float64(C)
		wantA, wantB, wantC := cf*aPrefill, cf*(prefix+ingest), prefix+cf*ingest
		wantDec := cf * decodePerAgent

		cell := turnbench.ProjectLongContext(s, tc.sh)
		if cell.A.PrefillFLOPs != wantA {
			t.Errorf("%s %+v: A.PrefillFLOPs = %g, want %g", tc.model, tc.sh, cell.A.PrefillFLOPs, wantA)
		}
		if cell.B.PrefillFLOPs != wantB {
			t.Errorf("%s %+v: B.PrefillFLOPs = %g, want %g", tc.model, tc.sh, cell.B.PrefillFLOPs, wantB)
		}
		if cell.C.PrefillFLOPs != wantC {
			t.Errorf("%s %+v: C.PrefillFLOPs = %g, want %g", tc.model, tc.sh, cell.C.PrefillFLOPs, wantC)
		}
		// Decode is identical across arms and equals the closed form; total = prefill + decode.
		if cell.A.DecodeFLOPs != wantDec || cell.B.DecodeFLOPs != wantDec || cell.C.DecodeFLOPs != wantDec {
			t.Errorf("%s %+v: decode FLOPs A/B/C = %g/%g/%g, want %g in all arms",
				tc.model, tc.sh, cell.A.DecodeFLOPs, cell.B.DecodeFLOPs, cell.C.DecodeFLOPs, wantDec)
		}
		if cell.A.TotalFLOPs != cell.A.PrefillFLOPs+cell.A.DecodeFLOPs ||
			cell.B.TotalFLOPs != cell.B.PrefillFLOPs+cell.B.DecodeFLOPs ||
			cell.C.TotalFLOPs != cell.C.PrefillFLOPs+cell.C.DecodeFLOPs {
			t.Errorf("%s %+v: TotalFLOPs must equal PrefillFLOPs+DecodeFLOPs per arm", tc.model, tc.sh)
		}
	}
}

// TestRatiosAreConsistentAndAntiInflated checks the ratios are exactly the
// quotients of the floors they display (no hidden rescaling), that the floor
// only ever eliminates work (every ratio >= 1), and the load-bearing honesty
// gate: with C=1 there are no peers to share the prefix with, so B/C MUST be
// exactly 1 — a cross-agent win cannot be manufactured where there are no
// agents.
func TestRatiosAreConsistentAndAntiInflated(t *testing.T) {
	t.Parallel()
	cases := []struct {
		model string
		sh    turnbench.SessionShape
	}{
		{"smollm2-135m", turnbench.SessionShape{Prefix: 100, Turns: 3, Agents: 2, Decode: 10, Result: 20}},
		{"qwen25-7b", turnbench.SessionShape{Prefix: 100_000, Turns: 10, Agents: 5, Decode: 200, Result: 500}},
		{"qwen25-7b", turnbench.SessionShape{Prefix: 100_000, Turns: 10, Agents: 1, Decode: 200, Result: 500}},
	}
	for _, tc := range cases {
		s, _ := turnbench.NamedShape(tc.model)
		cell := turnbench.ProjectLongContext(s, tc.sh)
		if cell.FlopAOverC != cell.A.TotalFLOPs/cell.C.TotalFLOPs {
			t.Errorf("%s %+v: FlopAOverC = %v, want A.Total/C.Total = %v",
				tc.model, tc.sh, cell.FlopAOverC, cell.A.TotalFLOPs/cell.C.TotalFLOPs)
		}
		if cell.FlopBOverC != cell.B.TotalFLOPs/cell.C.TotalFLOPs {
			t.Errorf("%s %+v: FlopBOverC = %v, want B.Total/C.Total = %v",
				tc.model, tc.sh, cell.FlopBOverC, cell.B.TotalFLOPs/cell.C.TotalFLOPs)
		}
		if cell.FlopAOverB != cell.A.TotalFLOPs/cell.B.TotalFLOPs {
			t.Errorf("%s %+v: FlopAOverB = %v, want A.Total/B.Total = %v",
				tc.model, tc.sh, cell.FlopAOverB, cell.A.TotalFLOPs/cell.B.TotalFLOPs)
		}
		for name, r := range map[string]float64{
			"token A/C": cell.TokenAOverC, "token B/C": cell.TokenBOverC,
			"flop A/C": cell.FlopAOverC, "flop B/C": cell.FlopBOverC, "flop A/B": cell.FlopAOverB,
		} {
			if r < 1 {
				t.Errorf("%s %+v: %s = %v < 1 — the fused arm must never do MORE work", tc.model, tc.sh, name, r)
			}
		}
		if tc.sh.Agents == 1 && (cell.TokenBOverC != 1.0 || cell.FlopBOverC != 1.0) {
			t.Errorf("C=1 must give B/C exactly 1.0 (got tok=%v flop=%v): no cross-agent reuse exists",
				cell.TokenBOverC, cell.FlopBOverC)
		}
	}
}

// TestLadderReportIsDeterministic pins the canonical ladder's structure and
// the report's regenerability: the artifact main.go writes must be
// byte-identical across runs, and the two headline regime picks must land on
// the cells the ladder defines (first single-agent ultra-long, then first
// multi-agent ultra-long).
func TestLadderReportIsDeterministic(t *testing.T) {
	t.Parallel()
	s, ok := turnbench.NamedShape("qwen25-7b")
	if !ok {
		t.Fatal("qwen25-7b shape must resolve")
	}
	wantShapes := turnbench.CanonicalLadder()
	r1 := turnbench.RunLongContextLadder(s, turnbench.CanonicalLadder(), turnbench.DefaultCostModel())
	r2 := turnbench.RunLongContextLadder(s, turnbench.CanonicalLadder(), turnbench.DefaultCostModel())
	if len(r1.Cells) != len(wantShapes) {
		t.Fatalf("ladder report has %d cells for %d input shapes", len(r1.Cells), len(wantShapes))
	}
	for i, want := range wantShapes {
		if got := r1.Cells[i].Shape; got != want {
			t.Errorf("cell %d shape = %+v, want input shape %+v", i, got, want)
		}
	}
	if string(r1.JSON()) != string(r2.JSON()) {
		t.Fatal("ladder report must be byte-identical across runs")
	}
	if r1.Cells[0].UltraLong {
		t.Error("anchor cell (P=2048, ~6.8k context) must NOT be ultra-long")
	}
	if r1.SingleUltraLongIdx != 1 || r1.MultiUltraLongIdx != 2 {
		t.Errorf("headline picks = (single %d, multi %d), want (1, 2)", r1.SingleUltraLongIdx, r1.MultiUltraLongIdx)
	}
	if !r1.Cells[5].UltraLong {
		t.Error("idle-heavy agent-city cell must be ultra-long")
	}
	if r1.Cost.Version != turnbench.CostModelVersion {
		t.Errorf("cost version = %q, want %q", r1.Cost.Version, turnbench.CostModelVersion)
	}
}

// TestNamedShapeLookupContract: unknown and empty names fail closed, and lookup
// is case-insensitive — main.go's -model flag accepts operator typos like
// QWEN25-7B while reporting the canonical Name.
func TestNamedShapeLookupContract(t *testing.T) {
	t.Parallel()
	if _, ok := turnbench.NamedShape("bogus-model"); ok {
		t.Error("unknown model name must return ok=false")
	}
	if _, ok := turnbench.NamedShape(""); ok {
		t.Error("empty model name must return ok=false")
	}
	s, ok := turnbench.NamedShape("QWEN25-7B")
	if !ok {
		t.Fatal("QWEN25-7B must resolve (case-insensitive)")
	}
	if s.Name != "qwen25-7b" {
		t.Errorf("resolved Name = %q, want %q", s.Name, "qwen25-7b")
	}
}

// TestZeroTurnEdge: a session with no turns still carries its prefix — the
// context peak is P, arm A re-prefills nothing, and its ratio collapses to 0
// (there is no re-prefill left to eliminate).
func TestZeroTurnEdge(t *testing.T) {
	t.Parallel()
	s, _ := turnbench.NamedShape("smollm2-135m")
	cell := turnbench.ProjectLongContext(s, turnbench.SessionShape{Prefix: 512, Turns: 0, Agents: 3, Decode: 10, Result: 20})
	if cell.MaxContextTokens != 512 {
		t.Errorf("zero-turn MaxContextTokens = %d, want 512 (the prefix alone)", cell.MaxContextTokens)
	}
	if cell.A.PrefillTokens != 0 {
		t.Errorf("zero-turn arm A prefill tokens = %d, want 0", cell.A.PrefillTokens)
	}
	if cell.TokenAOverC != 0 {
		t.Errorf("zero-turn token A/C = %v, want 0 (no turns ⇒ no re-prefill to eliminate)", cell.TokenAOverC)
	}
	if cell.A.DecodeFLOPs != 0 {
		t.Errorf("zero-turn decode FLOPs = %g, want 0", cell.A.DecodeFLOPs)
	}
}

func withIdle(sh turnbench.SessionShape, f float64) turnbench.SessionShape {
	sh.IdleFraction = f
	return sh
}

// TestIdleFractionClamping: out-of-range idle fractions must stay well-defined
// floors — >=1 leaves no active lanes (decode floor zeroed in every arm) and a
// negative fraction clamps back to the all-active floor, reproduced EXACTLY:
// it is the default path, so a clamp bug would silently move every number.
func TestIdleFractionClamping(t *testing.T) {
	t.Parallel()
	s, _ := turnbench.NamedShape("qwen25-7b")
	base := turnbench.SessionShape{Prefix: 100_000, Turns: 10, Agents: 5, Decode: 200, Result: 500}
	def := turnbench.ProjectLongContext(s, base)
	full := turnbench.ProjectLongContext(s, withIdle(base, 2.0))
	if full.A.DecodeFLOPs != 0 || full.B.DecodeFLOPs != 0 || full.C.DecodeFLOPs != 0 {
		t.Errorf("IdleFraction=2.0 must zero the decode floor: A=%g B=%g C=%g",
			full.A.DecodeFLOPs, full.B.DecodeFLOPs, full.C.DecodeFLOPs)
	}
	neg := turnbench.ProjectLongContext(s, withIdle(base, -1.0))
	if neg.A != def.A || neg.B != def.B || neg.C != def.C {
		t.Errorf("IdleFraction=-1.0 must reproduce the all-active floor exactly:\n"+
			"clamped A/B/C = %+v / %+v / %+v\nall-active A/B/C = %+v / %+v / %+v",
			neg.A, neg.B, neg.C, def.A, def.B, def.C)
	}
}

// TestEmptySweepExitPath mirrors main.go's "no session shapes" exit: a
// digit-free sweep parses to nil, and projecting an empty ladder yields zero
// cells with both headline picks unset (-1) rather than panicking or
// inventing cells.
func TestEmptySweepExitPath(t *testing.T) {
	t.Parallel()
	if got := intlist.Parse(""); got != nil {
		t.Errorf(`intlist.Parse("") = %v, want nil`, got)
	}
	if got := intlist.Parse("abc"); got != nil {
		t.Errorf(`intlist.Parse("abc") = %v, want nil`, got)
	}
	s, _ := turnbench.NamedShape("qwen25-7b")
	rep := turnbench.RunLongContextLadder(s, nil, turnbench.DefaultCostModel())
	if len(rep.Cells) != 0 {
		t.Errorf("empty ladder must project 0 cells, got %d", len(rep.Cells))
	}
	if rep.SingleUltraLongIdx != -1 || rep.MultiUltraLongIdx != -1 {
		t.Errorf("empty ladder must leave both headline picks at -1, got (%d, %d)",
			rep.SingleUltraLongIdx, rep.MultiUltraLongIdx)
	}
}

// TestFmtRatioGlyph pins the human-summary rendering exactly, including the
// multiplication sign U+00D7 (not 'x'): the stderr table is the tool's only
// non-machine output, so a glyph drift is a visible regression.
func TestFmtRatioGlyph(t *testing.T) {
	t.Parallel()
	const mul = "\u00d7"
	for in, want := range map[float64]string{
		42.1: "42.1" + mul,
		0:    "0.0" + mul,
		1:    "1.0" + mul,
	} {
		if got := turnbench.FmtRatio(in); got != want {
			t.Errorf("FmtRatio(%v) = %q, want %q", in, got, want)
		}
	}
}

// BenchmarkLongContextLadder times one full closed-form projection of the
// canonical ladder — the cost of a `longctxbench -ladder` run minus I/O.
func BenchmarkLongContextLadder(b *testing.B) {
	shape, ok := turnbench.NamedShape("qwen25-7b")
	if !ok {
		b.Fatal("qwen25-7b shape must resolve")
	}
	wantCells := b.N * len(turnbench.CanonicalLadder())
	gotCells := 0
	for i := 0; i < b.N; i++ {
		rep := turnbench.RunLongContextLadder(shape, turnbench.CanonicalLadder(), turnbench.DefaultCostModel())
		gotCells += len(rep.Cells)
	}
	if gotCells != wantCells {
		b.Fatalf("ladder lost cells: got %d over %d runs, want %d", gotCells, b.N, wantCells)
	}
}
