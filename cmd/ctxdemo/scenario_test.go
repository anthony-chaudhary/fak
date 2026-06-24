package main

import "testing"

// The token accounting is the honest, load-independent floor the whole demo rests on,
// so it gets a hand-computed check on a tiny scenario plus invariants on the catalog.

func TestPrefillTokensHandComputed(t *testing.T) {
	// A fixed, all-equal-result workload so the math is checkable by hand. Two agents,
	// three turns, prefix 100, decode 10, every tool result 20 tokens.
	w := Workload{
		Scn:     Scenario{Prefix: 100, Agents: 2, Turns: 3, Decode: 10},
		Results: [][]int{{20, 20}, {20, 20}},
	}
	a, b, c := w.prefillTokens()

	// stateless re-prefill, per agent: turn0 ctx=100; turn1 ctx=100+(10+20)=130;
	// turn2 ctx=130+(10+20)=160. per agent = 100+130+160 = 390; two agents → 780.
	if a != 780 {
		t.Errorf("stateless re-prefill floor = %d, want 780", a)
	}
	// per-agent KV: each agent prefills prefix once (2×100) + all result tokens (4×20=80) → 280.
	if b != 280 {
		t.Errorf("per-agent KV = %d, want 280", b)
	}
	// fak fused: prefix once (100) + all result tokens (80) → 180.
	if c != 180 {
		t.Errorf("fak fused = %d, want 180", c)
	}
	// The value ordering the demo rests on: fak (c) does no more prefill work than the WARM
	// per-agent KV baseline (b — the real serving baseline), which itself does no more than
	// the stateless re-prefill floor (a). fak's headline win is b/c (vs the warm cache).
	if !(c <= b && b <= a) {
		t.Errorf("ordering violated: fak=%d perAgentKV=%d floor=%d (want fak<=perAgentKV<=floor)", c, b, a)
	}
}

func TestBuildIsDeterministic(t *testing.T) {
	s, ok := findScenario("coding-agent")
	if !ok {
		t.Fatal("coding-agent missing from catalog")
	}
	w1, w2 := s.Build(), s.Build()
	for c := range w1.Results {
		for tt := range w1.Results[c] {
			if w1.Results[c][tt] != w2.Results[c][tt] || w1.Tools[c][tt] != w2.Tools[c][tt] {
				t.Fatalf("Build not deterministic at agent %d turn %d", c, tt)
			}
		}
	}
}

func TestWorkloadIsHeterogeneous(t *testing.T) {
	// The whole point of this demo vs demorace is that results are NOT a single
	// constant. Assert the deep-research workload actually varies across (agent,turn).
	s, _ := findScenario("deep-research")
	w := s.Build()
	first := w.Results[0][0]
	seen := map[int]bool{}
	for _, row := range w.Results {
		for _, v := range row {
			seen[v] = true
		}
	}
	if len(seen) < 3 {
		t.Errorf("expected heterogeneous result sizes, got %d distinct values (first=%d)", len(seen), first)
	}
	// and every result must respect its tool's declared bounds
	bounds := map[string][2]int{}
	for _, tl := range s.Tools {
		bounds[tl.Name] = [2]int{tl.MinTok, tl.MaxTok}
	}
	for c, row := range w.Results {
		for tt, v := range row {
			name := w.Tools[c][tt]
			bd := bounds[name]
			if v < bd[0] || v > bd[1] {
				t.Errorf("agent %d turn %d tool %s result %d out of bounds [%d,%d]", c, tt, name, v, bd[0], bd[1])
			}
		}
	}
}

func TestResultsLengthIsTurnsMinusOne(t *testing.T) {
	for _, s := range catalog() {
		w := s.Build()
		if len(w.Results) != s.Agents {
			t.Errorf("%s: rows=%d, want agents=%d", s.ID, len(w.Results), s.Agents)
		}
		for c := range w.Results {
			if len(w.Results[c]) != s.Turns-1 {
				t.Errorf("%s agent %d: results=%d, want turns-1=%d", s.ID, c, len(w.Results[c]), s.Turns-1)
			}
		}
	}
}

func TestCatalogRatiosClimb(t *testing.T) {
	// HEADLINE: every scenario must show fak doing strictly less prefill work than the WARM
	// per-agent KV baseline (TunedOverFak > 1 — the real cross-agent serving win). The
	// stateless re-prefill floor (a from prefillTokens) grows with context length, so the
	// long-context scenario must exceed the short-lookup one.
	rawFloor := func(id string) int { s, _ := findScenario(id); a, _, _ := s.Build().prefillTokens(); return a }
	for _, s := range catalog() {
		tk := viewOf(s).Tokens
		if tk.TunedOverFak <= 1.0 {
			t.Errorf("%s: warmKV/fak = %.2f, want > 1 (the cross-agent serving win — the headline)", s.ID, tk.TunedOverFak)
		}
	}
	if rawFloor("deep-research") <= rawFloor("support-bot") {
		t.Errorf("expected deep-research re-prefill floor (%d) > support-bot (%d)",
			rawFloor("deep-research"), rawFloor("support-bot"))
	}
}
