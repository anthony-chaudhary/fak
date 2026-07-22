package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// TestSpecDecodeGreedyMatchesGreedyAndAccepts is the #5098 witness for the pure engine
// binding: greedy speculative decode over live Sessions is TOKEN-IDENTICAL to plain greedy
// decode (Session.Generate), and — because a same-weights drafter predicts the target's own
// argmax at every position — the mean acceptance length is > 1 (drafting bought throughput).
func TestSpecDecodeGreedyMatchesGreedyAndAccepts(t *testing.T) {
	m := NewSynthetic(cfgV(64, 4, 4, 2, 16, 128))
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	n, k := 24, 4

	want := m.NewSession().Generate(prompt, n) // reference: plain greedy decode

	target := m.NewSession()
	drafter := m.NewSession()
	run, err := SpecDecodeGreedy(target, drafter, prompt, n, k)
	if err != nil {
		t.Fatalf("SpecDecodeGreedy: %v", err)
	}
	if len(run.Output) != len(want) {
		t.Fatalf("len(output)=%d want %d", len(run.Output), len(want))
	}
	for i := range want {
		if run.Output[i] != want[i] {
			t.Fatalf("token %d: spec=%d greedy=%d (speculative decode is NOT lossless)", i, run.Output[i], want[i])
		}
	}
	if run.AcceptedDrafts == 0 {
		t.Fatalf("AcceptedDrafts=0: a same-weights drafter must accept drafts")
	}
	if run.MeanAcceptanceLength <= 1.0 {
		t.Fatalf("MeanAcceptanceLength=%v, want >1 (a same-weights drafter accepts every draft)", run.MeanAcceptanceLength)
	}
}

// coResidentPool builds a residency Pool with two same-family, prefill-shareable models — a
// big verifier and a cheaper small drafter — so PickDrafter/BridgeRoles resolve them as a
// co-resident speculation pair. Both map to sessions of the SAME synthetic weights in the
// tests below, which makes the drafter a perfect predictor (acceptance > 1) while keeping the
// residency descriptors genuinely distinct.
func coResidentPool() *polymodel.Pool {
	pool := polymodel.NewPool(1 << 30)
	pool.Admit(polymodel.Model{ID: "big", Family: "fam", WeightBytes: 100, PrefixDigest: "d"})
	pool.Admit(polymodel.Model{ID: "small", Family: "fam", WeightBytes: 10, PrefixDigest: "d"})
	return pool
}

// TestSpecDecodeGreedyResolvedGateDefaultOff is the #5098 gate assertion: with FAK_POLYMODEL
// unset, the request-path entry NEVER speculates (ok=false), even with a fully co-resident
// drafter available — the caller falls back to plain self-decode.
func TestSpecDecodeGreedyResolvedGateDefaultOff(t *testing.T) {
	if polymodel.Enabled() {
		t.Skip("FAK_POLYMODEL is set in the environment; the default-off assertion needs it unset")
	}
	m := NewSynthetic(cfgV(32, 2, 2, 1, 16, 64))
	pool := coResidentPool()
	sessions := map[polymodel.ModelID]*Session{"big": m.NewSession(), "small": m.NewSession()}

	run, drafter, ok, err := SpecDecodeGreedyResolved([]int{1, 2, 3, 4}, 8, 4, "big", pool, sessions)
	if err != nil {
		t.Fatalf("resolved (gate off): %v", err)
	}
	if ok {
		t.Fatalf("gate default-off VIOLATED: speculation ran with FAK_POLYMODEL unset (drafter=%q, run=%+v)", drafter, run)
	}
}

// TestSpecDecodeGreedyResolvedRunsWhenEnabled proves the gated request path, once opted in via
// FAK_POLYMODEL, resolves the cheapest co-resident drafter (PickDrafter → "small") and runs a
// lossless speculative decode with acceptance > 1.
func TestSpecDecodeGreedyResolvedRunsWhenEnabled(t *testing.T) {
	t.Setenv(polymodel.FlagEnv, "on")
	m := NewSynthetic(cfgV(64, 4, 4, 2, 16, 128))
	pool := coResidentPool()
	sessions := map[polymodel.ModelID]*Session{"big": m.NewSession(), "small": m.NewSession()}
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	n, k := 24, 4

	want := m.NewSession().Generate(prompt, n)
	run, drafter, ok, err := SpecDecodeGreedyResolved(prompt, n, k, "big", pool, sessions)
	if err != nil {
		t.Fatalf("resolved (gate on): %v", err)
	}
	if !ok {
		t.Fatalf("gate on + co-resident drafter: expected speculation to run")
	}
	if drafter != "small" {
		t.Fatalf("PickDrafter chose %q, want the cheapest co-resident 'small'", drafter)
	}
	if len(run.Output) != len(want) {
		t.Fatalf("len(output)=%d want %d", len(run.Output), len(want))
	}
	for i := range want {
		if run.Output[i] != want[i] {
			t.Fatalf("token %d: spec=%d greedy=%d (resolved path is NOT lossless)", i, run.Output[i], want[i])
		}
	}
	if run.MeanAcceptanceLength <= 1.0 {
		t.Fatalf("MeanAcceptanceLength=%v, want >1", run.MeanAcceptanceLength)
	}
}
