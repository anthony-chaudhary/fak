package spec

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// scriptedDrafter proposes the precomputed greedy continuation `want` — a perfect
// drafter (100% acceptance while want lasts), the deterministic upper regime for the
// ensemble routing tests. Commit advances by the truly committed tokens, keeping it
// aligned even on rounds another member drafted.
type scriptedDrafter struct {
	want []int
	pos  int
}

func (d *scriptedDrafter) Draft(k int) []int {
	drafts := make([]int, 0, k)
	for j := 0; j < k && d.pos+j < len(d.want); j++ {
		drafts = append(drafts, d.want[d.pos+j])
	}
	return drafts
}

func (d *scriptedDrafter) Commit(committed []int) { d.pos += len(committed) }

// ---------------------------------------------------------------------------
// ModelDrafter — the production co-resident draft model.
// ---------------------------------------------------------------------------

// TestModelDrafterLossless: the production ModelDrafter through the full
// SpeculativeGreedy loop is token-identical to plain greedy, and leaks no open
// speculation.
func TestModelDrafterLossless(t *testing.T) {
	target := model.NewSynthetic(cfg(64, 4, 4, 2, 16, 128))
	draft := model.NewSynthetic(cfg(32, 2, 2, 1, 16, 64))
	prompt := bytesToIDs([]byte("the production drafter is the test drafter, promoted"))
	const N, K = 24, 4
	want := greedyDecode(target, prompt, N)

	sink := NewSink()
	got, drafted, accepted, rolled := SpeculativeGreedy(
		context.Background(), sink, target.NewSession(), prompt, N, K,
		NewModelDrafter(draft, prompt))
	assertEqualTokens(t, "model-drafter", got, want)
	if sink.OpenCount() != 0 {
		t.Fatalf("model-drafter: %d speculations left unresolved", sink.OpenCount())
	}
	t.Logf("model drafter: proposed %d, accepted %d, rolled back %d", drafted, accepted, rolled)
}

// TestModelDrafterCommitRealigns pins the ensemble fan contract directly: a Commit
// on a round the drafter did NOT draft (no open span) must not evict committed
// context, and a Commit after a Draft must drop exactly the speculative span — in
// both cases leaving the drafter's session bit-exact to a never-drafted reference
// fed the same committed tokens.
func TestModelDrafterCommitRealigns(t *testing.T) {
	m := model.NewSynthetic(cfg(48, 3, 3, 1, 16, 96))
	prompt := bytesToIDs([]byte("stay aligned with the true context"))
	d := NewModelDrafter(m, prompt)

	// Round 1: another member drafted; this drafter only re-threads the commit.
	committed1 := []int{10, 20}
	d.Commit(committed1)

	// Round 2: this drafter drafts (its KV grows), then the round commits different
	// tokens; Commit must evict the whole draft span and re-thread.
	_ = d.Draft(3)
	committed2 := []int{30, 40, 50}
	d.Commit(committed2)

	ref := m.NewSession()
	ref.Prefill(prompt)
	for _, tok := range append(append([]int{}, committed1...), committed2...) {
		ref.Step(tok)
	}
	if d.s.Cache.Len() != ref.Cache.Len() {
		t.Fatalf("cache len %d after commits, want %d (draft span not evicted or commit not threaded)",
			d.s.Cache.Len(), ref.Cache.Len())
	}
	assertContinuationsMatch(t, "model-drafter-realign", d.s, ref, 42, 4)
}

// ---------------------------------------------------------------------------
// LookupDrafter — prompt-lookup / n-gram multi-token prediction.
// ---------------------------------------------------------------------------

// TestLookupDrafterProposals pins the lookup rule on hand-built sequences: the
// longest suffix n-gram wins, the most recent occurrence supplies the continuation,
// k clamps the proposal, references are consulted after the context, and a miss
// returns nil.
func TestLookupDrafterProposals(t *testing.T) {
	// Context ...[7 8 9] appeared earlier twice; most recent occurrence (index 5)
	// is followed by [4 5 6].
	d := NewLookupDrafter([]int{7, 8, 9, 1, 2, 7, 8, 9, 4, 5, 6, 7, 8, 9}, 1, 3)
	if got := d.Draft(3); len(got) != 3 || got[0] != 4 || got[1] != 5 || got[2] != 6 {
		t.Fatalf("Draft(3) = %v, want [4 5 6] (most recent trigram continuation)", got)
	}
	if got := d.Draft(2); len(got) != 2 || got[0] != 4 || got[1] != 5 {
		t.Fatalf("Draft(2) = %v, want [4 5] (k clamps the continuation)", got)
	}

	// All-distinct context: no n-gram repeats, so no proposal.
	d2 := NewLookupDrafter([]int{1, 2, 3, 4, 5}, 1, 3)
	if got := d2.Draft(4); got != nil {
		t.Fatalf("Draft on a distinct context = %v, want nil (degrade to plain greedy)", got)
	}

	// Reference-primed: the context suffix continues inside a reference text.
	ref := []int{100, 4, 5, 200, 201, 202}
	d3 := NewLookupDrafter([]int{1, 2, 3, 4, 5}, 2, 3, ref)
	if got := d3.Draft(3); len(got) != 3 || got[0] != 200 || got[1] != 201 || got[2] != 202 {
		t.Fatalf("Draft with reference = %v, want [200 201 202]", got)
	}

	// Commit grows the context, so lookups see the committed tokens.
	d2.Commit([]int{1, 2, 3})
	if got := d2.Draft(2); len(got) != 2 || got[0] != 4 || got[1] != 5 {
		t.Fatalf("Draft after Commit = %v, want [4 5] (committed tokens joined the index)", got)
	}
}

// TestLookupDrafterLossless: model-free n-gram drafting through the full loop stays
// token-identical to greedy whatever the acceptance — a miss round degrades to plain
// greedy (empty draft), a hit round is verified like any other draft.
func TestLookupDrafterLossless(t *testing.T) {
	target := model.NewSynthetic(cfg(64, 4, 4, 2, 16, 128))
	prompt := bytesToIDs([]byte("lookup lookup lookup decoding needs no second model at all"))
	const N, K = 32, 4
	want := greedyDecode(target, prompt, N)

	sink := NewSink()
	got, drafted, accepted, rolled := SpeculativeGreedy(
		context.Background(), sink, target.NewSession(), prompt, N, K,
		NewLookupDrafter(prompt, 1, 3))
	assertEqualTokens(t, "lookup-drafter", got, want)
	if sink.OpenCount() != 0 {
		t.Fatalf("lookup-drafter: %d speculations left unresolved", sink.OpenCount())
	}
	t.Logf("lookup drafter: proposed %d, accepted %d, rolled back %d", drafted, accepted, rolled)
}

// TestLookupDrafterReferencePrimed: priming the lookup with a reference that
// contains the true continuation (here: the precomputed greedy output — the
// retrieval regime, where the answer quotes a known text) must yield accepted
// drafts, and stay lossless. This is the deterministic acceptance>0 witness the
// self-lookup test cannot promise.
func TestLookupDrafterReferencePrimed(t *testing.T) {
	target := model.NewSynthetic(cfg(64, 4, 4, 2, 16, 128))
	prompt := bytesToIDs([]byte("the reference corpus carries the continuation"))
	const N, K = 24, 4
	want := greedyDecode(target, prompt, N)
	ref := append(append([]int{}, prompt...), want...)

	sink := NewSink()
	got, drafted, accepted, _ := SpeculativeGreedy(
		context.Background(), sink, target.NewSession(), prompt, N, K,
		NewLookupDrafter(prompt, 1, 3, ref))
	assertEqualTokens(t, "lookup-reference-primed", got, want)
	if accepted == 0 {
		t.Fatalf("reference-primed lookup accepted 0 of %d drafted — the reference continuation was never proposed", drafted)
	}
	t.Logf("reference-primed lookup: proposed %d, accepted %d (%.0f%%)",
		drafted, accepted, 100*float64(accepted)/float64(drafted))
}

// ---------------------------------------------------------------------------
// MultiDrafter — the multi-draft-model ensemble.
// ---------------------------------------------------------------------------

// TestMultiDrafterLossless: a mixed ensemble (real draft model, model-free lookup,
// and an adversarial member) through the full loop is token-identical to greedy;
// every member is warmed up; the routed rounds account exactly; and the per-member
// meters reconcile with the run totals.
func TestMultiDrafterLossless(t *testing.T) {
	target := model.NewSynthetic(cfg(64, 4, 4, 2, 16, 128))
	draft := model.NewSynthetic(cfg(32, 2, 2, 1, 16, 64))
	prompt := bytesToIDs([]byte("the ensemble routes each round to the measured best drafter"))
	const N, K = 48, 4
	want := greedyDecode(target, prompt, N)

	md := NewMultiDrafter(4,
		NamedDrafter{Name: "draft-model", D: NewModelDrafter(draft, prompt)},
		NamedDrafter{Name: "lookup", D: NewLookupDrafter(prompt, 1, 3)},
		NamedDrafter{Name: "adversary", D: &advDrafter{}},
	)
	sink := NewSink()
	var meter AcceptanceMeter
	got, drafted, accepted, _ := SpeculativeGreedyMetered(
		context.Background(), sink, target.NewSession(), prompt, N, K, md, &meter)
	assertEqualTokens(t, "multi-drafter", got, want)
	if sink.OpenCount() != 0 {
		t.Fatalf("multi-drafter: %d speculations left unresolved", sink.OpenCount())
	}

	stats := md.Stats()
	if len(stats) != 3 {
		t.Fatalf("Stats() returned %d members, want 3", len(stats))
	}
	sumRouted, sumDrafted, sumAccepted := 0, 0, 0
	for _, s := range stats {
		if s.RoundsRouted == 0 {
			t.Errorf("member %q was never routed a round (warmup must reach every member)", s.Name)
		}
		sumRouted += s.RoundsRouted
		sumDrafted += s.Stats.Drafted
		sumAccepted += s.Stats.Accepted
		t.Logf("member %-11s routed %2d rounds, drafted %3d, accepted %3d (rate %.2f)",
			s.Name, s.RoundsRouted, s.Stats.Drafted, s.Stats.Accepted, s.Stats.AcceptanceRate)
	}
	if sumRouted != meter.Rounds() {
		t.Errorf("routed rounds sum %d != run rounds %d", sumRouted, meter.Rounds())
	}
	if sumDrafted != drafted {
		t.Errorf("member drafted sum %d != run drafted %d", sumDrafted, drafted)
	}
	// The ensemble derives accepted from the committed prefix, which the final round
	// may truncate (n reached mid-round) — so the sums may differ by at most one
	// round's drafts, never more.
	if sumAccepted > accepted || accepted-sumAccepted > K {
		t.Errorf("member accepted sum %d irreconcilable with run accepted %d (K=%d)", sumAccepted, accepted, K)
	}
}

// TestMultiDrafterRoutesToBest: with a perfect (scripted greedy continuation)
// member against an adversarial one, the exploit rule must route the bulk of the
// rounds to the perfect member and report it as Best — while periodic probes keep
// the adversary measured rather than starved forever.
func TestMultiDrafterRoutesToBest(t *testing.T) {
	target := model.NewSynthetic(cfg(64, 4, 4, 2, 16, 128))
	prompt := bytesToIDs([]byte("exploit the measured best, probe the rest"))
	const N, K = 40, 4
	want := greedyDecode(target, prompt, N)

	md := NewMultiDrafter(4,
		NamedDrafter{Name: "perfect", D: &scriptedDrafter{want: want}},
		NamedDrafter{Name: "adversary", D: &advDrafter{}},
	)
	sink := NewSink()
	got, _, _, _ := SpeculativeGreedy(
		context.Background(), sink, target.NewSession(), prompt, N, K, md)
	assertEqualTokens(t, "multi-drafter-routing", got, want)

	stats := md.Stats()
	var perfect, adversary MemberStats
	for _, s := range stats {
		switch s.Name {
		case "perfect":
			perfect = s
		case "adversary":
			adversary = s
		}
	}
	if perfect.RoundsRouted <= adversary.RoundsRouted {
		t.Fatalf("routing failed to exploit: perfect routed %d rounds, adversary %d",
			perfect.RoundsRouted, adversary.RoundsRouted)
	}
	if adversary.RoundsRouted < 2 {
		t.Fatalf("adversary routed %d rounds — the periodic probe never ran (warmup only)", adversary.RoundsRouted)
	}
	if md.Best() != "perfect" {
		t.Fatalf("Best() = %q, want \"perfect\"", md.Best())
	}
	t.Logf("routing: perfect %d rounds (rate %.2f), adversary %d rounds (rate %.2f)",
		perfect.RoundsRouted, perfect.Stats.AcceptanceRate,
		adversary.RoundsRouted, adversary.Stats.AcceptanceRate)
}
