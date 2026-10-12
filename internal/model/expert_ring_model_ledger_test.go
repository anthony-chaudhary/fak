package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The production constructor, demand path, and Close must bound request bookkeeping even
// when device weights remain hot for the entire model lifetime. This test also compiles
// against the unrepaired implementation: retained memberships then grow to 3 * requests.
// fak-test:runtime fast est=5s lane=default
func TestModelSharedExpertRingRequestLedgerBounded(t *testing.T) {
	t.Parallel()
	f := newCheckpointSessionFixture(t, true)
	sh, err := f.m.EnableSharedExpertRing(f.be)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = f.m.CloseWeights() })
	want := f.reference(0, checkpointSessionInput())
	const requests = 64
	quantUploads := func() int { return f.be.uploads[compute.Q2_K] + f.be.uploads[compute.Q3_K] }
	firstReads, firstUploads := 0, 0
	for request := 0; request < requests; request++ {
		s := f.session(t)
		for repeat := 0; repeat < 2; repeat++ {
			modelLedgerCheckOutput(t, checkpointSessionForward(t, s, 0), want)
		}
		s.Close()
		if request == 0 {
			firstReads, firstUploads = f.reads(), quantUploads()
		}
		if f.reads() != firstReads || quantUploads() != firstUploads {
			t.Fatalf("request %d reread or reuploaded hot weights", request)
		}
		st := sh.Stats()
		if st.Agents != 0 || st.Ring.ResidentCount != 3 || st.Ring.PageIns != 3 || st.Ring.Evictions != 0 || st.Ring.ResidentBytes != f.budget {
			t.Fatalf("request %d lost residency or attachment discipline: %+v", request, st)
		}
	}
	st := sh.Stats()
	if st.DistinctServes != 3*requests || st.SharedResidencies != 3 || st.CrossAgentHits != 6*(requests-1) || st.Demands != 6*requests || st.Ring.Hits != 6*requests-3 || st.Refusals != 0 || st.PageInBytes != f.budget {
		t.Fatalf("request history accounting changed: %+v", st)
	}
	// Inspect retained identity membership, not process RSS or fixture allocations. No new
	// implementation types are referenced, so this assertion is usable on the exact base.
	sh.mu.Lock()
	retained := 0
	for _, set := range sh.ledger.served {
		retained += len(set)
	}
	retainedSets := len(sh.ledger.served)
	sh.mu.Unlock()
	if retained != 0 || retainedSets != 0 {
		t.Fatalf("closed requests retained %d memberships in %d maps; want none", retained, retainedSets)
	}
	if err := f.m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	if err := f.m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
	if st := sh.Stats(); st.Enabled || st.Ring.ResidentBytes != 0 || st.Ring.ResidentCount != 0 {
		t.Fatalf("weight close retained ring residency: %+v", st)
	}
}

func modelLedgerCheckOutput(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("output length %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("output[%d]=%v, want %v", i, got[i], want[i])
		}
	}
}

// Stable public identities retain historical deduplication. An explicit name equal to a
// former automatic label is still a distinct identity, including for payer attribution.
// Eviction starts a fresh accounting span for both identity kinds.
// fak-test:runtime fast est=1s lane=default
func TestModelSharedExpertRingNamedLedgerIdentity(t *testing.T) {
	t.Parallel()
	f := newCheckpointSessionFixture(t, true)
	sh, err := f.m.EnableSharedExpertRing(f.be)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = f.m.CloseWeights() })
	forward := func(s *Session, expert, repeats int) {
		t.Helper()
		want := f.reference(expert, checkpointSessionInput())
		for repeat := 0; repeat < repeats; repeat++ {
			modelLedgerCheckOutput(t, checkpointSessionForward(t, s, expert), want)
		}
	}
	check := func(distinct, shared, cross int64) {
		t.Helper()
		st := sh.Stats()
		if st.DistinctServes != distinct || st.SharedResidencies != shared || st.CrossAgentHits != cross {
			t.Fatalf("ledger=%+v, want distinct=%d shared=%d cross=%d", st, distinct, shared, cross)
		}
	}
	named := func(name string) *Session {
		t.Helper()
		s := f.session(t)
		sh.Detach(s)
		if err := sh.Attach(s, name); err != nil {
			t.Fatal(err)
		}
		return s
	}
	automatic := f.session(t)
	alias := automatic.ringAgent
	forward(automatic, 0, 2)
	automatic.Close()
	check(3, 0, 0)
	stable := named(alias)
	forward(stable, 0, 2)
	stable.Close()
	check(6, 3, 6)
	stable = named(alias)
	forward(stable, 0, 2)
	check(6, 3, 12)
	peer := f.session(t)
	forward(peer, 0, 2)
	peer.Close()
	check(9, 3, 18)
	forward(stable, 0, 2)
	stable.Close()
	check(9, 3, 24)

	// The fixture fits exactly one expert. Another expert evicts the old three
	// projections; demanding the original again must start fresh history.
	evictor := f.session(t)
	forward(evictor, 1, 1)
	forward(evictor, 0, 1)
	evictor.Close()
	check(15, 3, 24)
	stable = named(alias)
	forward(stable, 0, 1)
	stable.Close()
	check(18, 6, 27)
	if st := sh.Stats(); st.Ring.PageIns != 9 || st.Ring.Evictions != 6 || st.Agents != 0 {
		t.Fatalf("unexpected residency epochs: %+v", st)
	}
}

// Live automatic agents still need independent deduplication while their requests overlap.
// fak-test:runtime fast est=1s lane=default
func TestModelSharedExpertRingLiveLedgerDedup(t *testing.T) {
	t.Parallel()
	f := newCheckpointSessionFixture(t, true)
	sh, err := f.m.EnableSharedExpertRing(f.be)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = f.m.CloseWeights() })
	a, b := f.session(t), f.session(t)
	for _, s := range []*Session{a, b, a, b} {
		checkpointSessionForward(t, s, 0)
	}
	a.Close()
	checkpointSessionForward(t, b, 0)
	st := sh.Stats()
	if st.Agents != 1 || st.DistinctServes != 6 || st.SharedResidencies != 3 || st.CrossAgentHits != 9 {
		t.Fatalf("overlapping requests lost deduplication: %+v", st)
	}
	sh.mu.Lock()
	retained := 0
	for _, set := range sh.ledger.served {
		retained += len(set)
	}
	sh.mu.Unlock()
	if retained != 3 {
		t.Fatalf("retained memberships=%d, want the surviving agent's three", retained)
	}
	b.Close()
	sh.mu.Lock()
	sets := len(sh.ledger.served)
	sh.mu.Unlock()
	if sets != 0 {
		t.Fatalf("last detach retained %d empty or historical maps", sets)
	}
}
