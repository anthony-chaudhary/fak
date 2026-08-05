package model

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expert_ring_pins_test.go — the R2 witnesses for #5613 (epic #5606): the bounded routed-expert ring
// now consults a DURABLE, workload-personalized pin-set instead of a static pinned=false, so a hot
// expert stops paying a cold page-in after every eviction and after every restart.
//
// The two claims the plan names, one test each:
//
//	warm start   — a second process, sharing only the dumped usage histogram, pages in strictly
//	               fewer routed-expert weights than the first on the SAME workload;
//	survival     — the pinned set that second process starts with is the one the first dumped.
//
// Plus the gates the wiring depends on: default-off is R0 byte-for-byte, the (layer,expert) identity
// parse refuses everything that is not a routed expert projection, and a corrupt dump degrades to a
// cold start while still reporting itself.

// expertPinSession is expertRingSession plus the R2 pin knobs — a session that warm-starts its
// pin-set from usagePath and dumps back to it at each turn boundary.
func expertPinSession(m *Model, ringBytes int64, pinBudget int, usagePath string) *Session {
	be := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	return &Session{
		M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{},
		ExpertRingBytes: ringBytes,
		ExpertPinBudget: pinBudget,
		ExpertUsagePath: usagePath,
	}
}

// expertPinRoundRobin is the pathological case for pure recency: a sweep longer than the ring, so
// every expert is evicted exactly before its next use and LRU scores ZERO hits. Expert 0 leads by
// one activation, making it the unambiguous hottest in the dumped prior.
var expertPinRoundRobin = []int{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0}

// runExpertPinTurn drives one process's worth of work — build a session, activate the window, close
// the turn (repin + dump), tear down — and reports the ring stats at the end of the window. Each
// call is a fresh session over a fresh device, so nothing but the file at usagePath carries between
// them: that is what makes the comparison a RESTART rather than a warm cache.
func runExpertPinTurn(t *testing.T, m *Model, ringBytes int64, pinBudget int, usagePath string, window []int) (ExpertRingStats, []ExpertPinSwap) {
	t.Helper()
	s := expertPinSession(m, ringBytes, pinBudget, usagePath)
	defer s.Close()
	x := expertRingTestInput(m.Cfg.HiddenSize)
	for _, e := range window {
		expertSwiGLU(m, 0, e, x, sessionQ4KKernel{s: s})
	}
	st := s.ExpertRing()
	swaps, err := s.ExpertRingEndTurn(0.9, 2)
	if err != nil {
		t.Fatalf("ExpertRingEndTurn: %v", err)
	}
	return st, swaps
}

// TestExpertRingWarmStartCutsColdPageIns is the leverage witness. Two runs of the SAME workload over
// the same ring budget, sharing only a dumped usage histogram: the second warm-starts its pin-set
// from the first's routing and pages in strictly fewer weights. Under pure LRU this workload scores
// zero hits by construction, so every hit in run 2 is attributable to the pin-set.
func TestExpertRingWarmStartCutsColdPageIns(t *testing.T) {
	const H, E = 256, 4
	m := expertRingTestModel(t, H, E)
	perWeight := expertRingWeightBytes(t, m)
	budget := perWeight * 6 // exactly two whole experts; the sweep touches four
	usage := filepath.Join(t.TempDir(), "expert-usage.json")

	// Run 1: no prior on disk. The pin-set warm-starts EMPTY, so this is plain LRU — and a sweep
	// longer than the ring means every activation misses.
	cold, coldChanges := runExpertPinTurn(t, m, budget, 1, usage, expertPinRoundRobin)
	// The turn boundary fills the free pin slot from what the turn actually routed — a cold run must
	// not have to wait for a restart to pin anything, so this is a FILL (nothing displaced), not a swap.
	if len(coldChanges) != 1 || coldChanges[0].OutLayer != -1 || coldChanges[0].InExpert != 0 {
		t.Fatalf("run 1 pin-set changes = %+v, want one fill of expert 0", coldChanges)
	}
	if cold.PinnedCount != 0 {
		t.Fatalf("run 1 started with %d pins; a cold first run has no prior to pin from", cold.PinnedCount)
	}
	if cold.Hits != 0 {
		t.Fatalf("run 1 scored %d ring hits; the round-robin window is supposed to defeat pure LRU, so the comparison would be confounded", cold.Hits)
	}
	if want := len(expertPinRoundRobin) * 3; cold.PageIns != want {
		t.Fatalf("run 1 page-ins=%d, want %d (every one of the %d activations misses, three projections each)", cold.PageIns, want, len(expertPinRoundRobin))
	}
	if _, err := os.Stat(usage); err != nil {
		t.Fatalf("run 1 left no usage dump for the next process to warm-start from: %v", err)
	}

	// Run 2: a fresh session over a fresh device, carrying nothing but that file.
	warm, _ := runExpertPinTurn(t, m, budget, 1, usage, expertPinRoundRobin)
	if warm.PinnedCount != 1 {
		t.Fatalf("run 2 warm-started with %d pins, want 1 (the budget)", warm.PinnedCount)
	}
	if warm.PageIns >= cold.PageIns {
		t.Fatalf("run 2 paged in %d weights vs run 1's %d — the warm-started pin-set bought nothing", warm.PageIns, cold.PageIns)
	}
	if warm.Hits == 0 {
		t.Fatal("run 2 scored no ring hits; the pinned expert was still being evicted")
	}
	// Expert 0 leads the prior by one activation, so it is what gets pinned — and being pinned, its
	// four activations cost ONE page-in instead of four. Three saved activations x three projections.
	if want := cold.PageIns - 9; warm.PageIns != want {
		t.Fatalf("run 2 page-ins=%d, want %d (expert 0's three re-activations become hits, three projections each)", warm.PageIns, want)
	}
	if warm.PeakBytes > warm.BudgetBytes {
		t.Fatalf("peak resident %d exceeds budget %d — pinning must not breach the bound", warm.PeakBytes, warm.BudgetBytes)
	}
}

// TestExpertRingPinSetSurvivesRestart is the survival witness: the identity the second process pins
// is the one the first process's routing selected, not a default or a re-derivation.
func TestExpertRingPinSetSurvivesRestart(t *testing.T) {
	const H, E = 256, 4
	m := expertRingTestModel(t, H, E)
	perWeight := expertRingWeightBytes(t, m)
	usage := filepath.Join(t.TempDir(), "expert-usage.json")

	// A window that makes expert 2 — NOT expert 0, which would also win a tie-break — the hottest,
	// so a surviving pin on (0,2) can only have come from the dumped routing.
	window := []int{2, 0, 2, 1, 2, 3, 2, 0, 2, 1}
	if _, _ = runExpertPinTurn(t, m, perWeight*6, 1, usage, window); true {
		hist, err := LoadExpertUsageHistogram(usage)
		if err != nil {
			t.Fatalf("LoadExpertUsageHistogram: %v", err)
		}
		if hot := hist.HotSet(1); len(hot) != 1 || hot[0].Layer != 0 || hot[0].Expert != 2 {
			t.Fatalf("dumped prior's hottest = %+v, want layer 0 expert 2", hot)
		}
	}

	s := expertPinSession(m, perWeight*6, 1, usage)
	defer s.Close()
	// The ring (and with it the warm start) is built lazily on the first routed staging.
	expertSwiGLU(m, 0, 0, expertRingTestInput(H), sessionQ4KKernel{s: s})
	if !s.expertRing.isExpertPinned(0, 2) {
		t.Fatalf("the restarted session did not pin (0,2); pins=%+v", s.expertRing.pins.Pins())
	}
	if s.expertRing.isExpertPinned(0, 0) {
		t.Fatal("the restarted session pinned (0,0), which the prior did not select")
	}
	if got := s.ExpertRing().PinnedCount; got != 1 {
		t.Fatalf("PinnedCount=%d, want 1", got)
	}
}

// TestExpertRingWithoutPinKnobsIsUnchanged is the default-off gate: a session that declares only a
// ring budget builds no pin-set, observes nothing, and its turn boundary is a no-op — R0 exactly.
func TestExpertRingWithoutPinKnobsIsUnchanged(t *testing.T) {
	const H = 256
	m := expertRingTestModel(t, H, 4)
	perWeight := expertRingWeightBytes(t, m)
	s := expertRingSession(m, perWeight*6) // ring budget only: no ExpertPinBudget, no ExpertUsagePath
	defer s.Close()
	x := expertRingTestInput(H)
	for _, e := range expertPinRoundRobin {
		expertSwiGLU(m, 0, e, x, sessionQ4KKernel{s: s})
	}
	if s.expertRing.pins != nil {
		t.Fatal("a session declaring no pin knobs built a pin-set")
	}
	if s.expertRing.turn != nil {
		t.Fatal("a session declaring no pin knobs is accumulating a usage histogram")
	}
	if st := s.ExpertRing(); st.PinnedCount != 0 || st.Hits != 0 {
		t.Fatalf("default ring: pinned=%d hits=%d, want 0/0 (plain LRU on a sweep it cannot cache)", st.PinnedCount, st.Hits)
	}
	swaps, err := s.ExpertRingEndTurn(0.9, 2)
	if err != nil || swaps != nil {
		t.Fatalf("ExpertRingEndTurn on a pin-less session: swaps=%v err=%v, want nil/nil", swaps, err)
	}
}

// TestRoutedExpertIdentity pins the parse that joins a NAME-keyed ring to an (layer,expert)-keyed
// pin-set. The refusals matter more than the accepts: a silent (0,0) is a real identity, so anything
// that is not a routed expert projection must be rejected rather than credited to layer 0 expert 0.
func TestRoutedExpertIdentity(t *testing.T) {
	for _, tc := range []struct {
		name          string
		layer, expert int
		ok            bool
	}{
		{name: expertName(0, 0, "gate_proj.weight"), layer: 0, expert: 0, ok: true},
		{name: expertName(7, 129, "down_proj.weight"), layer: 7, expert: 129, ok: true},
		{name: expertName(3, 5, "up_proj.bias"), layer: 3, expert: 5, ok: true},
		{name: "model.layers.3.mlp.shared_experts.gate_proj.weight"}, // shared, not routed (#3212)
		{name: "model.layers.3.mlp.gate.weight"},                     // the router
		{name: "model.layers.3.self_attn.q_proj.weight"},
		{name: "lm_head.weight"},
		{name: "model.layers.x.mlp.experts.2.gate_proj.weight"},  // non-numeric layer
		{name: "model.layers.3.mlp.experts.x.gate_proj.weight"},  // non-numeric expert
		{name: "model.layers.3.mlp.experts.1234567890.gate_p.w"}, // >9 digits: refuse, never overflow
	} {
		layer, expert, ok := routedExpertIdentity(tc.name)
		if ok != tc.ok {
			t.Fatalf("routedExpertIdentity(%q) ok=%v, want %v", tc.name, ok, tc.ok)
		}
		if ok && (layer != tc.layer || expert != tc.expert) {
			t.Fatalf("routedExpertIdentity(%q) = (%d,%d), want (%d,%d)", tc.name, layer, expert, tc.layer, tc.expert)
		}
	}
}

// TestExpertRingCorruptUsageDumpDegradesNotFails: a torn or foreign cache file must not take a serve
// down — the session cold-starts — but it must not vanish either, or a pin cache that silently
// stopped loading looks exactly like one that is working.
func TestExpertRingCorruptUsageDumpDegradesNotFails(t *testing.T) {
	const H = 256
	m := expertRingTestModel(t, H, 4)
	perWeight := expertRingWeightBytes(t, m)
	usage := filepath.Join(t.TempDir(), "expert-usage.json")
	if err := os.WriteFile(usage, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}

	s := expertPinSession(m, perWeight*6, 2, usage)
	defer s.Close()
	x := expertRingTestInput(H)
	for _, e := range expertPinRoundRobin {
		expertSwiGLU(m, 0, e, x, sessionQ4KKernel{s: s}) // must not panic or fail the forward
	}
	if s.expertRing.pins == nil {
		t.Fatal("a corrupt prior left the session with no pin-set at all; it must cold-start, not opt out")
	}
	if got := s.ExpertRing().PinnedCount; got != 0 {
		t.Fatalf("PinnedCount=%d after a corrupt prior, want 0 (a cold start pins nothing)", got)
	}
	swaps, err := s.ExpertRingEndTurn(0.9, 2)
	if err == nil {
		t.Fatal("the corrupt prior was never reported; a pin cache that stopped loading must be visible")
	}
	// ...and the turn still completed past the reported failure: the pin-set filled from what this
	// turn routed, and the dump replaced the corrupt file with a usable prior.
	if len(swaps) != 2 {
		t.Fatalf("pin-set changes = %+v, want the two-slot budget filled from this turn's routing", swaps)
	}
	if got := s.ExpertRing().PinnedCount; got != 2 {
		t.Fatalf("PinnedCount=%d after the boundary, want 2 (a corrupt prior costs one turn, not the session)", got)
	}
	if _, err := LoadExpertUsageHistogram(usage); err != nil {
		t.Fatalf("the turn boundary did not replace the corrupt file with a readable dump: %v", err)
	}
	// The error is reported ONCE — the next boundary is clean.
	if _, err := s.ExpertRingEndTurn(0.9, 2); err != nil {
		t.Fatalf("the warm-start error was reported twice: %v", err)
	}
}
