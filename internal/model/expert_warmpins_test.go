package model

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// expert_warmpins_test.go — the two-phase witness the #4358 first-checkable-step names: a crash-safe
// per-(layer,expert) histogram dumped per turn and summed at boot warm-starts pagedRing's pins, and a
// between-turns RepinPass actuator drifts that pinned hot-set from a phase-1 workload to a phase-2 one
// under bounded swaps and decaying heat. Every witness here is deterministic — the heat arithmetic is
// pinned to exact integers/halves so the swap order and the drift are reproducible, not statistical.

// wpTouch is n routed touches of one (layer,expert) — the compact way a test states a synthetic phase.
type wpTouch struct{ layer, expert, n int }

// wpTrace builds a deterministic ExpertAccessTrace from touch multiplicities, so ObserveTrace folds a
// known per-(layer,expert) count into a histogram. WeightBytes is a constant — ObserveTrace weights each
// event +1 regardless — so it only stands in for the resident size a real recorder would size.
func wpTrace(name string, touches ...wpTouch) ExpertAccessTrace {
	tr := ExpertAccessTrace{
		Schema: ExpertReplayTraceSchema, Name: name, Source: "warmpins-test",
		BudgetBytes: q4kBlockBytes * 4,
	}
	for _, t := range touches {
		for i := 0; i < t.n; i++ {
			tr.Events = append(tr.Events, ExpertAccessTraceEvent{Layer: t.layer, Expert: t.expert, WeightBytes: q4kBlockBytes})
		}
	}
	return tr
}

// TestExpertUsageHistogramPersistLoadRoundTrip witnesses the durable prior: a per-turn dump is written
// crash-safely (tmp+rename, no torn leftover), reloads byte-for-byte, a missing dump is an empty prior
// (not an error), and two dumps sum at boot — the "atomic dump per turn, summed at boot" the axis names.
func TestExpertUsageHistogramPersistLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()

	h := NewExpertUsageHistogram()
	h.ObserveTrace(wpTrace("turn1", wpTouch{0, 0, 10}, wpTouch{0, 1, 9}, wpTouch{1, 2, 3}))
	path := filepath.Join(dir, "turn1.json")
	if err := h.Persist(path); err != nil {
		t.Fatalf("Persist: %v", err)
	}

	// Crash-safety leaves no torn temp sibling behind — the rename either landed or cleaned up.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".expert-usage-") && strings.HasSuffix(e.Name(), ".tmp") {
			t.Fatalf("Persist left a temp file behind: %s", e.Name())
		}
	}

	back, err := LoadExpertUsageHistogram(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if back.Len() != 3 {
		t.Fatalf("reloaded Len = %d, want 3", back.Len())
	}
	for _, want := range []struct {
		layer, expert int
		count         float64
	}{{0, 0, 10}, {0, 1, 9}, {1, 2, 3}} {
		if got := back.Count(want.layer, want.expert); got != want.count {
			t.Fatalf("reloaded Count(%d,%d) = %v, want %v", want.layer, want.expert, got, want.count)
		}
	}

	// A missing dump is a cold first run: an empty prior, not an error.
	empty, err := LoadExpertUsageHistogram(filepath.Join(dir, "does-not-exist.json"))
	if err != nil {
		t.Fatalf("Load missing: unexpected error %v", err)
	}
	if empty.Len() != 0 {
		t.Fatalf("missing dump Len = %d, want 0", empty.Len())
	}

	// Summed at boot: a second turn's dump folds into the first.
	h2 := NewExpertUsageHistogram()
	h2.ObserveTrace(wpTrace("turn2", wpTouch{0, 0, 5}, wpTouch{2, 3, 4}))
	path2 := filepath.Join(dir, "turn2.json")
	if err := h2.Persist(path2); err != nil {
		t.Fatalf("Persist turn2: %v", err)
	}
	sum, err := SumExpertUsageHistograms(path, path2)
	if err != nil {
		t.Fatalf("Sum: %v", err)
	}
	for _, want := range []struct {
		layer, expert int
		count         float64
	}{{0, 0, 15}, {0, 1, 9}, {1, 2, 3}, {2, 3, 4}} {
		if got := sum.Count(want.layer, want.expert); got != want.count {
			t.Fatalf("summed Count(%d,%d) = %v, want %v", want.layer, want.expert, got, want.count)
		}
	}
	if sum.Len() != 4 {
		t.Fatalf("summed Len = %d, want 4", sum.Len())
	}
}

// TestExpertUsageHistogramLoadRejectsForeignSchema pins the "must not silently zero the prior" contract:
// a present-but-foreign file is an error, never an empty histogram that would erase yesterday's routing.
func TestExpertUsageHistogramLoadRejectsForeignSchema(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "foreign.json")
	if err := os.WriteFile(path, []byte(`{"schema":"not-fak/v9","counts":[]}`), 0o644); err != nil {
		t.Fatalf("seed foreign file: %v", err)
	}
	if _, err := LoadExpertUsageHistogram(path); err == nil {
		t.Fatal("Load of a foreign-schema file returned nil error; want a refusal")
	}
}

// TestWarmStartPinsSelectHottestPrior witnesses the startup warm-start: pins are the budget hottest
// experts of the summed prior, deterministically, with the boundary budgets behaving.
func TestWarmStartPinsSelectHottestPrior(t *testing.T) {
	prior := NewExpertUsageHistogram()
	prior.ObserveTrace(wpTrace("prior", wpTouch{0, 0, 10}, wpTouch{0, 1, 9}, wpTouch{0, 2, 1}, wpTouch{0, 3, 1}))

	pins := WarmStartExpertPins(prior, 2)
	if pins.Len() != 2 {
		t.Fatalf("pinned Len = %d, want 2", pins.Len())
	}
	if !pins.IsPinned(0, 0) || !pins.IsPinned(0, 1) {
		t.Fatal("warm-start did not pin the two hottest experts")
	}
	if pins.IsPinned(0, 2) || pins.IsPinned(0, 3) {
		t.Fatal("warm-start pinned a cold expert")
	}

	if n := WarmStartExpertPins(prior, 99).Len(); n != 4 {
		t.Fatalf("budget past population pinned %d, want all 4", n)
	}
	if n := WarmStartExpertPins(prior, 0).Len(); n != 0 {
		t.Fatalf("budget 0 pinned %d, want 0", n)
	}
}

// TestRepinPassTwoPhaseDriftBounded is the core witness: a phase-1 prior warm-starts pins on experts
// (0,0),(0,1); a phase-2 burst on (0,2),(0,3) drives the actuator to swap the coldest pins for the
// hottest recent experts — bounded by maxSwaps, quiescent when maxSwaps=0, and every swap improving
// (InHeat strictly exceeds OutHeat). Heat is pinned to exact halves so the drift is reproducible:
//
//	warm-start heat  {(0,0):10,(0,1):9,(0,2):1,(0,3):1}
//	after Decay(0.5) {(0,0):5, (0,1):4.5,(0,2):0.5,(0,3):0.5}
//	after +recent    {(0,0):5, (0,1):4.5,(0,2):50.5,(0,3):50.5}  (recent = +50 on 0,2 and 0,3)
func TestRepinPassTwoPhaseDriftBounded(t *testing.T) {
	dir := t.TempDir()
	p1 := NewExpertUsageHistogram()
	p1.ObserveTrace(wpTrace("phase1", wpTouch{0, 0, 10}, wpTouch{0, 1, 9}, wpTouch{0, 2, 1}, wpTouch{0, 3, 1}))
	dump := filepath.Join(dir, "phase1.json")
	if err := p1.Persist(dump); err != nil {
		t.Fatalf("Persist phase1: %v", err)
	}
	prior, err := SumExpertUsageHistograms(dump)
	if err != nil {
		t.Fatalf("Sum phase1: %v", err)
	}

	recent := NewExpertUsageHistogram()
	recent.ObserveTrace(wpTrace("phase2", wpTouch{0, 2, 50}, wpTouch{0, 3, 50}))

	// A — quiescent: maxSwaps=0 ages the heat but drifts no pins.
	quiet := WarmStartExpertPins(prior, 2)
	if sw := quiet.RepinPass(recent, 0.5, 0); sw != nil {
		t.Fatalf("maxSwaps=0 performed swaps: %+v", sw)
	}
	if !quiet.IsPinned(0, 0) || !quiet.IsPinned(0, 1) {
		t.Fatal("quiescent pass drifted the pinned set")
	}

	// B — bounded: maxSwaps=1 evicts only the single coldest pin (0,1) for the hottest recent (0,2).
	one := WarmStartExpertPins(prior, 2)
	sw1 := one.RepinPass(recent, 0.5, 1)
	if len(sw1) != 1 {
		t.Fatalf("maxSwaps=1 performed %d swaps, want 1", len(sw1))
	}
	if sw1[0].InHeat <= sw1[0].OutHeat {
		t.Fatalf("swap not improving: InHeat %v <= OutHeat %v", sw1[0].InHeat, sw1[0].OutHeat)
	}
	if sw1[0].OutLayer != 0 || sw1[0].OutExpert != 1 || sw1[0].InLayer != 0 || sw1[0].InExpert != 2 {
		t.Fatalf("bounded swap = out(%d,%d) in(%d,%d), want out(0,1) in(0,2)",
			sw1[0].OutLayer, sw1[0].OutExpert, sw1[0].InLayer, sw1[0].InExpert)
	}
	if !one.IsPinned(0, 0) || !one.IsPinned(0, 2) || one.IsPinned(0, 1) {
		t.Fatal("bounded pass did not drift by exactly one pin")
	}

	// C — full drift: maxSwaps=2 personalizes the whole set to the phase-2 pair.
	two := WarmStartExpertPins(prior, 2)
	sw2 := two.RepinPass(recent, 0.5, 2)
	if len(sw2) != 2 {
		t.Fatalf("maxSwaps=2 performed %d swaps, want 2", len(sw2))
	}
	for i, s := range sw2 {
		if s.InHeat <= s.OutHeat {
			t.Fatalf("swap %d not improving: InHeat %v <= OutHeat %v", i, s.InHeat, s.OutHeat)
		}
	}
	if !two.IsPinned(0, 2) || !two.IsPinned(0, 3) {
		t.Fatal("full drift did not pin the phase-2 pair")
	}
	if two.IsPinned(0, 0) || two.IsPinned(0, 1) {
		t.Fatal("full drift left a phase-1 pin resident")
	}
}

// TestPagedRingWarmStartAndRepin witnesses the ring-side seam: a never-warm-started ring reports no pin
// and its RepinPass is a no-op; after WarmStartPins the ring pins the phase-1 hot pair; and RepinPass
// drifts it to the phase-2 pair — the pagedRing.RepinPass(maxSwaps) turn-boundary call the issue names.
func TestPagedRingWarmStartAndRepin(t *testing.T) {
	prior := NewExpertUsageHistogram()
	prior.ObserveTrace(wpTrace("prior", wpTouch{0, 0, 10}, wpTouch{0, 1, 9}, wpTouch{0, 2, 1}, wpTouch{0, 3, 1}))
	recent := NewExpertUsageHistogram()
	recent.ObserveTrace(wpTrace("recent", wpTouch{0, 2, 50}, wpTouch{0, 3, 50}))

	ring := newPagedRing(nil, q4kBlockBytes*8)
	if ring.isExpertPinned(0, 0) {
		t.Fatal("cold ring reports a pin before any warm-start")
	}
	if sw := ring.RepinPass(recent, 0.5, 2); sw != nil {
		t.Fatalf("cold ring RepinPass returned %+v, want nil", sw)
	}

	ring.WarmStartPins(prior, 2)
	if !ring.isExpertPinned(0, 0) || !ring.isExpertPinned(0, 1) {
		t.Fatal("WarmStartPins did not pin the hot pair")
	}

	sw := ring.RepinPass(recent, 0.5, 2)
	if len(sw) != 2 {
		t.Fatalf("ring RepinPass performed %d swaps, want 2", len(sw))
	}
	if !ring.isExpertPinned(0, 2) || !ring.isExpertPinned(0, 3) {
		t.Fatal("ring RepinPass did not drift to the phase-2 pair")
	}
	if ring.isExpertPinned(0, 0) || ring.isExpertPinned(0, 1) {
		t.Fatal("ring RepinPass left a phase-1 pin resident")
	}
}
