package model

import (
	"sync"
	"testing"
)

// recordingQwen35PanelRunner is a Go-only test double for the prefill panel walk.
//
// It implements qwen35MetalForwardSequenceRunner without any Metal/cgo dependency so
// the panel-walk arithmetic in tryPrefillQwen35HybridQ4K can be exercised on every host.
// The loop is pure Go: it only requires s.qwen35HAL.sequenceAccepted and a
// sequenceBackend satisfying the runner interface, and it calls
// Qwen35MetalForwardSequence once per panel, aggregating the per-panel receipt into a
// single session receipt.
//
// The double reports each accepted panel as exactly one command buffer and one terminal
// wait — the real resident-Q4_K panel body's contract
// (metal_prefill_hybrid.go, Qwen35MetalForwardSequence receipt:
// CommandBuffers:1, TerminalWaits:1) — so the aggregate receipt is a faithful count
// witness for the #13042 budget assertion without a GPU.
type recordingQwen35PanelRunner struct {
	// Embed the interface so this double satisfies the static backend type on every
	// build (the concrete Metal backend is cgo-gated). Only the runner methods below
	// are ever invoked by the panel walk.
	Qwen35GDNPreprojectedSequenceBackend
	mu      sync.Mutex
	calls   int
	lengths []int
}

func (r *recordingQwen35PanelRunner) Qwen35MetalForwardSequence(_ *Session, ids []int) ([]float32, Qwen35MetalForwardSequenceReceipt, bool, error) {
	r.mu.Lock()
	r.calls++
	r.lengths = append(r.lengths, len(ids))
	r.mu.Unlock()
	receipt := Qwen35MetalForwardSequenceReceipt{
		Path:           Qwen35MetalGDNSequenceForwardPath,
		Available:      true,
		SelectorState:  Qwen35MetalSequenceSelectorOn,
		EvidenceState:  Qwen35MetalSequenceEvidenceExecuted,
		Tokens:         len(ids),
		CommandBuffers: 1,
		TerminalWaits:  1,
		Committed:      true,
		CompletedWait:  true,
	}
	return make([]float32, 256), receipt, true, nil
}

func (r *recordingQwen35PanelRunner) Qwen35MetalForwardSequenceReceipt() (Qwen35MetalForwardSequenceReceipt, bool) {
	return Qwen35MetalForwardSequenceReceipt{}, false
}

func (r *recordingQwen35PanelRunner) snapshot() (int, []int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.calls, append([]int(nil), r.lengths...)
}

// newPanelBudgetSession wires a fresh synthetic Qwen3.8 hybrid session to the Go-only
// runner with the resident sequence owner pre-accepted, so the production panel walk in
// tryPrefillQwen35HybridQ4K is reached with zero Metal.
func newPanelBudgetSession(t *testing.T) (*Session, *recordingQwen35PanelRunner) {
	t.Helper()
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	runner := &recordingQwen35PanelRunner{}
	s := m.NewSession()
	t.Cleanup(s.Close)
	s.Q4K, s.MetalQ4K = true, true
	s.qwen35HAL = &qwen35HALState{
		sequenceAccepted: true,
		sequenceBackend:  runner,
		sequenceLayers:   make([]Qwen35GDNAuxState, cfg.NumLayers),
	}
	return s, runner
}

// TestPrefillPanelRoundTripBudget is the L1 regression guard for issue #13042: a fixed
// long prompt's resident-Q4_K Metal prefill must not issue one terminal GPU wait per
// 32-token panel. It sources the aggregate round-trip count from the same
// Qwen35MetalForwardSequenceReceipt fields the physical Mac path fills
// (CommandBuffers, TerminalWaits) and asserts they stay within the named
// PrefillPanelRoundTripBudget, which is a small constant — NOT a function of len/32.
//
// Control arm: the OLD serial panelization issued ceil(cover/32) command buffers and
// terminal waits (one per 32-token panel); for the fixed 256-token prompt below that is
// 8, comfortably above PrefillPanelRoundTripBudget=2, so the assertion is not vacuous.
// A regression to the per-panel barrier turns this test red instead of shipping.
//
// Count-only, [SW-VERIFIED]: no wall-time and no GPU-utilization claim.
func TestPrefillPanelRoundTripBudget(t *testing.T) {
	// Fixed prompt = a multiple of 32, so the whole prompt is panel-covered and the
	// host-served tail is empty; only the panel round-trips are measured.
	const promptTokens = PrefillPanelRoundTripBudgetPromptTokens
	const oldPanelTokens = 32
	const oldPanelRoundTrips = promptTokens / oldPanelTokens // the pre-#13041 shape: 8

	s, runner := newPanelBudgetSession(t)
	ids := make([]int, promptTokens)
	for i := range ids {
		ids[i] = i % 512
	}
	if _, accepted := s.tryPrefillQwen35HybridQ4K(ids, false); !accepted {
		t.Fatalf("tryPrefillQwen35HybridQ4K declined a %d-token prompt, want accepted", promptTokens)
	}

	calls, lengths := runner.snapshot()
	if calls == 0 {
		t.Fatal("panel walk issued no panels")
	}
	for _, l := range lengths {
		if l < 1 || l > 128 {
			t.Fatalf("panel length %d outside the witnessed [1,128] range (lengths=%v)", l, lengths)
		}
	}

	agg := s.Qwen35MetalForwardSequenceReceipt()
	if agg.EvidenceState != Qwen35MetalSequenceEvidenceExecuted {
		t.Fatalf("aggregate receipt evidence=%q, want executed (receipt=%+v)", agg.EvidenceState, agg)
	}
	if agg.Tokens != promptTokens {
		t.Fatalf("aggregate Tokens=%d, want %d", agg.Tokens, promptTokens)
	}

	// The L1 budget assertion, sourced from the receipt the production path fills.
	if agg.CommandBuffers > PrefillPanelRoundTripBudget {
		t.Fatalf("prefill CommandBuffers=%d exceeds PrefillPanelRoundTripBudget=%d for a %d-token prompt",
			agg.CommandBuffers, PrefillPanelRoundTripBudget, promptTokens)
	}
	if agg.TerminalWaits > PrefillPanelRoundTripBudget {
		t.Fatalf("prefill TerminalWaits=%d exceeds PrefillPanelRoundTripBudget=%d for a %d-token prompt",
			agg.TerminalWaits, PrefillPanelRoundTripBudget, promptTokens)
	}

	// Control: the OLD per-32 shape would have spent oldPanelRoundTrips round-trips,
	// strictly above the budget, so a regression cannot pass this assertion.
	if oldPanelRoundTrips <= PrefillPanelRoundTripBudget {
		t.Fatalf("control is vacuous: old panel round-trips=%d <= budget=%d", oldPanelRoundTrips, PrefillPanelRoundTripBudget)
	}
	if agg.CommandBuffers >= oldPanelRoundTrips {
		t.Fatalf("round-trips did not collapse: CommandBuffers=%d >= old per-32 count=%d", agg.CommandBuffers, oldPanelRoundTrips)
	}
}

// TestPrefillPanelRoundTripBudgetDeclineFallsBack pins the fail-open edge of the #13042
// assertion: a declined admission leaves no executed-panel evidence and the walk must not
// fabricate a collapsed receipt.
func TestPrefillPanelRoundTripBudgetDeclineFallsBack(t *testing.T) {
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	s := m.NewSession()
	t.Cleanup(s.Close)
	s.Q4K, s.MetalQ4K = true, true
	s.qwen35HAL = &qwen35HALState{
		sequenceAccepted: false,
		sequenceBackend:  &recordingQwen35PanelRunner{},
		sequenceLayers:   make([]Qwen35GDNAuxState, cfg.NumLayers),
	}
	if _, accepted := s.tryPrefillQwen35HybridQ4K(make([]int, 96), false); !accepted {
		t.Fatal("declined-sequence session returned accepted=false, want the fail-open host path to accept")
	}
	if r := s.Qwen35MetalForwardSequenceReceipt(); r.EvidenceState == Qwen35MetalSequenceEvidenceExecuted {
		t.Fatalf("declined admission fabricated executed panel evidence: %+v", r)
	}
}
