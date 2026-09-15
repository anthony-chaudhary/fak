package model

import (
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
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

// deviceKVRecorder is the Go-only device-resident-KV admission double for #13087.
// It satisfies qwen35MetalDeviceKVAdmitter without Metal or cgo, so the walk's
// device-resident branch (admit -> attach -> reconcile -> detach) is exercised on
// every host. admit==false models a backend declining the device path, which must
// leave the host counters zero (fail-open).
type deviceKVRecorder struct {
	recordingQwen35PanelRunner
	admit bool

	mu             sync.Mutex
	admits         int
	detaches       int
	reconciles     int
	reconcileBase  int
	reconcileRows  int
	reconcileErr   error
	attachedTokens int
}

// AdmitDeviceKV is the admission seam. It records the request and, when admit is
// false, returns nil so the walk keeps the historical host-append path.
func (d *deviceKVRecorder) AdmitDeviceKV(_ *Session, tokens int) *metalgemm.DeviceKV {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.admits++
	if !d.admit {
		return nil
	}
	d.attachedTokens = tokens
	// A non-nil, well-formed pair stands in for the real device allocation; the
	// walk only requires a non-nil handle to take the device branch.
	return &metalgemm.DeviceKV{}
}

// Qwen35MetalForwardSequence mirrors the REAL backend's device-branch bookkeeping
// (metal_prefill_hybrid.go): when this double admitted the walk, every executed
// panel increments the same production counters the native path increments. It
// does NOT fabricate them independently of admission — a declined double leaves
// them at zero — so the walk-owner wiring (admit -> attach -> per-panel device
// bookkeeping -> reconcile -> detach) is exercised on every host. The REAL
// criterion-1 readback evidence is the GPU witness
// TestPrefillQwen35HybridQ4KDeviceKVPanelHostReadbackFlat; this double only proves
// the wiring, not the byte count.
func (d *deviceKVRecorder) Qwen35MetalForwardSequence(s *Session, ids []int) ([]float32, Qwen35MetalForwardSequenceReceipt, bool, error) {
	if s != nil {
		d.mu.Lock()
		admitted := d.admit
		d.mu.Unlock()
		if admitted {
			s.q4kHybridPrefillDevicePanels++
			s.q4kHybridPrefillDeviceRows += len(ids)
		}
	}
	return d.recordingQwen35PanelRunner.Qwen35MetalForwardSequence(s, ids)
}

func (d *deviceKVRecorder) DetachDeviceKV() *metalgemm.DeviceKV {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.detaches++
	return &metalgemm.DeviceKV{}
}

func (d *deviceKVRecorder) ReconcileDeviceKV(_ *Session, base, rows int) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.reconciles++
	d.reconcileBase, d.reconcileRows = base, rows
	return d.reconcileErr
}

func (d *deviceKVRecorder) deviceSnapshot() (admits, detaches, reconciles, base, rows, tokens int) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.admits, d.detaches, d.reconciles, d.reconcileBase, d.reconcileRows, d.attachedTokens
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

// newDeviceKVSession wires a fresh synthetic Qwen3.8 hybrid session to the Go-only
// device-KV admission double with the resident sequence owner pre-accepted.
func newDeviceKVSession(t *testing.T, admit bool) (*Session, *deviceKVRecorder) {
	t.Helper()
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	rec := &deviceKVRecorder{admit: admit}
	s := m.NewSession()
	t.Cleanup(s.Close)
	s.Q4K, s.MetalQ4K = true, true
	s.qwen35HAL = &qwen35HALState{
		sequenceAccepted: true,
		sequenceBackend:  rec,
		sequenceLayers:   make([]Qwen35GDNAuxState, cfg.NumLayers),
	}
	return s, rec
}

// TestPrefillQwen35HybridQ4KDeviceKVReceiptCollapse is the #13087 receipt witness: on the
// admitted device-resident path the panel walk attaches ONE device KV pair for the whole
// prompt, reconciles it ONCE (base = the host prefix at walk start, rows = the rows the
// panels appended), and records a device panel/row count matching what the walk executed.
// The per-panel host KV readback/re-upload is gone, so the host KV-append work no longer
// scales with panel count; the aggregate CommandBuffers still stays ceil(cover/128).
//
// The counters (s.q4kHybridPrefillDevicePanels/DeviceRows) are the typed receipt the
// production backend fills; this double emulates that same increment on admission, so a
// regression that silently drops the device branch turns this red instead of passing. It
// is a WIRING witness only: the non-fabricated criterion-1 readback number comes from the
// real GPU path in TestPrefillQwen35HybridQ4KDeviceKVPanelHostReadbackFlat.
//
// [SW-VERIFIED] — Go-only, no GPU.
func TestPrefillQwen35HybridQ4KDeviceKVReceiptCollapse(t *testing.T) {
	// A prompt that is a multiple of 32 but not of 128, so it rides multiple panels
	// (cover=256 -> 2 panels of 128) and the device branch is unmistakably exercised.
	const promptTokens = 256
	s, rec := newDeviceKVSession(t, true)
	ids := make([]int, promptTokens)
	for i := range ids {
		ids[i] = i % 512
	}
	if _, accepted := s.tryPrefillQwen35HybridQ4K(ids, false); !accepted {
		t.Fatalf("tryPrefillQwen35HybridQ4K declined a %d-token prompt, want accepted", promptTokens)
	}

	calls, _ := rec.snapshot()
	admits, detaches, reconciles, base, rows, tokens := rec.deviceSnapshot()
	if admits != 1 {
		t.Fatalf("AdmitDeviceKV calls=%d, want 1 (one pair per walk)", admits)
	}
	if tokens != promptTokens {
		t.Fatalf("AdmitDeviceKV tokens=%d, want %d", tokens, promptTokens)
	}
	if reconciles != 1 {
		t.Fatalf("ReconcileDeviceKV calls=%d, want exactly 1 terminal reconcile", reconciles)
	}
	// The walk detaches once explicitly after reconcile and once from a deferred
	// guard that also covers the panic/early-return path; DetachDeviceKV is defined
	// to return nil on the second call (so the real pair is freed exactly once), so
	// the witness only requires at least one detach and no leaked pair.
	if detaches < 1 {
		t.Fatalf("DetachDeviceKV calls=%d, want at least 1", detaches)
	}
	// base is the host prefix length at walk start (a fresh session's cache is empty);
	// rows is exactly the rows the panels appended, which is the panel-walk total.
	if base != 0 {
		t.Fatalf("reconcile base=%d, want 0 for a fresh session", base)
	}
	if rows != promptTokens {
		t.Fatalf("reconcile rows=%d, want %d (the panel-covered rows)", rows, promptTokens)
	}
	// These counters are the double's emulation of the real backend's per-panel
	// bookkeeping, so this asserts the walk owner threads the device pair through
	// every panel and reconciles once — a wiring check, NOT production byte
	// evidence. The real, non-fabricated readback witness is the GPU test
	// TestPrefillQwen35HybridQ4KDeviceKVPanelHostReadbackFlat.
	if got := s.q4kHybridPrefillDevicePanels; got != calls {
		t.Fatalf("emulated device panels=%d, want %d (every executed panel)", got, calls)
	}
	if got := s.q4kHybridPrefillDeviceRows; got != promptTokens {
		t.Fatalf("emulated device rows=%d, want %d", got, promptTokens)
	}
	if calls != 2 {
		t.Fatalf("panel calls=%d, want 2 (ceil(256/128))", calls)
	}

	// The panel round-trips still collapse (no regression from this change): the
	// aggregate command buffers stay at the widened-panel count, not the per-panel
	// KV round-trip count.
	agg := s.Qwen35MetalForwardSequenceReceipt()
	if agg.CommandBuffers > PrefillPanelRoundTripBudget {
		t.Fatalf("prefill CommandBuffers=%d exceeds PrefillPanelRoundTripBudget=%d", agg.CommandBuffers, PrefillPanelRoundTripBudget)
	}
}

// TestPrefillQwen35HybridQ4KDeviceKVDeclineStaysHost pins the fail-open edge: a backend
// that declines the device-resident admission keeps the historical host-append walk, the
// device counters stay zero, and it never calls reconcile/detach. KV/state are untouched
// by the decline.
//
// [SW-VERIFIED] — Go-only, no GPU.
func TestPrefillQwen35HybridQ4KDeviceKVDeclineStaysHost(t *testing.T) {
	const promptTokens = 256
	s, rec := newDeviceKVSession(t, false)
	beforeLen := s.Cache.Len()
	ids := make([]int, promptTokens)
	for i := range ids {
		ids[i] = i % 512
	}
	if _, accepted := s.tryPrefillQwen35HybridQ4K(ids, false); !accepted {
		t.Fatalf("declined device path returned accepted=false, want the fail-open host path to accept")
	}
	admits, detaches, reconciles, _, _, _ := rec.deviceSnapshot()
	if admits != 1 {
		t.Fatalf("AdmitDeviceKV calls=%d, want 1 (the walk always probes admission)", admits)
	}
	if reconciles != 0 || detaches != 0 {
		t.Fatalf("declined admission reconciled/detached: reconciles=%d detaches=%d", reconciles, detaches)
	}
	if s.q4kHybridPrefillDevicePanels != 0 || s.q4kHybridPrefillDeviceRows != 0 {
		t.Fatalf("declined admission recorded device counters (%d, %d), want 0",
			s.q4kHybridPrefillDevicePanels, s.q4kHybridPrefillDeviceRows)
	}
	// The host walk still advanced the session state (positions booked by the
	// panel runner); the device counters are the only thing the decline changed.
	if got := s.Cache.Len(); got != beforeLen {
		t.Fatalf("declined admission mutated the host cache prefix: len=%d, want %d", got, beforeLen)
	}
}
