package model

import (
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// failNthPanelRunner is a Go-only prefill-panel double that models the #13473
// PRE-COMMIT failure: the Nth Qwen35MetalForwardSequence call reports
// accepted==true with an error and an UNAVAILABLE receipt, which is exactly what the
// resident backend returns for an encode/graph failure before its single FinishRead
// commit (host KV and Cache positions are unmutated). Before the fix the panel loop
// panicked on any error and killed the serving process; after the fix the walk breaks
// on the pre-commit shape and fails open to the host path.
type failNthPanelRunner struct {
	*recordingQwen35PanelRunner
	failOn int

	mu    sync.Mutex
	calls int
}

func (r *failNthPanelRunner) Qwen35MetalForwardSequence(s *Session, ids []int) ([]float32, Qwen35MetalForwardSequenceReceipt, bool, error) {
	r.mu.Lock()
	r.calls++
	n := r.calls
	r.mu.Unlock()
	if n == r.failOn {
		// Pre-commit shape: accepted==true, zero receipt (Available==false), error.
		return nil, Qwen35MetalForwardSequenceReceipt{}, true, errors.New("metalgemm: Qwen full-attention device-KV encode failed")
	}
	return r.recordingQwen35PanelRunner.Qwen35MetalForwardSequence(s, ids)
}

func (r *failNthPanelRunner) callCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.calls
}

// newFailureSessionRunner wires a recording double to a fake preprojected-sequence
// backend so a failing wrapper embedding it promotes the lifecycle/operation methods
// the host fallback calls.
func newFailureSessionRunner() *recordingQwen35PanelRunner {
	return &recordingQwen35PanelRunner{Qwen35GDNPreprojectedSequenceBackend: newFakeQwen35GDNSequenceBackend()}
}

// failDeviceKVPanelRunner is the device-resident variant: it implements the device-KV
// admitter seam and the preprojected-sequence backend, and fails a chosen panel with
// the PRE-COMMIT decline shape (accepted=false). It records the reconcile geometry so
// the test can prove the walk reconciles the ACTUAL executed rows (`covered`), not the
// planned panel cover, when a device panel declines mid-walk — the #13473 path.
type failDeviceKVPanelRunner struct {
	*fakeQwen35GDNSequenceBackend
	mu            sync.Mutex
	calls         int
	failOn        int
	reconciles    int
	reconcileBase int
	reconcileRows int
}

func (r *failDeviceKVPanelRunner) Qwen35MetalForwardSequence(s *Session, ids []int) ([]float32, Qwen35MetalForwardSequenceReceipt, bool, error) {
	r.mu.Lock()
	r.calls++
	n := r.calls
	r.mu.Unlock()
	if n == r.failOn {
		// Pre-commit shape: accepted==true with an unavailable receipt.
		return nil, Qwen35MetalForwardSequenceReceipt{}, true, errors.New("metalgemm: Qwen full-attention device-KV encode failed")
	}
	if s != nil {
		s.q4kHybridPrefillDevicePanels++
		s.q4kHybridPrefillDeviceRows += len(ids)
	}
	return make([]float32, 256), Qwen35MetalForwardSequenceReceipt{
		Path: Qwen35MetalGDNSequenceForwardPath, Available: true,
		SelectorState: Qwen35MetalSequenceSelectorOn, EvidenceState: Qwen35MetalSequenceEvidenceExecuted,
		Tokens: len(ids), CommandBuffers: 1, TerminalWaits: 1, Committed: true, CompletedWait: true,
	}, true, nil
}

func (r *failDeviceKVPanelRunner) Qwen35MetalForwardSequenceReceipt() (Qwen35MetalForwardSequenceReceipt, bool) {
	return Qwen35MetalForwardSequenceReceipt{}, false
}

func (r *failDeviceKVPanelRunner) AdmitDeviceKV(_ *Session, _ int) *metalgemm.DeviceKV {
	return &metalgemm.DeviceKV{}
}

func (r *failDeviceKVPanelRunner) DetachDeviceKV() *metalgemm.DeviceKV {
	return &metalgemm.DeviceKV{}
}

func (r *failDeviceKVPanelRunner) ReconcileDeviceKV(_ *Session, base, rows int) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.reconciles++
	r.reconcileBase, r.reconcileRows = base, rows
	return nil
}

func (r *failDeviceKVPanelRunner) geometry() (reconciles, base, rows int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.reconciles, r.reconcileBase, r.reconcileRows
}

// newPanelFailureSession builds a synthetic Qwen3.8 hybrid session whose admitted
// sequence owner has working auxiliary state and a working preprojected-sequence
// backend, so the host fallback a pre-commit panel decline drops into is a real,
// executable path rather than the zero-state panic a bare recording double would hit.
// The model is quantized so the host projections resolve.
func newPanelFailureSession(t *testing.T, runner Qwen35GDNPreprojectedSequenceBackend) *Session {
	t.Helper()
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	s := m.NewSession()
	t.Cleanup(s.Close)
	s.Q4K, s.MetalQ4K = true, true
	backend := newFakeQwen35GDNSequenceBackend()
	layers := make([]Qwen35GDNAuxState, cfg.NumLayers)
	geom := s.qwen35GDNSequenceGeometry()
	for l := 0; l < cfg.NumLayers; l++ {
		if !cfg.isLinearAttnLayer(l) {
			continue
		}
		st, err := backend.NewQwen35GDNAuxState(l, geom)
		if err != nil {
			t.Fatalf("allocate aux state for layer %d: %v", l, err)
		}
		layers[l] = st
	}
	s.qwen35HAL = &qwen35HALState{sequenceAccepted: true, sequenceBackend: runner, sequenceLayers: layers}
	return s
}

// TestPrefillQwen35HybridQ4KPanelFailureDoesNotPanic is the #13473 regression guard:
// a PRE-COMMIT panel decline on the SECOND panel must NOT panic out of
// tryPrefillQwen35HybridQ4K. The walk stops at the decline and replays the remaining
// rows through the host path, so the process and the 8080 router endpoint survive.
//
// [SW-VERIFIED] — Go-only, no GPU.
func TestPrefillQwen35HybridQ4KPanelFailureDoesNotPanic(t *testing.T) {
	runner := &failNthPanelRunner{recordingQwen35PanelRunner: newFailureSessionRunner(), failOn: 2}
	s := newPanelFailureSession(t, runner)

	defer func() {
		if rec := recover(); rec != nil {
			t.Fatalf("pre-commit panel decline panicked out of tryPrefillQwen35HybridQ4K: %v", rec)
		}
	}()
	ids := make([]int, 256)
	for i := range ids {
		ids[i] = i % 512
	}
	if _, accepted := s.tryPrefillQwen35HybridQ4K(ids, false); !accepted {
		t.Fatal("panel-decline walk returned accepted=false, want the fail-open host path to accept")
	}
	if s.q4kHybridPrefillChunks < 1 {
		t.Fatalf("host fallback did not run after a mid-walk panel decline: chunks=%d", s.q4kHybridPrefillChunks)
	}
	// Panel 0 (128 rows) executed and was reconciled to the device triple before panel 1
	// declined; this portable recording double appends no host rows for its stub panels,
	// so the host cache holds exactly the unexecuted remainder (256-128) replayed by
	// prefillQwen35HybridQ4KHidden. The aggregate accounting below is the no-skip witness.
	const executedRows = 128
	if got := s.Cache.Len(); got != len(ids)-executedRows {
		t.Fatalf("host remainder len after a mid-walk decline=%d, want %d", got, len(ids)-executedRows)
	}
	agg := s.Qwen35MetalForwardSequenceReceipt()
	if agg.Tokens != len(ids) {
		t.Fatalf("aggregate receipt Tokens=%d, want %d (no token skipped across the declined panel)", agg.Tokens, len(ids))
	}
}

// TestPrefillQwen35HybridQ4KDeviceKVPanelFailureReconcilesExecutedRows is the
// device-resident half of the #13473 guard and the direct witness for the
// `covered := executedRows` fix. Prompt 256 rides two 128-token device panels; panel 0
// executes and appends its rows to the device triple, panel 1 declines pre-commit. The
// walk must reconcile the rows that ACTUALLY executed (128), not the planned cover
// (256), then replay the unexecuted remainder through the host path.
//
// A regression to the planned panelCover download would ask the device triple for rows
// it never wrote and desynchronize the host cache; a regression to the pre-fix backend
// contract (accepted==true on a pre-commit failure) would panic and kill the process.
//
// [SW-VERIFIED] — Go-only, no GPU.
func TestPrefillQwen35HybridQ4KDeviceKVPanelFailureReconcilesExecutedRows(t *testing.T) {
	runner := &failDeviceKVPanelRunner{
		fakeQwen35GDNSequenceBackend: newFakeQwen35GDNSequenceBackend(),
		failOn:                       2,
	}
	s := newPanelFailureSession(t, runner)

	defer func() {
		if rec := recover(); rec != nil {
			t.Fatalf("device-KV panel decline panicked out of tryPrefillQwen35HybridQ4K: %v", rec)
		}
	}()
	ids := make([]int, 256)
	for i := range ids {
		ids[i] = i % 512
	}
	if _, accepted := s.tryPrefillQwen35HybridQ4K(ids, false); !accepted {
		t.Fatal("device-KV panel-decline walk returned accepted=false, want the fail-open host path to accept")
	}

	reconciles, base, rows := runner.geometry()
	if reconciles != 1 {
		t.Fatalf("ReconcileDeviceKV calls=%d, want exactly 1", reconciles)
	}
	if base != 0 {
		t.Fatalf("reconcile base=%d, want 0 for a fresh session", base)
	}
	// The load-bearing assertion: exactly the executed panel's rows (128), not
	// panelCover (256), because panel 1 declined before appending to the device triple.
	if rows != 128 {
		t.Fatalf("reconcile rows=%d, want 128 (executed panel only, not the planned 256)", rows)
	}
	// This portable double stubs the device download, so the host cache holds exactly the
	// unexecuted remainder (256-128) replayed by prefillQwen35HybridQ4KHidden. The real
	// ReconcileDeviceKV appends the 128 device rows, making the total 256; reconcile
	// rows==128 + host remainder==128 is the portable witness that no token was skipped
	// or duplicated.
	if got := s.Cache.Len(); got != len(ids)-rows {
		t.Fatalf("host remainder len after a device-panel decline=%d, want %d", got, len(ids)-rows)
	}
	if s.q4kHybridPrefillChunks < 1 {
		t.Fatal("host remainder path did not run after the device panel declined")
	}
	agg := s.Qwen35MetalForwardSequenceReceipt()
	if agg.Tokens != len(ids) {
		t.Fatalf("aggregate receipt Tokens=%d, want %d", agg.Tokens, len(ids))
	}
}

// TestPrefillQwen35HybridQ4KFirstPanelFailureFallsBack pins the fail-open edge: a panel
// decline on the FIRST call executes no panel, so the device counters stay zero and the
// whole prompt replays through the host path — no panic, accepted.
//
// [SW-VERIFIED] — Go-only, no GPU.
func TestPrefillQwen35HybridQ4KFirstPanelFailureFallsBack(t *testing.T) {
	runner := &failNthPanelRunner{recordingQwen35PanelRunner: newFailureSessionRunner(), failOn: 1}
	s := newPanelFailureSession(t, runner)

	defer func() {
		if rec := recover(); rec != nil {
			t.Fatalf("first-panel decline panicked: %v", rec)
		}
	}()
	ids := make([]int, 256)
	for i := range ids {
		ids[i] = i % 512
	}
	if _, accepted := s.tryPrefillQwen35HybridQ4K(ids, false); !accepted {
		t.Fatal("first-panel decline returned accepted=false, want the fail-open host path to accept")
	}
	if s.q4kHybridPrefillDevicePanels != 0 || s.q4kHybridPrefillDeviceRows != 0 {
		t.Fatalf("first-panel decline recorded device work (%d, %d), want 0",
			s.q4kHybridPrefillDevicePanels, s.q4kHybridPrefillDeviceRows)
	}
	if s.q4kHybridPrefillChunks < 1 {
		t.Fatalf("full host path did not run after a first-panel decline: chunks=%d", s.q4kHybridPrefillChunks)
	}
	if got := s.Cache.Len(); got != len(ids) {
		t.Fatalf("cache len after first-panel decline=%d, want %d", got, len(ids))
	}
}

// postCommitFailRunner models the POST-COMMIT failure shape: accepted==true with a
// non-nil error and an AVAILABLE receipt (the backend fills its terminal receipt only
// after FinishRead commits). That route has advanced device GDN state and must stay
// fail-closed rather than replaying a half-advanced prefix on the host.
type postCommitFailRunner struct{ *recordingQwen35PanelRunner }

func (r *postCommitFailRunner) Qwen35MetalForwardSequence(s *Session, ids []int) ([]float32, Qwen35MetalForwardSequenceReceipt, bool, error) {
	return nil, Qwen35MetalForwardSequenceReceipt{
		Path:      Qwen35MetalGDNSequenceForwardPath,
		Available: true,
		Committed: true,
	}, true, errors.New("metalgemm: incomplete Qwen graph receipt")
}

// TestPrefillQwen35HybridQ4KPostCommitFailureStaysFailClosed documents the boundary of
// the #13473 fix: only a PRE-COMMIT decline fails open. A post-commit failure
// (accepted==true) still retires the sequence owner and fails closed, because the
// device GDN state has advanced and a host replay of the same rows would be corrupt. It
// deliberately asserts the typed fail-closed panic so a future change that silently
// fail-opens this branch turns the test red.
//
// [SW-VERIFIED] — Go-only, no GPU.
func TestPrefillQwen35HybridQ4KPostCommitFailureStaysFailClosed(t *testing.T) {
	runner := &postCommitFailRunner{recordingQwen35PanelRunner: newFailureSessionRunner()}
	s := newPanelFailureSession(t, runner)

	defer func() {
		rec := recover()
		if rec == nil {
			t.Fatal("post-commit failure fail-opened; want the fail-closed panic that retires the poisoned route")
		}
		var seq *Qwen35GDNSequenceOperationError
		if !errors.As(rec.(error), &seq) {
			t.Fatalf("post-commit failure panicked with %T (%v), want *Qwen35GDNSequenceOperationError", rec, rec)
		}
	}()
	s.tryPrefillQwen35HybridQ4K(make([]int, 256), false)
	t.Fatal("post-commit failure returned normally instead of failing closed")
}
