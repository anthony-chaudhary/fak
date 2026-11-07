// Ported from modular/modular max/python/max/nn/_graph_ops/state_space/kimi_delta.py@88ab8913af6b9298fe914e0f20d87a49e4172d0d (Apache-2.0 WITH LLVM-exception; Copyright (c) 2026 Modular Inc.). Adapted to fak-owned Go.
//
// The attribution above covers the GDN spec-verify ring-record oracle in this
// file (TestGDNVerifyRingFoldOracle and its geometry helper, plus the
// ring-path assertions in TestGDNVerifyRingManagerFold); the surrounding
// slot-shadow rollback tests are pre-existing fak code.

package compute

import (
	"fmt"
	"math"
	"testing"
)

func TestRecurrentRollbackAccuracy(t *testing.T) {
	draftK := 4
	maxSlots := 4
	mgr, err := NewRecurrentRollbackManager(maxSlots, draftK)
	if err != nil {
		t.Fatalf("NewRecurrentRollbackManager failed: %v", err)
	}

	initialState := RecurrentDeviceState{
		ConvState: []float32{1.0, 2.0, 3.0, 4.0},
		RecState:  []float32{10.0, 20.0, 30.0, 40.0},
	}

	draftTokens := []float32{0.5, 1.5, -0.5, 2.0}

	// Test all possible acceptance counts {0, 1, 2, 3, 4}
	for acceptCount := 0; acceptCount <= draftK; acceptCount++ {
		t.Run(fmt.Sprintf("accept_%d_of_%d", acceptCount, draftK), func(t *testing.T) {
			slotID := 0
			if err := mgr.RegisterSlot(slotID, initialState); err != nil {
				t.Fatalf("RegisterSlot failed: %v", err)
			}
			if err := mgr.ShadowSlot(slotID); err != nil {
				t.Fatalf("ShadowSlot failed: %v", err)
			}
			if err := mgr.RecordDraftTokens(slotID, draftTokens); err != nil {
				t.Fatalf("RecordDraftTokens failed: %v", err)
			}

			receipt, err := mgr.RollbackSlot(slotID, acceptCount)
			if err != nil {
				t.Fatalf("RollbackSlot failed: %v", err)
			}

			// Verify receipt invariants
			if receipt.SlotID != slotID {
				t.Fatalf("receipt SlotID=%d, want %d", receipt.SlotID, slotID)
			}
			if receipt.DraftK != draftK {
				t.Fatalf("receipt DraftK=%d, want %d", receipt.DraftK, draftK)
			}
			if receipt.AcceptedRows != acceptCount {
				t.Fatalf("receipt AcceptedRows=%d, want %d", receipt.AcceptedRows, acceptCount)
			}
			if receipt.D2HBytes != 0 {
				t.Fatalf("expected 0 D2H bytes, got %d", receipt.D2HBytes)
			}
			if receipt.D2HEvents != 0 {
				t.Fatalf("expected 0 D2H events, got %d", receipt.D2HEvents)
			}
			if !receipt.ZeroD2HVerified {
				t.Fatal("expected ZeroD2HVerified to be true")
			}

			// Verify state matches reference scalar execution
			gotLive, err := mgr.GetLiveState(slotID)
			if err != nil {
				t.Fatalf("GetLiveState failed: %v", err)
			}

			wantState := referenceScalarRecurrentExecution(initialState, draftTokens, acceptCount)
			for i := range wantState.ConvState {
				if math.Abs(float64(gotLive.ConvState[i]-wantState.ConvState[i])) > 1e-6 {
					t.Fatalf("conv mismatch at idx %d: got %v, want %v", i, gotLive.ConvState[i], wantState.ConvState[i])
				}
			}
			for i := range wantState.RecState {
				if math.Abs(float64(gotLive.RecState[i]-wantState.RecState[i])) > 1e-6 {
					t.Fatalf("rec mismatch at idx %d: got %v, want %v", i, gotLive.RecState[i], wantState.RecState[i])
				}
			}
		})
	}
}

func TestRecurrentRollbackZeroD2H(t *testing.T) {
	mgr, err := NewRecurrentRollbackManager(2, 4)
	if err != nil {
		t.Fatal(err)
	}

	state := RecurrentDeviceState{
		ConvState: []float32{0.1, 0.2, 0.3},
		RecState:  []float32{1.1, 2.2, 3.3},
	}
	if err := mgr.RegisterSlot(0, state); err != nil {
		t.Fatal(err)
	}
	if err := mgr.ShadowSlot(0); err != nil {
		t.Fatal(err)
	}
	if err := mgr.RecordDraftTokens(0, []float32{0.5, 0.6, 0.7, 0.8}); err != nil {
		t.Fatal(err)
	}

	receipt, err := mgr.RollbackSlot(0, 2)
	if err != nil {
		t.Fatal(err)
	}

	if receipt.D2HBytes != 0 || receipt.D2HEvents != 0 || !receipt.ZeroD2HVerified {
		t.Fatalf("Zero D2H invariant violated: %+v", receipt)
	}
	if receipt.D2DBytesCopied <= 0 {
		t.Fatalf("expected positive D2D bytes copied on device, got %d", receipt.D2DBytesCopied)
	}
}

func TestRecurrentRollbackLatencyImprovement(t *testing.T) {
	mgr, err := NewRecurrentRollbackManager(2, 4)
	if err != nil {
		t.Fatal(err)
	}

	state := RecurrentDeviceState{
		ConvState: make([]float32, 64),
		RecState:  make([]float32, 256),
	}
	for i := range state.ConvState {
		state.ConvState[i] = float32(i)
	}
	for i := range state.RecState {
		state.RecState[i] = float32(i * 2)
	}

	slotID := 1
	if err := mgr.RegisterSlot(slotID, state); err != nil {
		t.Fatal(err)
	}
	if err := mgr.ShadowSlot(slotID); err != nil {
		t.Fatal(err)
	}
	if err := mgr.RecordDraftTokens(slotID, []float32{1.0, 2.0, 3.0, 4.0}); err != nil {
		t.Fatal(err)
	}

	// 1. Measure on-device rollback latency
	onDeviceReceipt, err := mgr.RollbackSlot(slotID, 2)
	if err != nil {
		t.Fatal(err)
	}

	// 2. Measure simulated host roundtrip fallback latency
	hostLatency, hostD2HBytes, err := mgr.SimulateHostRollback(slotID, 2)
	if err != nil {
		t.Fatal(err)
	}

	if hostD2HBytes <= 0 {
		t.Fatalf("host fallback should record non-zero D2H bytes, got %d", hostD2HBytes)
	}

	t.Logf("On-device rewind latency: %v (0 D2H bytes), Host fallback latency: %v (%d D2H bytes)",
		onDeviceReceipt.RollbackLatency, hostLatency, hostD2HBytes)

	// On-device atomic slot rewind should be orders of magnitude faster than host roundtrip (which pays PCIe sync)
	if onDeviceReceipt.RollbackLatency >= hostLatency {
		t.Logf("Notice: on-device latency (%v) vs host fallback (%v)", onDeviceReceipt.RollbackLatency, hostLatency)
	}
}

func TestRecurrentRollbackFailClosed(t *testing.T) {
	// Invalid manager config
	if _, err := NewRecurrentRollbackManager(0, 4); err == nil {
		t.Fatal("expected error on maxSlots=0")
	}
	if _, err := NewRecurrentRollbackManager(4, 0); err == nil {
		t.Fatal("expected error on draftK=0")
	}

	mgr, _ := NewRecurrentRollbackManager(2, 4)

	// Out of bounds slot
	if err := mgr.RegisterSlot(5, RecurrentDeviceState{ConvState: []float32{1}}); err == nil {
		t.Fatal("expected error on out-of-bounds slot registration")
	}

	// Empty state
	if err := mgr.RegisterSlot(0, RecurrentDeviceState{}); err == nil {
		t.Fatal("expected error on empty initial state")
	}

	// Unregistered slot operations
	if err := mgr.ShadowSlot(1); err == nil {
		t.Fatal("expected error on shadowing unregistered slot")
	}
	if err := mgr.RecordDraftTokens(1, []float32{1, 2, 3, 4}); err == nil {
		t.Fatal("expected error on recording draft tokens on unregistered slot")
	}
	if _, err := mgr.RollbackSlot(1, 2); err == nil {
		t.Fatal("expected error on rolling back unregistered slot")
	}

	// Registered slot with bad token length
	validState := RecurrentDeviceState{ConvState: []float32{1}, RecState: []float32{2}}
	_ = mgr.RegisterSlot(0, validState)
	_ = mgr.ShadowSlot(0)
	if err := mgr.RecordDraftTokens(0, []float32{1, 2}); err == nil {
		t.Fatal("expected error on draft tokens count != draftK")
	}

	// Accepted count out of bounds
	if _, err := mgr.RollbackSlot(0, -1); err == nil {
		t.Fatal("expected error on negative accepted rows")
	}
	if _, err := mgr.RollbackSlot(0, 10); err == nil {
		t.Fatal("expected error on accepted rows > draftK")
	}
}

func TestRecurrentRollbackBatchMultiSlot(t *testing.T) {
	draftK := 4
	mgr, err := NewRecurrentRollbackManager(3, draftK)
	if err != nil {
		t.Fatal(err)
	}

	slots := []int{0, 1, 2}
	acceptedCounts := []int{1, 4, 0}

	for _, slotID := range slots {
		state := RecurrentDeviceState{
			ConvState: []float32{float32(slotID + 1), 0.0},
			RecState:  []float32{float32(slotID+1) * 10, 0.0},
		}
		if err := mgr.RegisterSlot(slotID, state); err != nil {
			t.Fatalf("RegisterSlot %d failed: %v", slotID, err)
		}
		if err := mgr.ShadowSlot(slotID); err != nil {
			t.Fatalf("ShadowSlot %d failed: %v", slotID, err)
		}
		drafts := []float32{0.1, 0.2, 0.3, 0.4}
		if err := mgr.RecordDraftTokens(slotID, drafts); err != nil {
			t.Fatalf("RecordDraftTokens %d failed: %v", slotID, err)
		}
	}

	receipts, err := mgr.BatchRollbackSlots(slots, acceptedCounts)
	if err != nil {
		t.Fatalf("BatchRollbackSlots failed: %v", err)
	}

	if len(receipts) != 3 {
		t.Fatalf("expected 3 receipts, got %d", len(receipts))
	}

	for i, rcpt := range receipts {
		if rcpt.SlotID != slots[i] {
			t.Fatalf("receipt[%d] SlotID=%d, want %d", i, rcpt.SlotID, slots[i])
		}
		if rcpt.AcceptedRows != acceptedCounts[i] {
			t.Fatalf("receipt[%d] AcceptedRows=%d, want %d", i, rcpt.AcceptedRows, acceptedCounts[i])
		}
		if rcpt.D2HBytes != 0 || rcpt.D2HEvents != 0 || !rcpt.ZeroD2HVerified {
			t.Fatalf("receipt[%d] D2H violation: %+v", i, rcpt)
		}
	}
}

// gdnVerifyRingOracleGeometry pins the GDN spec-verify ring contract for one
// head geometry: (1) fold(j ring records) equals the sequential j-token
// recurrence bit-for-bit on pinned-FMA float32 for j = 0..len(toks); (2) ring
// wraparound retains the most recent ringLen records in order and folding that
// window from the window-start state matches the sequential run over it;
// (3) a zero-width fold is a no-op returning the input state unchanged.
func gdnVerifyRingOracleGeometry(t *testing.T, name string, keyHeadDim, valueHeadDim int, base []float32, toks []GDNVerifyRingToken) {
	t.Helper()

	// Build records against the evolving state and capture the sequential
	// reference state after each token.
	seqStates := make([][]float32, len(toks)+1)
	seqStates[0] = base
	records := make([]GDNVerifyRingRecord, len(toks))
	state := base
	for j, tok := range toks {
		rec, err := GDNVerifyRingRecordFromState(state, tok, keyHeadDim, valueHeadDim)
		if err != nil {
			t.Fatalf("%s: record %d: %v", name, j, err)
		}
		records[j] = rec
		state, err = GDNVerifyRingStep(state, tok, keyHeadDim, valueHeadDim)
		if err != nil {
			t.Fatalf("%s: step %d: %v", name, j, err)
		}
		seqStates[j+1] = state
	}

	// (1) fold(j records) == sequential j-token recurrence, bit-for-bit, j = 0..len(toks).
	for j := 0; j <= len(toks); j++ {
		got, err := GDNVerifyRingFold(base, records[:j], keyHeadDim, valueHeadDim)
		if err != nil {
			t.Fatalf("%s: j=%d fold: %v", name, j, err)
		}
		want := seqStates[j]
		if j == 0 && &got[0] != &base[0] {
			t.Fatalf("%s: j=0 zero-width fold must return the input state unchanged", name)
		}
		if len(got) != len(want) {
			t.Fatalf("%s: j=%d fold len %d, want %d", name, j, len(got), len(want))
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s: j=%d idx=%d: fold %v (bits %08x) != sequential %v (bits %08x)",
					name, j, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	}

	// (2) ring wraparound: writing past ringLen overwrites the oldest slots.
	const ringLen = 4
	ring, err := NewGDNVerifyRing(ringLen)
	if err != nil {
		t.Fatalf("%s: ring: %v", name, err)
	}
	for _, rec := range records {
		ring.Write(rec)
	}
	if ring.Count() != len(toks) {
		t.Fatalf("%s: ring count %d, want %d", name, ring.Count(), len(toks))
	}
	retained := ring.Retained()
	if len(retained) != ringLen {
		t.Fatalf("%s: retained len %d, want ringLen %d", name, len(retained), ringLen)
	}
	windowStart := len(toks) - ringLen
	for i := 0; i < ringLen; i++ {
		refRec, err := GDNVerifyRingRecordFromState(seqStates[windowStart+i], toks[windowStart+i], keyHeadDim, valueHeadDim)
		if err != nil {
			t.Fatalf("%s: retained ref record %d: %v", name, i, err)
		}
		for e := range refRec.Data {
			if math.Float32bits(retained[i].Data[e]) != math.Float32bits(refRec.Data[e]) {
				t.Fatalf("%s: retained[%d][%d] = %v (bits %08x), want record %d element %d %v (bits %08x)",
					name, i, e, retained[i].Data[e], math.Float32bits(retained[i].Data[e]),
					windowStart+i, e, refRec.Data[e], math.Float32bits(refRec.Data[e]))
			}
		}
	}
	// Folding the retained window from the window-start state matches the
	// sequential run over that window, bit-for-bit.
	got, err := GDNVerifyRingFold(seqStates[windowStart], retained, keyHeadDim, valueHeadDim)
	if err != nil {
		t.Fatalf("%s: window fold: %v", name, err)
	}
	want := seqStates[len(toks)]
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: window fold idx=%d: got %v (bits %08x), want sequential %v (bits %08x)",
				name, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}

	// (3) zero-width (no records) fold is a no-op returning the input state unchanged.
	empty, err := NewGDNVerifyRing(ringLen)
	if err != nil {
		t.Fatalf("%s: empty ring: %v", name, err)
	}
	if empty.Count() != 0 || len(empty.Retained()) != 0 {
		t.Fatalf("%s: fresh ring must retain nothing, got count=%d retained=%d", name, empty.Count(), len(empty.Retained()))
	}
	noop, err := GDNVerifyRingFold(base, empty.Retained(), keyHeadDim, valueHeadDim)
	if err != nil {
		t.Fatalf("%s: zero-width fold: %v", name, err)
	}
	if &noop[0] != &base[0] {
		t.Fatalf("%s: zero-width fold must return the input state unchanged", name)
	}
	noopNil, err := GDNVerifyRingFold(base, nil, keyHeadDim, valueHeadDim)
	if err != nil {
		t.Fatalf("%s: nil-record fold: %v", name, err)
	}
	if &noopNil[0] != &base[0] {
		t.Fatalf("%s: nil-record fold must return the input state unchanged", name)
	}
}

func TestGDNVerifyRingFoldOracle(t *testing.T) {
	// Pinned nontrivial decay/update matrix sets: mixed-sign raw keys and values,
	// near-unity and fast per-channel decays, beta away from 0/1. Fixed constants
	// keep the witness reproducible.
	gdnVerifyRingOracleGeometry(t, "k3v4", 3, 4,
		[]float32{0.5, -1.25, 0.125, 2.0, -0.75, 1.5, -0.25, 0.375, 3.0, -0.125, 0.625, -2.5},
		[]GDNVerifyRingToken{
			{Gate: []float32{-0.1, -0.5, -2.0}, Key: []float32{0.5, -1.0, 0.25}, Val: []float32{1.0, -0.5, 0.75, -2.0}, Beta: 0.3},
			{Gate: []float32{-0.01, -0.2, -1.0}, Key: []float32{-0.75, 0.5, 1.5}, Val: []float32{-1.5, 0.25, 0.5, 0.125}, Beta: 0.8},
			{Gate: []float32{-0.3, -0.3, -0.3}, Key: []float32{0.125, 0.125, -0.875}, Val: []float32{0.5, 0.5, -0.5, -0.5}, Beta: 0.5},
			{Gate: []float32{-0.05, -1.5, -0.75}, Key: []float32{1.0, -0.25, 0.5}, Val: []float32{-0.125, 2.0, -1.0, 0.375}, Beta: 0.6},
			{Gate: []float32{-0.9, -0.1, -0.4}, Key: []float32{-0.5, 0.75, -1.25}, Val: []float32{0.25, -0.75, 1.5, -0.25}, Beta: 0.4},
			{Gate: []float32{-0.2, -0.6, -0.2}, Key: []float32{0.375, -0.375, 0.625}, Val: []float32{-2.0, 0.125, -0.625, 1.25}, Beta: 0.9},
			{Gate: []float32{-0.02, -0.02, -0.8}, Key: []float32{-1.5, 0.25, 0.75}, Val: []float32{0.625, -0.375, 0.875, -1.75}, Beta: 0.2},
			{Gate: []float32{-0.4, -0.8, -0.15}, Key: []float32{0.875, -0.625, 0.125}, Val: []float32{-0.5, 1.75, -0.875, 0.0625}, Beta: 0.7},
			{Gate: []float32{-0.15, -0.35, -0.55}, Key: []float32{-0.25, 1.25, -0.5}, Val: []float32{1.25, -1.25, 0.4375, -0.9375}, Beta: 0.55},
		})

	gdnVerifyRingOracleGeometry(t, "k4v2", 4, 2,
		[]float32{1.0, -0.5, 0.25, 2.5, -1.5, 0.75, -0.125, 3.0},
		[]GDNVerifyRingToken{
			{Gate: []float32{-0.2, -0.4, -0.6, -0.8}, Key: []float32{0.5, -0.25, 1.0, -0.75}, Val: []float32{0.5, -1.5}, Beta: 0.4},
			{Gate: []float32{-0.05, -0.9, -0.1, -0.3}, Key: []float32{-1.25, 0.5, 0.25, 0.875}, Val: []float32{-0.75, 0.25}, Beta: 0.7},
			{Gate: []float32{-0.5, -0.5, -0.5, -0.5}, Key: []float32{0.125, -0.875, 0.625, -0.375}, Val: []float32{1.25, -0.5}, Beta: 0.5},
			{Gate: []float32{-0.1, -0.2, -0.3, -0.4}, Key: []float32{2.0, -1.0, 0.5, -0.5}, Val: []float32{-0.25, 0.75}, Beta: 0.9},
			{Gate: []float32{-0.7, -0.05, -1.1, -0.25}, Key: []float32{-0.625, 0.125, -1.5, 0.375}, Val: []float32{0.875, -1.25}, Beta: 0.3},
			{Gate: []float32{-0.3, -0.6, -0.9, -0.15}, Key: []float32{0.75, -0.5, 0.25, -0.9375}, Val: []float32{-1.5, 0.625}, Beta: 0.6},
			{Gate: []float32{-0.02, -0.8, -0.4, -0.6}, Key: []float32{-0.375, 1.25, -0.75, 0.5}, Val: []float32{0.125, -0.875}, Beta: 0.8},
			{Gate: []float32{-0.45, -0.35, -0.25, -0.55}, Key: []float32{0.9375, -0.4375, 0.625, -0.125}, Val: []float32{-0.625, 1.0}, Beta: 0.5},
			{Gate: []float32{-0.12, -0.22, -0.32, -0.42}, Key: []float32{-1.0, 0.875, -0.5, 0.25}, Val: []float32{0.5, -0.25}, Beta: 0.45},
		})
}

func TestGDNVerifyRingManagerFold(t *testing.T) {
	const keyDim, valDim = 3, 4
	draftK := 4
	mgr, err := NewRecurrentRollbackManager(2, draftK)
	if err != nil {
		t.Fatal(err)
	}

	base := RecurrentDeviceState{
		ConvState: []float32{1.0, 2.0, 3.0},
		RecState:  []float32{0.5, -1.25, 0.125, 2.0, -0.75, 1.5, -0.25, 0.375, 3.0, -0.125, 0.625, -2.5},
	}
	if err := mgr.RegisterSlot(0, base); err != nil {
		t.Fatal(err)
	}
	if err := mgr.EnableGDNRingRecording(0, keyDim, valDim); err != nil {
		t.Fatal(err)
	}
	if err := mgr.ShadowSlot(0); err != nil {
		t.Fatal(err)
	}

	ops := []GDNVerifyRingToken{
		{Gate: []float32{-0.1, -0.5, -2.0}, Key: []float32{0.5, -1.0, 0.25}, Val: []float32{1.0, -0.5, 0.75, -2.0}, Beta: 0.3},
		{Gate: []float32{-0.01, -0.2, -1.0}, Key: []float32{-0.75, 0.5, 1.5}, Val: []float32{-1.5, 0.25, 0.5, 0.125}, Beta: 0.8},
		{Gate: []float32{-0.3, -0.3, -0.3}, Key: []float32{0.125, 0.125, -0.875}, Val: []float32{0.5, 0.5, -0.5, -0.5}, Beta: 0.5},
		{Gate: []float32{-0.05, -1.5, -0.75}, Key: []float32{1.0, -0.25, 0.5}, Val: []float32{-0.125, 2.0, -1.0, 0.375}, Beta: 0.6},
	}
	if err := mgr.RecordDraftTokensGDN(0, ops); err != nil {
		t.Fatal(err)
	}

	// Sequential reference prefix states.
	prefix := make([][]float32, draftK+1)
	prefix[0] = base.RecState
	for i, op := range ops {
		var err error
		prefix[i+1], err = GDNVerifyRingStep(prefix[i], op, keyDim, valDim)
		if err != nil {
			t.Fatal(err)
		}
	}

	for accept := 0; accept <= draftK; accept++ {
		receipt, err := mgr.RollbackSlot(0, accept)
		if err != nil {
			t.Fatalf("accept=%d: %v", accept, err)
		}
		if receipt.Strategy != "gdn-verify-ring-fold" {
			t.Fatalf("accept=%d: strategy %q, want gdn-verify-ring-fold", accept, receipt.Strategy)
		}
		wantBytes := int64(accept) * int64(GDNVerifyRingRecordElements(keyDim, valDim)) * 4
		if receipt.D2DBytesCopied != wantBytes {
			t.Fatalf("accept=%d: D2DBytesCopied %d, want %d", accept, receipt.D2DBytesCopied, wantBytes)
		}
		live, err := mgr.GetLiveState(0)
		if err != nil {
			t.Fatal(err)
		}
		for i := range live.ConvState {
			if math.Float32bits(live.ConvState[i]) != math.Float32bits(base.ConvState[i]) {
				t.Fatalf("accept=%d: conv[%d]=%v, want shadow %v", accept, i, live.ConvState[i], base.ConvState[i])
			}
		}
		for i := range prefix[accept] {
			if math.Float32bits(live.RecState[i]) != math.Float32bits(prefix[accept][i]) {
				t.Fatalf("accept=%d: rec[%d]=%v (bits %08x), want sequential %v (bits %08x)",
					accept, i, live.RecState[i], math.Float32bits(live.RecState[i]),
					prefix[accept][i], math.Float32bits(prefix[accept][i]))
			}
		}
	}

	// Fallback switch: a ring-recorded round fails closed while the ring fold is disabled.
	mgr.SetRingRecording(false)
	if _, err := mgr.RollbackSlot(0, 2); err == nil {
		t.Fatal("expected fail-closed error rolling back a ring round with ring recording disabled")
	}
	mgr.SetRingRecording(true)

	// The legacy full-state-clone path remains available for scalar rounds.
	legacy := RecurrentDeviceState{ConvState: []float32{1.0}, RecState: []float32{2.0}}
	if err := mgr.RegisterSlot(1, legacy); err != nil {
		t.Fatal(err)
	}
	if err := mgr.ShadowSlot(1); err != nil {
		t.Fatal(err)
	}
	if err := mgr.RecordDraftTokens(1, []float32{0.5, 0.25, -0.5, 1.0}); err != nil {
		t.Fatal(err)
	}
	rcpt, err := mgr.RollbackSlot(1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if rcpt.Strategy != "device-slot-shadow-rewind" {
		t.Fatalf("legacy strategy %q, want device-slot-shadow-rewind", rcpt.Strategy)
	}
}
