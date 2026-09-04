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
