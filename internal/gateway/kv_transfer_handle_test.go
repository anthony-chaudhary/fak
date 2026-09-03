package gateway

import (
	"testing"
	"time"
)

func TestKVTransferHandleWitness(t *testing.T) {
	// First witness requirements (#9916):
	// 1. Starts an async transfer (e.g. offload).
	// 2. Proves eviction strictly refuses every touched block while transfer is in flight.
	// 3. Completes the transfer and proves eviction is permitted once handle completes.
	// 4. Handle reports exact direction and touched device blocks.

	mgr := NewKVTransferManager()

	touchedBlocks := []int{101, 102, 103}
	untouchedBlock := 999

	// 1. Start an asynchronous offload
	handle, err := mgr.StartTransfer(KVTransferOffload, touchedBlocks)
	if err != nil {
		t.Fatalf("StartTransfer failed: %v", err)
	}

	// 4. Verify handle metadata
	if handle.Direction != KVTransferOffload {
		t.Fatalf("expected direction %s, got %s", KVTransferOffload, handle.Direction)
	}
	if len(handle.DeviceBlocks) != 3 {
		t.Fatalf("expected 3 touched blocks, got %d", len(handle.DeviceBlocks))
	}

	// 2. Verify reclamation/eviction refuses every touched block while in-flight
	for _, b := range touchedBlocks {
		canReclaim, reason := mgr.CanReclaimBlock(b)
		if canReclaim {
			t.Fatalf("expected block %d reclamation to be refused while in-flight", b)
		}
		if reason == "" {
			t.Fatalf("expected refusal reason for in-flight block %d", b)
		}
	}

	// Untouched block can be reclaimed freely
	if canReclaim, _ := mgr.CanReclaimBlock(untouchedBlock); !canReclaim {
		t.Fatalf("expected untouched block %d to be reclaimable", untouchedBlock)
	}

	// Simulate async work
	go func() {
		time.Sleep(10 * time.Millisecond)
		_ = mgr.FinishTransfer(handle, nil)
	}()

	// Wait for completion
	<-handle.Done()

	if !handle.Completed {
		t.Fatal("expected handle to be completed")
	}

	// 3. Verify reclamation is now permitted on all previously touched blocks
	for _, b := range touchedBlocks {
		canReclaim, reason := mgr.CanReclaimBlock(b)
		if !canReclaim {
			t.Fatalf("expected block %d to be reclaimable after completion, refused: %s", b, reason)
		}
	}
}
