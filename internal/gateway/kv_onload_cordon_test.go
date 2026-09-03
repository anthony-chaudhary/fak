package gateway

import (
	"errors"
	"testing"
	"time"
)

func TestKVCordonDecodeWitness(t *testing.T) {
	// First witness requirements (#9915):
	// 1. Overlap H2D prefix restoration while preventing decode/read during in-flight onload.
	// 2. Cancel request during delayed in-flight onload.
	// 3. Assert blocks are NEITHER decoded from NOR recycled until completion.
	// 4. Once transfer completes, blocks can be safely released and recycled.

	cordonMgr := NewKVCordonManager()
	transferMgr := NewKVTransferManager()

	reqID := "session-delayed-onload"
	cordonedBlocks := []int{201, 202}

	// Start asynchronous onload
	handle, err := transferMgr.StartTransfer(KVTransferOnload, cordonedBlocks)
	if err != nil {
		t.Fatalf("StartTransfer failed: %v", err)
	}

	// Register decode cordon
	_, err = cordonMgr.CordonOnload(reqID, cordonedBlocks, handle)
	if err != nil {
		t.Fatalf("CordonOnload failed: %v", err)
	}

	// 1. Verify decode is refused while onload is in flight
	canDecode, err := cordonMgr.CanDecode(reqID)
	if canDecode || !errors.Is(err, ErrDecodeCordoned) {
		t.Fatalf("expected decode to be refused while onload in-flight, got canDecode=%t, err=%v", canDecode, err)
	}

	// 2. Cancel during delayed onload
	if err := cordonMgr.CancelRequest(reqID); err != nil {
		t.Fatalf("CancelRequest failed: %v", err)
	}

	// 3. Assert blocks are NEITHER decoded from NOR recycled until completion:
	// (a) Decode must be refused
	canDecodeAfterCancel, _ := cordonMgr.CanDecode(reqID)
	if canDecodeAfterCancel {
		t.Fatal("expected decode refused after cancellation")
	}

	// (b) Recycling must be strictly refused while transfer is in flight
	for _, b := range cordonedBlocks {
		canRecycle, err := cordonMgr.CanRecycleBlock(b)
		if canRecycle || !errors.Is(err, ErrRecycleCordoned) {
			t.Fatalf("block %d prematurely recycled while transfer in-flight: canRecycle=%t, err=%v", b, canRecycle, err)
		}
	}

	// Simulate async completion of the H2D transfer
	time.Sleep(10 * time.Millisecond)
	_ = transferMgr.FinishTransfer(handle, nil)

	// 4. Now that transfer is completed, verify blocks can be safely recycled
	for _, b := range cordonedBlocks {
		canRecycle, err := cordonMgr.CanRecycleBlock(b)
		if !canRecycle || err != nil {
			t.Fatalf("block %d should be safe to recycle after transfer complete: %v", b, err)
		}
	}

	// Release the cordon cleanly
	if err := cordonMgr.ReleaseOnload(reqID); err != nil {
		t.Fatalf("ReleaseOnload failed: %v", err)
	}
}
