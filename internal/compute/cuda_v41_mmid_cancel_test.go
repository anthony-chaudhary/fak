//go:build cuda && cgo && fak_cuda_v41_routed_mmid

package compute

import (
	"errors"
	"testing"
	"unsafe"
)

// fak-test:runtime fast est=1ms lane=default
// One early-abort lifecycle case of the actual ownership core. The substituted
// native operations do not prove CUDA completion, quarantine or pool behavior.
func TestCUDAV41MMIDCancelAfterEarlyAbort(t *testing.T) {
	savedOps, savedIssued := cudaV41MMIDBatchNative, cudaV41MMIDIssuedStatuses
	t.Cleanup(func() {
		cudaV41MMIDBatchNative, cudaV41MMIDIssuedStatuses = savedOps, savedIssued
	})
	cudaV41MMIDIssuedStatuses = make(map[*cudaBuf]cudaV41MMIDIssued)
	c, foreign := &cudaBackend{}, &cudaBackend{}
	makeStatus := func(owner *cudaBackend, issue bool) V41DeferredStatus {
		storage := make([]byte, 16)
		buf := &cudaBuf{ptr: unsafe.Pointer(&storage[0]), n: len(storage)}
		st := V41DeferredStatus{Kind: V41DeferredRoutedSwiGLU, Count: 3, Records: 1,
			Record: makeTensor(owner, F32, RowMajor, []int{4}, nil, buf)}
		if issue {
			cudaMu.Lock()
			owner.cudaV41MMIDIssue(st)
			cudaMu.Unlock()
		}
		return st
	}
	first, second := makeStatus(c, true), makeStatus(c, true)
	other, unissued := makeStatus(foreign, true), makeStatus(c, false)
	otherBuf, unknownBuf := other.Record.buf.(*cudaBuf), unissued.Record.buf.(*cudaBuf)
	otherPtr, unknownPtr := otherBuf.ptr, unknownBuf.ptr
	// Cancellation owns the issued storage even if a later descriptor's public
	// metadata is invalid. It must still reach the second accepted owner.
	second.Count = -1
	statuses := []V41DeferredStatus{first, other, first, unissued, second}
	reads, holds, retirements := 0, 0, 0
	cudaV41MMIDBatchNative = cudaV41MMIDBatchOperations{
		admit: func(*cudaBackend) error {
			t.Fatal("cancellation retried poisoned producer admission")
			return nil
		},
		read: func(host []byte, ptrs []unsafe.Pointer, sizes, offsets []int) (int, bool) {
			reads++
			if len(host) != 0 || len(ptrs) != 0 || len(sizes) != 0 || len(offsets) != 0 {
				t.Fatal("cancellation submitted copies or decoded unknown status")
			}
			return 40029, false // original error must survive an uncertain drain
		},
		hold: func() { holds++ },
		retire: func(unsafe.Pointer) int {
			retirements++
			t.Fatal("unknown completion returned storage to the pool")
			return 0
		},
	}
	primary := errors.New("later ordinary producer failed before enqueue")
	if err := c.cudaV41MMIDCancel(statuses, primary); err != primary {
		t.Fatalf("lost original failure: %v", err)
	}
	if reads != 1 || holds != 1 || retirements != 0 || c.fenceGen != 0 ||
		first.Record.buf.(*cudaBuf).ptr != nil || second.Record.buf.(*cudaBuf).ptr != nil ||
		otherBuf.ptr != otherPtr || unknownBuf.ptr != unknownPtr || len(cudaV41MMIDIssuedStatuses) != 1 {
		t.Fatal("abort did not uniquely consume own statuses while preserving foreign/unissued storage")
	}
	if _, ok := cudaV41MMIDIssuedStatuses[otherBuf]; !ok {
		t.Fatal("foreign issuance authority was consumed")
	}
	if err := c.cudaV41MMIDCancel(statuses, primary); err != primary || reads != 1 || holds != 1 || retirements != 0 {
		t.Fatal("repeated abort acquired a second ownership/completion grant")
	}
}
