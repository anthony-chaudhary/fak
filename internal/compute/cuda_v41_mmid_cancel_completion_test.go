//go:build cuda && cgo && fak_cuda_v41_routed_mmid

package compute

import (
	"errors"
	"testing"
	"unsafe"
)

// fak-test:runtime fast est=1ms lane=default
// Actual adapter/core completion disposition with substituted terminal ops;
// no physical CUDA completion or output numerical validity is established.
func TestCUDAV41MMIDCancelCompletionDisposition(t *testing.T) {
	savedOps, savedIssued := cudaV41MMIDBatchNative, cudaV41MMIDIssuedStatuses
	t.Cleanup(func() { cudaV41MMIDBatchNative, cudaV41MMIDIssuedStatuses = savedOps, savedIssued })
	for _, mode := range []string{"completed", "unknown", "cleanup-panic"} {
		t.Run(mode, func(t *testing.T) {
			cudaV41MMIDIssuedStatuses = make(map[*cudaBuf]cudaV41MMIDIssued)
			c := &cudaBackend{}
			q := &cudaV41DeferredQualification{owner: c}
			reads, holds := 0, 0
			cudaV41MMIDBatchNative = cudaV41MMIDBatchOperations{
				admit: func(*cudaBackend) error { t.Fatal("cancel retried admission"); return nil },
				read: func(host []byte, ptrs []unsafe.Pointer, sizes, offsets []int) (int, bool) {
					reads++
					if len(host)+len(ptrs)+len(sizes)+len(offsets) != 0 {
						t.Fatal("cancel copied data")
					}
					if mode == "cleanup-panic" {
						panic("terminal drain")
					}
					if mode == "unknown" {
						return 40029, false
					}
					return 0, true
				},
				hold:   func() { holds++ },
				retire: func(unsafe.Pointer) int { t.Fatal("no-owner drain retired storage"); return 0 },
			}
			primary := errors.New("failure before first status")
			completed, err := q.V41CancelDeferred(nil, primary)
			if err != primary || completed != (mode == "completed") || reads != 1 || c.fenceGen != 0 {
				t.Fatalf("completion=%v err=%v reads=%d generation=%d", completed, err, reads, c.fenceGen)
			}
			if mode == "completed" && holds != 0 {
				t.Fatal("completed drain was retained")
			}
			if mode != "completed" && holds != 1 {
				t.Fatal("unknown drain was not retained")
			}
			// Compatibility: old private Cancel acquires no new completion
			// grant for no issued owners, including after an earlier consume.
			if err := q.cancel(nil, primary); err != primary || reads != 1 {
				t.Fatal("legacy cancellation changed")
			}
		})
	}
}
