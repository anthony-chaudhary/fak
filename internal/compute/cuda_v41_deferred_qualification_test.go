//go:build cuda && cgo && fak_cuda_v41_routed_mmid

package compute

import (
	"errors"
	"testing"
	"unsafe"
)

// fak-test:runtime fast est=1ms lane=default
func TestCUDAV41DeferredQualificationIsolationAndCancel(t *testing.T) {
	c := &cudaBackend{}
	q := &cudaV41DeferredQualification{owner: c}
	if _, ok := any(c).(V41DeferredStatusBackend); ok {
		t.Fatal("ordinary CUDA acquired deferred methods through qualification work")
	}
	if _, ok := any(q).(Backend); ok {
		t.Fatal("qualification adapter can escape into model Backend selection")
	}
	var absent *cudaV41DeferredQualification
	if absent.SupportsV41DeferredStatus(1) || q.SupportsV41DeferredStatus(0) {
		t.Fatal("invalid qualification domain admitted")
	}
	if out, st, err := absent.V41RoutedSwiGLUDeferred(Tensor{}, Tensor{}, 0, 1); err == nil || out.buf != nil || st.Record.buf != nil {
		t.Fatal("nil owner exposed SwiGLU output")
	}
	if out, st, err := absent.V41BF16PublishDeferred(Tensor{}, 0); err == nil || out.buf != nil || st.Record.buf != nil {
		t.Fatal("nil owner exposed publication output")
	}
	if verdicts, rows, err := absent.V41ReadDeferred(nil, nil); err == nil || verdicts != nil || rows != nil {
		t.Fatal("nil owner exposed readback")
	}
	// Exercise actual adapter -> recovered cancellation core. Substituting the
	// terminal operations checks delegation/ownership, not device completion.
	savedOps, savedIssued := cudaV41MMIDBatchNative, cudaV41MMIDIssuedStatuses
	t.Cleanup(func() { cudaV41MMIDBatchNative, cudaV41MMIDIssuedStatuses = savedOps, savedIssued })
	cudaV41MMIDIssuedStatuses = make(map[*cudaBuf]cudaV41MMIDIssued)
	storage := make([]byte, 16)
	buf := &cudaBuf{ptr: unsafe.Pointer(&storage[0]), n: 16}
	st := V41DeferredStatus{Kind: V41DeferredRoutedSwiGLU, Count: 1, Records: 1,
		Record: makeTensor(c, F32, RowMajor, []int{4}, nil, buf)}
	cudaMu.Lock()
	c.cudaV41MMIDIssue(st)
	cudaMu.Unlock()
	fences, holds := 0, 0
	cudaV41MMIDBatchNative = cudaV41MMIDBatchOperations{
		admit: func(*cudaBackend) error { t.Fatal("cancel retried admission"); return nil },
		read: func(host []byte, ptrs []unsafe.Pointer, sizes, offsets []int) (int, bool) {
			fences++
			if len(host)+len(ptrs)+len(sizes)+len(offsets) != 0 {
				t.Fatal("cancel issued readback")
			}
			return 40029, false
		},
		hold:   func() { holds++ },
		retire: func(unsafe.Pointer) int { t.Fatal("uncertain storage retired"); return 0 },
	}
	primary := errors.New("primary producer failure")
	for i := 0; i < 2; i++ {
		if err := q.cancel([]V41DeferredStatus{st}, primary); err != primary {
			t.Fatalf("primary error replaced: %v", err)
		}
	}
	if fences != 1 || holds != 1 || len(cudaV41MMIDIssuedStatuses) != 0 || buf.ptr != nil {
		t.Fatal("adapter did not preserve exactly-once uncertain cancellation")
	}
}
