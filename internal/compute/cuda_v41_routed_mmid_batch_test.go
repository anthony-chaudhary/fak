//go:build cuda && cgo && fak_cuda_v41_routed_mmid

package compute

import (
	"encoding/binary"
	"errors"
	"math"
	"strings"
	"testing"
	"unsafe"
)

// fak-test:runtime fast est=5ms lane=default
// This is a software lifetime witness of the actual Go reader. Native transport,
// pinned-host retention and numerical CUDA kernels require separate witnesses.
func TestCUDAV41MMIDBatchLifecycle(t *testing.T) {
	savedOps, savedIssued := cudaV41MMIDBatchNative, cudaV41MMIDIssuedStatuses
	t.Cleanup(func() {
		cudaV41MMIDBatchNative, cudaV41MMIDIssuedStatuses = savedOps, savedIssued
	})
	newRow := func(c *cudaBackend, n int) Tensor {
		storage := make([]byte, n*4)
		return makeTensor(c, F32, RowMajor, []int{n}, nil, &cudaBuf{ptr: unsafe.Pointer(&storage[0]), n: len(storage)})
	}
	issue := func(c *cudaBackend, kind V41DeferredStatusKind) V41DeferredStatus {
		st := V41DeferredStatus{Kind: kind, Count: 3, Records: 2, Record: newRow(c, 8)}
		if kind == V41DeferredBF16Publish {
			st.Site = V41BF16PublishRoutedDown
		}
		cudaMu.Lock()
		c.cudaV41MMIDIssue(st)
		cudaMu.Unlock()
		return st
	}
	for _, mode := range []string{"success", "late-invalid", "transfer-failed", "drain-failed", "duplicate-row-alias", "retirement-panic", "stale-address"} {
		t.Run(mode, func(t *testing.T) {
			cudaV41MMIDIssuedStatuses = make(map[*cudaBuf]cudaV41MMIDIssued)
			c := &cudaBackend{}
			first, second := issue(c, V41DeferredRoutedSwiGLU), issue(c, V41DeferredBF16Publish)
			a, b := first.Record.buf.(*cudaBuf), second.Record.buf.(*cudaBuf)
			ap, bp := a.ptr, b.ptr
			row := newRow(c, 3)
			rowBuf, rowPtr := row.buf.(*cudaBuf), row.buf.(*cudaBuf).ptr
			reads, holds, retireCalls := 0, 0, 0
			released := make(map[unsafe.Pointer]int)
			cudaV41MMIDBatchNative = cudaV41MMIDBatchOperations{
				admit: func(*cudaBackend) error { return nil },
				hold:  func() { holds++ },
				retire: func(ptr unsafe.Pointer) int {
					retireCalls++
					if ptr == rowPtr {
						t.Fatal("caller row entered status retirement")
					}
					if mode == "retirement-panic" && retireCalls == 1 {
						panic("retirement injected")
					}
					released[ptr]++
					return 0
				},
				read: func(host []byte, ptrs []unsafe.Pointer, sizes, offsets []int) (int, bool) {
					reads++
					if mode == "late-invalid" || mode == "duplicate-row-alias" {
						if len(host) != 0 || len(ptrs) != 0 {
							t.Fatal("invalid batch submitted copies instead of completion only")
						}
						return 0, true
					}
					if len(ptrs) != 3 || ptrs[0] != ap || ptrs[1] != bp || ptrs[2] != rowPtr ||
						len(sizes) != 3 || sizes[0] != 32 || sizes[1] != 32 || sizes[2] != 12 ||
						offsets[0] != 0 || offsets[1] != 32 || offsets[2] != 64 || len(host) != sizes[0]+sizes[1]+sizes[2] {
						t.Fatal("descriptor/record/row ordering or bounded image layout changed")
					}
					if mode == "transfer-failed" || mode == "drain-failed" {
						return 20017, mode == "transfer-failed"
					}
					for r := 0; r < 2; r++ {
						at := r * 16
						binary.LittleEndian.PutUint32(host[at:], v41RoutedSwiGLUStatusMagic)
						binary.LittleEndian.PutUint32(host[at+4:], 3)
						binary.LittleEndian.PutUint32(host[at+8:], math.MaxUint32)
						binary.LittleEndian.PutUint32(host[at+12:], math.MaxUint32)
					}
					for r := 2; r < 4; r++ {
						at := r * 16
						binary.LittleEndian.PutUint32(host[at:], v41BF16CheckedStatusMagic)
						binary.LittleEndian.PutUint32(host[at+4:], V41BF16PublishRoutedDown)
						binary.LittleEndian.PutUint32(host[at+8:], math.MaxUint32)
						binary.LittleEndian.PutUint32(host[at+12:], 3)
					}
					// Typed routed and BF16 causes keep descriptor/pick order;
					// only the BF16 cause gains backend attribution.
					binary.LittleEndian.PutUint32(host[24:], 2)
					binary.LittleEndian.PutUint32(host[40:], 1)
					for i, value := range []float32{1, -2, 3} {
						binary.LittleEndian.PutUint32(host[64+i*4:], math.Float32bits(value))
					}
					return 0, true
				},
			}
			statuses := []V41DeferredStatus{first, second}
			rows := []Tensor{row}
			switch mode {
			case "late-invalid":
				// The first error is deliberately in the second descriptor;
				// both issued owners must already be in the retirement set.
				statuses[1].Count++
			case "duplicate-row-alias":
				statuses = []V41DeferredStatus{first, first, second}
				rows = []Tensor{first.Record}
			case "stale-address":
				// Free clears this opaque handle. A different handle can reuse
				// its address without inheriting its status issuance authority.
				a.ptr = nil
				reused := *a
				reused.ptr = ap
				statuses = []V41DeferredStatus{first}
				rows = nil
				verdicts, values, err := c.cudaV41MMIDReadBatch(statuses, rows)
				if err == nil || verdicts != nil || values != nil || reads != 0 || retireCalls != 0 || reused.ptr != ap {
					t.Fatal("stale status consumed recycled storage")
				}
				forged := first
				forged.Record.buf = &reused
				if _, _, err := c.cudaV41MMIDReadBatch([]V41DeferredStatus{forged}, nil); err == nil || reads != 0 || retireCalls != 0 {
					t.Fatal("unissued replacement handle inherited status authority")
				}
				return
			}
			verdicts, values, err := c.cudaV41MMIDReadBatch(statuses, rows)
			if reads != 1 || a.ptr != nil || b.ptr != nil || len(cudaV41MMIDIssuedStatuses) != 0 || rowBuf.ptr != rowPtr {
				t.Fatalf("lifecycle: reads=%d records=%v/%v issued=%d row changed=%t", reads, a.ptr, b.ptr, len(cudaV41MMIDIssuedStatuses), rowBuf.ptr != rowPtr)
			}
			if mode == "success" {
				var fault *V41RoutedSwiGLUFault
				var bf16 *V41BF16CheckedArithmeticError
				var attributed *BackendError
				if err != nil || len(verdicts) != 4 || verdicts[0] != nil ||
					!errors.As(verdicts[1], &fault) || fault.Index != 2 ||
					!errors.As(verdicts[2], &bf16) || bf16.Index != 1 || bf16.Site != V41BF16PublishRoutedDown ||
					!errors.As(verdicts[2], &attributed) || attributed.Backend != "cuda" || attributed.Site != "V41BF16Publish" || verdicts[3] != nil ||
					len(values) != 1 || len(values[0]) != 3 || values[0][0] != 1 || values[0][1] != -2 || values[0][2] != 3 || c.fenceGen != 1 {
					t.Fatalf("ordered successful result: %v %v %v", verdicts, values, err)
				}
			} else if err == nil || verdicts != nil || values != nil || c.fenceGen != 0 {
				t.Fatalf("failed batch exposed result/readiness: %v %v %v", verdicts, values, err)
			}
			if mode == "drain-failed" {
				if holds != 1 || retireCalls != 0 || !strings.Contains(err.Error(), "20017") {
					t.Fatal("unknown completion reused records or lost first failure")
				}
			} else if mode == "duplicate-row-alias" {
				if released[ap] != 0 || released[bp] != 1 || retireCalls != 1 {
					t.Fatal("duplicate/row alias was pooled or later owner stranded")
				}
			} else if mode == "retirement-panic" {
				if retireCalls != 2 || released[bp] != 1 {
					t.Fatal("retirement panic stranded later owner")
				}
			} else if released[ap] != 1 || released[bp] != 1 || holds != 0 {
				t.Fatal("known-complete owners were not uniquely retired")
			}
		})
	}
}
