//go:build cuda && cgo && fak_cuda_v41_routed_mmid

package compute

import (
	"encoding/binary"
	"math"
	"reflect"
	"testing"
	"unsafe"
)

// fak-test:runtime fast est=5ms lane=default
// Actual reader validation/flattening with substituted terminal transport;
// this is not physical transfer, kernel or lifetime qualification.
func TestCUDAV41MMIDPanelReadDomain(t *testing.T) {
	savedOps, savedIssued := cudaV41MMIDBatchNative, cudaV41MMIDIssuedStatuses
	t.Cleanup(func() { cudaV41MMIDBatchNative, cudaV41MMIDIssuedStatuses = savedOps, savedIssued })
	makeRow := func(c *cudaBackend, shape []int, bytes int) Tensor {
		storage := make([]byte, bytes)
		return makeTensor(c, F32, RowMajor, shape, nil, &cudaBuf{ptr: unsafe.Pointer(&storage[0]), n: bytes})
	}
	for _, tc := range []struct {
		name  string
		shape []int
		bytes int
		want  bool
	}{
		{"vector", []int{4}, 16, true},
		{"panel", []int{2, 3}, 24, true},
		{"zero", []int{2, 0}, 16, false},
		{"negative", []int{-1, 4}, 16, false},
		{"rank3", []int{1, 2, 3}, 24, false},
		{"product-overflow", []int{math.MaxInt, 2}, 16, false},
		{"bounded-product", []int{math.MaxInt32, 2}, 16, false},
		{"short-storage", []int{2, 3}, 20, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cudaV41MMIDIssuedStatuses = make(map[*cudaBuf]cudaV41MMIDIssued)
			c := &cudaBackend{}
			row := makeRow(c, tc.shape, tc.bytes)
			ptr := row.buf.(*cudaBuf).ptr
			reads := 0
			cudaV41MMIDBatchNative = cudaV41MMIDBatchOperations{
				admit: func(*cudaBackend) error { return nil },
				read: func(host []byte, ptrs []unsafe.Pointer, sizes, offsets []int) (int, bool) {
					reads++
					if !tc.want || len(ptrs) != 2 || sizes[0] != tc.bytes || sizes[1] != 4 {
						t.Fatal("wrong row-only descriptor domain")
					}
					for i, size := range sizes {
						for j := 0; j < size/4; j++ {
							binary.LittleEndian.PutUint32(host[offsets[i]+j*4:], math.Float32bits(float32(10*i+j)))
						}
					}
					return 0, true
				},
				hold:   func() { t.Fatal("known row-only read was retained") },
				retire: func(unsafe.Pointer) int { t.Fatal("borrowed row retired"); return 0 },
			}
			verdicts, values, err := c.cudaV41MMIDReadBatch(nil, []Tensor{row, makeRow(c, []int{1}, 4)})
			if tc.want {
				want := make([]float32, tc.bytes/4)
				for i := range want {
					want[i] = float32(i)
				}
				if err != nil || len(verdicts) != 0 || len(values) != 2 || !reflect.DeepEqual(values[0], want) || !reflect.DeepEqual(values[1], []float32{10}) || reads != 1 || c.fenceGen != 1 {
					t.Fatalf("flat per-tensor row-only read: %v %v %v", verdicts, values, err)
				}
			} else if err == nil || verdicts != nil || values != nil || reads != 0 || c.fenceGen != 0 {
				t.Fatal("invalid panel reached copy or exposed result")
			}
			if row.buf.(*cudaBuf).ptr != ptr {
				t.Fatal("borrowed row ownership changed")
			}
		})
	}
	// A rank-2 view of issued status storage is still a forbidden status alias.
	cudaV41MMIDIssuedStatuses = make(map[*cudaBuf]cudaV41MMIDIssued)
	c := &cudaBackend{}
	record := makeRow(c, []int{4}, 16)
	st := V41DeferredStatus{Kind: V41DeferredRoutedSwiGLU, Count: 1, Records: 1, Record: record}
	cudaMu.Lock()
	c.cudaV41MMIDIssue(st)
	cudaMu.Unlock()
	panel := record
	panel.Shape = []int{2, 2}
	fences := 0
	cudaV41MMIDBatchNative = cudaV41MMIDBatchOperations{
		admit: func(*cudaBackend) error { return nil },
		read: func(host []byte, ptrs []unsafe.Pointer, sizes, offsets []int) (int, bool) {
			fences++
			if len(host)+len(ptrs)+len(sizes)+len(offsets) != 0 {
				t.Fatal("alias reached copy")
			}
			return 0, true
		},
		hold:   func() { t.Fatal("known alias completion held") },
		retire: func(unsafe.Pointer) int { t.Fatal("aliased storage returned to pool"); return 0 },
	}
	verdicts, values, err := c.cudaV41MMIDReadBatch([]V41DeferredStatus{st}, []Tensor{panel})
	if err == nil || verdicts != nil || values != nil || fences != 1 || len(cudaV41MMIDIssuedStatuses) != 0 || record.buf.(*cudaBuf).ptr != nil {
		t.Fatal("panel alias did not retain and uniquely consume issued ownership")
	}
}
