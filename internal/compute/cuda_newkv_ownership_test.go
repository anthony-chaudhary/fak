//go:build cuda && cgo

package compute

import (
	"fmt"
	"testing"
	"unsafe"
)

// Synthetic allocation/release callbacks only; no CUDA operation is executed.
// Runtime estimate only, not measured.
// fak-test:runtime fast est=1ms lane=default
func TestCUDANewKVGraphAllocationOwnership(t *testing.T) {
	for failAt := 0; failAt <= 6; failAt++ {
		t.Run(fmt.Sprintf("failAt%d", failAt), func(t *testing.T) {
			k := &cudaKV{
				cfg:    KVConfig{NumLayers: 2, NumKVHeads: 1, HeadDim: 2},
				maxPos: 7,
				K:      make([]dslice, 2), Kraw: make([]dslice, 2), V: make([]dslice, 2),
			}
			const capF = 14
			primary := &DeviceAllocError{Bytes: capF * 4, Site: "synthetic preallocation", Class: MemoryKVCache}
			var backing [6]byte
			var allocated []unsafe.Pointer
			released := map[unsafe.Pointer]int{}
			attempts := 0
			var got any
			func() {
				defer func() { got = recover() }()
				k.preallocateGraphWithOperations(capF, func(bytes int, site string) unsafe.Pointer {
					attempts++
					prefix := []string{"kv-key-prealloc layer ", "kv-pre-rope-key-prealloc layer ", "kv-value-prealloc layer "}[(attempts-1)%3]
					if bytes != capF*4 || site != prefix+itoaC((attempts-1)/3) {
						t.Errorf("allocation contract: bytes=%d site=%q", bytes, site)
					}
					if attempts == failAt {
						panic(primary)
					}
					ptr := unsafe.Pointer(&backing[attempts-1])
					allocated = append(allocated, ptr)
					return ptr
				}, func(ptr unsafe.Pointer) {
					released[ptr]++
					// Cleanup must attempt all owned rows and preserve primary identity.
					panic("secondary cleanup panic")
				})
			}()
			if failAt == 0 {
				if got != nil || attempts != 6 || len(released) != 0 {
					t.Fatalf("successful ownership: panic=%v allocations=%d releases=%d", got, attempts, len(released))
				}
			} else {
				if got != primary || attempts != failAt || len(released) != failAt-1 {
					t.Fatalf("failed ownership: panic=%v allocations=%d releases=%d", got, attempts, len(released))
				}
				for _, ptr := range allocated {
					if released[ptr] != 1 {
						t.Errorf("allocated pointer released %d times", released[ptr])
					}
				}
			}
			for layer := range k.K {
				for column, row := range []dslice{k.K[layer], k.Kraw[layer], k.V[layer]} {
					want := dslice{}
					if failAt == 0 {
						want = dslice{ptr: allocated[3*layer+column], cap: capF}
					}
					if row != want {
						t.Errorf("layer %d column %d metadata = %+v, want %+v", layer, column, row, want)
					}
				}
			}
			if k.maxPos != 7 || len(k.pos) != 0 {
				t.Fatal("constructor capacity or empty token positions changed")
			}
		})
	}
}
