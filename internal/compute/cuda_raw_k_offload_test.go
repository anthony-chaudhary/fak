//go:build cuda && cgo

package compute

import (
	"reflect"
	"testing"
	"unsafe"
)

// Exercises the real upload owner through injected allocation/copy operations.
// Host sentinel pointers never reach native CUDA allocation, copy, or Free.
// The old host ring regression is a separate ownership-contract witness.
// fak-test:runtime fast est=10ms lane=default
func TestCUDARawKOffloadIndependentOwnership(t *testing.T) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	saved := uploadCache
	uploadCache = map[ucKey]Tensor{}
	defer func() { uploadCache = saved }()

	for _, tc := range []struct {
		dt       Dtype
		bytes    int
		bits     uint8
		makeHost func(Backend, []int, []byte) Tensor
	}{
		{Q2_K, 84, 2, NewQ2K}, {Q4_K, 144, 4, NewQ4K},
		{Q5_K, 176, 5, NewQ5K}, {Q6_K, 210, 6, NewQ6K},
	} {
		t.Run(tc.dt.String(), func(t *testing.T) {
			c := &cudaBackend{budgetBytes: 1000, dlUsed: 77, managedN: 3,
				immutableWeightUploadCalls: 5, immutableWeightUploadTransferBytes: 9, immutableWeightUploadResidentBytes: 11}
			raw := make([]byte, tc.bytes)
			for i := range raw {
				raw[i] = byte(197 + i)
			}
			host := tc.makeHost(Default(), []int{1, 256}, raw)
			hb := host.buf.(HostBuffer)
			key := ucKey{uintptr(unsafe.Pointer(&hb.I8()[0])), tc.dt, host.Layout}
			cachedStorage := make([]byte, tc.bytes)
			cachedBuf := &cudaBuf{ptr: unsafe.Pointer(&cachedStorage[0]), n: tc.bytes, class: MemoryWeights}
			cached := makeTensor(c, tc.dt, RowMajor, []int{1, 256}, host.Quant, cachedBuf)
			uploadCache[key] = cached
			defer delete(uploadCache, key)
			weightCalls, allocs, copies, releases := 0, 0, 0, 0
			memory := map[unsafe.Pointer][]byte{}
			var owners []*cudaBuf
			weight := func(src Tensor, source HostBuffer) Tensor {
				weightCalls++
				// The unchanged actual weight method is safe here only because
				// this exact hit returns before every native allocator/copy.
				hit, ok := uploadCache[key]
				if !ok || hit.Numel() != src.Numel() {
					t.Fatal("missing safe cache-hit precondition")
				}
				return c.uploadRawKQuant(src, source)
			}
			allocate := func(n int) *cudaBuf {
				allocs++
				if n != tc.bytes {
					t.Fatalf("allocation=%d want=%d", n, tc.bytes)
				}
				b := make([]byte, n)
				p := unsafe.Pointer(&b[0])
				memory[p] = b
				owner := &cudaBuf{ptr: p, n: n, class: MemoryOffload, managed: tc.dt == Q6_K}
				owners = append(owners, owner)
				return owner
			}
			copyPayload := func(p unsafe.Pointer, src []int8) {
				copies++
				dst, ok := memory[p]
				if !ok {
					t.Fatal("copy destination not owned by fixture")
				}
				for i, v := range src {
					dst[i] = byte(v)
				}
			}
			release := func(p unsafe.Pointer) {
				releases++
				b, ok := memory[p]
				if !ok {
					t.Fatal("unowned or duplicate release")
				}
				clear(b)
				delete(memory, p)
			}
			call := func(class MemoryClass) Tensor {
				return c.uploadRawKQuantClassWithOperations(host, hb, class, weight, allocate, copyPayload, release)
			}
			if call(MemoryWeights).buf != cachedBuf || call(MemoryWeights).buf != cachedBuf || allocs != 0 || copies != 0 {
				t.Fatal("normal immutable-weight cache sharing changed")
			}
			first, second := call(MemoryOffload), call(MemoryOffload)
			a, b := first.buf.(*cudaBuf), second.buf.(*cudaBuf)
			if a == b || a == cachedBuf || b == cachedBuf || allocs != 2 || copies != 2 || weightCalls != 2 || releases != 0 {
				t.Fatal("offload upload did not create two independent owners")
			}
			for _, owner := range owners {
				if owner.class != MemoryOffload || owner.host != 0 || owner.hostKeep != nil || owner.budgetedWeightBytes != 0 || owner.managedWeight {
					t.Fatal("offload owner acquired shared-cache or immutable-placement metadata")
				}
			}
			if first.Dtype != tc.dt || first.Layout != RowMajor || !reflect.DeepEqual(first.Shape, []int{1, 256}) || first.Quant.Block != 256 || first.Quant.Axis != 2 || first.Quant.Bits != tc.bits || first.Quant.Symmetric {
				t.Fatal("packed tensor descriptor changed")
			}
			// Model caller retirement without sending host storage to CUDA Free.
			p := a.ptr
			a.ptr = nil
			release(p)
			if !reflect.DeepEqual(memory[b.ptr], raw) || uploadCache[key].buf != cachedBuf || releases != 1 {
				t.Fatal("one offload retirement changed its peer or immutable cache")
			}
			if c.dlUsed != 77 || c.managedN != 3 || c.immutableWeightUploadCalls != 5 || c.immutableWeightUploadTransferBytes != 9 || c.immutableWeightUploadResidentBytes != 11 {
				t.Fatal("offload operation changed immutable-weight accounting")
			}
			host.Shape[0] = 99
			if first.Shape[0] != 1 || second.Shape[0] != 1 {
				t.Fatal("shape aliases caller")
			}
		})
	}

	for _, phase := range []string{"allocation", "copy", "copy_and_cleanup"} {
		t.Run(phase, func(t *testing.T) {
			c := &cudaBackend{budgetBytes: 1000, dlUsed: 77, managedN: 3}
			host := NewQ4K(Default(), []int{1, 256}, make([]byte, 144))
			primary := &DeviceAllocError{Bytes: 144, Site: "offload-test", Class: MemoryOffload}
			storage := make([]byte, 144)
			var owner *cudaBuf
			allocs, copies, releases := 0, 0, 0
			var got any
			func() {
				defer func() { got = recover() }()
				c.uploadRawKQuantClassWithOperations(host, host.buf.(HostBuffer), MemoryOffload,
					func(Tensor, HostBuffer) Tensor { t.Fatal("offload entered weight cache"); return Tensor{} },
					func(n int) *cudaBuf {
						allocs++
						if phase == "allocation" {
							panic(primary)
						}
						owner = &cudaBuf{ptr: unsafe.Pointer(&storage[0]), n: n, class: MemoryOffload}
						return owner
					},
					func(unsafe.Pointer, []int8) { copies++; panic(primary) },
					func(p unsafe.Pointer) {
						releases++
						if owner == nil || owner.ptr != nil || p != unsafe.Pointer(&storage[0]) {
							t.Error("release did not detach exact owner")
						}
						if phase == "copy_and_cleanup" {
							panic("secondary cleanup")
						}
					},
				)
				t.Error("failure returned normally")
			}()
			wantRelease := 1
			if phase == "allocation" {
				wantRelease = 0
			}
			if got != primary || allocs != 1 || copies != wantRelease || releases != wantRelease {
				t.Fatalf("failure identity/ownership changed: panic=%v alloc=%d copy=%d release=%d", got, allocs, copies, releases)
			}
			if c.dlUsed != 77 || c.managedN != 3 || c.immutableWeightUploadCalls != 0 || len(uploadCache) != 0 {
				t.Fatal("failed offload published cache/accounting state")
			}
		})
	}
}
