//go:build cuda && cgo

package compute

import (
	"encoding/binary"
	"errors"
	"testing"
)

// TestCUDAGEMMScratchMissFailsClosed arms the dequant + fp16 GEMM scratch fault at both scratch
// sites (the two fp16 weight halves) for each P>=128 entry (Q4_K, Q2_K and Q8_0
// BatchedMatMul). Each must raise a typed DeviceAllocError naming the allocation that missed
// instead of returning an output tensor holding a previous op's bytes, and the next call must
// compute normally.
func TestCUDAGEMMScratchMissFailsClosed(t *testing.T) {
	cb := cudaOrSkip(t)
	var seed lcg = 0x5ca7
	g := &seed
	out, in, P := 320, 256, 128
	X := mkResident(cb, []int{P, in}, rscale(g, P*in, 1.0))
	weights := map[string]Tensor{
		"Q4_K": cb.Upload(newQ4KHost(cb, []int{out, in}, randQ4KBytes(g, out, in)), Q4_K),
		"Q8_0": cb.Upload(NewF32(cb, []int{out, in}, rscale(g, out*in, 1.0)), Q8_0),
	}
	q2k := make([]byte, out*(in/q2kSuper)*q2kSuperBlock)
	for i := range q2k {
		q2k[i] = byte(int32(g.f() * 512))
	}
	for base := 0; base < len(q2k); base += q2kSuperBlock {
		binary.LittleEndian.PutUint16(q2k[base+80:base+82], 0x3000) // d = 0.125
		binary.LittleEndian.PutUint16(q2k[base+82:base+84], 0x2c00) // dmin = 0.0625
	}
	weights["Q2_K"] = cb.Upload(NewQ2K(cb, []int{out, in}, q2k), Q2_K)
	// All three large-P routes split resident weights directly; the first two
	// allocations are FP16 weight halves, with no F32 dequant image (#14097).
	sites := []struct {
		name  string
		nth   int
		bytes int
	}{
		{"fp16-hi", 1, cudaDequantScratch(out, in).split},
		{"fp16-split", 2, cudaDequantScratch(out, in).split},
	}
	for name, w := range weights {
		for _, site := range sites {
			t.Run(name+"/"+site.name, func(t *testing.T) {
				var y Tensor
				err := func() (err error) {
					defer func() {
						if r := recover(); r != nil {
							e, ok := r.(error)
							if !ok {
								t.Fatalf("panic value %T %v, want an error", r, r)
							}
							err = e
						}
					}()
					cudaArmScratchAllocFaultForTest(site.nth)
					y = cb.BatchedMatMul(w, X, P)
					return nil
				}()
				cudaArmScratchAllocFaultForTest(0)
				var alloc *DeviceAllocError
				if !errors.As(err, &alloc) {
					t.Fatalf("scratch miss returned err=%v y=%v, want a DeviceAllocError and no output", err, y.Shape)
				}
				if alloc.Bytes != site.bytes || alloc.Class != MemoryScratchpad {
					t.Fatalf("DeviceAllocError %+v, want %d scratchpad bytes", alloc, site.bytes)
				}
				if got := cb.Read(cb.BatchedMatMul(w, X, P)); len(got) != P*out {
					t.Fatalf("GEMM after the fault returned %d values, want %d", len(got), P*out)
				}
			})
		}
	}
}
