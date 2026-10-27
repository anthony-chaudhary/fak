//go:build cuda && cgo

package compute

import (
	"math"
	"testing"
)

// The chunk expansion changes FP32 summation order, so it has its own numerical
// witness rather than relaxing the existing serial panel's bitwise contract.
// For dispatch proof, collect FAK_CUDA_TIMESTAMP_PROFILE=1 and require
// k_qwen35_gdn_recurrent_chunks in the physical-device trace for panels >=16.
// fak-test:runtime fast est=3s
func TestCUDAQwen35GDNChunkRecurrenceMatchesSerial(t *testing.T) {
	t.Setenv("FAK_CUDA_GDN_RECURRENCE", "serial")
	be := cudaGDNBackend(t)
	for _, g := range []cudaGDNGeometry{
		{hidden: 16, nK: 2, nV: 6, kHd: 24, vHd: 11, kernel: 4, eps: 1e-5},
		{hidden: 32, nK: 16, nV: 48, kHd: 128, vHd: 128, kernel: 4, eps: 1e-6},
	} {
		serial := newCUDAGDNOperands(t, be, g)
		chunk := serial
		rng := cudaGDNLCG(0x9359)
		convInit := rng.vector((g.kernel-1)*g.convDim(), 1.0)
		stateInit := rng.vector(g.nV*g.kHd*g.vHd, 0.2)
		serial.convState = uploadCUDAGDN(t, be, []int{g.kernel - 1, g.convDim()}, convInit, MemoryKVCache, "gdn-serial-conv")
		chunk.convState = uploadCUDAGDN(t, be, []int{g.kernel - 1, g.convDim()}, convInit, MemoryKVCache, "gdn-chunk-conv")
		serial.recurrentState = uploadCUDAGDN(t, be, []int{g.nV, g.kHd, g.vHd}, stateInit, MemoryKVCache, "gdn-serial-state")
		chunk.recurrentState = uploadCUDAGDN(t, be, []int{g.nV, g.kHd, g.vHd}, stateInit, MemoryKVCache, "gdn-chunk-state")
		// Sequential calls deliberately retain both histories across tile boundaries.
		for _, tokens := range []int{15, 16, 17, 33} {
			panel := uploadCUDAGDN(t, be, []int{tokens, g.hidden}, rng.vector(tokens*g.hidden, 1), MemoryActivation, "gdn-chunk-input")
			run := func(mode string, o cudaGDNOperands) []float32 {
				t.Setenv("FAK_CUDA_GDN_RECURRENCE", mode)
				out, _, _, err := be.Qwen35GDNSequence(panel, o.inQKV, o.inZ, o.inB, o.inA, o.convW,
					o.aLog, o.dtBias, o.norm, o.outW, o.convState, o.recurrentState,
					g.nK, g.nV, g.kHd, g.vHd, g.kernel, g.eps)
				if err != nil {
					t.Fatalf("%s tokens=%d: %v", mode, tokens, err)
				}
				return be.Read(out)
			}
			want, got := run("serial", serial), run("chunk16", chunk)
			for _, pair := range []struct {
				name      string
				want, got []float32
			}{
				{"output", want, got},
				{"conv-state", be.Read(serial.convState), be.Read(chunk.convState)},
				{"recurrent-state", be.Read(serial.recurrentState), be.Read(chunk.recurrentState)},
			} {
				if len(pair.want) != len(pair.got) {
					t.Fatalf("%s lengths differ", pair.name)
				}
				for i, w := range pair.want {
					x := float64(pair.got[i])
					err := math.Abs(x - float64(w))
					limit := 2e-5 + 2e-4*math.Abs(float64(w))
					if math.IsNaN(x) || math.IsInf(x, 0) || math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) || err > limit {
						t.Fatalf("%s tokens=%d K=%d V=%d index=%d got=%g want=%g error=%g limit=%g", pair.name, tokens, g.kHd, g.vHd, i, x, w, err, limit)
					}
				}
			}
		}
	}
}
