//go:build cuda && cgo

package compute

import (
	"errors"
	"math"
	"testing"
)

// The prompt panel (tiled conv, hoisted gates, column-split recurrence, split out-norm; #9359)
// keeps every element's expression and order, so a whole-prompt sequence must reproduce the
// token-by-token decode loop bit for bit: per-token output, final conv state and final
// recurrent state. Qwen3.8-27B axes put 4 column blocks on each value head and 3 conv tiles
// (one ragged) on 130 tokens; the ragged geometry leaves part of a column block idle; a
// 2-token prompt shifts part of the old conv history; K=1 has no history at all.
// fak-test:runtime fast est=2s
func TestCUDAQwen35GDNSequencePanelBitwiseDecodeLoop(t *testing.T) {
	for _, tc := range []struct {
		name   string
		g      cudaGDNGeometry
		tokens int
	}{
		{"qwen38-27b-axes", cudaGDNGeometry{hidden: 64, nK: 16, nV: 48, kHd: 128, vHd: 128, kernel: 4, eps: 1e-6}, 130},
		{"ragged-columns", cudaGDNGeometry{hidden: 16, nK: 2, nV: 6, kHd: 24, vHd: 40, kernel: 4, eps: 1e-5}, 70},
		{"prompt-shorter-than-history", cudaGDNGeometry{hidden: 16, nK: 2, nV: 4, kHd: 16, vHd: 32, kernel: 4, eps: 1e-5}, 2},
		{"pointwise-conv", cudaGDNGeometry{hidden: 16, nK: 2, nV: 4, kHd: 16, vHd: 32, kernel: 1, eps: 1e-5}, 5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			be := cudaGDNBackend(t)
			g := tc.g
			seq := newCUDAGDNOperands(t, be, g)
			ref := seq
			rng := cudaGDNLCG(0x9359)
			convInit := rng.vector((g.kernel-1)*g.convDim(), 1.0)
			recurrentInit := rng.vector(g.nV*g.kHd*g.vHd, 0.2)
			seq.convState = uploadCUDAGDN(t, be, []int{g.kernel - 1, g.convDim()}, convInit, MemoryKVCache, "qwen35-gdn-panel-conv-state")
			seq.recurrentState = uploadCUDAGDN(t, be, []int{g.nV, g.kHd, g.vHd}, recurrentInit, MemoryKVCache, "qwen35-gdn-panel-recurrent-state")
			ref.convState = uploadCUDAGDN(t, be, []int{g.kernel - 1, g.convDim()}, convInit, MemoryKVCache, "qwen35-gdn-loop-conv-state")
			ref.recurrentState = uploadCUDAGDN(t, be, []int{g.nV, g.kHd, g.vHd}, recurrentInit, MemoryKVCache, "qwen35-gdn-loop-recurrent-state")

			inputs := rng.vector(tc.tokens*g.hidden, 1.0)
			panel := uploadCUDAGDN(t, be, []int{tc.tokens, g.hidden}, inputs, MemoryActivation, "qwen35-gdn-panel-input")
			got, _, _, err := be.Qwen35GDNSequence(
				panel, seq.inQKV, seq.inZ, seq.inB, seq.inA, seq.convW,
				seq.aLog, seq.dtBias, seq.norm, seq.outW, seq.convState,
				seq.recurrentState, g.nK, g.nV, g.kHd, g.vHd, g.kernel, g.eps,
			)
			if err != nil {
				t.Fatalf("sequence: %v", err)
			}
			gotOut := be.Read(got)
			want := make([]float32, 0, len(gotOut))
			for token := 0; token < tc.tokens; token++ {
				x := uploadCUDAGDN(t, be, []int{g.hidden}, inputs[token*g.hidden:(token+1)*g.hidden], MemoryActivation, "qwen35-gdn-loop-input")
				out, _, _, err := be.Qwen35GDNDecode(
					x, ref.inQKV, ref.inZ, ref.inB, ref.inA, ref.convW,
					ref.aLog, ref.dtBias, ref.norm, ref.outW, ref.convState,
					ref.recurrentState, g.nK, g.nV, g.kHd, g.vHd, g.kernel, g.eps,
				)
				if err != nil {
					t.Fatalf("decode token %d: %v", token, err)
				}
				want = append(want, be.Read(out)...)
			}
			requireFiniteNonzeroCUDA(t, "panel output", gotOut)
			for _, pair := range []struct {
				name      string
				got, want []float32
			}{
				{"output", gotOut, want},
				{"conv state", be.Read(seq.convState), be.Read(ref.convState)},
				{"recurrent state", be.Read(seq.recurrentState), be.Read(ref.recurrentState)},
			} {
				if len(pair.got) != len(pair.want) {
					t.Fatalf("%s len %d != %d", pair.name, len(pair.got), len(pair.want))
				}
				for i := range pair.want {
					if math.Float32bits(pair.got[i]) != math.Float32bits(pair.want[i]) {
						t.Fatalf("%s[%d] = %g (%08x), decode loop %g (%08x)", pair.name, i,
							pair.got[i], math.Float32bits(pair.got[i]), pair.want[i], math.Float32bits(pair.want[i]))
					}
				}
			}
		})
	}
}

// The panel holds one key-dimension column of state in registers and the q/k rows in fixed
// shared arrays, so a key head wider than 128 is a typed geometry refusal before any launch,
// never a truncation and never a device fault that poisons the session.
// fak-test:runtime fast est=1s
func TestCUDAQwen35GDNSequenceRefusesWideKeyHead(t *testing.T) {
	be := cudaGDNBackend(t)
	t.Cleanup(be.Recycle)
	prodLatch := be.faultLatch
	be.faultLatch = NewDeviceFaultLatch("cuda", cudaFaultReconstructBudget)
	t.Cleanup(func() { be.faultLatch = prodLatch })
	g := cudaGDNGeometry{hidden: 8, nK: 1, nV: 1, kHd: 129, vHd: 4, kernel: 2, eps: 1e-5}
	o := newCUDAGDNOperands(t, be, g)
	panel := uploadCUDAGDN(t, be, []int{2, g.hidden}, make([]float32, 2*g.hidden), MemoryActivation, "qwen35-gdn-wide-key-input")
	_, _, _, err := be.Qwen35GDNSequence(
		panel, o.inQKV, o.inZ, o.inB, o.inA, o.convW, o.aLog, o.dtBias, o.norm, o.outW,
		o.convState, o.recurrentState, g.nK, g.nV, g.kHd, g.vHd, g.kernel, g.eps,
	)
	var geometry *Qwen35GDNGeometryError
	if !errors.As(err, &geometry) {
		t.Fatalf("key_head_dim=129 sequence error = %T %v, want *Qwen35GDNGeometryError", err, err)
	}
	if be.faultLatch.Snapshot().Refusing() {
		t.Fatal("a geometry refusal poisoned the device session")
	}
}
