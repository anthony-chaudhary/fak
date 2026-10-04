package model

import (
	"math"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model/ffn"
)

// moe_host_batch_ffn_test.go pins the migration of batchedExpertDelta's per-expert
// activation*up fusion onto the shared ffn.ApplyInPlace helper (fak#13450). The batched
// host-expert tail used to carry its own activation-times-up loop over each expert's
// intermediate row; it now delegates that loop to ffn. The migration must be numerically
// invisible: gate/up row fusion, non-overlap ownership and the gate-weighted accumulation
// order are unchanged.

// legacyBatchedExpertDelta is the PRE-refactor body of batchedExpertDelta, transcribed as
// the independent original-arithmetic oracle. It runs the per-expert q4k gate/up rows,
// fuses them with the inline activation*up loop, projects through the k-quant down tensor
// and accumulates in route order. The production batchedExpertDelta must match it bit for
// bit, because every named operation runs in the identical order.
func legacyBatchedExpertDelta(cfg Config, picks []routePick, gate, up []*q4kTensor, down []*kQuantTensor, xn, delta []float32) {
	H := cfg.HiddenSize
	MI := cfg.expertIntermediate()
	for i := range picks {
		g := q4kMatRows(gate[i], xn)
		u := q4kMatRows(up[i], xn)
		for j := 0; j < MI; j++ {
			g[j] = act(g[j], cfg) * u[j]
		}
		d := kQuantMatRows(down[i], g)
		w := picks[i].weight
		for j := 0; j < H; j++ {
			delta[j] += w * d[j]
		}
	}
}

// hostBatchFFNMkQ4K and hostBatchFFNMkQ5K build deterministic resident tensors for the
// parity fixture, matching the shapes batchedExpertDelta consumes (gate/up are Q4_K q4k,
// down is Q5_K kQuant).
func hostBatchFFNMkQ4K(rng *rand.Rand, out, in int) *q4kTensor {
	nblk := in / qkK
	raw := make([]byte, out*nblk*q4kBlockBytes)
	blk := make([]byte, q4kBlockBytes)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			randQ4KBlock(rng, blk)
			copy(raw[(o*nblk+b)*q4kBlockBytes:], blk)
		}
	}
	return quantizeQ4KFromRaw(raw, out, in)
}

func hostBatchFFNMkQ5K(rng *rand.Rand, out, in int) *kQuantTensor {
	nblk := in / qkK
	raw := make([]byte, out*nblk*q5kBlockBytes)
	for i := range raw {
		raw[i] = byte(rng.Intn(256))
	}
	return quantizeKQuantFromRaw(raw, out, in, kindQ5K)
}

// TestHostBatchedFFNApplyInPlaceParity proves the migrated batchedExpertDelta is bit-identical
// to the pre-refactor inline loop over every expert's intermediate row, for SiLU and GeGLU
// and for both int8-SDOT states, using math.Float32bits so signed zeros and NaN payloads must
// match exactly. It exercises the real consumer (batchedExpertDelta), not a bare helper.
func TestHostBatchedFFNApplyInPlaceParity(t *testing.T) {
	const H, MI, K = 768, 512, 4 // realistic; qkK multiples
	cases := []struct {
		name    string
		actGelu bool
	}{
		{name: "silu"},
		{name: "gelu-tanh", actGelu: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{HiddenSize: H, IntermediateSize: MI, MoEIntermediateSize: MI, ActGeluTanh: tc.actGelu}
			rng := rand.New(rand.NewSource(13450))

			gate := make([]*q4kTensor, K)
			up := make([]*q4kTensor, K)
			down := make([]*kQuantTensor, K)
			for e := 0; e < K; e++ {
				gate[e] = hostBatchFFNMkQ4K(rng, MI, H)
				up[e] = hostBatchFFNMkQ4K(rng, MI, H)
				down[e] = hostBatchFFNMkQ5K(rng, H, MI)
			}
			xn := make([]float32, H)
			for i := range xn {
				xn[i] = float32(rng.NormFloat64())
			}
			picks := make([]routePick, K)
			for i := range picks {
				picks[i] = routePick{expert: i, weight: float32(rng.NormFloat64())}
			}

			for _, int8on := range []bool{false, true} {
				setQ4KSDOTForTest(int8on)
				setKQuantSDOTForTest(int8on)

				want := make([]float32, H)
				legacyBatchedExpertDelta(cfg, picks, gate, up, down, xn, want)
				got := make([]float32, H)
				batchedExpertDelta(cfg, picks, gate, up, down, xn, got)
				q4kSDOTForce, kQuantSDOTForce = 0, 0

				for j := 0; j < H; j++ {
					if math.Float32bits(got[j]) != math.Float32bits(want[j]) {
						t.Fatalf("int8=%v hidden[%d]: got bits=%#x legacy=%#x", int8on, j,
							math.Float32bits(got[j]), math.Float32bits(want[j]))
					}
				}
			}
		})
	}
}

// TestHostBatchedFFNApplyInPlaceOwnership proves the delegated fusion keeps the gate row
// in place and reads the up row without mutating it: ApplyInPlace's contract is that the
// caller owns gate, up is read-only, and a successful call writes only gate. Byte-equality
// of up before and after, plus the expected fused gate, pins the ownership the migration
// must preserve.
func TestHostBatchedFFNApplyInPlaceOwnership(t *testing.T) {
	const MI = 64
	cfg := Config{HiddenSize: 8, IntermediateSize: MI, MoEIntermediateSize: MI}
	gate := make([]float32, MI)
	up := make([]float32, MI)
	for i := range gate {
		gate[i] = float32(i%17-8) / 4
		up[i] = float32(i%13-6) / 8
	}
	upCopy := append([]float32(nil), up...)
	wantGate := make([]float32, MI)
	for i := range gate {
		wantGate[i] = act(gate[i], cfg) * up[i]
	}

	activate := func(v float32) float32 { return act(v, cfg) }
	if err := ffn.ApplyInPlace(gate, up, activate); err != nil {
		t.Fatalf("ApplyInPlace: %v", err)
	}
	for i := range wantGate {
		if math.Float32bits(gate[i]) != math.Float32bits(wantGate[i]) {
			t.Fatalf("gate[%d] bits=%#x want=%#x", i, math.Float32bits(gate[i]), math.Float32bits(wantGate[i]))
		}
	}
	for i := range upCopy {
		if math.Float32bits(up[i]) != math.Float32bits(upCopy[i]) {
			t.Fatalf("up[%d] mutated: bits=%#x want=%#x", i, math.Float32bits(up[i]), math.Float32bits(upCopy[i]))
		}
	}
}

// TestHostBatchedFFNApplyInPlaceZeroAllocs pins the allocation invariant the hot decode
// path depends on: the delegated fusion over a preallocated row allocates nothing.
func TestHostBatchedFFNApplyInPlaceZeroAllocs(t *testing.T) {
	const MI = 256
	cfg := Config{HiddenSize: 8, IntermediateSize: MI, MoEIntermediateSize: MI}
	gate := make([]float32, MI)
	up := make([]float32, MI)
	activate := func(v float32) float32 { return act(v, cfg) }

	allocs := testing.AllocsPerRun(50, func() {
		if err := ffn.ApplyInPlace(gate, up, activate); err != nil {
			t.Fatalf("ApplyInPlace: %v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyInPlace allocations = %v, want 0", allocs)
	}
}
