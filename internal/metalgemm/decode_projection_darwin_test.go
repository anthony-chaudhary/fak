//go:build darwin && arm64 && cgo

// fak-test:runtime fast est=3s

package metalgemm

import (
	"math"
	"math/rand"
	"testing"
)

// nativeProjectionFixture is one Q5_K matrix resident twice: as native rows and as the Q8_0
// requantization of its exact dequantization, so a native and a Q8 decode can be compared.
type nativeProjectionFixture struct {
	native *Q5KWeight
	q8     *Q8Weight
}

func newNativeProjectionFixture(t *testing.T, rng *rand.Rand, out, in int) nativeProjectionFixture {
	t.Helper()
	raw := randomQ5KRows(rng, out, in)
	native := UploadQ5KGoOwned(raw, out, in)
	if native == nil {
		t.Fatalf("Q5_K upload %dx%d", out, in)
	}
	rowBytes := (in / 256) * Q5KBlockBytes
	codes := make([]int8, out*in)
	scales := make([]float32, out*(in/32))
	for o := 0; o < out; o++ {
		row := q5kRefRow(raw[o*rowBytes:(o+1)*rowBytes], in)
		for b := 0; b < in/32; b++ {
			var amax float64
			for _, v := range row[b*32 : (b+1)*32] {
				amax = math.Max(amax, math.Abs(v))
			}
			d := amax / 127
			scales[o*(in/32)+b] = float32(d)
			for i, v := range row[b*32 : (b+1)*32] {
				if d > 0 {
					codes[o*in+b*32+i] = int8(math.Round(v / d))
				}
			}
		}
	}
	q8 := UploadQ8(codes, scales, out, in)
	if q8 == nil {
		t.Fatalf("Q8 upload %dx%d", out, in)
	}
	return nativeProjectionFixture{native: native, q8: q8}
}

func (f nativeProjectionFixture) release() {
	f.native.Release()
	f.q8.Release()
}

func requireCosine(t *testing.T, name string, want, got []float32, floor float64) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: len %d, want %d", name, len(got), len(want))
	}
	var dot, ww, gg float64
	for i := range want {
		w, g := float64(want[i]), float64(got[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("%s: non-finite at %d", name, i)
		}
		dot, ww, gg = dot+w*g, ww+w*w, gg+g*g
	}
	if ww == 0 || gg == 0 {
		t.Fatalf("%s: zero vector", name)
	}
	if cos := dot / math.Sqrt(ww*gg); cos < floor {
		t.Fatalf("%s: cosine %.6f < %.6f", name, cos, floor)
	}
}

// TestQwen35DecodeNativeProjections pins the native k-quant decode route: a mixer whose
// in_proj_qkv/z/out_proj are served from Q5_K rows (b/a stay Q8) decodes like the all-Q8 mixer
// over the same weights, skips the quantizer for every all-native source, and the batched
// encoder tracks the serial native decode row by row.
func TestQwen35DecodeNativeProjections(t *testing.T) {
	if !Q5KAvailable() {
		t.Skip("Q5_K Metal kernels unavailable")
	}
	geometry := GDNGeometry{NumKeyHeads: 2, NumValueHeads: 4, KeyHeadDim: 128, ValueHeadDim: 128, ConvKernel: 4}
	const hidden = 512
	rng := rand.New(rand.NewSource(11))
	qkv := newNativeProjectionFixture(t, rng, geometry.convDim(), hidden)
	z := newNativeProjectionFixture(t, rng, geometry.valueDim(), hidden)
	out := newNativeProjectionFixture(t, rng, hidden, geometry.valueDim())
	defer qkv.release()
	defer z.release()
	defer out.release()
	q8 := newQwen35DecodeBatchWeights(t, hidden, geometry)
	defer releaseQwen35DecodeBatchWeights(q8)
	reference := Qwen35DecodeWeights{InQKV: qkv.q8, InZ: z.q8, InB: q8.InB, InA: q8.InA, Out: out.q8}
	native := Qwen35DecodeWeights{InB: q8.InB, InA: q8.InA}
	native.Native[0] = KQuantProjection{Q5K: qkv.native}
	native.Native[1] = KQuantProjection{Q5K: z.native}
	native.Native[4] = KQuantProjection{Q5K: out.native}
	panel := qwen35DecodeBatchPanel(geometry)
	const batch = 3
	input := qwen35DecodeBatchInput(batch, hidden)

	run := func(weights Qwen35DecodeWeights, row int) ([]float32, Qwen35DecodeReceipt) {
		t.Helper()
		states := newQwen35DecodeBatchStates(t, geometry, 1)
		defer closeQwen35DecodeBatchStates(states)
		seedQwen35DecodeBatchState(t, geometry, states[0], row+1)
		got, receipt, accepted, err := RunQwen35Decode(Qwen35DecodeRequest{
			Input: input[row*hidden : (row+1)*hidden], Weights: weights, State: states[0], Panel: panel,
		})
		if err != nil || !accepted {
			t.Fatalf("row %d decode accepted=%v err=%v", row, accepted, err)
		}
		return append([]float32(nil), got...), receipt
	}
	serial := make([][]float32, batch)
	for row := 0; row < batch; row++ {
		want, wantReceipt := run(reference, row)
		got, gotReceipt := run(native, row)
		requireCosine(t, "native vs Q8 mixer", want, got, 0.999)
		if wantReceipt.Quantizers != 2 || gotReceipt.Quantizers != 1 || gotReceipt.ProjectionDispatches != 5 ||
			gotReceipt.Encoders != wantReceipt.Encoders-1 {
			t.Fatalf("receipts Q8=%+v native=%+v, want the all-native out_proj source to skip its quantizer", wantReceipt, gotReceipt)
		}
		serial[row] = got
	}

	states := newQwen35DecodeBatchStates(t, geometry, batch)
	defer closeQwen35DecodeBatchStates(states)
	for row, state := range states {
		seedQwen35DecodeBatchState(t, geometry, state, row+1)
	}
	outputs, receipt, accepted, err := RunQwen35DecodeBatch(Qwen35DecodeBatchRequest{Input: input, Weights: native, States: states, Panel: panel})
	if err != nil || !accepted {
		t.Fatalf("native batch accepted=%v err=%v", accepted, err)
	}
	if receipt.Quantizers != 1 || receipt.Encoders != 7 || receipt.ProjectionDispatches != 5 {
		t.Fatalf("native batch receipt=%+v", receipt)
	}
	// The panel runs q5k_mul_mm (weights rounded to half), the serial decode q5k_mul_mv (f32).
	for row := range outputs {
		requireCosine(t, "native batch vs serial", serial[row], outputs[row], 0.99999)
	}

	both := native
	both.InQKV = qkv.q8
	if _, _, accepted, err := RunQwen35Decode(Qwen35DecodeRequest{Input: input[:hidden], Weights: both, State: states[0], Panel: panel}); accepted || err == nil {
		t.Fatalf("a projection with both Q8 and native handles was accepted (err=%v)", err)
	}
}

// TestQwen35FullAttentionBatchNativeQK serves the gated-Q and K projections from Q5_K rows and
// pins the batch to the Q8 run over the requantized weights.
func TestQwen35FullAttentionBatchNativeQK(t *testing.T) {
	if !Q5KAvailable() {
		t.Skip("Q5_K Metal kernels unavailable")
	}
	const modelWidth, nH, nKV, hd, rotary = 512, 4, 2, 64, 32
	rng := rand.New(rand.NewSource(17))
	q := newNativeProjectionFixture(t, rng, 2*nH*hd, modelWidth)
	k := newNativeProjectionFixture(t, rng, nKV*hd, modelWidth)
	defer q.release()
	defer k.release()
	v := attentionBatchQ4K(t, nKV*hd, modelWidth, 5)
	defer v.Release()
	reference := attentionBatchRequest(3, modelWidth, nH, nKV, hd, rotary, Qwen35FullAttentionWeights{Q: q.q8, K: k.q8, V: v})
	native := reference
	native.Weights = Qwen35FullAttentionWeights{V: v, NativeQ: KQuantProjection{Q5K: q.native}, NativeK: KQuantProjection{Q5K: k.native}}
	want, _, accepted, err := RunQwen35FullAttentionDecodeBatch(reference)
	if err != nil || !accepted {
		t.Fatalf("Q8 attention batch accepted=%v err=%v", accepted, err)
	}
	got, receipt, accepted, err := RunQwen35FullAttentionDecodeBatch(native)
	if err != nil || !accepted || receipt.ProjectionDispatches != 3 {
		t.Fatalf("native attention batch accepted=%v receipt=%+v err=%v", accepted, receipt, err)
	}
	for row := range want.Output {
		requireCosine(t, "attention output", want.Output[row], got.Output[row], 0.999)
		requireCosine(t, "kraw", want.KRaw[row], got.KRaw[row], 0.999)
	}
}
