package ggufload

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func inkling2MiniFullShape() model.Inkling2LoopHParams {
	hp := inkling2MiniKVShape()
	hp.Hidden = 2304
	hp.RelLength = 128
	hp.ExpertsUsed = 8
	hp.ExpertWidth = 768
	return hp
}

// fak-test:runtime fast est=200ms
func TestInkling2PrefillArenaFP8Budget(t *testing.T) {
	hp := inkling2MiniFullShape()
	const T = 512
	const padK = 768

	f32Keep, f32Scr := inkling2PrefillArenaBytesFor(hp, T, padK, false)
	fp8Keep, fp8Scr := inkling2PrefillArenaBytesFor(hp, T, padK, true)
	f32Total := inkling2PrefillArenaTotalBytes(hp, T, padK, false)
	fp8Total := inkling2PrefillArenaTotalBytes(hp, T, padK, true)

	const maxBytes = 180 << 20 // 180 MiB gate
	if fp8Total > maxBytes {
		t.Fatalf("T=%d FP8 prefill arena total = %d bytes (%.2f MiB), want <= %d bytes (180 MiB)",
			T, fp8Total, float64(fp8Total)/(1<<20), maxBytes)
	}
	if f32Total <= 3*fp8Total {
		t.Fatalf("expected >=3x reduction from F32 (%d) to FP8 (%d)", f32Total, fp8Total)
	}
	t.Logf("T=%d prefill arenas: F32 keep=%.2f MiB scr=%.2f MiB total=%.2f MiB | FP8 keep=%.2f MiB scr=%.2f MiB total=%.2f MiB",
		T,
		float64(f32Keep)/(1<<20), float64(f32Scr)/(1<<20), float64(f32Total)/(1<<20),
		float64(fp8Keep)/(1<<20), float64(fp8Scr)/(1<<20), float64(fp8Total)/(1<<20))

	// Fit planner must admit T=512 with FP8 inside 180 MiB headroom and T=256 inside 100 MiB.
	if got := Inkling2PrefillChunkForBudget(hp, 180<<20, true); got != 512 {
		t.Fatalf("Inkling2PrefillChunkForBudget(180 MiB, fp8=true) = %d, want 512", got)
	}
	if got := Inkling2PrefillChunkForBudget(hp, 100<<20, true); got != 256 {
		t.Fatalf("Inkling2PrefillChunkForBudget(100 MiB, fp8=true) = %d, want 256", got)
	}
	if got := Inkling2PrefillChunkForBudget(hp, 200<<20, false); got >= 512 {
		t.Fatalf("Inkling2PrefillChunkForBudget(200 MiB, fp8=false) = %d, should not admit 512 in F32", got)
	}

	// Verify 128k NVFP4 KV + T=512 FP8 prefill arenas fit inside RTX 5080 2.083 GiB session budget.
	kv128k := inkling2ResidentKVBytesFor(hp, 131072, true, 512, Inkling2KVNVFP4)
	otherFP8 := inkling2ResidentOtherBytesFor(hp, inkling2ArenaFloats, 512, padK, true)
	const sessionBudget = int64(2236600000) // ~2.083 GiB
	if kv128k+otherFP8 > sessionBudget {
		t.Fatalf("128k NVFP4 KV + T=512 FP8 prefill = %.3f GiB exceeds 2.083 GiB session budget",
			float64(kv128k+otherFP8)/(1<<30))
	}
}

// fakeInk2FP8Prefill wraps fakeInk2Cohort with deterministic FP8 E4M3FN arena ops
// and simple linear/RMS/head ops so s.prefill exercises the real FP8 keep/scr path.
type fakeInk2FP8Prefill struct {
	*fakeInk2Cohort
	quantCalls   int
	dequantCalls int
	lastLogits   []float32
}

func newFakeInk2FP8Prefill() *fakeInk2FP8Prefill {
	return &fakeInk2FP8Prefill{fakeInk2Cohort: newFakeInk2Cohort()}
}

func (d *fakeInk2FP8Prefill) Caps() compute.Caps        { return compute.Caps{} }
func (d *fakeInk2FP8Prefill) Ink2PFP8ArenaReady() error { return nil }

func (d *fakeInk2FP8Prefill) Ink2PQuantizeFP8(src, codes, scales compute.Ink2Ref, rows, cols int) error {
	d.quantCalls++
	s := d.rows(src)[:rows*cols]
	c, sc := compute.Ink2FP8QuantizeRows(s, rows, cols)
	dstCodes := d.rows(codes)
	nWords := (len(c) + 3) / 4
	for w := 0; w < nWords; w++ {
		var word [4]byte
		for b := 0; b < 4 && w*4+b < len(c); b++ {
			word[b] = c[w*4+b]
		}
		dstCodes[w] = math.Float32frombits(binary.LittleEndian.Uint32(word[:]))
	}
	copy(d.rows(scales)[:rows], sc)
	return nil
}

func (d *fakeInk2FP8Prefill) Ink2PDequantFP8(codes, scales, dst compute.Ink2Ref, rows, cols int) error {
	d.dequantCalls++
	nBytes := rows * cols
	srcCodes := d.rows(codes)
	raw := make([]byte, nBytes)
	nWords := (nBytes + 3) / 4
	for w := 0; w < nWords; w++ {
		var word [4]byte
		binary.LittleEndian.PutUint32(word[:], math.Float32bits(srcCodes[w]))
		for b := 0; b < 4 && w*4+b < nBytes; b++ {
			raw[w*4+b] = word[b]
		}
	}
	sc := d.rows(scales)[:rows]
	out := compute.Ink2FP8DequantRows(raw, sc, rows, cols)
	copy(d.rows(dst)[:nBytes], out)
	return nil
}

func (d *fakeInk2FP8Prefill) Ink2PRMS(x, w, y compute.Ink2Ref, rows, n int, eps float32) error {
	xr, yr := d.rows(x), d.rows(y)
	var wr []float32
	if w.T.Buf() != nil {
		wr = d.rows(w)
	}
	for r := 0; r < rows; r++ {
		off := r * n
		var ss float64
		for i := 0; i < n; i++ {
			v := float64(xr[off+i])
			ss += v * v
		}
		inv := float32(1.0 / math.Sqrt(ss/float64(n)+float64(eps)))
		for i := 0; i < n; i++ {
			v := xr[off+i] * inv
			if wr != nil {
				v *= wr[i]
			}
			yr[off+i] = v
		}
	}
	return nil
}

func (d *fakeInk2FP8Prefill) Ink2Mix(states []compute.Ink2Ref, _ compute.Ink2Ref, nq int, _ []compute.Ink2Ref, _ float32, out compute.Ink2Ref, n int) error {
	dst := d.rows(out)
	for q := 0; q < nq; q++ {
		off := q * n
		for i := 0; i < n; i++ {
			var sum float32
			for _, st := range states {
				sum += d.rows(st)[i]
			}
			dst[off+i] = sum / float32(max(1, len(states)))
		}
	}
	return nil
}

func (d *fakeInk2FP8Prefill) Ink2MatVec(w compute.Tensor, x, y compute.Ink2Ref) error {
	rows, cols := w.Shape[0], w.Shape[1]
	wm, xr, yr := d.rows(compute.Ink2At(w, 0)), d.rows(x), d.rows(y)
	for r := 0; r < rows; r++ {
		var sum float32
		for c := 0; c < cols; c++ {
			sum += wm[r*cols+c] * xr[c]
		}
		yr[r] = sum
	}
	return nil
}

func (d *fakeInk2FP8Prefill) Ink2SoftcapArgmax(logits compute.Ink2Ref, n int, softcap float32, outIdx compute.Ink2Ref) error {
	lr := d.rows(logits)[:n]
	d.lastLogits = append([]float32(nil), lr...)
	best := 0
	for i := 1; i < n; i++ {
		if lr[i] > lr[best] {
			best = i
		}
	}
	d.rows(outIdx)[0] = math.Float32frombits(uint32(int32(best)))
	return nil
}

type fakeInk2DeterministicEmbed struct{ hidden int }

func (e fakeInk2DeterministicEmbed) Row(tok int, dst []float32) {
	for i := range dst {
		dst[i] = float32(math.Sin(float64(tok*13+i*7+1))) * 0.5
	}
}

// fak-test:runtime fast est=200ms
func TestInkling2PrefillArenaFP8(t *testing.T) {
	const hidden = 16
	const vocab = 8
	hp := model.Inkling2LoopHParams{
		Hidden: hidden, Layers: 0, Heads: 2, KVHeads: 2, HeadDim: 4,
		ConvKernel: 3, ExpertsUsed: 2, ExpertWidth: 8, RelLength: 4,
		VocabUnpadded: vocab, SlidingWindow: 64, NormEps: 1e-5,
	}
	outData := make([]float32, vocab*hidden)
	for r := 0; r < vocab; r++ {
		for c := 0; c < hidden; c++ {
			outData[r*hidden+c] = float32(math.Cos(float64((r+1)*(c+3)))) * 0.4
		}
	}
	m := &Inkling2LoopModel{
		HP: hp,
		Weights: &model.Inkling2LoopWeights{
			TokEmbd:    fakeInk2DeterministicEmbed{hidden: hidden},
			TokNorm:    make([]float32, hidden),
			OutputNorm: make([]float32, hidden),
			Output:     &model.Inkling2LoopDenseMatrix{R: vocab, C: hidden, Data: outData},
		},
	}
	for i := 0; i < hidden; i++ {
		m.Weights.TokNorm[i] = 1.0
		m.Weights.OutputNorm[i] = 1.0
	}

	prompt := []int{1, 3, 5, 7, 2, 4, 6, 0}

	runArm := func(fp8Env, arenaEnv string) (int, []float32, int, int, int) {
		t.Setenv("FAK_INKLING2_PREFILL_FP8", fp8Env)
		t.Setenv("FAK_INKLING2_PREFILL_ARENA", arenaEnv)
		be := newFakeInk2FP8Prefill()
		s, err := newInkling2ResidentRing(m, be, 64, 64, 64, 64)
		if err != nil {
			t.Fatalf("newInkling2ResidentRing(fp8=%s, arena=%s): %v", fp8Env, arenaEnv, err)
		}
		defer s.close()
		tok, err := s.prefill(context.Background(), prompt)
		if err != nil {
			t.Fatalf("prefill(fp8=%s, arena=%s): %v", fp8Env, arenaEnv, err)
		}
		keepFloats, _ := s.prefillArenas(64)
		return tok, be.lastLogits, be.quantCalls, be.dequantCalls, keepFloats
	}

	tokF32, logitsF32, qF32, dqF32, keepF32 := runArm("0", "")
	tokFP8, logitsFP8, qFP8, dqFP8, keepFP8 := runArm("1", "")
	tokArenaF32, _, qArenaF32, dqArenaF32, keepArenaF32 := runArm("1", "f32")
	tokArenaFP8, _, qArenaFP8, dqArenaFP8, keepArenaFP8 := runArm("0", "fp8")

	if qF32 != 0 || dqF32 != 0 || qArenaF32 != 0 || dqArenaF32 != 0 || keepArenaF32 != keepF32 || tokArenaF32 != tokF32 {
		t.Fatalf("F32 arm ran FP8 ops or mismatched: q=(%d,%d) dq=(%d,%d) keep=(%d,%d) tok=(%d,%d)",
			qF32, qArenaF32, dqF32, dqArenaF32, keepF32, keepArenaF32, tokF32, tokArenaF32)
	}
	if qFP8 == 0 || dqFP8 == 0 || qArenaFP8 == 0 || dqArenaFP8 == 0 || keepArenaFP8 != keepFP8 || tokArenaFP8 != tokFP8 {
		t.Fatalf("FP8 arm did not exercise FP8 ops or mismatched: q=(%d,%d) dq=(%d,%d) keep=(%d,%d) tok=(%d,%d)",
			qFP8, qArenaFP8, dqFP8, dqArenaFP8, keepFP8, keepArenaFP8, tokFP8, tokArenaFP8)
	}
	if keepFP8 >= keepF32/2 {
		t.Fatalf("FP8 keep floats %d not significantly smaller than F32 %d", keepFP8, keepF32)
	}
	if tokFP8 != tokF32 {
		t.Fatalf("greedy token mismatch: FP8=%d, F32=%d (logitsFP8=%v, logitsF32=%v)",
			tokFP8, tokF32, logitsFP8, logitsF32)
	}

	// Compute KL(p_F32 || p_FP8) and assert <= 0.0015.
	softmax := func(x []float32) []float64 {
		out := make([]float64, len(x))
		mx := float64(x[0])
		for _, v := range x[1:] {
			if float64(v) > mx {
				mx = float64(v)
			}
		}
		var sum float64
		for i, v := range x {
			out[i] = math.Exp(float64(v) - mx)
			sum += out[i]
		}
		for i := range out {
			out[i] /= sum
		}
		return out
	}
	p := softmax(logitsF32)
	q := softmax(logitsFP8)
	var kl float64
	for i := range p {
		if p[i] > 0 && q[i] > 0 {
			kl += p[i] * math.Log(p[i]/q[i])
		}
	}
	if kl > 0.0015 {
		t.Fatalf("KL(F32 || FP8) = %.6f > 0.0015", kl)
	}
	t.Logf("greedy token=%d match, KL=%.6f, keepFloats F32=%d -> FP8=%d (quant=%d, dequant=%d)",
		tokFP8, kl, keepF32, keepFP8, qFP8, dqFP8)
}
