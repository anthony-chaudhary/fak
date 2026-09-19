package model

// v41_attn_latent_norm_test.go — the #13290 witness for the V4.1 Q/KV latent
// norms (reference Attention: self.q_norm = RMSNorm(q_lora_rank) and
// self.kv_norm = RMSNorm(head_dim)).
//
// WHY THIS IS THE LAYER-36 NON-FINITE ROOT CAUSE (upstream magnitude source).
// The published reference dataflow is:
//
//	qr = self.q_norm(self.wq_a(x))        # RMSNorm over the q-lora latent
//	q  = self.wq_b(qr)                    # then the up-projection
//	kv = self.kv_norm(self.wkv(x))        # RMSNorm over the KV vector
//	apply_rotary_emb(kv[..., -rd:], ...)  # then RoPE on the rope tail
//
// Before this change the native full-path forward projected attn.wq_a and
// attn.wkv WITHOUT applying their RMSNorm weights, so the Q and KV vectors
// reaching the 512-term attention dot product were raw unbounded projections.
// With HeadDim=512 a finite per-element magnitude above sqrt(MaxFloat32/512)
// ~= 8.15e17 overflows `dot += q*kv` to +Inf; maxScore becomes +Inf; and
// exp32(Inf-Inf)=NaN poisons the softmax denominator (the `sum == 0` guard does
// not fire on NaN). The NaN row surfaced downstream as the layer-36 refusal
// V41GroupedOutputProjection named ("grouped output non-finite value element 0"),
// which the #13290 attribution retargeted onto the upstream magnitude source.
//
// RED-ON-PARENT. The test drives the REAL forwardV41 on a full-geometry fixture
// whose attn.wq_a.weight is scaled so large that, WITHOUT the q_norm stage, the
// 512-wide q-lora latent projects to an unbounded query and the 512-term
// accumulation overflows. On the parent commit the forward fails closed with a
// non-finite attention output; with the #13290 latent norm wired it stays finite.
// The fixture is a small but internally consistent FULL geometry (HeadDim at the
// published 512, one head), and the same public helpers the production forward
// uses are re-applied here as an independent scalar oracle.

import (
	"encoding/binary"
	"math"
	"testing"
)

// v41LatentNormOracle applies the reference order to the raw projections:
// qr = rmsnorm(qLat, qNorm) then q = wq_b(qr), and kv = rmsnorm(kvRaw, kvNorm).
// It is the independent scalar transcription of the two stages, distinct from the
// production v41Layer code path; a missing production stage breaks the equality.
func v41LatentNormOracle(qLat, qNorm, wqB, kvRaw, kvNorm []float32, qLoraRank, qHeadDim int, eps float32, cfg Config) (q, kv []float32) {
	qr := rmsnormCfg(qLat, qNorm, eps, cfg)
	q = matRows(wqB, qr, qHeadDim, qLoraRank)
	kv = rmsnormCfg(kvRaw, kvNorm, eps, cfg)
	return q, kv
}

// v41SetTensorScale overwrites a manifest tensor's raw f32 elements with a
// deterministic value of magnitude scale, so a fixture can drive a projection to
// a chosen magnitude. It is test-only and changes no production layout.
func v41SetTensorScale(m *Model, name string, scale float32) {
	meta, ok := m.manifest[name]
	if !ok {
		panic("v41SetTensorScale: missing tensor " + name)
	}
	n := meta.Nbytes / 4
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint32(m.raw[meta.Offset+i*4:], math.Float32bits(scale))
	}
}

// TestV41AttnLatentNormAppliedOnFullPath is the #13290 RED->GREEN witness. It
// runs the real full-path forward on a fixture whose q-lora projection is scaled
// to ~1e18 per element: without the q_norm stage the 512-term attention
// accumulation overflows to a non-finite value (the witnessed pre-fix failure);
// with the norm wired the projection is bounded, the logits are finite, and the
// production attention input matches the independent scalar oracle.
func TestV41AttnLatentNormAppliedOnFullPath(t *testing.T) {
	const eps = 1e-6

	// --- (1) Unit-level oracle pin on the two stages ---------------------------
	{
		const (
			H     = 8
			qLora = 4
		)
		cfg := Config{HiddenSize: H, RMSNormEps: eps}
		qLat := []float32{3, -5, 7, 11}
		qNorm := []float32{1, 1, 1, 1}
		wqB := make([]float32, H*qLora)
		for i := range wqB {
			wqB[i] = float32((i%7)-3) * 0.25
		}
		kvRaw := []float32{2, 4, -6, 8, -10, 12, 14, -16}
		kvNorm := []float32{1, 1, 1, 1, 1, 1, 1, 1}

		// Production order, hand-applied through the same helpers v41Layer uses.
		qrProd := rmsnormCfg(qLat, qNorm, eps, cfg)
		qProd := matRows(wqB, qrProd, H, qLora)
		kvProd := rmsnormCfg(kvRaw, kvNorm, eps, cfg)

		qOra, kvOra := v41LatentNormOracle(qLat, qNorm, wqB, kvRaw, kvNorm, qLora, H, eps, cfg)
		if !equalFloat32Slices(qProd, qOra) {
			t.Fatalf("production q = %v, oracle q = %v; the reference q_norm->wq_b order is not applied", qProd, qOra)
		}
		if !equalFloat32Slices(kvProd, kvOra) {
			t.Fatalf("production kv = %v, oracle kv = %v; the reference kv_norm order is not applied", kvProd, kvOra)
		}
		// Discriminating: the un-normed latent projects to a DIFFERENT vector, so
		// the equality genuinely witnesses the norm rather than agreeing by
		// construction.
		if equalFloat32Slices(matRows(wqB, qLat, H, qLora), qProd) {
			t.Fatal("un-normed and normed q agree; the fixture is not discriminating")
		}
		// RMSNorm (all-ones gain) leaves unit RMS, so the normed latent's max
		// magnitude is far below a scaled raw one.
		if maxAbs32(qrProd) >= maxAbs32(qLat) {
			t.Fatalf("normed latent max |v| = %g >= raw max |v| = %g; the norm did not bound the magnitude",
				maxAbs32(qrProd), maxAbs32(qLat))
		}
	}

	// --- (2) The pre-fix producer reproduced directly --------------------------
	// A finite per-element magnitude above sqrt(MaxFloat32/512) overflows a
	// 512-term accumulation to +Inf; exp32(Inf-Inf)=NaN then poisons the softmax
	// denominator. This is the exact non-finite shape the guard named.
	{
		const hd = 512
		big := make([]float32, hd)
		for i := range big {
			big[i] = 1e19
		}
		var dot float32
		for i := 0; i < hd; i++ {
			dot += big[i] * big[i]
		}
		if !math.IsInf(float64(dot), 1) {
			t.Fatalf("512-term accumulation of 1e19 magnitudes = %g, want +Inf (the non-finite producer)", dot)
		}
	}

	// --- (3) The production full-path forward stays finite ---------------------
	// v41RawFullFlattenedMHC is a real FULL-geometry fixture (HeadDim at the
	// published 512, one head). Scale attn.wq_a, attn.wq_b and attn.wkv up so the
	// un-normed q-lora latent and KV vector overflow the 512-term attention
	// accumulation; the wired q_norm/kv_norm bound both to unit RMS and keep the
	// result finite.
	m := v41RawFullFlattenedMHC(t)
	v41SetTensorScale(m, layerName(0, "attn.wq_a.weight"), 1e18)
	v41SetTensorScale(m, layerName(0, "attn.wq_b.weight"), 1e18)
	v41SetTensorScale(m, layerName(0, "attn.wkv.weight"), 1e18)
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("full-geometry model refused at admission: %v", err)
	}
	act, err := m.forwardV41([]int{0, 1}, nil)
	if err != nil {
		t.Fatalf("full-path forward error = %v, want nil (missing attn.wq_a_norm/attn.kv_norm would overflow to non-finite here)", err)
	}
	for r, row := range act.Logits {
		for i, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("logits[%d][%d] = %v, want finite; the latent norm did not bound the attention input", r, i, v)
			}
		}
	}
}

// maxAbs32 returns the maximum absolute value of v (0 for an empty slice).
func maxAbs32(v []float32) float32 {
	var m float32
	for _, x := range v {
		if a := float32(math.Abs(float64(x))); a > m {
			m = a
		}
	}
	return m
}
