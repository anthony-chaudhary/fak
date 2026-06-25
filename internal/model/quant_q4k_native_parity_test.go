package model

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// TestQ4KNativeMatchesGGUFToQ8Reference is the acceptance gate #439 calls for: it pins the
// resident NATIVE Q4_K forward against the current GGUF->Q8 reference forward on a tiny
// Qwen3.5/Qwen3.6 hybrid fixture, with argmax (next-token id) parity AND a cosine-similarity
// floor on the logits, BEFORE the path is trusted on the 27B (where cmd/q4kdiag runs the
// 248068 first-token check). The two paths are built from the SAME source weights so the only
// difference under test is the residency lever the issue introduces — the identity-normalized
// matmul majority held as raw Q4_K (dequantized on the fly to f32) vs expanded to Q8_0:
//
//   - For every eligible-majority projection (mlp gate/up/down on every layer, self_attn
//     v_proj/o_proj on full-attn layers — exactly ResidentQ4KEligible / fillQ4KMajority), we
//     synthesize a finite Q4_K payload, dequantize it into the model's f32 manifest, and keep
//     the raw bytes. So the model's f32 weight for each majority tensor IS dequant(rawQ4K).
//   - The GGUF->Q8 reference (sRef, Quant=true) then runs Quantize() over that f32 — i.e. it
//     expands the SAME q4_k_m source to Q8_0, the exact path #439 is removing. Its body
//     matmuls dispatch through sessionQ8Kernel.
//   - The native path (sNat, Q4K=true) holds those majority tensors as raw Q4_K (q4kw) and
//     dispatches them through sessionQ4KKernel's resident GEMV; the normalize-sensitive
//     minority (self_attn q/k, the whole linear_attn.* family) and the tied LM head stay Q8
//     in BOTH sessions, so the logit delta is attributable solely to the majority residency.
//
// The native path is strictly the more faithful of the two (no Q8 weight-quantization error
// on the majority, f32 activation into those GEMVs), so a high cosine + argmax agreement is
// the right shape of gate: it proves dropping the Q8 expansion does not move the next token,
// while a real wiring bug (wrong store, transposed/mis-laid weights, a dispatch mismatch)
// would collapse the cosine and flip the argmax.
func TestQ4KNativeMatchesGGUFToQ8Reference(t *testing.T) {
	// Pin the f32 resident-Q4_K GEMV (q4kMatRowsRange) so this gate is deterministic and
	// arch-stable: the default int8 SDOT decode path is on for arm64 and off elsewhere, which
	// would make the measured cosine arch-dependent. The production int8 path is held to this
	// same f32 path bit-for-bit by TestQ4KInt8DotMatchesF32 / TestQ4KGemmInt8MatchesMatRowsInt8,
	// so gating the f32 native path against the Q8 reference here transitively covers the int8
	// path the M3 Pro actually decodes with. Same scoping knob the sibling Q4_K parity tests use.
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)

	// The eligible-majority projections, same set ResidentQ4KEligible holds raw and
	// fillQ4KMajority populates: identity-normalized matmul weights only.
	type proj struct {
		name string
		out  int
	}
	var projs []proj
	nKVhd := cfg.NumKVHeads * cfg.HeadDim
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		projs = append(projs,
			proj{p + "mlp.gate_proj.weight", cfg.IntermediateSize},
			proj{p + "mlp.up_proj.weight", cfg.IntermediateSize},
			proj{p + "mlp.down_proj.weight", cfg.HiddenSize},
		)
		if !cfg.isLinearAttnLayer(l) {
			projs = append(projs,
				proj{p + "self_attn.v_proj.weight", nKVhd},
				proj{p + "self_attn.o_proj.weight", cfg.HiddenSize},
			)
		}
	}

	// Build one shared q4kw store AND rewrite the f32 manifest so the GGUF->Q8 reference
	// quantizes the IDENTICAL source weights the native path holds raw. randQ4KBlockBounded
	// keeps d/min small so the dequanted weights sit in the ~0.1 band NewSynthetic uses and
	// the multi-layer forward stays finite.
	rng := rand.New(rand.NewSource(20260625))
	m.q4kw = make(map[string]*q4kTensor, len(projs))
	blk := make([]byte, q4kBlockBytes)
	for _, pr := range projs {
		meta, ok := m.manifest[pr.name]
		if !ok {
			t.Fatalf("manifest missing %s", pr.name)
		}
		in := meta.Shape[len(meta.Shape)-1]
		if in%qkK != 0 {
			t.Fatalf("%s reduction dim %d not a multiple of %d", pr.name, in, qkK)
		}
		nblk := in / qkK
		raw := make([]byte, pr.out*nblk*q4kBlockBytes)
		for i := 0; i < pr.out*nblk; i++ {
			randQ4KBlockBounded(rng, blk, 2, 6)
			copy(raw[i*q4kBlockBytes:(i+1)*q4kBlockBytes], blk)
		}
		// Dequantize the raw Q4_K straight into the model's f32 bytes so f32 == dequant(raw):
		// this is the GGUF->Q8 reference's source, and the native path dequants the same raw.
		f32 := make([]float32, pr.out*in)
		dequantQ4KRef(f32, raw)
		for i, v := range f32 {
			binary.LittleEndian.PutUint32(m.raw[meta.Offset+i*4:], math.Float32bits(v))
		}
		m.q4kw[pr.name] = quantizeQ4KFromRaw(raw, pr.out, in)
	}

	// GGUF->Q8 reference: expand every matmul weight (incl. the now-Q4K-consistent majority)
	// to Q8_0. This must run AFTER the f32 rewrite so the Q8 codes encode dequant(raw).
	m.Quantize()

	prompt := []int{5, 11, 23, 7, 41, 3, 59, 17, 31, 47, 13, 29, 53, 19, 61, 37}
	for _, id := range prompt {
		if id >= cfg.VocabSize {
			t.Fatalf("prompt id %d >= vocab %d", id, cfg.VocabSize)
		}
	}

	// Reference: Quant=true -> sessionQ8Kernel body + headQ (the GGUF->Q8 path).
	sRef := m.NewSession()
	sRef.Quant = true
	var wantLogits []float32
	for pos, id := range prompt {
		wantLogits = sRef.token(id, pos)
	}

	// Native: Q4K=true -> sessionQ4KKernel (raw Q4_K majority + Q8 minority) + headResident.
	sNat := m.NewSession()
	sNat.Q4K = true
	var gotLogits []float32
	for pos, id := range prompt {
		gotLogits = sNat.token(id, pos)
	}

	if len(wantLogits) != cfg.VocabSize || len(gotLogits) != cfg.VocabSize {
		t.Fatalf("logit lengths ref=%d native=%d, want %d", len(wantLogits), len(gotLogits), cfg.VocabSize)
	}

	cos := cosine(wantLogits, gotLogits)
	amRef, amNat := argmax(wantLogits), argmax(gotLogits)
	t.Logf("native-q4 vs GGUF->Q8: cosine=%.6f argmax ref=%d native=%d", cos, amRef, amNat)

	// Cosine floor: with the f32 native GEMV pinned the comparison is deterministic, and the
	// only logit-moving difference is the majority's Q8 weight-quantization error in the
	// reference (the native path keeps it exact-from-raw), which is small over a 4-layer hybrid
	// forward — measured cosine 0.99947 at this fixture. 0.999 leaves margin for cross-arch f32
	// rounding while still collapsing on any real layout/dispatch bug (a transposed weight or
	// wrong store drops cosine well below 0.9 and flips the argmax).
	if cos < 0.999 {
		t.Fatalf("native-q4 vs GGUF->Q8 cosine %.6f < 0.999 — residency path diverges from the Q8 reference", cos)
	}
	// Next-token argmax must agree: dropping the Q8 expansion may not change which token the
	// model would emit. (Both sessions share the identical Q8 LM head, so a flip here is a body
	// divergence, not head noise.)
	if amRef != amNat {
		t.Fatalf("native-q4 argmax %d != GGUF->Q8 argmax %d (cosine %.6f)", amNat, amRef, cos)
	}
}
