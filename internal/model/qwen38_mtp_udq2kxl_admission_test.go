package model

import (
	"errors"
	"math/rand"
	"testing"
)

// qwen38MTPUDQ2KXLTestModels mirrors the exact UD-Q2_K_XL retained head on the
// Q4_K mechanism fixture's config: Q6_K/Q8_0 matrices in the quant stores, F32
// norms in the manifest and a Q4_K output head, beside a dequantized F32 oracle.
func qwen38MTPUDQ2KXLTestModels(t *testing.T) (ud, ref *Model) {
	t.Helper()
	base, _ := qwen38MTPQ4KTestModels(t)
	cfg := base.Cfg
	shapes, err := qwen35MTPExpectedShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ud = &Model{
		Cfg: cfg, manifest: map[string]tensorMeta{},
		q4kw: map[string]*q4kTensor{}, q8w: map[string]*q8Tensor{}, kqw: map[string]*kQuantTensor{},
	}
	ref = &Model{Cfg: cfg, manifest: map[string]tensorMeta{}}
	for i, name := range qwen38MTPNormTensors {
		data := make([]float32, shapes[name][0])
		for j := range data {
			data[j] = 0.75 + float32((i+j)%5)/16
		}
		appendQwen38MTPF32Tensor(ud, name, shapes[name], data)
		appendQwen38MTPF32Tensor(ref, name, shapes[name], data)
	}
	rng := rand.New(rand.NewSource(12721))
	for _, name := range qwen38MTPMatrixTensors {
		appendQwen38MTPMixedQuantTensor(t, ud, ref, name, shapes[name], qwen38MTPUDQ2KXLArtifactMatrixTypes[name], rng)
	}
	appendQwen38MTPMixedQuantTensor(t, ud, ref, "lm_head.weight", []int{cfg.VocabSize, cfg.HiddenSize}, Qwen38MTPFormatQ4K, rng)
	return ud, ref
}

// fak-test:runtime fast est=300ms lane=default
func TestQwen38MTPResidentUDQ2KXLHeadAdmitsThroughResidentLayout(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	ud, ref := qwen38MTPUDQ2KXLTestModels(t)

	layout, present, err := ud.qwen38MTPTensorLayout()
	if !present || err != nil || layout.Format != Qwen38MTPFormatQ4K {
		t.Fatalf("UD-Q2_K_XL layout = %+v present=%v err=%v; want the resident Q4_K route", layout, present, err)
	}
	for name, want := range qwen38MTPUDQ2KXLArtifactMatrixTypes {
		if _, ok := ud.manifest[name]; ok {
			t.Fatalf("fixture keeps %s in the F32 manifest; the resident route must not", name)
		}
		if got := layout.TensorTypes[name]; got != string(want) {
			t.Fatalf("%s tensor type = %q, want %q", name, got, want)
		}
	}
	mode, err := ud.Qwen35MTPMode(false)
	if err != nil || !mode.Eligible || !mode.Enabled {
		t.Fatalf("Qwen35MTPMode = %+v, %v; want eligible and enabled", mode, err)
	}
	if mode, err := ud.Qwen35MTPMode(true); err != nil || !mode.Eligible || mode.Enabled {
		t.Fatalf("disabled Qwen35MTPMode = %+v, %v; want eligible and not enabled", mode, err)
	}
	if err := ud.validateQwen35MTPForwardTensors(); err != nil {
		t.Fatalf("forward tensor validation: %v", err)
	}

	// Admission is only sound if the admitted head executes: the resident CPU
	// forward must match the dequantized oracle.
	udForward, err := ud.NewQwen35MTPForward()
	if err != nil {
		t.Fatalf("construct UD-Q2_K_XL MTP forward: %v", err)
	}
	t.Cleanup(udForward.Close)
	udForward.draft.MetalQ4K = false
	refForward, err := ref.NewQwen35MTPForward()
	if err != nil {
		t.Fatalf("construct F32 oracle MTP forward: %v", err)
	}
	t.Cleanup(refForward.Close)
	prior, embedding := qwen38MTPInputs(ud.Cfg.HiddenSize, 0)
	got, err := udForward.Forward(0, prior, embedding)
	if err != nil {
		t.Fatalf("execute UD-Q2_K_XL MTP forward: %v", err)
	}
	want, err := refForward.Forward(0, prior, embedding)
	if err != nil {
		t.Fatalf("execute dequantized oracle MTP forward: %v", err)
	}
	// Six Q6_K/Q8_0 matrices (two in the Q4_K_M head) widen the resident
	// rounding, so the bound is 1e-4 rather than that fixture's 1e-5.
	if cos := cosine(got, want); cos < 0.9999 || argmaxF32(got) != argmaxF32(want) {
		t.Fatalf("UD-Q2_K_XL MTP forward cosine=%.8f argmax=%d, oracle argmax=%d", cos, argmaxF32(got), argmaxF32(want))
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestQwen38MTPResidentUDQ2KXLIncompleteHeadStaysRefused(t *testing.T) {
	ud, _ := qwen38MTPUDQ2KXLTestModels(t)
	delete(ud.kqw, "mtp.layers.0.self_attn.q_proj.weight")
	mode, err := ud.Qwen35MTPMode(false)
	var incomplete *Qwen35MTPIncompleteError
	if !errors.As(err, &incomplete) || mode.Enabled {
		t.Fatalf("Qwen35MTPMode = %+v, %v; want *Qwen35MTPIncompleteError and not enabled", mode, err)
	}
	if err := ud.validateQwen35MTPForwardTensors(); err == nil {
		t.Fatal("forward tensor validation admitted a resident head missing q_proj")
	}
}
