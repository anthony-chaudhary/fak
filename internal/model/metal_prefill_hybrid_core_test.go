package model

import (
	"io"
	"os"
	"strings"
	"testing"
)

// TestQwen35HybridViaMMMatchesCPUTemplate is the host-independent correctness witness for the
// backend-agnostic hybrid prefill core (prefillQwen35HybridViaMM). The Metal twin
// (prefillBatchedMetalQwen35Hybrid, Apple Silicon+cgo) is a thin wrapper that feeds this same core a
// GPU f16 GEMM, so its CPU-side logic — the conv1d+SiLU mixer, the q/k L2-norm, the delta-rule
// recurrent scan, the gated RMSNorm readout, the full-attention RoPE/GQA/output-gate, both
// RMSNorms and every residual — IS this file's, and is provable WITHOUT a Mac Metal runtime:
// drive the core with a CPU mm that reproduces the proven prefillQwen35HybridQHidden path's
// per-projection qGemm8 and assert the whole prefill (logits + KV cache + linear-attn cache)
// matches that proven path.
//
// This catches the exact bug class the Metal lane is otherwise blind to off-device: a transcription
// error in the recurrence/attention/orchestration when the twin was hand-copied from the CPU
// template. Such an error diverges O(1) per layer and blows past the close-helper tolerances; the
// only residual under those tolerances is the documented grouped-vs-ungrouped Q8 GEMM float-order
// drift (qGemm8IntoMany in the template vs per-call qGemm8Into here), which is ~1e-6. What this
// does NOT witness — the GPU f16 GEMM numerics and on-device throughput — is the irreducibly
// Mac-gated residual that closes #71 (the on-device parity run on the M3 Pro).
func TestQwen35HybridViaMMMatchesCPUTemplate(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	m.Quantize()
	// 16 tokens meets qwen35HybridQBatchMinPrompt — the same prompt the batched-prefill gates use.
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}

	// Reference: the proven Q8 CPU hybrid prefill (prefillQwen35HybridQHidden), the template the
	// twin structurally copies.
	ref := m.NewSession()
	ref.Quant = true
	want := ref.headQ(ref.prefillQwen35HybridQHidden(prompt))

	// Under test: the backend-agnostic core fed a CPU mm. The mm reproduces the template's
	// per-projection math exactly — quantize the activation panel, then qGemm8Into against the
	// same m.q8 weight — so a faithful core is numerically identical to the template up to the
	// grouped-vs-ungrouped GEMM float-order drift the close-helpers tolerate.
	got := m.NewSession()
	got.Quant = true
	P := len(prompt)
	cpuMM := func(name string, X []float32, out int) []float32 {
		width := len(X) / P
		var panel q8Panel
		quantizeBatchPanelInto(&panel, X, P, width)
		Y := make([]float32, P*out)
		qGemm8Into(got.M.q8(name), &panel, Y)
		return Y
	}
	gotLogits := got.headQ(got.prefillQwen35HybridViaMM(prompt, cpuMM))

	assertQuantLogitsClose(t, "hybrid via-mm core vs CPU template logits", want, gotLogits)
	assertKVCacheQuantClose(t, "hybrid via-mm core vs CPU template", ref.Cache, got.Cache)
	assertLinearAttnCacheQuantClose(t, "hybrid via-mm core vs CPU template", ref.Cache.linear, got.Cache.linear)
}

func TestQwen35HybridViaMMProfilePrintsHybridSplit(t *testing.T) {
	t.Setenv("FAK_QPROFILE", "1")

	oldStderr := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	os.Stderr = w
	writerClosed := false
	defer func() {
		os.Stderr = oldStderr
		if !writerClosed {
			_ = w.Close()
		}
	}()

	out := make(chan string, 1)
	go func() {
		var b strings.Builder
		_, _ = io.Copy(&b, r)
		out <- b.String()
	}()

	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	s := m.NewSession()
	prompt := []int{3, 7, 11}
	P := len(prompt)
	zeroMM := func(name string, X []float32, out int) []float32 {
		return make([]float32, P*out)
	}
	s.prefillQwen35HybridViaMM(prompt, zeroMM)

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	writerClosed = true
	got := <-out
	for _, want := range []string{
		"[metalprof-hybrid P=3]",
		"total=",
		"gemm+roundtrip=",
		"gdn-recurrence=",
		"full-attn=",
		"qk-norm=",
		"norm+act=",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("profile output %q missing %q", got, want)
		}
	}
}

// TestQwen35HybridPrefillIsolationGatesZeroTheStage witnesses the #2725 isolation split: with
// FAK_PREFILL_NO_GDN set, the Gated-DeltaNet recurrence layers are skipped, so the profile's
// gdn-recurrence stage reads exactly 0.0 ms — the operator's on-device lever for measuring a
// stage's share of the prefill wall (mirroring #67's FAK_DECODE_NO_ATTN). It also confirms the
// gate genuinely changes the forward (skipping the recurrence perturbs the logits), so the gate is
// wired to the real body, not a no-op. This runs host-independently; the wall-clock ladder the gate
// feeds is the Mac-gated residual documented in docs/notes.
func TestQwen35HybridPrefillIsolationGatesZeroTheStage(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	m.Quantize()
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}
	P := len(prompt)
	// The real Q8 CPU GEMM (as TestQwen35HybridViaMMMatchesCPUTemplate uses) so the recurrence has a
	// non-trivial contribution to zero out — an all-zero mm would make every stage's output zero and
	// hide whether the gate is wired.
	cpuMM := func(name string, X []float32, out int) []float32 {
		width := len(X) / P
		var panel q8Panel
		quantizeBatchPanelInto(&panel, X, P, width)
		Y := make([]float32, P*out)
		qGemm8Into(m.q8(name), &panel, Y)
		return Y
	}

	// Baseline forward (no gate) — the recurrence contributes.
	base := func() []float32 {
		s := m.NewSession()
		s.Quant = true
		return s.prefillQwen35HybridViaMM(prompt, cpuMM)
	}()

	// With NO_GDN the recurrence layers are dropped: the profile's gdn-recurrence stage is 0.0, and
	// the output must differ from the baseline (the gate reaches the real body).
	t.Setenv("FAK_QPROFILE", "1")
	t.Setenv("FAK_PREFILL_NO_GDN", "1")
	out := captureStderr(t, func() {
		gs := m.NewSession()
		gs.Quant = true
		gated := gs.prefillQwen35HybridViaMM(prompt, cpuMM)
		if len(gated) != len(base) {
			t.Fatalf("gated hidden len %d != baseline %d", len(gated), len(base))
		}
		same := true
		for i := range base {
			if base[i] != gated[i] {
				same = false
				break
			}
		}
		if same {
			t.Fatal("FAK_PREFILL_NO_GDN did not change the forward — the isolation gate is not wired to the GDN body")
		}
	})
	if !strings.Contains(out, "gdn-recurrence=0.0") {
		t.Fatalf("profile %q missing zeroed stage %q under FAK_PREFILL_NO_GDN", out, "gdn-recurrence=0.0")
	}
}

// captureStderr runs fn with os.Stderr redirected to a pipe and returns everything it wrote.
func captureStderr(t *testing.T, fn func()) string {
	t.Helper()
	oldStderr := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	os.Stderr = w
	out := make(chan string, 1)
	go func() {
		var b strings.Builder
		_, _ = io.Copy(&b, r)
		out <- b.String()
	}()
	fn()
	if err := w.Close(); err != nil {
		os.Stderr = oldStderr
		t.Fatal(err)
	}
	os.Stderr = oldStderr
	return <-out
}
