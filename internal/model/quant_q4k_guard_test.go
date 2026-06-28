package model

// quant_q4k_guard_test.go — the #1067 guardrail gate. metalQ4KWeight can free a Q4_K weight's
// CPU copy (qt.raw) after a Metal upload under FAK_Q4K_FREE_CPU=1 for single residency. Decode
// then takes the Metal GEMV, but the batched prefill GEMM stays on the CPU and reads qt.raw — so
// a multi-thousand-token prompt reached a nil raw and died with a cryptic "slice bounds out of
// range [N:0]" deep in a parFor worker. requireRawCPU turns that into a legible failure naming the
// misconfig. These tests pin: (a) a freed weight panics legibly from BOTH CPU entry points
// (decode GEMV + prefill GEMM), (b) the message names the op and #1067, (c) a degenerate empty
// tensor is a no-op (no false positive), and (d) a resident weight runs unguarded. No build tags
// or device: the guardrail is the arch-independent CPU path, so this gates on every box.

import (
	"strings"
	"testing"
)

// freedQ4KTensor is a Q4_K weight whose CPU bytes were dropped after a Metal upload: out>0 (it
// holds real rows) but raw==nil. This is exactly the state metalQ4KWeight leaves under
// FAK_Q4K_FREE_CPU=1, and the state the CPU GEMM/GEMV must refuse instead of slice-panicking.
func freedQ4KTensor(out, in int) *q4kTensor {
	return &q4kTensor{out: out, in: in, nblk: in / qkK, raw: nil}
}

// residentQ4KTensor builds an [out,in] Q4_K weight with a full resident raw buffer — any byte
// pattern is a valid block (the dequant is total), so the guardrail must let it through.
func residentQ4KTensor(out, in int) *q4kTensor {
	nblk := in / qkK
	raw := make([]byte, out*nblk*q4kBlockBytes)
	for i := range raw {
		raw[i] = byte(i * 31)
	}
	// Keep each super-block's f16 scales (d, dmin) in a sane finite range so the f32 dot stays
	// finite — mirrors randomQ4KTensor's clamp; not load-bearing for the guardrail, just hygiene.
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			base := (o*nblk + b) * q4kBlockBytes
			raw[base+1] = 0x2C | (raw[base+1] & 0x03)
			raw[base+3] = 0x2C | (raw[base+3] & 0x03)
		}
	}
	return &q4kTensor{out: out, in: in, nblk: nblk, raw: raw}
}

// recoverContains runs fn and returns the recovered panic message (empty string if fn did not
// panic), so a test can assert both that a panic fired and what it said.
func recoverContains(fn func()) (msg string) {
	defer func() {
		if r := recover(); r != nil {
			if s, ok := r.(string); ok {
				msg = s
			} else if e, ok := r.(error); ok {
				msg = e.Error()
			}
		}
	}()
	fn()
	return ""
}

func TestRequireRawCPU_FreedDecodeGEMVPanicsLegibly(t *testing.T) {
	qt := freedQ4KTensor(8, 256)
	x := make([]float32, qt.in)
	msg := recoverContains(func() { q4kMatRows(qt, x) })
	if msg == "" {
		t.Fatal("decode GEMV on a freed Q4_K weight did not panic — the #1067 guardrail is not wired")
	}
	for _, want := range []string{"freed CPU weight", "decode GEMV", "#1067"} {
		if !strings.Contains(msg, want) {
			t.Errorf("decode panic message missing %q; got: %s", want, msg)
		}
	}
	if strings.Contains(msg, "slice bounds out of range") {
		t.Errorf("guardrail should pre-empt the cryptic slice panic, got: %s", msg)
	}
}

func TestRequireRawCPU_FreedPrefillGEMMPanicsLegibly(t *testing.T) {
	qt := freedQ4KTensor(8, 256)
	const P = 4 // a multi-token prefill batch — the exact path that crashed on an ~8K prompt
	X := make([]float32, P*qt.in)
	msg := recoverContains(func() { q4kGemm(qt, X, P) })
	if msg == "" {
		t.Fatal("prefill GEMM on a freed Q4_K weight did not panic — the #1067 guardrail is not wired")
	}
	for _, want := range []string{"freed CPU weight", "prefill GEMM", "#1067"} {
		if !strings.Contains(msg, want) {
			t.Errorf("prefill panic message missing %q; got: %s", want, msg)
		}
	}
}

func TestRequireRawCPU_EmptyTensorIsNoOp(t *testing.T) {
	// A degenerate out==0 tensor holds no rows to read either way, so the guardrail must NOT fire
	// (out>0 is what distinguishes a real-but-freed weight from an empty one).
	qt := &q4kTensor{out: 0, in: 256, nblk: 1, raw: nil}
	if msg := recoverContains(func() { qt.requireRawCPU("decode GEMV") }); msg != "" {
		t.Errorf("requireRawCPU panicked on a degenerate empty tensor: %s", msg)
	}
}

func TestRequireRawCPU_ResidentWeightRunsUnguarded(t *testing.T) {
	// A weight with its raw bytes present must pass the guardrail and produce correctly-sized
	// output from both CPU entry points — the guardrail must not false-positive on a live weight.
	qt := residentQ4KTensor(8, 256)
	if msg := recoverContains(func() {
		if y := q4kMatRows(qt, make([]float32, qt.in)); len(y) != qt.out {
			t.Errorf("decode GEMV returned %d rows, want %d", len(y), qt.out)
		}
	}); msg != "" {
		t.Errorf("decode GEMV panicked on a resident weight: %s", msg)
	}
	const P = 4
	if msg := recoverContains(func() {
		if y := q4kGemm(qt, make([]float32, P*qt.in), P); len(y) != P*qt.out {
			t.Errorf("prefill GEMM returned %d entries, want %d", len(y), P*qt.out)
		}
	}); msg != "" {
		t.Errorf("prefill GEMM panicked on a resident weight: %s", msg)
	}
}
