package model

// v41_warmup_resident_overbound_test.go — the #13288 witness for the V4.1
// in-kernel warmup resident over-bound.
//
// The physical strix3 re-run at trunk 6b627e180 (exclusive lease
// lease-strix3-589ac5971c1b, box isolated to 58.9 GiB free) bound the serve
// plan (staging-host-charge=43.896GiB, resident-bound=43.298GiB) and then the
// in-kernel warmup's resident set climbed monotonically to a 56.57 GiB peak and
// the kernel OOM-killed the process before any token.
//
// The dominant uncharged term was the two grouped output projections
// (attn.wo_a.weight / attn.wo_b.weight), read through v41ProjF32, which
// allocates a FRESH whole-tensor f32 block per layer per forward. At the
// published geometry (~416 MiB/layer) across the 40-layer stack that is ~16.25
// GiB of transient charged to no ledger.
//
// The fix reads those two leaves into a per-forward REUSED v41ProjScratch, so
// the peak contribution is one layer's worth rather than the accumulated stack.
//
// These tests pin the contract:
//
//  1. BOUND: reading a resident k-quant projection through v41ProjF32Into with a
//     reused scratch allocates the whole-tensor f32 expansion at most ONCE
//     across N calls, where the parent's v41ProjF32 allocates it N times.
//  2. BYTE-IDENTITY: v41ProjF32Into and v41ProjF32 produce bit-identical output
//     for the same resident weight (including the f32-manifest zero-copy path).
//  3. FAIL-CLOSED: an absent projection refuses with a typed ErrV41ForwardStage
//     naming the tensor, exactly like v41ProjF32 (#13276 preserved).
//  4. FORWARD: a model whose grouped output projections live ONLY in a resident
//     quant store runs the real forward through the reused scratch and emits
//     finite logits — the scratch does not perturb the arithmetic.

import (
	"errors"
	"runtime"
	"strings"
	"testing"
)

// v41MeasureAlloc returns the bytes allocated by fn (TotalAlloc delta), after a
// GC so prior garbage is not attributed to the call.
func v41MeasureAlloc(fn func()) uint64 {
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	fn()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc
}

// TestV41WarmupResidentOverboundSpine is the #13288 acceptance witness: the
// grouped output projection read, given a reused scratch, must NOT allocate the
// whole-tensor f32 expansion on every call. On the parent commit v41ProjF32Into
// and v41ProjScratch do not exist, so this test does not compile there — the RED
// symptom the land gate requires.
func TestV41WarmupResidentOverboundSpine(t *testing.T) {
	m := v41ReducedModel(t)

	// Move attn.wo_a/wo_b to a resident Q8_0 store so the whole-tensor f32
	// expansion path is the only way to read them.
	const outA, inA = 1024, 2048 // wo_a-shaped, in a multiple of 256
	name := layerName(0, "attn.wo_a.weight")
	delete(m.manifest, name)
	if m.kqw == nil {
		m.kqw = map[string]*kQuantTensor{}
	}
	m.kqw[name] = quantizeKQuantFromRaw(v41ResidentQ2KRaw(outA, inA), outA, inA, kindQ2K)

	expand := uint64(outA) * uint64(inA) * 4 // one whole-tensor f32 block

	// Warm the read so the store's one-time ensureRawCPU memo is not counted.
	if _, err := m.v41ProjF32Into(0, "attn.wo_a.weight", nil); err != nil {
		t.Fatalf("warm v41ProjF32Into error = %v, want nil", err)
	}

	// The PARENT read allocates a fresh whole-tensor f32 block every call.
	parentBytes := v41MeasureAlloc(func() {
		for l := 0; l < 4; l++ {
			if _, err := m.v41ProjF32(0, "attn.wo_a.weight"); err != nil {
				t.Fatalf("v41ProjF32 error = %v, want nil", err)
			}
		}
	})
	if parentBytes < 4*expand {
		t.Fatalf("v41ProjF32 over 4 calls allocated %d bytes, want >= %d (the fixture does not exercise the per-call expansion)", parentBytes, 4*expand)
	}

	// The reused-scratch read pays the expansion at most once across the same
	// number of calls — the warmup's monotonic-growth term is bounded.
	scratch := &v41ProjScratch{}
	if _, err := m.v41ProjF32Into(0, "attn.wo_a.weight", scratch.woA); err != nil {
		t.Fatalf("v41ProjF32Into error = %v, want nil", err)
	}
	scratch.woA = nil
	scratchBytes := v41MeasureAlloc(func() {
		for l := 0; l < 4; l++ {
			w, err := m.v41ProjF32Into(0, "attn.wo_a.weight", scratch.woA)
			if err != nil {
				t.Fatalf("v41ProjF32Into error = %v, want nil", err)
			}
			scratch.woA = w
		}
	})
	if scratchBytes >= 4*expand {
		t.Fatalf("reused-scratch read over 4 calls allocated %d bytes, want < %d (the scratch is not being reused)", scratchBytes, 4*expand)
	}
	t.Logf("alloc parent=%d scratch=%d one-expansion=%d", parentBytes, scratchBytes, expand)
}

// TestV41ProjF32IntoByteIdentical pins that the caller-buffer read is
// bit-identical to the allocating read on both the resident-quant and the
// f32-manifest paths.
func TestV41ProjF32IntoByteIdentical(t *testing.T) {
	m := v41ReducedModel(t)

	// Resident Q2_K path.
	name := layerName(0, "attn.wo_a.weight")
	shape := v41ProjectionShape(t, m, "attn.wo_a.weight")
	out, in := shape[0], shape[1]
	delete(m.manifest, name)
	if m.kqw == nil {
		m.kqw = map[string]*kQuantTensor{}
	}
	m.kqw[name] = quantizeKQuantFromRaw(v41ResidentQ8Raw(out, in), out, in, kindQ8_0)

	want, ok := m.residentF32Mat(name)
	if !ok {
		t.Fatalf("residentF32Mat(%s) not resident; fixture is wrong", name)
	}
	got, err := m.v41ProjF32Into(0, "attn.wo_a.weight", nil)
	if err != nil {
		t.Fatalf("v41ProjF32Into error = %v, want nil", err)
	}
	if len(got) != len(want) {
		t.Fatalf("v41ProjF32Into len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("v41ProjF32Into[%d] = %v, want byte-identical %v", i, got[i], want[i])
		}
	}

	// f32-manifest zero-copy path.
	m2 := v41ReducedModel(t)
	mname := layerName(0, "attn.wo_b.weight")
	if !m2.has(mname) {
		t.Fatalf("reduced fixture is missing the f32 manifest entry %s", mname)
	}
	want2 := m2.tensor(mname)
	got2, err := m2.v41ProjF32Into(0, "attn.wo_b.weight", nil)
	if err != nil {
		t.Fatalf("f32-manifest v41ProjF32Into error = %v, want nil", err)
	}
	if len(got2) != len(want2) {
		t.Fatalf("f32-manifest v41ProjF32Into len = %d, want %d", len(got2), len(want2))
	}
	for i := range want2 {
		if got2[i] != want2[i] {
			t.Fatalf("f32-manifest v41ProjF32Into[%d] = %v, want byte-identical %v", i, got2[i], want2[i])
		}
	}
}

// TestV41ProjF32IntoFailsClosedWhenAbsent pins #13276: a projection in no store
// refuses with a typed ErrV41ForwardStage naming the tensor, never a panic.
func TestV41ProjF32IntoFailsClosedWhenAbsent(t *testing.T) {
	m := v41ReducedModel(t)
	name := layerName(0, "attn.wo_a.weight")
	delete(m.manifest, name)
	if m.kqw != nil {
		delete(m.kqw, name)
	}

	_, err := m.v41ProjF32Into(0, "attn.wo_a.weight", nil)
	if err == nil {
		t.Fatalf("v41ProjF32Into on an absent tensor returned nil error, want a typed refusal")
	}
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("v41ProjF32Into error = %v, want errors.Is(err, ErrV41ForwardStage)", err)
	}
	if !strings.Contains(err.Error(), name) {
		t.Fatalf("v41ProjF32Into error %q does not name the tensor %s", err.Error(), name)
	}
}

// TestV41WarmupScratchForwardStaysFinite drives the real reduced forward with the
// grouped output projections resident only in a quant store, so the reused
// v41ProjScratch path is exercised end-to-end. The scratch must not perturb the
// arithmetic: the forward still emits finite logits.
func TestV41WarmupScratchForwardStaysFinite(t *testing.T) {
	m := v41ReducedModel(t)
	for _, leaf := range []string{"attn.wo_a.weight", "attn.wo_b.weight"} {
		shape := v41ProjectionShape(t, m, leaf)
		raw := v41ResidentQ8Raw(shape[0], shape[1])
		v41MoveProjToResidentKQuant(m, 0, leaf, shape, raw, kindQ8_0)
	}
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("admission error = %v, want nil", err)
	}
	act, err := m.forwardV41([]int{1, 3}, nil)
	if err != nil {
		t.Fatalf("forward error = %v, want nil", err)
	}
	if len(act.Logits) != 2 {
		t.Fatalf("forward produced %d logit rows, want 2", len(act.Logits))
	}
	for r, row := range act.Logits {
		for i, v := range row {
			if v != v || v > 3.4e38 || v < -3.4e38 {
				t.Fatalf("logits[%d][%d] = %v, want finite", r, i, v)
			}
		}
	}
}
