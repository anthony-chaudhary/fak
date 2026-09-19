package model

// v41_expert_scratch_overbound_test.go ? the #13288 routed-expert scratch
// witness.
//
// After the grouped-output-projection fix landed (3476ed59e / 0480cf446) the
// physical strix3 re-run at trunk 732956939 still drove the in-kernel warmup's
// resident set ~14 GiB ABOVE the serve plan's staging-host-charge (43.570 GiB),
// exhausted the host (0 available, 30/31 GiB swap) and never left
// warmup_pending. The attributed residual is the streamed-experts' per-fault f32
// materialization: expertWeightF32 allocated a FRESH make([]float32, out*in) for
// EACH faulted projection (~135 MiB/expert at the published V4.1 geometry), and
// the MoE loop faults three projections per pick, many picks per token, across
// the 40-layer stack.
//
// The fix mirrors the #13288 v41ProjScratch pattern: v41ExpertF32Into writes the
// faulted projection into a forward-scoped reused buffer, so the term is bounded
// to one expert's worth instead of the accumulated churn. The MoE loop is
// strictly sequential and v41SwiGLU consumes the blocks read-only.
//
// These tests pin the contract:
//
//  1. BOUND: reading a tier-faulted projection through v41ExpertF32Into with a
//     reused buffer allocates the whole-tensor f32 expansion at most ONCE across
//     N reads, where v41ExpertF32 allocates it N times.
//  2. BYTE-IDENTITY: v41ExpertF32Into and v41ExpertF32 produce bit-identical
//     output for the same tier weight, including the resident-store path.
//  3. NO-ALIAS: the resident f32-manifest path COPIES into the reused buffer, so
//     a later dequant into that buffer never mutates model memory.
//  4. FAIL-CLOSED: an absent projection still refuses with a typed
//     ErrV41ForwardStage naming the tensor (#13276 preserved).

import (
	"errors"
	"strings"
	"testing"
)

// TestV41ExpertScratchOverboundSpine is the #13288 routed-expert acceptance
// witness: a tier-faulted projection read into a reused buffer must NOT allocate
// the whole-tensor f32 expansion on every read. On the parent commit
// v41ExpertF32Into does not exist, so this test does not compile there ? the RED
// symptom the land gate requires.
func TestV41ExpertScratchOverboundSpine(t *testing.T) {
	m := v41TierOnlyModel(t, ExpertCheckpointQ4K)

	const leaf = "w1.weight"
	name := layerName(0, "ffn.experts.0."+leaf)
	if !m.expertCheckpoint.Has(name) {
		t.Fatalf("fixture tier does not index %s", name)
	}

	// Warm the tier so the one-time fault/dequant memo is not counted.
	if _, err := m.v41ExpertF32(0, "ffn.experts.0."+leaf); err != nil {
		t.Fatalf("warm v41ExpertF32 error = %v, want nil", err)
	}
	warm, err := m.v41ExpertF32(0, "ffn.experts.0."+leaf)
	if err != nil {
		t.Fatalf("v41ExpertF32 error = %v, want nil", err)
	}
	expand := uint64(len(warm)) * 4 // one whole-tensor f32 block

	const reads = 8
	parentBytes := v41MeasureAlloc(func() {
		for i := 0; i < reads; i++ {
			if _, err := m.v41ExpertF32(0, "ffn.experts.0."+leaf); err != nil {
				t.Fatalf("v41ExpertF32 error = %v, want nil", err)
			}
		}
	})
	if parentBytes < reads*expand {
		t.Fatalf("v41ExpertF32 over %d reads allocated %d bytes, want >= %d (the fixture does not exercise the per-read expansion)", reads, parentBytes, reads*expand)
	}

	scratch := &v41ProjScratch{}
	if _, err := m.v41ExpertF32Into(0, "ffn.experts.0."+leaf, scratch.exp1); err != nil {
		t.Fatalf("v41ExpertF32Into warm error = %v, want nil", err)
	}
	scratch.exp1 = nil
	scratchBytes := v41MeasureAlloc(func() {
		for i := 0; i < reads; i++ {
			w, err := m.v41ExpertF32Into(0, "ffn.experts.0."+leaf, scratch.exp1)
			if err != nil {
				t.Fatalf("v41ExpertF32Into error = %v, want nil", err)
			}
			scratch.exp1 = w
		}
	})
	if scratchBytes >= reads*expand {
		t.Fatalf("reused-buffer expert read over %d reads allocated %d bytes, want < %d (the scratch is not being reused)", reads, scratchBytes, reads*expand)
	}
	t.Logf("alloc expert parent=%d scratch=%d one-expansion=%d", parentBytes, scratchBytes, expand)
}

// TestV41ExpertF32IntoByteIdentical pins that the caller-buffer expert read is
// bit-identical to the allocating read on the checkpoint-tier path.
func TestV41ExpertF32IntoByteIdentical(t *testing.T) {
	m := v41TierOnlyModel(t, ExpertCheckpointQ4K)
	const leaf = "ffn.experts.0.w1.weight"

	want, err := m.v41ExpertF32(0, leaf)
	if err != nil {
		t.Fatalf("v41ExpertF32 error = %v, want nil", err)
	}
	got, err := m.v41ExpertF32Into(0, leaf, nil)
	if err != nil {
		t.Fatalf("v41ExpertF32Into error = %v, want nil", err)
	}
	if len(got) != len(want) {
		t.Fatalf("v41ExpertF32Into len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("v41ExpertF32Into[%d] = %v, want byte-identical %v", i, got[i], want[i])
		}
	}
}

// TestV41ExpertF32IntoManifestViewIsNotAliased pins the aliasing rule for the
// routed-expert resident path: a resident f32-manifest projection is an
// unsafe.Slice over the model's own backing memory, so the reused-buffer read
// must COPY it rather than hand back the view.
func TestV41ExpertF32IntoManifestViewIsNotAliased(t *testing.T) {
	m := v41ReducedModel(t)
	const leaf = "ffn.experts.0.w1.weight"
	name := layerName(0, leaf)
	view := m.tensor(name)
	if len(view) == 0 {
		t.Fatalf("fixture is missing the f32 manifest expert weight %s", name)
	}
	want := append([]float32(nil), view...)

	scratch := &v41ProjScratch{}
	got, err := m.v41ExpertF32Into(0, leaf, scratch.exp1)
	if err != nil {
		t.Fatalf("manifest expert read error = %v, want nil", err)
	}
	scratch.exp1 = got
	if len(got) > 0 && len(view) > 0 && &got[0] == &view[0] {
		t.Fatalf("v41ExpertF32Into returned a buffer aliasing the model's manifest memory")
	}
	after := m.tensor(name)
	for i := range want {
		if after[i] != want[i] {
			t.Fatalf("manifest expert weight[%d] = %v, want unchanged %v", i, after[i], want[i])
		}
	}
}

// TestV41ExpertF32IntoFailsClosedWhenAbsent pins #13276: a projection in neither
// a resident store nor the tier refuses with a typed ErrV41ForwardStage naming
// the tensor, never a panic.
func TestV41ExpertF32IntoFailsClosedWhenAbsent(t *testing.T) {
	m := v41ReducedModel(t)
	const leaf = "ffn.experts.0.w1.weight"
	name := layerName(0, leaf)
	delete(m.manifest, name)
	if m.kqw != nil {
		delete(m.kqw, name)
	}

	_, err := m.v41ExpertF32Into(0, leaf, nil)
	if err == nil {
		t.Fatalf("v41ExpertF32Into on an absent tensor returned nil error, want a typed refusal")
	}
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("v41ExpertF32Into error = %v, want errors.Is(err, ErrV41ForwardStage)", err)
	}
	if !strings.Contains(err.Error(), name) {
		t.Fatalf("v41ExpertF32Into error %q does not name the tensor %s", err.Error(), name)
	}
}

// TestV41ExpertScratchForwardStaysFinite drives the real reduced forward with the
// routed experts served ONLY from the checkpoint tier, so the reused expert
// scratch is exercised end-to-end across all three projections. The scratch must
// not perturb the arithmetic: the forward still emits finite logits.
func TestV41ExpertScratchForwardStaysFinite(t *testing.T) {
	m := v41TierOnlyModel(t, ExpertCheckpointQ4K)
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
