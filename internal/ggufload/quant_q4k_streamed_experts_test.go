package ggufload

import (
	"os"
	"path/filepath"
	"testing"
)

// 13143: the serve streamed-expert arm threaded WithStreamedExperts into an entry point that
// refuses it by contract, so the bounded-resident NVMe-streamed expert path could never reach
// model load. These regressions pin the two halves of the fix at the loader seam:
//   1. the option EFFECT surface must report the streamed-expert request, because that is what
//      serve's loadResidentQ4KProfiled reads to pick the right entry point; and
//   2. LoadModelQ4KStreamedExperts must admit the option (no quant_q4k_loader.go:284 refusal)
//      and transfer the checkpoint lifetime to the model instead of closing it on return.

// TestIssue13143StreamedExpertsOptionEffectIsObservable is the serve-routing half: an option list
// carrying WithStreamedExperts(bound) must be observable through ApplyQ4KLoadOptions with the exact
// bound, or the loader entry-point selector cannot see the arm it must route.
func TestIssue13143StreamedExpertsOptionEffectIsObservable(t *testing.T) {
	const bound = int64(7 << 30)
	effects := ApplyQ4KLoadOptions([]Q4KLoadOption{WithStreamedExperts(bound)})
	if !effects.StreamedExperts {
		t.Fatal("WithStreamedExperts(bound) is not observable via ApplyQ4KLoadOptions; the serve loader-entry selector cannot route the streamed-expert arm")
	}
	if effects.StreamedExpertBytes != bound {
		t.Fatalf("streamed-expert bound = %d, want %d", effects.StreamedExpertBytes, bound)
	}
	if effects.StreamedDenseQ4K {
		t.Fatal("WithStreamedExperts also reported StreamedDenseQ4K; the two arms must stay distinguishable")
	}
	// The dense arm must report its own effect and NOT the expert one (no cross-talk).
	dense := ApplyQ4KLoadOptions([]Q4KLoadOption{WithStreamedDenseQ4K(true)})
	if !dense.StreamedDenseQ4K || dense.StreamedExperts {
		t.Fatalf("streamed-dense effects = %+v, want StreamedDenseQ4K=true StreamedExperts=false", dense)
	}
}

// TestIssue13143StreamedExpertsEntryTransfersCheckpoint is the loader half: over a real stageable
// Q2_K routed-expert checkpoint, the lifetime-transferring streamed-experts entry must reach model
// load (the quant_q4k_loader.go:284 refusal is gone) and the returned model must own the checkpoint
// (its weight closer is bound; the checkpoint is not closed on return). The lifetime-CLOSING entry
// must still refuse the same options, proving the new entry is the one that admits them.
func TestIssue13143StreamedExpertsEntryTransfersCheckpoint(t *testing.T) {
	// H and I are the expert slab's reduction/row dims and must be whole super-blocks
	// (qkK = 256), or FusedExpertTensors declines the slab as unstaged (eager path) and the
	// tier has nothing to serve -- the same gate the serve streamed fixture satisfies.
	// Every declared dim is a multiple of 32 so the eager dense Q8_0 path (which the
	// non-expert tensors take) accepts the reduction dims; H and I are also whole expert
	// super-blocks (qkK = 256) so FusedExpertTensors stages the routed slabs.
	const H, V, qLora, kvLora, qkNope, qkRope, vHead, nH, idxHeads, idxDim, E, I, sharedI = 256, 64, 32, 32, 32, 32, 32, 2, 32, 32, 4, 256, 256
	blob := glmMoeDsaFullGGUFTyped(H, V, qLora, kvLora, qkNope, qkRope, vHead, nH, idxHeads, idxDim, E, I, sharedI, TensorQ2_K)
	path := filepath.Join(t.TempDir(), "13143-streamed-experts.gguf")
	if err := os.WriteFile(path, blob, 0o600); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	// Baseline: the lifetime-CLOSING entry refuses, which is the defect's signature.
	if _, err := LoadModelQ4KProfileOptions(path, nil, WithStreamedExperts(1<<30)); err == nil {
		t.Fatal("LoadModelQ4KProfileOptions admitted WithStreamedExperts; the lifetime-closing contract is broken")
	}

	// The fix: the lifetime-TRANSFERRING entry admits the option and returns a model.
	m, err := LoadModelQ4KStreamedExperts(path, nil, 1<<30)
	if err != nil {
		t.Fatalf("LoadModelQ4KStreamedExperts refused the streamed-expert arm (the fak#13143 defect): %v", err)
	}
	if m == nil {
		t.Fatal("LoadModelQ4KStreamedExperts returned a nil model with no error")
	}
	// The model owns the checkpoint: closing the model must release it exactly once and not
	// report active sessions (none were opened).
	if err := m.CloseWeights(); err != nil {
		t.Fatalf("CloseWeights on the streamed-experts model: %v", err)
	}
}
