package model

import (
	"bytes"
	"testing"
)

// expert_sparse_overlay_test.go — the witnesses for #13030 (the disk-backed Engram sparse-overlay
// expert layout with per-rank row copies), the MoE disk-backed serving spine's #13008 parent.
//
// The rung's claims are byte claims, so the tests are byte tests. The central witness registers a
// per-rank overlay over ONE shared checkpoint shard, then proves two things at once: an expert the
// overlay bands is served from its copied rows (and the overlay ledger counts exactly those bytes),
// while an expert outside the band falls through to the dense stride byte-for-byte. The default-off
// witness is the one that would catch the easy mistake: with no overlay registered, the tier must
// be byte-identical to the historical dense-stride path.

// overlayTestModel builds the same streamed MoE the checkpoint-tier witnesses use, then registers a
// per-rank sparse overlay on the gate fused tensor carrying ONLY the named expert indices' rows.
// It returns the model, the tier and the per-expert byte stride.
func overlayTestModel(t *testing.T, hidden, experts, topK int, presentExperts []int) (*Model, *ExpertCheckpointTier, int64) {
	t.Helper()
	m, tier, stride := expertCheckpointTestModel(t, hidden, experts, topK, 0)

	// Rebuild the gate projection's fused slab so the overlay carries the SAME bytes the shard
	// does; a divergence would make the overlay-vs-dense comparison meaningless.
	resident := expertPrefetchModel(t, hidden, experts, topK)
	overlay, err := NewExpertSparseOverlay("blk.0.ffn_gate_exps.weight", experts)
	if err != nil {
		t.Fatalf("NewExpertSparseOverlay: %v", err)
	}
	for _, e := range presentExperts {
		name := expertName(0, e, "gate_proj.weight")
		raw := resident.q4kw[name].raw
		if err := overlay.SetRow(e, name, raw); err != nil {
			t.Fatalf("SetRow(%d, %s): %v", e, name, err)
		}
	}
	if err := tier.SetExpertSparseOverlay(overlay); err != nil {
		t.Fatalf("SetExpertSparseOverlay: %v", err)
	}
	return m, tier, stride
}

// TestEngramSparseOverlayServesPresentRowsAndFallsThrough is this rung's headline. On ONE shared
// shard, expert 0's gate projection is registered in the rank's overlay and expert 1's is not:
//   - expert 0 faults from the overlay's copied rows and moves exactly its stride via the overlay
//     ledger (AC1, AC2), and it is byte-identical to the dense-stride read (so the overlay copy is
//     the honest rows, not a placeholder);
//   - expert 1, outside the band, falls through to the dense stride and is byte-identical to the
//     same expert read before any overlay existed (AC3).
func TestEngramSparseOverlayServesPresentRowsAndFallsThrough(t *testing.T) {
	const H, E, K = 256, 8, 4

	// Baseline: the dense-stride bytes for both experts on an overlay-free tier, so every later
	// comparison is against the ACTUAL historical path rather than a restatement of the overlay.
	_, plain, stride := expertCheckpointTestModel(t, H, E, K, 0)
	wantPresent, err := plain.fault(expertName(0, 0, "gate_proj.weight"))
	if err != nil {
		t.Fatalf("baseline fault of the overlay-present expert: %v", err)
	}
	wantAbsent, err := plain.fault(expertName(0, 1, "gate_proj.weight"))
	if err != nil {
		t.Fatalf("baseline fault of the fall-through expert: %v", err)
	}

	m, tier, _ := overlayTestModel(t, H, E, K, []int{0})
	_ = m

	present, err := tier.fault(expertName(0, 0, "gate_proj.weight"))
	if err != nil {
		t.Fatalf("overlay fault of a bitmap-present expert: %v", err)
	}
	if !bytes.Equal(present.q4.raw, wantPresent.q4.raw) {
		t.Fatalf("the overlay-served rows differ from the dense-stride read; the rank was served " +
			"bytes the shared shard does not hold")
	}

	absent, err := tier.fault(expertName(0, 1, "gate_proj.weight"))
	if err != nil {
		t.Fatalf("fall-through fault of an expert outside the overlay: %v", err)
	}
	if !bytes.Equal(absent.q4.raw, wantAbsent.q4.raw) {
		t.Fatalf("an expert OUTSIDE the overlay did not fall through to the dense stride byte-for-byte")
	}

	st := tier.Stats()
	if st.OverlayTensors != 1 || st.OverlayPresent != 1 {
		t.Fatalf("overlay ledger = %d tensors / %d present rows, want 1 / 1", st.OverlayTensors, st.OverlayPresent)
	}
	if st.OverlayRows != 1 {
		t.Fatalf("OverlayRows=%d, want 1 (only the bitmap-present expert was overlay-served)", st.OverlayRows)
	}
	if st.OverlayBytesRead != stride {
		t.Fatalf("OverlayBytesRead=%d, want the %d-byte stride the overlay row moved", st.OverlayBytesRead, stride)
	}
	// The fall-through expert must be booked on the DENSE ledger, not the overlay one, or the
	// overlay ledger would over-report the offload win.
	if st.BytesRead != stride {
		t.Fatalf("BytesRead=%d, want %d — exactly the one fall-through expert's stride", st.BytesRead, stride)
	}
	if st.Reads != 1 {
		t.Fatalf("Reads=%d, want 1 — the overlay-served expert must not issue a device read", st.Reads)
	}
}

// TestEngramSparseOverlayDefaultOffIsByteIdentical is the default-off gate (AC4). A tier with no
// overlay registered must read exactly as it did before the overlay existed: same bytes, same
// dense ledger, and an overlay ledger that stays at zero.
func TestEngramSparseOverlayDefaultOffIsByteIdentical(t *testing.T) {
	const H, E, K = 256, 8, 4

	_, tier, stride := expertCheckpointTestModel(t, H, E, K, 0)
	names := []string{
		expertName(0, 0, "gate_proj.weight"),
		expertName(0, 3, "up_proj.weight"),
		expertName(0, 5, "down_proj.weight"),
	}
	for _, name := range names {
		if _, err := tier.fault(name); err != nil {
			t.Fatalf("fault %s on an overlay-free tier: %v", name, err)
		}
	}
	st := tier.Stats()
	if st.OverlayTensors != 0 || st.OverlayPresent != 0 || st.OverlayRows != 0 || st.OverlayBytesRead != 0 {
		t.Fatalf("an overlay-free tier reported overlay activity: %+v", st)
	}
	if st.Reads != len(names) || st.BytesRead != int64(len(names))*stride {
		t.Fatalf("dense ledger reads=%d bytes=%d, want %d reads / %d bytes — the historical path",
			st.Reads, st.BytesRead, len(names), int64(len(names))*stride)
	}
}

// TestEngramSparseOverlayBitmapRefusesMalformed drives the row-presence bitmap's fail-closed edge:
// an out-of-range row index and an empty row are refused at registration, and an overlay naming a
// fused tensor no shard carries is refused by the tier — a mis-registered band must not silently
// serve nothing.
func TestEngramSparseOverlayBitmapRefusesMalformed(t *testing.T) {
	const H, E = 256, 4

	o, err := NewExpertSparseOverlay("blk.0.ffn_gate_exps.weight", E)
	if err != nil {
		t.Fatalf("NewExpertSparseOverlay: %v", err)
	}
	if err := o.SetRow(E, expertName(0, 0, "gate_proj.weight"), []byte{1, 2}); err == nil {
		t.Fatal("SetRow accepted a row index outside the bitmap")
	}
	if err := o.SetRow(0, "", []byte{1, 2}); err == nil {
		t.Fatal("SetRow accepted an unnamed row")
	}
	if err := o.SetRow(0, expertName(0, 0, "gate_proj.weight"), nil); err == nil {
		t.Fatal("SetRow accepted an empty row")
	}
	if o.PresentRows() != 0 {
		t.Fatalf("a malformed overlay marked %d rows present, want 0", o.PresentRows())
	}
	if _, err := NewExpertSparseOverlay("", E); err == nil {
		t.Fatal("NewExpertSparseOverlay accepted an empty fused-tensor name")
	}
	if _, err := NewExpertSparseOverlay("blk.0.ffn_gate_exps.weight", 0); err == nil {
		t.Fatal("NewExpertSparseOverlay accepted a zero row count")
	}

	_, tier, _ := expertCheckpointTestModel(t, H, E, 2, 0)
	bad, err := NewExpertSparseOverlay("blk.9.ffn_absent_exps.weight", E)
	if err != nil {
		t.Fatalf("NewExpertSparseOverlay: %v", err)
	}
	if err := tier.SetExpertSparseOverlay(bad); err == nil {
		t.Fatal("the tier registered an overlay banding a fused tensor no shard carries")
	}
}
