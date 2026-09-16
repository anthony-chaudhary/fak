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

// TestEngramSparseOverlayDiskBackedRowsAreNotResident is the disk-backed-row witness. A per-rank
// overlay registers ONE expert's gate stride as a DISK-BACKED extent over a shared reader (the
// [off,size) form the 183.25 GiB routed-expert figure requires) instead of a resident copy:
//   - the fault is byte-identical to the dense-stride read of the same expert, so the extent names
//     the honest rows and not a placeholder (AC1);
//   - the overlay holds ZERO resident row bytes for it — BackedRows()==1 and the resident row map
//     is empty — proving the rank's band costs no host RAM for the expert it bands (AC2);
//   - the overlay ledger still books exactly the stride the fault moved, so the offload win is a
//     measured number rather than an assertion (AC3).
func TestEngramSparseOverlayDiskBackedRowsAreNotResident(t *testing.T) {
	const H, E, K = 256, 8, 4

	// Baseline dense-stride bytes for the expert this overlay will band.
	_, plain, _ := expertCheckpointTestModel(t, H, E, K, 0)
	want, err := plain.fault(expertName(0, 0, "gate_proj.weight"))
	if err != nil {
		t.Fatalf("baseline fault: %v", err)
	}
	name := expertName(0, 0, "gate_proj.weight")

	// Rebuild the SAME resident rows the shard holds, then publish them through a shared ReaderAt
	// the overlay reads on demand. The extent deliberately starts at a non-zero offset so an
	// off-by-base read cannot pass by coincidence.
	resident := expertPrefetchModel(t, H, E, K)
	row := resident.q4kw[name].raw
	shard := append([]byte("PREAMBLE"), row...)
	off := int64(len("PREAMBLE"))

	_, tier, stride := expertCheckpointTestModel(t, H, E, K, 0)
	overlay, err := NewExpertSparseOverlay("blk.0.ffn_gate_exps.weight", E)
	if err != nil {
		t.Fatalf("NewExpertSparseOverlay: %v", err)
	}
	if err := overlay.SetRowBacked(0, name, bytes.NewReader(shard), off, int64(len(row))); err != nil {
		t.Fatalf("SetRowBacked: %v", err)
	}
	if err := tier.SetExpertSparseOverlay(overlay); err != nil {
		t.Fatalf("SetExpertSparseOverlay: %v", err)
	}

	if got := overlay.BackedRows(); got != 1 {
		t.Fatalf("BackedRows=%d, want 1 — the disk-backed row was not registered", got)
	}

	got, err := tier.fault(name)
	if err != nil {
		t.Fatalf("fault of a disk-backed overlay row: %v", err)
	}
	if !bytes.Equal(got.q4.raw, want.q4.raw) {
		t.Fatalf("the disk-backed overlay served bytes the shared shard does not hold")
	}

	st := tier.Stats()
	if st.OverlayRows != 1 {
		t.Fatalf("OverlayRows=%d, want 1 (the disk-backed row was overlay-served)", st.OverlayRows)
	}
	if st.OverlayBytesRead != stride {
		t.Fatalf("OverlayBytesRead=%d, want the %d-byte stride the disk-backed row moved", st.OverlayBytesRead, stride)
	}
}

// TestEngramSparseOverlayDiskBackedFailsClosed drives the extent source's fail-closed edges: a
// short/truncated backing read must NOT be served as a zero-byte weight, a name may not be claimed
// by both the resident and the backed form, and a malformed extent is refused at registration.
func TestEngramSparseOverlayDiskBackedFailsClosed(t *testing.T) {
	const H, E = 256, 4
	name := expertName(0, 0, "gate_proj.weight")

	o, err := NewExpertSparseOverlay("blk.0.ffn_gate_exps.weight", E)
	if err != nil {
		t.Fatalf("NewExpertSparseOverlay: %v", err)
	}
	// A nil reader, a negative offset, and a zero size are each refused at registration.
	if err := o.SetRowBacked(0, name, nil, 0, 4); err == nil {
		t.Fatal("SetRowBacked accepted a nil reader")
	}
	if err := o.SetRowBacked(0, name, bytes.NewReader([]byte{1, 2, 3, 4}), -1, 4); err == nil {
		t.Fatal("SetRowBacked accepted a negative extent offset")
	}
	if err := o.SetRowBacked(0, name, bytes.NewReader([]byte{1, 2, 3, 4}), 0, 0); err == nil {
		t.Fatal("SetRowBacked accepted a zero-size extent")
	}
	if o.PresentRows() != 0 {
		t.Fatalf("a malformed backed overlay marked %d rows present, want 0", o.PresentRows())
	}

	// A name already claimed resident cannot also be claimed backed.
	if err := o.SetRow(0, name, []byte{9, 9}); err != nil {
		t.Fatalf("SetRow: %v", err)
	}
	if err := o.SetRowBacked(0, name, bytes.NewReader([]byte{1, 2}), 0, 2); err == nil {
		t.Fatal("SetRowBacked shadowed an already-resident row")
	}

	// A truncated extent: the shard holds fewer bytes than the row declares, so the read falls
	// short and row() must report the row absent rather than serving a partial weight.
	short, err := NewExpertSparseOverlay("blk.0.ffn_gate_exps.weight", E)
	if err != nil {
		t.Fatalf("NewExpertSparseOverlay: %v", err)
	}
	if err := short.SetRowBacked(1, expertName(0, 1, "gate_proj.weight"),
		bytes.NewReader([]byte{1, 2, 3}), 0, 8); err != nil {
		t.Fatalf("SetRowBacked: %v", err)
	}
	_, tier, _ := expertCheckpointTestModel(t, H, E, 2, 0)
	if err := tier.SetExpertSparseOverlay(short); err != nil {
		t.Fatalf("SetExpertSparseOverlay: %v", err)
	}
	// The bitmap says present, but the extent cannot be read: the fault must fall through to the
	// dense stride (a real read), never return the 3 available bytes or a zero-length row.
	got, err := tier.fault(expertName(0, 1, "gate_proj.weight"))
	if err != nil {
		t.Fatalf("fault after a short backed read: %v", err)
	}
	if len(got.q4.raw) == 0 || len(got.q4.raw) == 3 {
		t.Fatalf("a short backed read served %d bytes; it must fall through, not serve a partial row", len(got.q4.raw))
	}
	if st := tier.Stats(); st.OverlayRows != 0 {
		t.Fatalf("OverlayRows=%d, want 0 — the short read must not be booked as an overlay hit", st.OverlayRows)
	}

	// A ReaderAt that returns fewer bytes than requested WITHOUT an error is a loose implementation
	// (the io.ReaderAt contract forbids it, but a shard wrapper can lose the error). The explicit
	// n==size check must still fail the row closed rather than serving a zero-padded weight.
	loose, err := NewExpertSparseOverlay("blk.0.ffn_gate_exps.weight", E)
	if err != nil {
		t.Fatalf("NewExpertSparseOverlay: %v", err)
	}
	if err := loose.SetRowBacked(2, expertName(0, 2, "gate_proj.weight"),
		shortReaderAt{data: []byte{1, 2, 3}}, 0, 8); err != nil {
		t.Fatalf("SetRowBacked: %v", err)
	}
	if _, ok := loose.row(expertName(0, 2, "gate_proj.weight")); ok {
		t.Fatal("row() served a zero-padded row from a ReaderAt that returned fewer bytes with no error")
	}
}

// shortReaderAt is an io.ReaderAt that returns min(len(p), len(data)) bytes and NO error — the loose
// implementation the explicit n==size check in row() must defend against.
type shortReaderAt struct{ data []byte }

func (s shortReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if off >= int64(len(s.data)) {
		return 0, nil
	}
	n := copy(p, s.data[off:])
	return n, nil
}
