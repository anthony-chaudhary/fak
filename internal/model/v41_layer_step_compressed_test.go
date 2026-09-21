package model

// Independent regression for fak#13307: append one compressor source row to a
// compressed V4.1 layer WITHOUT replaying the historical inputs that already
// left the incomplete group.
//
// This test is written from the ISSUE CONTRACT, not from the implementation:
//   * at most ratio-1 pre-attention INPUT rows are retained while a group is
//     incomplete, and an incomplete group emits nothing;
//   * when the append completes a group the emitted row must equal the row
//     v41CompressedRows emits for that same group at the same absolute covered
//     range [start,end);
//   * the emitted row is attributed to the SOURCE layer and its absolute
//     covered range;
//   * an injected projection error leaves the retained group and emitted
//     length unchanged (atomic failure).
//
// It reuses the reduced V4.1 model (HiddenSize=64, HeadDim=32) so the
// production projection primitives are read exactly, and it uses the
// PRODUCTION full-sequence path (v41CompressedRows) as the parity oracle rather
// than a hand re-derivation.

import (
	"math"
	"testing"
)

// v41LCFixtureValues builds a tiny ratio-3 compressed source fixture with
// non-symmetric inputs so that a wrong group identity, a wrong covered range or
// a wrong pooling order is observable in the emitted row.
func v41LCFixtureValues() v41SRFixture {
	H, W := 64, 32
	ratio := 3
	proj := func(seed int) []float32 {
		out := make([]float32, W*H)
		for r := 0; r < W; r++ {
			for c := 0; c < H; c++ {
				out[r*H+c] = float32(((r+seed+1)*(c+3))%11-5) / 9
			}
		}
		return out
	}
	mkInput := func(base float64) []float32 {
		in := make([]float32, H)
		for c := 0; c < H; c++ {
			in[c] = float32(math.Sin(base + float64(c)*0.21))
		}
		return in
	}
	norm := make([]float32, W)
	for i := range norm {
		norm[i] = float32(0.7 + 0.11*float64(i%5))
	}
	inputs := make([][]float32, 8)
	for i := range inputs {
		inputs[i] = mkInput(0.2 + 0.6*float64(i))
	}
	return v41SRFixture{
		Ratio:      ratio,
		Width:      W,
		Hidden:     H,
		Eps:        1e-6,
		NormWeight: norm,
		Inputs:     inputs,
		WKV:        proj(0),
		WGate:      proj(2),
	}
}

// v41LCProjector mirrors the exact projection v41CompressedRows applies: a
// row-major [width x hidden] matRows against the pre-attention input row.
func v41LCProjector(w []float32, width, hidden int) func([]float32) ([]float32, error) {
	return func(in []float32) ([]float32, error) {
		return matRows(w, in, width, hidden), nil
	}
}

// TestV41LayerStepCompressedSourceAppend drives the incremental source append
// against the production full-sequence rows, one position at a time, and
// asserts exact row parity at every complete group boundary plus the
// retention/attribution contract in between.
func TestV41LayerStepCompressedSourceAppend(t *testing.T) {
	f := v41LCFixtureValues()
	m := v41SRBuildModel(t, f)

	state, err := NewV41AttentionState(f.Width, f.Ratio)
	if err != nil {
		t.Fatalf("NewV41AttentionState: %v", err)
	}
	projKV := v41LCProjector(f.WKV, f.Width, f.Hidden)
	projScore := v41LCProjector(f.WGate, f.Width, f.Hidden)

	var emittedRows [][]float32
	for pos, input := range f.Inputs {
		refRows, err := m.v41CompressedRows(0, f.Ratio, f.Inputs[:pos+1], f.Inputs[:pos+1])
		if err != nil {
			t.Fatalf("production v41CompressedRows pos=%d: %v", pos, err)
		}
		latent, emitted, start, end, err := state.appendCompressorSource(
			0, f.Ratio, pos, input, mustPool(t, f.Ratio, f.Width),
			projKV, projScore, f.NormWeight, float32(f.Eps))
		if err != nil {
			t.Fatalf("appendCompressorSource pos=%d: %v", pos, err)
		}

		wantIncomplete := (pos+1)%f.Ratio != 0
		if emitted == wantIncomplete {
			t.Fatalf("pos=%d emitted=%v, want incomplete=%v", pos, emitted, wantIncomplete)
		}
		if !emitted {
			if latent != nil {
				t.Fatalf("pos=%d incomplete group emitted a latent row", pos)
			}
			if got := len(state.partialInputs); got > f.Ratio-1 {
				t.Fatalf("pos=%d retained %d inputs, exceeds ratio-1=%d", pos, got, f.Ratio-1)
			}
			if rows, ok := state.KVSourceRows(0); ok && len(rows) != len(emittedRows) {
				t.Fatalf("pos=%d incomplete group published %d rows, want %d", pos, len(rows), len(emittedRows))
			}
			continue
		}

		if len(refRows) == 0 {
			t.Fatalf("pos=%d emitted before the production path closed a group", pos)
		}
		wantRow := refRows[len(refRows)-1]
		wantStart := pos + 1 - f.Ratio
		wantEnd := pos + 1
		if start != wantStart || end != wantEnd {
			t.Fatalf("pos=%d covered range [%d,%d), want [%d,%d)", pos, start, end, wantStart, wantEnd)
		}
		if !rowsEqual(latent, wantRow) {
			t.Fatalf("pos=%d emitted row does not match production v41CompressedRows newest row", pos)
		}
		emittedRows = append(emittedRows, append([]float32(nil), latent...))

		got, ok := state.KVSourceRows(0)
		if !ok || len(got) != len(emittedRows) {
			t.Fatalf("pos=%d KVSourceRows len=%d ok=%v, want %d", pos, len(got), ok, len(emittedRows))
		}
		if !rowsEqual(got[len(got)-1], wantRow) {
			t.Fatalf("pos=%d published row differs from emitted row", pos)
		}
		if len(state.partialInputs) != 0 || len(state.partialPositions) != 0 {
			t.Fatalf("pos=%d completed group retained %d inputs / %d positions",
				pos, len(state.partialInputs), len(state.partialPositions))
		}
	}

	wantRows := len(f.Inputs) / f.Ratio
	if len(emittedRows) != wantRows {
		t.Fatalf("emitted %d compressed rows, want %d", len(emittedRows), wantRows)
	}
	// Bounded retention: after all complete groups the only retained copies are
	// the at-most ratio-1 tail rows, never the whole replayed history.
	if tail := len(f.Inputs) % f.Ratio; state.retainedCopies != tail {
		t.Fatalf("retainedCopies=%d after %d inputs, want the %d-row tail", state.retainedCopies, len(f.Inputs), tail)
	}
	if state.retainedCopies >= len(f.Inputs) {
		t.Fatalf("retention grew with the sequence: %d copies for %d inputs", state.retainedCopies, len(f.Inputs))
	}
}

// TestV41LayerStepCompressedSourceAtomicOnError proves an injected projection
// error leaves the retained group and emitted length unchanged: the seam must
// validate and project BEFORE mutating, and roll back otherwise.
func TestV41LayerStepCompressedSourceAtomicOnError(t *testing.T) {
	f := v41LCFixtureValues()

	state, err := NewV41AttentionState(f.Width, f.Ratio)
	if err != nil {
		t.Fatalf("NewV41AttentionState: %v", err)
	}
	projKV := v41LCProjector(f.WKV, f.Width, f.Hidden)
	projScore := v41LCProjector(f.WGate, f.Width, f.Hidden)

	for pos := 0; pos < f.Ratio; pos++ {
		_, _, _, _, err := state.appendCompressorSource(
			0, f.Ratio, pos, f.Inputs[pos], mustPool(t, f.Ratio, f.Width),
			projKV, projScore, f.NormWeight, float32(f.Eps))
		if err != nil {
			t.Fatalf("clean append pos=%d: %v", pos, err)
		}
	}
	beforeRows, _ := state.KVSourceRows(0)

	// Fill the next incomplete group to ratio-1 retained inputs.
	pos := state.nextCompressRow
	for i := 0; i < f.Ratio-1; i++ {
		_, emitted, _, _, err := state.appendCompressorSource(
			0, f.Ratio, pos+i, f.Inputs[pos+i], mustPool(t, f.Ratio, f.Width),
			projKV, projScore, f.NormWeight, float32(f.Eps))
		if err != nil {
			t.Fatalf("fill append pos=%d: %v", pos+i, err)
		}
		if emitted {
			t.Fatalf("fill append pos=%d emitted before the group completed", pos+i)
		}
	}
	afterFillRows, _ := state.KVSourceRows(0)
	retainedBefore := len(state.partialInputs)

	finalPos := pos + f.Ratio - 1
	failing := func([]float32) ([]float32, error) {
		return nil, v41StageErr(v41StageCompress, 0, nil)
	}
	if _, _, _, _, err := state.appendCompressorSource(
		0, f.Ratio, finalPos, f.Inputs[finalPos], mustPool(t, f.Ratio, f.Width),
		failing, projScore, f.NormWeight, float32(f.Eps)); err == nil {
		t.Fatalf("group-completing append with failing projection returned no error")
	}

	afterRows, _ := state.KVSourceRows(0)
	if len(afterRows) != len(afterFillRows) {
		t.Fatalf("failed append changed emitted length %d -> %d", len(afterFillRows), len(afterRows))
	}
	if state.nextCompressRow != finalPos {
		t.Fatalf("failed append advanced nextCompressRow to %d, want %d", state.nextCompressRow, finalPos)
	}
	if len(state.partialInputs) != retainedBefore {
		t.Fatalf("failed append changed retained group %d -> %d", retainedBefore, len(state.partialInputs))
	}
	if len(beforeRows) != 1 || retainedBefore != f.Ratio-1 {
		t.Fatalf("pre-injection state drifted: rows=%d retained=%d", len(beforeRows), retainedBefore)
	}
}

// TestV41LayerStepCompressedSourceRefusesBadPosition proves a non-contiguous
// absolute position is refused before mutation.
func TestV41LayerStepCompressedSourceRefusesBadPosition(t *testing.T) {
	f := v41LCFixtureValues()
	state, err := NewV41AttentionState(f.Width, f.Ratio)
	if err != nil {
		t.Fatalf("NewV41AttentionState: %v", err)
	}
	projKV := v41LCProjector(f.WKV, f.Width, f.Hidden)
	projScore := v41LCProjector(f.WGate, f.Width, f.Hidden)
	if _, _, _, _, err := state.appendCompressorSource(
		0, f.Ratio, 5, f.Inputs[0], mustPool(t, f.Ratio, f.Width),
		projKV, projScore, f.NormWeight, float32(f.Eps)); err == nil {
		t.Fatalf("non-contiguous first append returned no error")
	}
	if len(state.partialInputs) != 0 || state.nextCompressRow != 0 {
		t.Fatalf("refused append mutated state")
	}
}

func mustPool(t *testing.T, ratio, width int) *V41CompressorPool {
	t.Helper()
	p, err := NewV41CompressorPool(ratio, width)
	if err != nil {
		t.Fatalf("NewV41CompressorPool(%d,%d): %v", ratio, width, err)
	}
	return p
}

func rowsEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
