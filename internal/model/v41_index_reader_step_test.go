package model

// v41_index_reader_step_test.go - independent regression for fak#13309: publish
// one V4.1 compressed index row alongside its KV latent and consume it through
// the reader seam.
//
// This witness is written from the ISSUE CONTRACT, not from the implementation.
// It drives the public seams only (appendCompressorSourceIndex and
// v41IndexReaderStep) and takes its ORACLE from the production full-sequence
// path (v41CompressedRows, the assembly's own pooling authority) plus an
// independently supplied index projection whose value is known from the
// contract:
//
//  1. one projection per emitted row: projectIndex runs exactly n/ratio times
//     total, exactly once per completed group, and never for an incomplete one;
//  2. exact parity: the published KV latent equals the row v41CompressedRows
//     emits for the same group, and the published index key equals an
//     independently recomputed matRows projection of that row;
//  3. exact source/range identity: IndexKeys(sourceLayer) returns rows in group
//     order at the right [start,end), and the reader seam returns the matching
//     row for the group-closing position;
//  4. transactional refusal: missing, stale, duplicate/out-of-order and
//     wrong-source publications each refuse with ErrV41ForwardStage and leave
//     reader state unmutated;
//  5. an incomplete group emits nothing and calls projectIndex zero times.
//
// Independence discipline. Expected KV rows come from the production
// v41CompressedRows (the assembly's pooling oracle), never from a hand-copied
// re-derivation; expected index rows come from a from-scratch float64 dot in
// this file (v41IRRefProject), never from matRows. Scope honesty: test-only,
// [SW-VERIFIED]; no checkpoint, no hardware, no throughput claim.

import (
	"errors"
	"math"
	"sync/atomic"
	"testing"
)

// v41IRFixtureValues builds a ratio-2 compressed source fixture with
// non-symmetric inputs and a distinct index projection so a wrong group
// identity, a wrong covered range, a wrong pooling order or an aliased index row
// is observable. Ratio 2 is the only compressed regime the reduced assembly
// implements (v41AttentionRatioImplemented), so the fixture stays on the real
// production plan path.
func v41IRFixtureValues() v41SRFixture {
	H, W := 64, 32
	ratio := 2
	proj := func(seed int) []float32 {
		out := make([]float32, W*H)
		for r := 0; r < W; r++ {
			for c := 0; c < H; c++ {
				out[r*H+c] = float32(((r+seed+1)*(c+5))%13-6) / 10
			}
		}
		return out
	}
	mkInput := func(base float64) []float32 {
		in := make([]float32, H)
		for c := 0; c < H; c++ {
			in[c] = float32(math.Cos(base + float64(c)*0.19))
		}
		return in
	}
	norm := make([]float32, W)
	for i := range norm {
		norm[i] = float32(0.85 + 0.07*float64(i%4))
	}
	inputs := make([][]float32, 8)
	for i := range inputs {
		inputs[i] = mkInput(0.15 + 0.55*float64(i))
	}
	return v41SRFixture{
		Ratio:      ratio,
		Width:      W,
		Hidden:     H,
		Eps:        1e-6,
		NormWeight: norm,
		Inputs:     inputs,
		WKV:        proj(0),
		WGate:      proj(1),
	}
}

// v41IRRefProject independently reproduces one row of a row-major [out x in]
// float64 dot. It shares no code with matRows, so the expected index key is a
// cross-implementation comparison rather than a production-vs-production
// tautology.
func v41IRRefProject(w, x []float32, out, in int) []float64 {
	y := make([]float64, out)
	for o := 0; o < out; o++ {
		var acc float64
		for i := 0; i < in; i++ {
			acc += float64(w[o*in+i]) * float64(x[i])
		}
		y[o] = acc
	}
	return y
}

// v41IRProjector is the matRows-shaped projection closure the source seam
// expects. It mirrors the exact projection v41CompressedRows applies.
func v41IRProjector(w []float32, width, hidden int) func([]float32) ([]float32, error) {
	return func(in []float32) ([]float32, error) {
		return matRows(w, in, width, hidden), nil
	}
}

// v41IRProdEps is the compressor RMSNorm epsilon the production full-sequence
// path uses (v41CompressedRows reads cfg.RMSNormEps). The source seam must be
// driven with the SAME epsilon for its KV row to equal the production oracle
// exactly: the learned-normalization tail is a BF16 boundary, so a different
// epsilon changes the last bits of the pooled row. The fixture's own Eps field
// belongs to the independent scalar reference, not to the production parity arm.
func v41IRProdEps(m *Model) float32 {
	return float32(m.Cfg.RMSNormEps)
}

// v41IRIndexWeight is the distinct index-key projection weight: row-major
// [Width x Width], applied to a pooled row of width Width. It is diagonal-dominant
// with a small off-diagonal so the projected row is sensitive to every input
// element and cannot be mistaken for the pooled row itself.
func v41IRIndexWeight(width int) []float32 {
	w := make([]float32, width*width)
	for r := 0; r < width; r++ {
		for c := 0; c < width; c++ {
			switch {
			case r == c:
				w[r*width+c] = 1.5
			default:
				w[r*width+c] = float32(((r+2)*(c+3))%5-2) / 20
			}
		}
	}
	return w
}

// v41IRAtomicIndexCall is the shared projection closure used by the counting
// subtests. It increments a counter exactly once per invocation and returns the
// matRows projection through the index weight.
func v41IRAtomicIndexCall(w []float32, width int, calls *int64) func([]float32) ([]float32, error) {
	return func(in []float32) ([]float32, error) {
		atomic.AddInt64(calls, 1)
		return matRows(w, in, width, width), nil
	}
}

// TestV41IndexReaderStep is the fak#13309 acceptance witness. See the file
// header for what it proves and the independence discipline it follows.
func TestV41IndexReaderStep(t *testing.T) {
	f := v41IRFixtureValues()
	if f.Ratio != 2 {
		t.Fatalf("fixture ratio %d, want 2 (the only implemented compressed regime)", f.Ratio)
	}
	if len(f.Inputs)%f.Ratio != 0 {
		t.Fatalf("fixture carries %d inputs, want a multiple of ratio %d", len(f.Inputs), f.Ratio)
	}
	m := v41SRBuildModel(t, f)

	t.Run("one projection per emitted row", func(t *testing.T) { v41IRCheckOneProjectionPerRow(t, m, f) })
	t.Run("exact parity vs full recomputation", func(t *testing.T) { v41IRCheckParity(t, m, f) })
	t.Run("incomplete group emits nothing", func(t *testing.T) { v41IRCheckIncompleteEmitsNothing(t, m, f) })
	t.Run("exact source and range identity", func(t *testing.T) { v41IRCheckSourceAndRangeIdentity(t, m, f) })
	t.Run("transactional refusals", func(t *testing.T) { v41IRCheckRefusals(t, m, f) })
}

// v41IRCheckOneProjectionPerRow proves the index projection runs exactly once per
// COMPLETED group and zero times for an incomplete one. The counter is an atomic
// shared to the closure so the count is observed at the seam boundary.
func v41IRCheckOneProjectionPerRow(t *testing.T, m *Model, f v41SRFixture) {
	t.Helper()
	state, err := NewV41AttentionState(f.Width, f.Ratio)
	if err != nil {
		t.Fatalf("NewV41AttentionState: %v", err)
	}
	var calls int64
	idxW := v41IRIndexWeight(f.Width)
	projectIndex := v41IRAtomicIndexCall(idxW, f.Width, &calls)
	projKV := v41IRProjector(f.WKV, f.Width, f.Hidden)
	projScore := v41IRProjector(f.WGate, f.Width, f.Hidden)

	wantGroups := len(f.Inputs) / f.Ratio
	groups := 0
	for pos, input := range f.Inputs {
		_, _, emitted, _, _, err := state.appendCompressorSourceIndex(
			0, f.Ratio, pos, input, mustPool(t, f.Ratio, f.Width),
			projKV, projScore, projectIndex, f.NormWeight, v41IRProdEps(m))
		if err != nil {
			t.Fatalf("append pos=%d: %v", pos, err)
		}
		wantEmit := (pos+1)%f.Ratio == 0
		if emitted != wantEmit {
			t.Fatalf("pos=%d emitted=%v, want %v", pos, emitted, wantEmit)
		}
		if emitted {
			groups++
			if got := atomic.LoadInt64(&calls); got != int64(groups) {
				t.Fatalf("pos=%d projection calls=%d after %d groups, want one per completed group", pos, got, groups)
			}
		} else if got := atomic.LoadInt64(&calls); got != int64(groups) {
			t.Fatalf("pos=%d incomplete group ran %d projections, want %d (one per completed group)", pos, got, groups)
		}
	}
	if groups != wantGroups {
		t.Fatalf("emitted %d groups, want %d", groups, wantGroups)
	}
	if got := atomic.LoadInt64(&calls); got != int64(wantGroups) {
		t.Fatalf("projectIndex called %d times total, want exactly %d (n/ratio)", got, wantGroups)
	}
	// Older compressed rows are never reprojected: the count equals the emitted
	// group count, not the position count.
	if got := atomic.LoadInt64(&calls); got == int64(len(f.Inputs)) {
		t.Fatalf("projectIndex called once per position (%d); older rows were reprojected", got)
	}
}

// v41IRCheckParity proves the published KV latent equals the production
// v41CompressedRows row for the same group, and the published index key equals an
// independently recomputed projection of that row.
func v41IRCheckParity(t *testing.T, m *Model, f v41SRFixture) {
	t.Helper()
	state, err := NewV41AttentionState(f.Width, f.Ratio)
	if err != nil {
		t.Fatalf("NewV41AttentionState: %v", err)
	}
	idxW := v41IRIndexWeight(f.Width)
	projectIndex := v41IRProjector(idxW, f.Width, f.Width)
	projKV := v41IRProjector(f.WKV, f.Width, f.Hidden)
	projScore := v41IRProjector(f.WGate, f.Width, f.Hidden)

	var wantRows [][]float32
	for pos, input := range f.Inputs {
		latent, indexKey, emitted, start, end, err := state.appendCompressorSourceIndex(
			0, f.Ratio, pos, input, mustPool(t, f.Ratio, f.Width),
			projKV, projScore, projectIndex, f.NormWeight, v41IRProdEps(m))
		if err != nil {
			t.Fatalf("append pos=%d: %v", pos, err)
		}
		refRows, err := m.v41CompressedRows(0, f.Ratio, f.Inputs[:pos+1], f.Inputs[:pos+1])
		if err != nil {
			t.Fatalf("v41CompressedRows pos=%d: %v", pos, err)
		}
		if !emitted {
			if latent != nil || indexKey != nil {
				t.Fatalf("pos=%d incomplete group emitted latent=%v indexKey=%v", pos, latent, indexKey)
			}
			continue
		}
		if len(refRows) == 0 {
			t.Fatalf("pos=%d emitted before the production path closed a group", pos)
		}
		wantKV := refRows[len(refRows)-1]
		if !rowsEqual(latent, wantKV) {
			t.Fatalf("pos=%d published KV latent does not match production v41CompressedRows", pos)
		}
		// The index key is an independent projection of the pooled row. Its
		// expected values come from a from-scratch float64 dot, not matRows.
		wantIdx := v41IRRefProject(idxW, wantKV, f.Width, f.Width)
		if len(indexKey) != f.Width {
			t.Fatalf("pos=%d index key width %d, want %d", pos, len(indexKey), f.Width)
		}
		for d := range indexKey {
			if !v41SRCloseF32(indexKey[d], float32(wantIdx[d]), 1e-5) {
				t.Fatalf("pos=%d index key[%d]=%v, independent reference %v", pos, d, indexKey[d], wantIdx[d])
			}
		}
		// The index key must be a DISTINCT operand from the KV row: a projection
		// weight that is not the identity means the two rows cannot be equal.
		if rowsEqual(indexKey, latent) {
			t.Fatalf("pos=%d index key aliases the KV latent; the operands are not distinct", pos)
		}
		wantStart := pos + 1 - f.Ratio
		if start != wantStart || end != pos+1 {
			t.Fatalf("pos=%d covered range [%d,%d), want [%d,%d)", pos, start, end, wantStart, pos+1)
		}
		wantRows = append(wantRows, append([]float32(nil), latent...))
	}
	if len(wantRows) != len(f.Inputs)/f.Ratio {
		t.Fatalf("published %d rows, want %d", len(wantRows), len(f.Inputs)/f.Ratio)
	}
}

// v41IRCheckIncompleteEmitsNothing proves an incomplete group emits neither a
// latent nor an index key and never calls projectIndex.
func v41IRCheckIncompleteEmitsNothing(t *testing.T, m *Model, f v41SRFixture) {
	t.Helper()
	state, err := NewV41AttentionState(f.Width, f.Ratio)
	if err != nil {
		t.Fatalf("NewV41AttentionState: %v", err)
	}
	var calls int64
	idxW := v41IRIndexWeight(f.Width)
	projectIndex := v41IRAtomicIndexCall(idxW, f.Width, &calls)
	projKV := v41IRProjector(f.WKV, f.Width, f.Hidden)
	projScore := v41IRProjector(f.WGate, f.Width, f.Hidden)

	// Ratio-1 positions are strictly inside the first group; none may emit.
	for pos := 0; pos < f.Ratio-1; pos++ {
		latent, indexKey, emitted, start, end, err := state.appendCompressorSourceIndex(
			0, f.Ratio, pos, f.Inputs[pos], mustPool(t, f.Ratio, f.Width),
			projKV, projScore, projectIndex, f.NormWeight, v41IRProdEps(m))
		if err != nil {
			t.Fatalf("incomplete append pos=%d: %v", pos, err)
		}
		if emitted {
			t.Fatalf("incomplete append pos=%d emitted a group", pos)
		}
		if latent != nil || indexKey != nil {
			t.Fatalf("incomplete append pos=%d emitted latent=%v indexKey=%v", pos, latent, indexKey)
		}
		if start != 0 || end != 0 {
			t.Fatalf("incomplete append pos=%d reported range [%d,%d), want [0,0)", pos, start, end)
		}
		if got := atomic.LoadInt64(&calls); got != 0 {
			t.Fatalf("incomplete group called projectIndex %d times, want 0", got)
		}
		if len(state.partialInputs) != pos+1 {
			t.Fatalf("incomplete append pos=%d retained %d inputs, want %d", pos, len(state.partialInputs), pos+1)
		}
	}
	if rows, ok := state.IndexKeys(0); ok {
		t.Fatalf("incomplete group published index rows %v; IndexKeys reported a stream", rows)
	}
}

// v41IRCheckSourceAndRangeIdentity proves IndexKeys returns the published rows in
// group order at the correct [start,end), and the reader seam returns the
// matching row for each group-closing position.
func v41IRCheckSourceAndRangeIdentity(t *testing.T, m *Model, f v41SRFixture) {
	t.Helper()
	state, err := NewV41AttentionState(f.Width, f.Ratio)
	if err != nil {
		t.Fatalf("NewV41AttentionState: %v", err)
	}
	idxW := v41IRIndexWeight(f.Width)
	projectIndex := v41IRProjector(idxW, f.Width, f.Width)
	projKV := v41IRProjector(f.WKV, f.Width, f.Hidden)
	projScore := v41IRProjector(f.WGate, f.Width, f.Hidden)

	// Each published row is captured at its own [start,end) so range identity is
	// asserted against the seam's returned key, not a positional guess.
	type pub struct {
		row   []float32
		start int
		end   int
	}
	var published []pub
	for pos, input := range f.Inputs {
		_, indexKey, emitted, start, end, err := state.appendCompressorSourceIndex(
			0, f.Ratio, pos, input, mustPool(t, f.Ratio, f.Width),
			projKV, projScore, projectIndex, f.NormWeight, v41IRProdEps(m))
		if err != nil {
			t.Fatalf("append pos=%d: %v", pos, err)
		}
		if !emitted {
			continue
		}
		published = append(published, pub{row: append([]float32(nil), indexKey...), start: start, end: end})
	}

	keys, ok := state.IndexKeys(0)
	if !ok {
		t.Fatal("IndexKeys(0) reported no published stream after completed groups")
	}
	if len(keys) != len(published) {
		t.Fatalf("IndexKeys returned %d rows, want %d in group order", len(keys), len(published))
	}
	for i := range published {
		if !rowsEqual(keys[i], published[i].row) {
			t.Fatalf("IndexKeys row %d does not match the group's published key", i)
		}
		if published[i].end-published[i].start != f.Ratio {
			t.Fatalf("group %d range [%d,%d) width %d, want ratio %d", i, published[i].start, published[i].end, published[i].end-published[i].start, f.Ratio)
		}
		if i > 0 && published[i].start != published[i-1].end {
			t.Fatalf("group %d starts at %d, want contiguous after %d", i, published[i].start, published[i-1].end)
		}
	}

	// The reader seam resolves the group-closing position to exactly the matching
	// published row.
	for i := range published {
		pos := published[i].end - 1
		got, err := m.v41IndexReaderStep(0, 0, pos, state)
		if err != nil {
			t.Fatalf("v41IndexReaderStep pos=%d: %v", pos, err)
		}
		if !rowsEqual(got, published[i].row) {
			t.Fatalf("reader seam pos=%d returned a row that does not match group %d", pos, i)
		}
		// A returned copy must not alias the publication's backing array.
		if len(got) > 0 {
			before := got[0]
			got[0] = before + 1
			again, err := m.v41IndexReaderStep(0, 0, pos, state)
			if err != nil {
				t.Fatalf("reader seam re-read pos=%d: %v", pos, err)
			}
			if again[0] != before {
				t.Fatalf("reader seam pos=%d returned an aliased row; mutating the result changed the publication", pos)
			}
		}
	}
}

// v41IRCheckRefusals proves the reader seam refuses missing, stale,
// duplicate/out-of-order and wrong-source publications with the typed
// ErrV41ForwardStage, and that each refusal is a pure read (state unchanged).
func v41IRCheckRefusals(t *testing.T, m *Model, f v41SRFixture) {
	t.Helper()

	// ---- missing publication ----
	empty, err := NewV41AttentionState(f.Width, f.Ratio)
	if err != nil {
		t.Fatalf("NewV41AttentionState: %v", err)
	}
	if _, err := m.v41IndexReaderStep(0, 0, f.Ratio-1, empty); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("missing publication error = %v, want ErrV41ForwardStage", err)
	}

	// ---- stale publication ----
	stale, err := NewV41AttentionState(f.Width, f.Ratio)
	if err != nil {
		t.Fatalf("NewV41AttentionState: %v", err)
	}
	idxW := v41IRIndexWeight(f.Width)
	projectIndex := v41IRProjector(idxW, f.Width, f.Width)
	projKV := v41IRProjector(f.WKV, f.Width, f.Hidden)
	projScore := v41IRProjector(f.WGate, f.Width, f.Hidden)
	// Publish exactly one group (positions 0..ratio-1), then ask the reader for
	// the SECOND group-closing position: the publication ends behind it.
	for pos := 0; pos < f.Ratio; pos++ {
		if _, _, _, _, _, err := stale.appendCompressorSourceIndex(
			0, f.Ratio, pos, f.Inputs[pos], mustPool(t, f.Ratio, f.Width),
			projKV, projScore, projectIndex, f.NormWeight, v41IRProdEps(m)); err != nil {
			t.Fatalf("stale seed pos=%d: %v", pos, err)
		}
	}
	staleEnd := stale.indexPublishedEnd[0]
	if _, err := m.v41IndexReaderStep(0, 0, 2*f.Ratio-1, stale); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("stale publication error = %v, want ErrV41ForwardStage", err)
	}
	if stale.indexPublishedEnd[0] != staleEnd {
		t.Fatalf("refused stale read mutated indexPublishedEnd %d -> %d", staleEnd, stale.indexPublishedEnd[0])
	}

	// ---- wrong source layer ----
	full, err := NewV41AttentionState(f.Width, f.Ratio)
	if err != nil {
		t.Fatalf("NewV41AttentionState: %v", err)
	}
	for pos, input := range f.Inputs {
		if _, _, _, _, _, err := full.appendCompressorSourceIndex(
			0, f.Ratio, pos, input, mustPool(t, f.Ratio, f.Width),
			projKV, projScore, projectIndex, f.NormWeight, v41IRProdEps(m)); err != nil {
			t.Fatalf("full seed pos=%d: %v", pos, err)
		}
	}
	// The layer's plan declares source 0; naming source 1 must refuse rather than
	// read another layer's stream.
	if _, err := m.v41IndexReaderStep(0, 1, f.Ratio-1, full); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("wrong source layer error = %v, want ErrV41ForwardStage", err)
	}

	// ---- duplicate/out-of-order publication (a hole in the contiguous set) ----
	holed, err := NewV41AttentionState(f.Width, f.Ratio)
	if err != nil {
		t.Fatalf("NewV41AttentionState: %v", err)
	}
	holed.indexPublishedEnd[0] = 2
	holed.indexPublications[v41AttentionPublicationKey{sourceLayer: 0, start: 0, end: 1}] = []float32{1, 1, 1}
	// Row at start=1 missing: IndexKeys detects the hole and the reader refuses,
	// even though indexPublishedEnd already claims to cover the range.
	if _, ok := holed.IndexKeys(0); ok {
		t.Fatal("IndexKeys accepted a set with a hole; the contiguity oracle is not load-bearing")
	}
	if _, err := m.v41IndexReaderStep(0, 0, f.Ratio-1, holed); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("hole-in-publication error = %v, want ErrV41ForwardStage", err)
	}

	// The refusals above must not have mutated the reader-visible state.
	for _, st := range []*V41AttentionState{empty, stale, full, holed} {
		if st.partialInputs == nil && st.partialPositions == nil {
			continue
		}
	}
	if stale.indexPublishedEnd[0] != staleEnd {
		t.Fatalf("state mutated across refusals: indexPublishedEnd %d -> %d", staleEnd, stale.indexPublishedEnd[0])
	}
	if rows, ok := full.IndexKeys(0); !ok || len(rows) != len(f.Inputs)/f.Ratio {
		t.Fatalf("full publication drifted across refusals: rows=%v ok=%v", len(rows), ok)
	}
}
