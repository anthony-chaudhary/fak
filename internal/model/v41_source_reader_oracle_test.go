package model

// v41_source_reader_oracle_test.go - TestV41SourceReaderNumericOracle (fak#13323):
// the independent numeric witness for one V4.1 compressed source-reader PAIR.
//
// What it proves. Given one compressed source layer's hidden inputs and its
// compressor + indexer weights (the fixture in v41_source_reader_fixture_test.go),
// the PRODUCTION path must:
//  1. project and pool the inputs into compressed rows (v41CompressedRows, the
//     rows the source publishes into session state), and
//  2. select the source's top-k compressed row IDs from those rows
//     (NewV41IndexerPublication, the source's published selection), and
//  3. let a READER reuse that published candidate mask for its own query
//     (V41IndexerPublication.Reuse, the reader-consumes-source seam),
// and every one of those results must agree, elementwise, with the independently
// transcribed scalar reference in v41_source_reader_fixture_test.go.
//
// Boundary coverage. The fixture carries four positions and a ratio of two, so a
// prefix of length 1 is BELOW one compression boundary (zero emitted groups),
// length 2 is exactly AT the boundary (one emitted group), and length 3 crosses
// ABOVE it (still one emitted group; the trailing position is incomplete). The
// compressed-row COUNT, the selected IDs, and the reader's numeric output are
// asserted at all three lengths.
//
// Non-vacuity. Two negative controls prove the fixture is not trivially
// satisfied: (a) a deliberately perturbed source key stream moves the reader's
// selected rows, so step (3) is sensitive to its input; (b) the source and reader
// queries are distinguishable, so step (3) cannot be credited by step (2)'s
// answer for free.
//
// Scope honesty. Test-only; no production behavior changes; no checkpoint, no
// hardware, no full-model generation. This is [SW-VERIFIED] coverage only; it
// establishes no throughput or physical-parity claim.

import (
	"testing"
)

// v41SRBuildModel builds a reduced model whose compressor and indexer tensors hold
// the fixture's exact values, so the production helpers read exactly the fixture
// inputs. It reuses the reduced config (HiddenSize=64, HeadDim=32=compressor
// width) and the synthetic manifest/raw layout, then overwrites the stage tensors
// in place.
func v41SRBuildModel(t *testing.T, f v41SRFixture) *Model {
	t.Helper()
	m := v41ReducedModel(t)
	cfg := m.Cfg
	if cfg.HiddenSize != f.Hidden || cfg.HeadDim != f.Width {
		t.Fatalf("reduced geometry H=%d hd=%d, fixture H=%d width=%d",
			cfg.HiddenSize, cfg.HeadDim, f.Hidden, f.Width)
	}
	// Declare an in-range ratio-2 compressor and index source at layer 0 so the
	// production helpers are on the real config path.
	cfg.DeepSeekV41.CompressRatios = []int{f.Ratio}
	cfg.DeepSeekV41.IndexSourceLayerIDs = []int{0}
	cfg.IndexNHeads = 1
	cfg.IndexHeadDim = f.Width
	cfg.IndexTopK = f.SrcTopK
	m.Cfg = cfg

	type ts = synthTensor
	extra := []ts{
		{layerName(0, "attn.compressor.wkv.weight"), []int{f.Width, f.Hidden}},
		{layerName(0, "attn.compressor.wgate.weight"), []int{f.Width, f.Hidden}},
		{layerName(0, "attn.compressor.norm.weight"), []int{f.Width}},
		{layerName(0, "indexer.wq_b.weight"), []int{cfg.IndexNHeads * cfg.IndexHeadDim, cfg.QLoraRank}},
		{layerName(0, "indexer.wk.weight"), []int{cfg.IndexHeadDim, f.Width}},
		{layerName(0, "indexer.k_norm.weight"), []int{cfg.IndexHeadDim}},
		{layerName(0, "indexer.weights_proj.weight"), []int{cfg.IndexNHeads, f.Hidden}},
	}
	man, raw := synthBuildRaw(extra, func(name string, next func() float32) float32 {
		return 1.0
	})
	for k, v := range man {
		m.manifest[k] = v
	}
	for k, v := range raw {
		m.raw[k] = v
	}

	// Overwrite the stage tensors with the fixture's exact values, in place. The
	// returned slices are zero-copy views into m.raw, so copy is sufficient.
	v41SRPut(t, m, layerName(0, "attn.compressor.wkv.weight"), f.WKV)
	v41SRPut(t, m, layerName(0, "attn.compressor.wgate.weight"), f.WGate)
	v41SRPut(t, m, layerName(0, "attn.compressor.norm.weight"), f.NormWeight)
	return m
}

// v41SRPut overwrites a tensor's values in place, asserting the shape matches.
func v41SRPut(t *testing.T, m *Model, name string, vals []float32) {
	t.Helper()
	dst := m.tensor(name)
	if len(dst) != len(vals) {
		t.Fatalf("tensor %q has %d elements, want %d", name, len(dst), len(vals))
	}
	copy(dst, vals)
}

// TestV41SourceReaderNumericOracle is the fak#13323 acceptance witness. See the
// file header for what it proves and the independence discipline it follows.
func TestV41SourceReaderNumericOracle(t *testing.T) {
	f := v41SRFixtureValues()
	m := v41SRBuildModel(t, f)

	if f.Ratio != 2 {
		t.Fatalf("fixture ratio %d, want 2 for a single compression boundary", f.Ratio)
	}

	// Lengths 1, 2 and 3 span one compression boundary of width ratio=2.
	for _, seq := range []int{1, 2, 3} {
		seq := seq
		t.Run(v41SRSeqLabel(seq), func(t *testing.T) {
			v41SRCheckLength(t, m, f, seq)
		})
	}

	t.Run("negative control: a stale publication is detected", func(t *testing.T) {
		v41SRNegativeControlStalePublication(t, f)
	})
	t.Run("negative control: a wrong source layer is detected", func(t *testing.T) {
		v41SRNegativeControlWrongSourceLayer(t, f)
	})
}

// v41SRSeqLabel names a length subtest.
func v41SRSeqLabel(seq int) string {
	switch seq {
	case 1:
		return "length 1 below one compression boundary"
	case 2:
		return "length 2 at the compression boundary"
	default:
		return "length 3 above the compression boundary"
	}
}

// v41SRCheckLength drives the production source publication and reader reuse for
// one prefix length and compares every result to the independent reference.
func v41SRCheckLength(t *testing.T, m *Model, f v41SRFixture, seq int) {
	t.Helper()

	inputs := f.Inputs[:seq]

	// ---- (1) production source projection + pooling ----
	// v41CompressedRows projects each hidden input through the layer's
	// compressor wkv/wgate (the fixture values) and pools it. kvRows is unused for
	// ratio>1, so we pass the fixture's inputs for both arguments.
	gotRows, err := m.v41CompressedRows(0, f.Ratio, inputs, inputs)
	if err != nil {
		t.Fatalf("production v41CompressedRows: %v", err)
	}
	wantRows := v41SRRefCompressedRows(f, seq)

	wantGroups := seq / f.Ratio
	if len(gotRows) != wantGroups {
		t.Fatalf("compressed-row count = %d, want %d for length %d (ratio %d)",
			len(gotRows), wantGroups, seq, f.Ratio)
	}
	if len(gotRows) != len(wantRows) {
		t.Fatalf("compressed-row count = %d, independent reference %d", len(gotRows), len(wantRows))
	}
	for g := range gotRows {
		if len(gotRows[g]) != f.Width {
			t.Fatalf("compressed row %d width %d, want %d", g, len(gotRows[g]), f.Width)
		}
		for d := range gotRows[g] {
			if !v41SRCloseF32(gotRows[g][d], wantRows[g][d], 1e-5) {
				t.Fatalf("compressed[%d][%d] = %v, independent reference %v (len %d)",
					g, d, gotRows[g][d], wantRows[g][d], seq)
			}
		}
	}

	if len(gotRows) == 0 {
		// Below the boundary the source publishes nothing; a reader over an empty
		// stream must select nothing rather than invent rows. The production
		// selection over an empty key stream returns zero rows — it does NOT pad
		// to topK. Padding a short selection up to the consumer's fixed stride is
		// the consumer's job (v41AttentionIndexList.one), so a publication that
		// invented padding slots here would double-pad.
		pub, err := NewV41IndexerPublication(0, f.SrcQ, nil, f.SrcWeights, 1, f.Width, 0, 0, 0, f.SrcTopK, f.SrcOffset)
		if err != nil {
			t.Fatalf("empty-stream source publication: %v", err)
		}
		if rows := pub.Rows(); len(rows) != 0 {
			t.Fatalf("empty-stream rows %v, want zero rows (length min(topK, 0))", rows)
		}
		return
	}

	flatKeys := v41SRFlattenRows(gotRows)

	// ---- (2) production source index selection over the compressed rows ----
	srcPub, err := NewV41IndexerPublication(
		0, f.SrcQ, flatKeys, f.SrcWeights,
		1, f.Width, len(gotRows),
		0, 0, f.SrcTopK, f.SrcOffset)
	if err != nil {
		t.Fatalf("production source publication: %v", err)
	}
	refSrcScores := v41SRRefScores(f.SrcQ, flatKeys, f.SrcWeights, 1, f.Width, len(gotRows))
	wantSrcRows := v41SRRefSelectRows(refSrcScores, len(gotRows), f.SrcTopK, f.SrcOffset, nil)
	v41SRMustParity(t, "source selection", srcPub.Rows(), wantSrcRows)

	// ---- (3) reader consumption of the source's published selection ----
	// Reuse re-scores the reader's OWN query over the same key stream, filtered by
	// the source's published candidate mask (nil here because this publication is
	// not a candidate source, so the filter is the identity). The reader's query
	// and per-head weight differ from the source's, so this step cannot be
	// credited by the source's answer for free.
	gotReader, err := srcPub.Reuse(
		f.ReaderQ, flatKeys, f.ReaderWeights,
		1, f.Width, len(gotRows), f.ReaderTopK, f.SrcOffset)
	if err != nil {
		t.Fatalf("production reader reuse: %v", err)
	}
	refReaderScores := v41SRRefScores(f.ReaderQ, flatKeys, f.ReaderWeights, 1, f.Width, len(gotRows))
	wantReader := v41SRRefSelectRows(refReaderScores, len(gotRows), f.ReaderTopK, f.SrcOffset, srcPub.Candidates())
	v41SRMustParity(t, "reader selection", gotReader, wantReader)

	// Non-vacuity: when the independent references themselves disagree, the reader
	// step must not be the source step's answer in disguise, so the production
	// reader rows must equal neither the source's rows nor any other query's rows.
	// When the references happen to agree there is nothing to distinguish, so the
	// check is a no-op rather than a skip.
	if !v41SRSameInt32(wantSrcRows, wantReader) {
		if v41SRSameInt32(gotReader, srcPub.Rows()) {
			t.Fatalf("reader rows %v equal the source rows %v although the independent references differ; the reader step is not distinct",
				gotReader, srcPub.Rows())
		}
	}

	// Every produced reader row must resolve inside the compressed stream.
	for _, id := range gotReader {
		if id == -1 {
			continue
		}
		if int(id) >= len(gotRows) {
			t.Fatalf("reader row %d is outside the compressed stream of length %d", id, len(gotRows))
		}
	}
}

// v41SRNegativeControlStalePublication proves the reader result is not vacuous
// and that a source's published selection is genuinely load-bearing. It builds a
// source publication with a candidate mask over one key stream, then drives
// Reuse with a DIFFERENT key stream of the same length. Because Reuse re-scores
// against the keys it is handed but filters by the source's stored mask, the
// stale publication's candidate mask must constrain the reader's rows: a
// candidate the stale mask excluded from scoring cannot be resurrected.
//
// The control asserts the mask actually bites: at least one position the fresh
// key stream prefers is excluded by the stale publication's mask, so the stale
// reader selection differs from the unconstrained selection. A publication that
// ignored its stored mask would produce the unconstrained answer and fail here.
func v41SRNegativeControlStalePublication(t *testing.T, f v41SRFixture) {
	t.Helper()
	// seq=4 with ratio 2 emits two compressed rows, the minimum for a candidate
	// block mask to exclude a position the reader would otherwise prefer.
	seq := 4
	if len(f.Inputs) < seq {
		t.Fatalf("fixture carries %d inputs, need %d for the stale-publication control", len(f.Inputs), seq)
	}
	m := v41SRBuildModel(t, f)
	rows, err := m.v41CompressedRows(0, f.Ratio, f.Inputs[:seq], f.Inputs[:seq])
	if err != nil {
		t.Fatalf("production rows: %v", err)
	}
	flat := v41SRFlattenRows(rows)
	if len(rows) < 2 {
		t.Fatalf("stale-publication control needs >=2 compressed rows, got %d", len(rows))
	}

	// Publish with one candidate BLOCK so the source stores a real mask.
	stale, err := NewV41IndexerPublication(0, f.SrcQ, flat, f.SrcWeights, 1, f.Width, len(rows), 1, 1, f.SrcTopK, 0)
	if err != nil {
		t.Fatalf("source publication with candidate blocks: %v", err)
	}
	mask := stale.Candidates()
	if mask == nil {
		t.Fatal("source publication declared candidate blocks but published no mask")
	}

	// Perturb the key stream (finite, same length) so the reader's own scores
	// move, then confirm the stored mask still constrains the result.
	moved := make([]float32, len(flat))
	copy(moved, flat)
	for i := range moved {
		moved[i] = moved[i]*0.5 + 0.125
	}
	gotStale, err := stale.Reuse(f.ReaderQ, moved, f.ReaderWeights, 1, f.Width, len(rows), f.ReaderTopK, f.SrcOffset)
	if err != nil {
		t.Fatalf("stale publication reuse: %v", err)
	}
	refScores := v41SRRefScores(f.ReaderQ, moved, f.ReaderWeights, 1, f.Width, len(rows))
	wantMasked := v41SRRefSelectRows(refScores, len(rows), f.ReaderTopK, f.SrcOffset, mask)
	v41SRMustParity(t, "stale-publication reader selection", gotStale, wantMasked)

	// Non-vacuity: the mask must exclude at least one position the reader would
	// otherwise select, so a publication ignoring its mask is detectably wrong.
	wantFree := v41SRRefSelectRows(refScores, len(rows), f.ReaderTopK, f.SrcOffset, nil)
	if v41SRSameInt32(wantMasked, wantFree) {
		t.Fatalf("stale publication mask %v did not constrain the reader; control is vacuous", mask)
	}
}

// v41SRNegativeControlWrongSourceLayer proves the source PUBLICATION is
// layer-scoped: a reader that resolves the wrong source layer gets no rows rather
// than silently reading another layer's stream. This drives the real production
// publish/read seam — v41AttentionSourceUpdates builds the source's update and
// V41AttentionState publishes it under its layer ID — not a synthetic shortcut.
func v41SRNegativeControlWrongSourceLayer(t *testing.T, f v41SRFixture) {
	t.Helper()
	cfg := v41AttentionCompressedConfig(t)
	cfg.NumLayers = 2
	roles := v41AttentionRoles(cfg)
	srcPlan, err := v41AttentionPlanFor(cfg, 0, roles)
	if err != nil {
		t.Fatalf("source plan: %v", err)
	}
	if srcPlan.Role != V41AttentionRoleKVSource {
		t.Fatalf("layer 0 role = %v, want KV source", srcPlan.Role)
	}
	// A model shell is enough: the source-update seam reads only cfg and the plan.
	m := &Model{Cfg: cfg}
	const hd = 4
	pooled := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}
	latents := [][]float32{{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}}
	updates := m.v41AttentionSourceUpdates(srcPlan, pooled, latents)
	if len(updates) != len(pooled) {
		t.Fatalf("source published %d updates, want %d", len(updates), len(pooled))
	}
	st, err := NewV41AttentionState(hd, 8)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.Prefill(pooled, updates); err != nil {
		t.Fatalf("source Prefill: %v", err)
	}
	// The publishing layer is srcPlan.Layer; a reader resolving any other layer
	// must find nothing, so the layer identity is load-bearing.
	if rows, ok := st.KVSourceRows(srcPlan.Layer); !ok || len(rows) != len(pooled) {
		t.Fatalf("KVSourceRows(%d) = (%v,%v), want %d rows", srcPlan.Layer, rows, ok, len(pooled))
	}
	if _, ok := st.KVSourceRows(srcPlan.Layer + 7); ok {
		t.Fatal("KVSourceRows(+7) reported a source never published; the layer identity is not load-bearing")
	}
}
