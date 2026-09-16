package model

// v41_attention_test.go — the independent witness for the DeepSeek V4.1 CED/CSA2
// compressed/shared attention seam (issue #12896, leaf of parent #12640). It
// exercises the seam this leaf adds: the compressed contraction over a pooled KV
// stream (V41AttentionCompressedForward) with block-causal visibility, the
// shared KV/index source-layer resolution, the fail-closed refusal of an
// unimplemented compress ratio, and Prefill/Step continuity.
//
// Independence discipline (family_cpu_oracle_test.go). The compressed-attention
// oracle below reuses NONE of the production contraction machinery that it is
// checking — not V41AttentionCompressedForward, not v41CompressedCausalMask,
// not V41SparseAttentionSink. It hardcodes the reference's scalar dataflow over
// the fixture's own rows, so it is a genuine cross-implementation comparison,
// not a production-vs-production tautology.

import (
	"errors"
	"math"
	"testing"
)

// v41AttentionCompressedConfig is the reduced fixture's compressed/shared
// schedule: layer 0 is a compressed KV+index source (ratio 2), which the seam
// must execute as a real pooled contraction. The compressor and indexer tensors
// are supplied by v41AttentionModel.
func v41AttentionCompressedConfig(t *testing.T) Config {
	t.Helper()
	cfg := v41TestReducedConfig(t, 1, V41RouterExperts)
	cfg.NumExpertsPerTok = V41RouterTopK
	cfg.NSharedExperts = 1
	cfg.RoutedScalingFactor = 1.5
	cfg.RopeScaling = ""
	cfg.LongRope = nil
	cfg.RopeFactor = 0
	cfg.RopeOrigContext = 0
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000
	}
	cfg.DeepSeekV41.CompressRatios = []int{2}
	cfg.DeepSeekV41.KVSourceLayerIDs = []int{0}
	cfg.DeepSeekV41.IndexSourceLayerIDs = []int{0}
	cfg.IndexNHeads = 2
	cfg.IndexHeadDim = 4
	cfg.IndexTopK = 2
	return cfg
}

// v41AttentionModel builds a reduced V4.1 model whose layer 0 is a compressed
// KV+index source, wired with the compressor and indexer tensors the seam reads.
func v41AttentionModel(t *testing.T) *Model {
	t.Helper()
	m := v41ReducedModel(t)
	cfg := v41AttentionCompressedConfig(t)
	m.Cfg = cfg

	H := cfg.HiddenSize
	width := v41CompressorWidth(cfg)
	type ts = synthTensor
	extra := []ts{
		{layerName(0, "attn.compressor.wkv.weight"), []int{width, H}},
		{layerName(0, "attn.compressor.wgate.weight"), []int{width, H}},
		{layerName(0, "attn.compressor.norm.weight"), []int{width}},
		{layerName(0, "indexer.wq_b.weight"), []int{cfg.IndexNHeads * cfg.IndexHeadDim, cfg.QLoraRank}},
		{layerName(0, "indexer.wk.weight"), []int{cfg.IndexHeadDim, width}},
		{layerName(0, "indexer.k_norm.weight"), []int{cfg.IndexHeadDim}},
		{layerName(0, "indexer.weights_proj.weight"), []int{cfg.IndexNHeads, H}},
	}
	man, raw := synthBuildRaw(extra, func(name string, next func() float32) float32 {
		switch {
		case hasSuffix(name, "compressor.norm.weight") || hasSuffix(name, "indexer.k_norm.weight"):
			return 1.0
		default:
			return synthMatmulFill(name, next)
		}
	})
	for k, v := range man {
		m.manifest[k] = v
	}
	for k, v := range raw {
		m.raw[k] = v
	}
	return m
}

// v41OracleCompressedForward is the independent scalar reference for the CED/CSA2
// compressed contraction. It shares no code with V41AttentionCompressedForward.
func v41OracleCompressedForward(t *testing.T, ratio int, q, rows [][]float32, sink []float32, scale float32) [][]float32 {
	t.Helper()
	seq := len(q)
	groups := len(rows)
	hd := len(rows[0])
	nH := len(q[0]) / hd
	out := make([][]float32, seq)
	for pos := 0; pos < seq; pos++ {
		o := make([]float32, nH*hd)
		for h := 0; h < nH; h++ {
			qh := q[pos][h*hd : (h+1)*hd]
			var visible []int
			for g := 0; g < groups; g++ {
				if (g+1)*ratio-1 <= pos {
					visible = append(visible, g)
				}
			}
			maxScore := float64(sink[h])
			dots := make([]float64, len(visible))
			for i, g := range visible {
				var d float64
				for j := 0; j < hd; j++ {
					d += float64(qh[j]) * float64(rows[g][j])
				}
				d *= float64(scale)
				dots[i] = d
				if d > maxScore {
					maxScore = d
				}
			}
			if math.IsInf(maxScore, -1) {
				out[pos] = o
				break
			}
			sum := math.Exp(float64(sink[h]) - maxScore)
			for _, d := range dots {
				sum += math.Exp(d - maxScore)
			}
			if sum == 0 {
				continue
			}
			for i, g := range visible {
				w := math.Exp(dots[i]-maxScore) / sum
				for j := 0; j < hd; j++ {
					o[h*hd+j] += float32(w * float64(rows[g][j]))
				}
			}
		}
		out[pos] = o
	}
	return out
}

// TestV41AttentionCompressedOracleMatches is the #12896 scalar-oracle witness:
// the production compressed contraction matches an independent scalar
// transcription on a reduced fixture, including the block-causal boundary.
func TestV41AttentionCompressedOracleMatches(t *testing.T) {
	const (
		ratio  = 2
		groups = 3
		hd     = 4
		heads  = 2
		seq    = 6
	)
	rows := [][]float32{
		{0.5, -0.25, 0.75, 0.1},
		{-0.3, 0.4, -0.6, 0.2},
		{0.15, -0.5, 0.25, -0.35},
	}
	q := make([][]float32, seq)
	for pos := 0; pos < seq; pos++ {
		row := make([]float32, heads*hd)
		for i := range row {
			row[i] = float32(math.Sin(float64(pos*7+i))) * 0.5
		}
		q[pos] = row
	}
	sink := []float32{0.25, -0.1}
	scale := float32(0.125)

	flatQ := flatten(q)
	got, err := V41AttentionCompressedForward(flatQ, rows, V41AttentionSharedKVOptions{
		Layer: 0, Ratio: ratio, Groups: groups, HeadDim: hd, Heads: heads,
		TopK: 0, Softmax: scale, Sink: sink,
	})
	if err != nil {
		t.Fatalf("V41AttentionCompressedForward: %v", err)
	}
	want := v41OracleCompressedForward(t, ratio, q, rows, sink, scale)
	wantFlat := flatten(want)
	if len(got) != len(wantFlat) {
		t.Fatalf("output length = %d, want %d", len(got), len(wantFlat))
	}
	for i := range got {
		if d := math.Abs(float64(got[i] - wantFlat[i])); d > 1e-5 {
			t.Fatalf("compressed attention out[%d] = %g, want %g (|delta| = %.3e)", i, got[i], wantFlat[i], d)
		}
	}

	// The block-causal boundary: group g is visible only once its last position
	// (g+1)*ratio-1 has closed, so at pos 0 no group is visible and the
	// contraction must return an all-zero row (the reference's finite bound).
	// At pos ratio-1 (== 1) group 0 has just closed and IS visible.
	for j := 0; j < heads*hd; j++ {
		if got[j] != 0 {
			t.Fatalf("pos 0 (before any group closes) out[%d] = %g, want 0", j, got[j])
		}
	}
	// Conversely, the first closed group IS visible at pos ratio-1 and produces a
	// non-zero row, so the zero above is the causal boundary, not a dead path.
	nonzero := false
	for j := 0; j < heads*hd; j++ {
		if got[(ratio-1)*heads*hd+j] != 0 {
			nonzero = true
			break
		}
	}
	if !nonzero {
		t.Fatal("first closed group produced an all-zero row; the contraction never fired")
	}
}

// TestV41AttentionSharedSourceResolution witnesses the shared KV/index
// source-layer resolution the seam adds.
func TestV41AttentionSharedSourceResolution(t *testing.T) {
	t.Run("role schedule", func(t *testing.T) {
		cfg := v41AttentionCompressedConfig(t)
		cfg.NumLayers = 4
		cfg.DeepSeekV41.CompressRatios = []int{2, 0, 0, 0}
		cfg.DeepSeekV41.KVSourceLayerIDs = []int{0}
		cfg.DeepSeekV41.IndexSourceLayerIDs = []int{0}
		roles := v41AttentionRoles(cfg)
		if roles[0] != V41AttentionRoleKVSource {
			t.Fatalf("layer 0 role = %v, want KV source", roles[0])
		}
		for l := 1; l < 4; l++ {
			if roles[l] != V41AttentionRoleReader {
				t.Fatalf("layer %d role = %v, want reader of the preceding source", l, roles[l])
			}
		}
	})

	t.Run("plan source resolution", func(t *testing.T) {
		cfg := v41AttentionCompressedConfig(t)
		cfg.NumLayers = 5
		cfg.DeepSeekV41.CompressRatios = []int{2, 0, 2, 0, 0}
		cfg.DeepSeekV41.KVSourceLayerIDs = []int{0, 2}
		cfg.DeepSeekV41.IndexSourceLayerIDs = []int{0, 2}
		roles := v41AttentionRoles(cfg)
		for _, tc := range []struct{ layer, wantSrc int }{
			{0, 0}, {1, 0}, {2, 2}, {3, 2}, {4, 2},
		} {
			plan, err := v41AttentionPlanFor(cfg, tc.layer, roles)
			if err != nil {
				t.Fatalf("plan layer %d: %v", tc.layer, err)
			}
			if plan.KVSourceLayer != tc.wantSrc {
				t.Fatalf("layer %d KV source = %d, want %d", tc.layer, plan.KVSourceLayer, tc.wantSrc)
			}
			if plan.IndexSourceLayer != tc.wantSrc {
				t.Fatalf("layer %d index source = %d, want %d", tc.layer, plan.IndexSourceLayer, tc.wantSrc)
			}
		}
	})

	t.Run("published rows are what a reader reads", func(t *testing.T) {
		st, err := NewV41AttentionState(4, 8)
		if err != nil {
			t.Fatal(err)
		}
		published := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}
		if err := st.Prefill([][]float32{{1, 0, 0, 0}}, []V41AttentionStateUpdate{
			{Ref: V41AttentionStateRef{LayerID: 0, Ratio: 2, IsKVSource: true, IsIndexSource: true}, Latent: published[0], IndexKey: published[0]},
			{Ref: V41AttentionStateRef{LayerID: 0, Ratio: 2, IsKVSource: true, IsIndexSource: true}, Latent: published[1], IndexKey: published[1]},
		}); err != nil {
			t.Fatal(err)
		}
		rows, ok := st.KVSourceRows(0)
		if !ok || len(rows) != 2 {
			t.Fatalf("KVSourceRows(0) = (%v,%v), want 2 rows", rows, ok)
		}
		for g := range published {
			for d := range published[g] {
				if rows[g][d] != published[g][d] {
					t.Fatalf("reader row[%d][%d] = %g, want %g", g, d, rows[g][d], published[g][d])
				}
			}
		}
		if _, ok := st.KVSourceRows(3); ok {
			t.Fatal("KVSourceRows(3) reported a source that was never published")
		}
	})
}

// TestV41AttentionUnimplementedRatioFailsClosed witnesses the fail-closed
// discipline: a compress ratio the reduced assembly cannot represent as a V4.1
// CED/CSA2 contraction must refuse with a typed error wrapping
// ErrV41ForwardStage, never silently pool at a different width or fall through
// to the generic per-layer Q/K/V attention.
func TestV41AttentionUnimplementedRatioFailsClosed(t *testing.T) {
	if !v41AttentionRatioImplemented(2) || !v41AttentionRatioImplemented(0) || !v41AttentionRatioImplemented(1) {
		t.Fatal("the published V4.1 regimes 0/1/2 must be implemented")
	}
	for _, ratio := range []int{3, 5, 100, 128, 4096} {
		if v41AttentionRatioImplemented(ratio) {
			t.Fatalf("ratio %d reported implemented", ratio)
		}
		cfg := v41AttentionCompressedConfig(t)
		cfg.NumLayers = 1
		cfg.DeepSeekV41.CompressRatios = []int{ratio}
		cfg.DeepSeekV41.KVSourceLayerIDs = nil
		cfg.DeepSeekV41.IndexSourceLayerIDs = nil
		roles := v41AttentionRoles(cfg)
		_, err := v41AttentionPlanFor(cfg, 0, roles)
		if !errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("ratio %d plan error = %v, want ErrV41ForwardStage", ratio, err)
		}
	}

	cfg := v41AttentionCompressedConfig(t)
	cfg.DeepSeekV41.CompressRatios = []int{-4}
	_, err := v41AttentionPlanFor(cfg, 0, v41AttentionRoles(cfg))
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("negative ratio plan error = %v, want ErrV41ForwardStage", err)
	}
}

// TestV41AttentionPrefillStepDeterministic witnesses that the compressed/shared
// seam is deterministic across a two-token Prefill followed by a Step: the
// Step's logits equal a single longer Forward's last-position logits, and two
// identical runs produce bit-identical output.
func TestV41AttentionPrefillStepDeterministic(t *testing.T) {
	m := v41AttentionModel(t)
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("compressed/shared admission error = %v, want nil (the seam executes)", err)
	}

	ids := []int{2, 4}
	s := &Session{M: m}
	prefill := s.Prefill(ids)
	if len(prefill) == 0 {
		t.Fatal("Prefill returned no logits")
	}
	step := s.Step(6)
	long := m.Forward([]int{2, 4, 6})
	if long == nil || len(long.Logits) != 3 {
		t.Fatalf("long Forward returned %v positions, want 3", long)
	}
	v41LogitsClose(t, "v41 attention step-vs-forward", step, long.Logits[2])

	s2 := &Session{M: m}
	_ = s2.Prefill(ids)
	step2 := s2.Step(6)
	if len(step) != len(step2) {
		t.Fatalf("step logits length %d vs %d", len(step), len(step2))
	}
	for i := range step {
		if step[i] != step2[i] {
			t.Fatalf("step logits[%d] = %g vs %g; not deterministic", i, step[i], step2[i])
		}
	}
}

// TestV41AttentionSourcePublishReaderReuse is the end-to-end witness for the
// shared KV/index publication pair on a 3-layer schedule: layer 0 is a
// compressed source, layer 1 is a reader. It drives the exact seam functions
// the forward runs — v41AttentionSourceUpdates (the source's publication) and
// v41AttentionIndexList (the reader's reuse) — through a real
// V41AttentionState, proving the reader consumes the source's published rows
// and selection rather than recomputing them, and that a reader whose source
// published nothing fails closed with a typed error.
func TestV41AttentionSourcePublishReaderReuse(t *testing.T) {
	cfg := v41AttentionCompressedConfig(t)
	cfg.NumLayers = 2
	cfg.DeepSeekV41.CompressRatios = []int{2, 0}
	cfg.DeepSeekV41.KVSourceLayerIDs = []int{0}
	cfg.DeepSeekV41.IndexSourceLayerIDs = []int{0}
	// A model shell is enough: these seam functions read only cfg and the state.
	m := &Model{Cfg: cfg}

	roles := v41AttentionRoles(cfg)
	srcPlan, err := v41AttentionPlanFor(cfg, 0, roles)
	if err != nil {
		t.Fatalf("source plan: %v", err)
	}
	readerPlan, err := v41AttentionPlanFor(cfg, 1, roles)
	if err != nil {
		t.Fatalf("reader plan: %v", err)
	}
	if srcPlan.Role != V41AttentionRoleKVSource {
		t.Fatalf("layer 0 role = %v, want KV source", srcPlan.Role)
	}
	if readerPlan.Role != V41AttentionRoleReader {
		t.Fatalf("layer 1 role = %v, want reader", readerPlan.Role)
	}

	// The source publishes two completed compressed rows. The pooled rows are the
	// source's own output; the update carries them for later readers.
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
	// The reader resolves the source's published rows and reads them byte-equal.
	rows, ok := st.KVSourceRows(srcPlan.Layer)
	if !ok || len(rows) != len(pooled) {
		t.Fatalf("reader KVSourceRows = (%v,%v), want %d rows", rows, ok, len(pooled))
	}
	for g := range pooled {
		for d := range pooled[g] {
			if rows[g][d] != pooled[g][d] {
				t.Fatalf("reader row[%d][%d] = %g, want %g", g, d, rows[g][d], pooled[g][d])
			}
		}
	}

	// The source's own local selection is the authority; v41AttentionIndexList
	// replicates it across the published positions at the resolved stride.
	local := []int32{0, 1}
	indexList, err := m.v41AttentionIndexList(srcPlan, nil, local, hd, len(pooled), 2)
	if err != nil {
		t.Fatalf("source index list: %v", err)
	}
	if len(indexList) != 2*srcPlan.TopKWidth {
		t.Fatalf("source index list length = %d, want %d", len(indexList), 2*srcPlan.TopKWidth)
	}
	for t2 := 0; t2 < 2; t2++ {
		for i := 0; i < srcPlan.TopKWidth; i++ {
			if indexList[t2*srcPlan.TopKWidth+i] != local[i] {
				t.Fatalf("source indexList[%d][%d] = %d, want %d", t2, i, indexList[t2*srcPlan.TopKWidth+i], local[i])
			}
		}
	}

	// The reader reuses the source's published top-k selection. Publish it (as
	// the forward does for an index source) and confirm the reader resolves it.
	pubRows := [][]int32{{0, 1}, {0, 1}}
	if err := st.PublishTopK(srcPlan.Ratio, pubRows); err != nil {
		t.Fatal(err)
	}
	reused, err := m.v41AttentionIndexList(readerPlan, &v41ForwardState{attn: st}, nil, hd, len(pooled), 2)
	if err != nil {
		t.Fatalf("reader index list: %v", err)
	}
	if len(reused) != len(pubRows)*readerPlan.TopKWidth {
		t.Fatalf("reader index list length = %d, want %d", len(reused), len(pubRows)*readerPlan.TopKWidth)
	}
	for i := range reused {
		if reused[i] != pubRows[i/readerPlan.TopKWidth][i%readerPlan.TopKWidth] {
			t.Fatalf("reader reused indexList[%d] = %d, want %d", i, reused[i], pubRows[i/readerPlan.TopKWidth][i%readerPlan.TopKWidth])
		}
	}

	// Fail closed: a reader whose source published no top-k selection must refuse
	// rather than silently contract against an empty index list.
	empty, err2 := NewV41AttentionState(hd, 8)
	if err2 != nil {
		t.Fatal(err2)
	}
	if _, err := m.v41AttentionIndexList(readerPlan, &v41ForwardState{attn: empty}, nil, hd, len(pooled), 2); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("reader with no publication error = %v, want ErrV41ForwardStage", err)
	}
}
