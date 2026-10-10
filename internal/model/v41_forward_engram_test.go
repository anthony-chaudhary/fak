package model

// v41_forward_engram_test.go is the pub#13577 acceptance witness: on a full-
// geometry model the Engram injection must advance EACH of the four distinct
// persistent mHC streams from that stream's OWN independently computed gate,
// instead of collapsing all four gates onto the single vector x[t].
//
// The pre-fix production path computed one `h := x[t]` and, on the full path,
// wrote the same gated delta into stream 0 only, leaving streams 1..3 untouched.
// The candidate computes h_s = streams[t][s] per stream, so streams 1..3 move and
// no two per-stream deltas are equal. This test builds four ASYMMETRIC streams
// and a scalar oracle that never calls v41EngramInject, so it would fail on the
// parent (streams 1..3 unchanged) and pass on the candidate.
//
// It also pins the two compatibility contracts in the DONE condition: the
// reduced fixture still matches v41OracleEngramForward (reduced fixtures remain
// byte-for-byte compatible) and a deleted stage still fails closed with
// ErrV41NativeUnsupported.

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// v41FullEngramModel builds a small raw-backed FULL-geometry V4.1 model (H=64,
// NumLayers=1) whose single in-range layer 0 declares Engram. It carries the
// full-geometry attention tensors (attn.wq_a_norm / attn.kv_norm, 512-wide KV
// latent, OGroups=1) plus the three Engram mixing tensors, and wires a synthetic
// packed-row source. mhc.mixes.weight is stored in the artifact's [4H, 24]
// flattened orientation so the full forward uses v41MHCProjectFull.
func v41FullEngramModel(t *testing.T) (*Model, V41EngramLayout) {
	t.Helper()
	cfg := v41FullGeometryConfig(t)
	cfg.OGroups = 1
	cfg.DeepSeekV41.EngramLayerIDs = []int{0}
	cfg.DeepSeekV41.EngramNumEmbeddings = []int{48}
	cfg.DeepSeekV41.EngramMaxNgramSize = 4
	cfg.DeepSeekV41.EngramNHeads = 8
	cfg.DeepSeekV41.EngramHeadDim = cfg.HiddenSize

	H := cfg.HiddenSize
	I := cfg.MoEIntermediateSize
	qHeadDim := cfg.NumHeads * cfg.HeadDim
	oDim := cfg.OLoraRank * cfg.OGroups
	cols := (cfg.DeepSeekV41.EngramMaxNgramSize - 1) * cfg.DeepSeekV41.EngramNHeads

	type ts = synthTensor
	tensors := []ts{
		{"model.embed_tokens.weight", []int{cfg.VocabSize, H}},
		{"lm_head.weight", []int{cfg.VocabSize, H}},
		{"model.norm.weight", []int{H}},
		{layerName(0, "attn_norm.weight"), []int{H}},
		{layerName(0, "ffn_norm.weight"), []int{H}},
		{layerName(0, "mhc.mixes.weight"), []int{4 * H, v41MHCMixWidth}},
		{layerName(0, "mhc.ffn_mixes.weight"), []int{4 * H, v41MHCMixWidth}},
		{layerName(0, "mhc.ffn_base"), []int{v41MHCMixWidth}},
		{layerName(0, "mhc.ffn_scale"), []int{3}},
		{layerName(0, "mhc.base"), []int{v41MHCMixWidth}},
		{layerName(0, "mhc.scale"), []int{3}},
		{layerName(0, "attn.wq_a.weight"), []int{cfg.QLoraRank, H}},
		{layerName(0, "attn.wq_a_norm.weight"), []int{cfg.QLoraRank}},
		{layerName(0, "attn.wq_b.weight"), []int{qHeadDim, cfg.QLoraRank}},
		{layerName(0, "attn.wkv.weight"), []int{v41KVLoraRank, H}},
		{layerName(0, "attn.kv_norm.weight"), []int{v41KVLoraRank}},
		{layerName(0, "attn.wo_a.weight"), []int{cfg.OLoraRank, qHeadDim}},
		{layerName(0, "attn.wo_b.weight"), []int{H, oDim}},
		{layerName(0, "attn.sink"), []int{cfg.NumHeads}},
		{layerName(0, "ffn.gate.weight"), []int{cfg.NumExperts, H}},
		{layerName(0, "ffn.gate.e_score_correction_bias"), []int{cfg.NumExperts}},
		{layerName(0, "ffn.shared_experts.w1.weight"), []int{I, H}},
		{layerName(0, "ffn.shared_experts.w3.weight"), []int{I, H}},
		{layerName(0, "ffn.shared_experts.w2.weight"), []int{H, I}},
		{layerName(0, "engram_kv.weight"), []int{cols * cfg.DeepSeekV41.EngramHeadDim, (4 + 1) * H}},
		{layerName(0, "engram_q_norm.weight"), []int{4 * H}},
		{layerName(0, "engram_k_norm.weight"), []int{4 * H}},
	}
	for e := 0; e < cfg.NumExperts; e++ {
		stem := "ffn.experts." + itoa(e)
		tensors = append(tensors,
			ts{layerName(0, stem+".w1.weight"), []int{I, H}},
			ts{layerName(0, stem+".w3.weight"), []int{I, H}},
			ts{layerName(0, stem+".w2.weight"), []int{H, I}},
		)
	}

	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		switch {
		case name == "model.norm.weight" || hasSuffix(name, "attn_norm.weight") ||
			hasSuffix(name, "ffn_norm.weight") || hasSuffix(name, "attn.wq_a_norm.weight") ||
			hasSuffix(name, "attn.kv_norm.weight"):
			return 1.0
		case hasSuffix(name, "mhc.ffn_scale"):
			return .75
		case hasSuffix(name, "mhc.ffn_base"):
			return .125 * next()
		case hasSuffix(name, "mhc.scale"):
			return 1.0
		case hasSuffix(name, "mhc.base"):
			return 0.0
		case hasSuffix(name, "attn.sink"):
			return 0.25 * next()
		default:
			return synthMatmulFill(name, next)
		}
	})
	m := &Model{Cfg: cfg, manifest: man, raw: raw}

	layout := v41EngramTestLayout(cfg)
	src := &v41EngramMemorySource{
		packed: v41EngramTestPackedRows(int(layout.Rows[0])),
		rows:   int(layout.Rows[0]),
	}
	if err := m.wireV41Engram(layout, []V41EngramRowSource{src}, int64(V41EngramPackedRowBytes)*4); err != nil {
		t.Fatalf("wire full Engram stage: %v", err)
	}
	return m, layout
}

// v41FullEngramAsymmetricStreams builds len(ids) positions of four DISTINCT
// width-H persistent mHC vectors. Stream s is a deterministic ramp scaled by
// (s+1), so the four RMS/dot statistics are independent and produce four distinct
// gates. This fixture deliberately aliases x[t] with stream 0; production also
// supports independently owned x and residual buffers.
func v41FullEngramAsymmetricStreams(ids []int, H int) (x [][]float32, streams [][][]float32) {
	streams = make([][][]float32, len(ids))
	x = make([][]float32, len(ids))
	for tt := range ids {
		set := make([][]float32, 4)
		for s := 0; s < 4; s++ {
			v := make([]float32, H)
			for i := range v {
				v[i] = v41OracleBF16(float32(0.05*float64(s+1))*float32((i%7)-3) + float32(s)*0.1)
			}
			set[s] = v
		}
		streams[tt] = set
		x[tt] = set[0] // deliberate aliasing witness for the explicit synchronization
	}
	return x, streams
}

// v41OracleFullEngramUpdates independently transcribes the reference per-stream
// schedule WITHOUT calling v41EngramInject: hash -> gather -> dequant -> project
// -> per-stream gate from that stream's own vector -> single-stream write-back.
// h2/k2/dot are accumulated in float64; the F32 residual sum publishes BF16,
// matching the f32 forward within cpuOracleTol. It returns the expected post-call
// stream values as [position][stream][dim].
func v41OracleFullEngramUpdates(t *testing.T, m *Model, layout V41EngramLayout, l int, streams [][][]float32, ids []int, eps float32) [][][]float32 {
	t.Helper()
	cfg := m.Cfg
	H := cfg.HiddenSize
	dim := cfg.DeepSeekV41.EngramHeadDim
	hc := 4
	cols := (layout.MaxNgramSize - 1) * layout.HeadsPerNgram

	stage := m.v41EngramStageFor()
	if stage == nil {
		t.Fatal("oracle: Engram stage not wired")
	}
	hash, err := NewV41EngramHashState(layout)
	if err != nil {
		t.Fatal(err)
	}
	allRows, err := hash.Hash(ids, nil)
	if err != nil {
		t.Fatal(err)
	}
	layers := len(layout.Rows)
	cacheIdx := stage.cacheIndex(l)
	layerRows := make([]uint32, 0, len(ids)*cols)
	for tt := range ids {
		base := tt*layers*cols + cacheIdx*cols
		layerRows = append(layerRows, allRows[base:base+cols]...)
	}
	gathered, err := GatherV41EngramRows([]*V41EngramRowCache{stage.caches[cacheIdx]}, layerRows, cols)
	if err != nil {
		t.Fatal(err)
	}

	wKV := cpuOracleTensor(t, m, layerName(l, "engram_kv.weight"))
	qNorm := cpuOracleTensor(t, m, layerName(l, "engram_q_norm.weight"))
	kNorm := cpuOracleTensor(t, m, layerName(l, "engram_k_norm.weight"))

	want := make([][][]float32, len(ids))
	for tt := range ids {
		rowVec := make([]float32, cols*dim)
		for c := 0; c < cols; c++ {
			copy(rowVec[c*dim:], v41OracleEngramDequant(t, gathered[tt*cols+c], dim))
		}
		projected := cpuOracleMatVec(wKV, rowVec, (hc+1)*H, cols*dim)
		value := make([]float32, H)
		for i := 0; i < H; i++ {
			value[i] = v41OracleBF16(projected[hc*H+i])
		}
		want[tt] = make([][]float32, hc)
		for s := 0; s < hc; s++ {
			h := streams[tt][s]
			key := make([]float32, H)
			var h2, k2, dot float64
			for i := 0; i < H; i++ {
				key[i] = v41OracleBF16(projected[s*H+i])
				hh := float64(h[i])
				kk := float64(key[i])
				h2 += hh * hh
				k2 += kk * kk
				dot += hh * float64(qNorm[s*H+i]) * float64(kNorm[s*H+i]) * kk
			}
			dot *= 1 / math.Sqrt(h2/float64(H)+float64(eps))
			dot *= 1 / math.Sqrt(k2/float64(H)+float64(eps))
			dot *= 1 / math.Sqrt(float64(H))
			gate := 1 / (1 + math.Exp(-math.Copysign(math.Sqrt(math.Max(math.Abs(dot), 1e-6)), dot)))
			row := make([]float32, H)
			for i := 0; i < H; i++ {
				delta := float32(float32(gate) * value[i])
				row[i] = v41OracleBF16(float32(h[i] + delta))
			}
			want[tt][s] = row
		}
	}
	return want
}

// TestV41EngramFullPersistentStreams is the pub#13577 acceptance witness. On a
// full-geometry model it drives the production Engram injector against four
// asymmetric persistent mHC streams and checks that each stream is updated from
// its own gate, that x stays synchronized with stream 0, that the reduced fixture
// still matches its oracle, and that a missing source fails closed.
func TestV41EngramFullPersistentStreams(t *testing.T) {
	m, layout := v41FullEngramModel(t)
	cfg := m.Cfg
	H := cfg.HiddenSize
	hc := 4
	eps := float32(cfg.RMSNormEps)
	ids := []int{1, 3, 5}

	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		t.Fatalf("v41ForwardGeometry error = %v, want nil", err)
	}
	if !full {
		t.Fatal("fixture classified as reduced; the full persistent-stream path is not exercised")
	}
	if cfg.DeepSeekV41.EngramHeadDim != cfg.HiddenSize {
		t.Fatalf("EngramHeadDim=%d HiddenSize=%d, want equality", cfg.DeepSeekV41.EngramHeadDim, cfg.HiddenSize)
	}

	x, streams := v41FullEngramAsymmetricStreams(ids, H)
	// Snapshot the pre-call streams so the non-vacuity checks compare against the
	// caller's inputs, not against a value production already mutated.
	before := make([][][]float32, len(streams))
	for tt := range streams {
		before[tt] = make([][]float32, hc)
		for s := 0; s < hc; s++ {
			before[tt][s] = append([]float32(nil), streams[tt][s]...)
		}
	}

	want := v41OracleFullEngramUpdates(t, m, layout, 0, streams, ids, eps)

	if err := m.v41EngramInject(0, x, streams, true, ids, eps); err != nil {
		t.Fatalf("v41EngramInject(full) error = %v, want nil", err)
	}

	// (1) Every stream row matches the independent per-stream oracle.
	for tt := range ids {
		for s := 0; s < hc; s++ {
			for i := 0; i < H; i++ {
				if d := math.Abs(float64(streams[tt][s][i] - want[tt][s][i])); d > cpuOracleTol {
					t.Fatalf("streams[%d][%d][%d] = %v, want %v (|delta| = %.3e > tol %.0e)",
						tt, s, i, streams[tt][s][i], want[tt][s][i], d, cpuOracleTol)
				}
			}
		}
	}

	// (2) x[t] stays synchronized with stream 0 after the call.
	for tt := range ids {
		for i := 0; i < H; i++ {
			if x[tt][i] != streams[tt][0][i] {
				t.Fatalf("x[%d][%d] = %v, want stream 0 value %v", tt, i, x[tt][i], streams[tt][0][i])
			}
		}
	}

	// (3) Non-vacuity: streams 1..3 each moved from their pre-call values. On the
	// parent streams 1..3 are never advanced, so this assertion AND the
	// oracle-match assertion above both fail; the oracle-match check reports first
	// (streams 1..3 unchanged where the independent oracle expects them moved).
	for tt := range ids {
		for s := 1; s < hc; s++ {
			moved := false
			for i := 0; i < H; i++ {
				if streams[tt][s][i] != before[tt][s][i] {
					moved = true
					break
				}
			}
			if !moved {
				t.Fatalf("streams[%d][%d] is unchanged after full injection; stream %d ignored", tt, s, s)
			}
		}
	}

	// (4) The four per-stream deltas are mutually distinct (four distinct gates).
	// Compute each stream's delta over the first position and require pairwise
	// difference; equal deltas mean the gates collapsed onto one vector.
	deltas := make([][]float32, hc)
	for s := 0; s < hc; s++ {
		row := make([]float32, H)
		for i := 0; i < H; i++ {
			row[i] = streams[0][s][i] - before[0][s][i]
		}
		deltas[s] = row
	}
	for s := 0; s < hc; s++ {
		for u := s + 1; u < hc; u++ {
			equal := true
			for i := 0; i < H; i++ {
				if deltas[s][i] != deltas[u][i] {
					equal = false
					break
				}
			}
			if equal {
				t.Fatalf("per-stream deltas for streams %d and %d are identical; the four gates were not independently computed", s, u)
			}
		}
	}

	// (5) Reduced-compat: the existing reduced fixture must still match its
	// independent oracle (reduced fixtures remain byte-for-byte compatible).
	rm, rlayout := v41ReducedEngramModel(t)
	rids := []int{1, 3, 5}
	ract := rm.Forward(rids)
	if ract == nil || len(ract.Logits) != len(rids) {
		t.Fatalf("reduced Forward returned %v positions, want %d", ract, len(rids))
	}
	rwant := v41OracleEngramForward(t, rm, rlayout, rids)
	for tPos := range rids {
		v41LogitsClose(t, "reduced-engram", ract.Logits[tPos], rwant[tPos])
	}

	// (6) Missing source fails closed: with the stage deleted, a declared
	// in-range Engram layer must refuse with ErrV41NativeUnsupported.
	missing, _ := v41FullEngramModel(t)
	v41EngramStages.Delete(missing)
	if err := missing.v41ForwardAdmitted(); !errors.Is(err, ErrV41NativeUnsupported) {
		t.Fatalf("unwired full Engram admission error = %v, want ErrV41NativeUnsupported", err)
	}

	// (7) End-to-end integration: the full path's call site in v41Layer forwards
	// `streams, full` into v41EngramInject, but the direct call above only proves
	// the callee. Drive the REAL full assembly (forwardV41: attention + MoE + head)
	// twice on the same weights -- once with the Engram declaration live on layer
	// 0 and once with it moved out of range -- and require the logits to differ.
	// A silently-skipped injection at the call site would leave the two forwards
	// identical, so this is a real integration run, not a unit mock.
	live, _ := v41FullEngramModel(t)
	actLive, err := live.forwardV41(ids, nil)
	if err != nil {
		t.Fatalf("full forwardV41 with live Engram error = %v, want nil", err)
	}
	if actLive == nil || len(actLive.Logits) != len(ids) {
		t.Fatalf("full forwardV41 returned %v positions, want %d", actLive, len(ids))
	}
	for tPos := range ids {
		if len(actLive.Logits[tPos]) != cfg.VocabSize {
			t.Fatalf("full forwardV41 logits[%d] has %d values, want vocab %d", tPos, len(actLive.Logits[tPos]), cfg.VocabSize)
		}
		for i, v := range actLive.Logits[tPos] {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("full forwardV41 logits[%d][%d] = %v, want finite", tPos, i, v)
			}
		}
	}

	off, _ := v41FullEngramModel(t)
	off.Cfg.DeepSeekV41.EngramLayerIDs = []int{99}
	actOff, err := off.forwardV41(ids, nil)
	if err != nil {
		t.Fatalf("control full forwardV41 with out-of-range Engram error = %v, want nil", err)
	}
	if actOff == nil || len(actOff.Logits) != len(ids) {
		t.Fatalf("control full forwardV41 returned %v positions, want %d", actOff, len(ids))
	}
	moved := false
	for tPos := range ids {
		for i := range actLive.Logits[tPos] {
			if math.Abs(float64(actLive.Logits[tPos][i]-actOff.Logits[tPos][i])) > cpuOracleTol {
				moved = true
			}
		}
	}
	if !moved {
		t.Fatal("full-path Engram injection did not change the logits; the v41Layer/forwardV41 call site is a no-op")
	}
}

// v41RectangularEngramModel builds a full-geometry V4.1 model (H=64, NumLayers=1)
// whose single in-range layer 0 declares Engram with an EngramHeadDim (dim) that
// DIFFERS from HiddenSize (H). The packed-row source is unchanged (still 264 bytes
// per row), so the concatenated row width is cols*dim while the projection output
// is (hc+1)*H: a genuinely rectangular projection. The pre-fix production path
// refused any dim != H before projecting; this fixture drives the accepted path.
func v41RectangularEngramModel(t *testing.T, dim int) (*Model, V41EngramLayout) {
	t.Helper()
	cfg := v41FullGeometryConfig(t)
	cfg.OGroups = 1
	cfg.DeepSeekV41.EngramLayerIDs = []int{0}
	cfg.DeepSeekV41.EngramNumEmbeddings = []int{48}
	cfg.DeepSeekV41.EngramMaxNgramSize = 4
	cfg.DeepSeekV41.EngramNHeads = 8
	cfg.DeepSeekV41.EngramHeadDim = dim

	H := cfg.HiddenSize
	I := cfg.MoEIntermediateSize
	qHeadDim := cfg.NumHeads * cfg.HeadDim
	oDim := cfg.OLoraRank * cfg.OGroups
	cols := (cfg.DeepSeekV41.EngramMaxNgramSize - 1) * cfg.DeepSeekV41.EngramNHeads

	type ts = synthTensor
	tensors := []ts{
		{"model.embed_tokens.weight", []int{cfg.VocabSize, H}},
		{"lm_head.weight", []int{cfg.VocabSize, H}},
		{"model.norm.weight", []int{H}},
		{layerName(0, "attn_norm.weight"), []int{H}},
		{layerName(0, "ffn_norm.weight"), []int{H}},
		{layerName(0, "mhc.mixes.weight"), []int{4 * H, v41MHCMixWidth}},
		{layerName(0, "mhc.ffn_mixes.weight"), []int{4 * H, v41MHCMixWidth}},
		{layerName(0, "mhc.ffn_base"), []int{v41MHCMixWidth}},
		{layerName(0, "mhc.ffn_scale"), []int{3}},
		{layerName(0, "mhc.base"), []int{v41MHCMixWidth}},
		{layerName(0, "mhc.scale"), []int{3}},
		{layerName(0, "attn.wq_a.weight"), []int{cfg.QLoraRank, H}},
		{layerName(0, "attn.wq_a_norm.weight"), []int{cfg.QLoraRank}},
		{layerName(0, "attn.wq_b.weight"), []int{qHeadDim, cfg.QLoraRank}},
		{layerName(0, "attn.wkv.weight"), []int{v41KVLoraRank, H}},
		{layerName(0, "attn.kv_norm.weight"), []int{v41KVLoraRank}},
		{layerName(0, "attn.wo_a.weight"), []int{cfg.OLoraRank, qHeadDim}},
		{layerName(0, "attn.wo_b.weight"), []int{H, oDim}},
		{layerName(0, "attn.sink"), []int{cfg.NumHeads}},
		{layerName(0, "ffn.gate.weight"), []int{cfg.NumExperts, H}},
		{layerName(0, "ffn.gate.e_score_correction_bias"), []int{cfg.NumExperts}},
		{layerName(0, "ffn.shared_experts.w1.weight"), []int{I, H}},
		{layerName(0, "ffn.shared_experts.w3.weight"), []int{I, H}},
		{layerName(0, "ffn.shared_experts.w2.weight"), []int{H, I}},
		{layerName(0, "engram_kv.weight"), []int{cols * dim, (4 + 1) * H}},
		{layerName(0, "engram_q_norm.weight"), []int{4 * H}},
		{layerName(0, "engram_k_norm.weight"), []int{4 * H}},
	}
	for e := 0; e < cfg.NumExperts; e++ {
		stem := "ffn.experts." + itoa(e)
		tensors = append(tensors,
			ts{layerName(0, stem+".w1.weight"), []int{I, H}},
			ts{layerName(0, stem+".w3.weight"), []int{I, H}},
			ts{layerName(0, stem+".w2.weight"), []int{H, I}},
		)
	}

	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		switch {
		case name == "model.norm.weight" || hasSuffix(name, "attn_norm.weight") ||
			hasSuffix(name, "ffn_norm.weight") || hasSuffix(name, "attn.wq_a_norm.weight") ||
			hasSuffix(name, "attn.kv_norm.weight"):
			return 1.0
		case hasSuffix(name, "mhc.ffn_scale"):
			return .75
		case hasSuffix(name, "mhc.ffn_base"):
			return .125 * next()
		case hasSuffix(name, "mhc.scale"):
			return 1.0
		case hasSuffix(name, "mhc.base"):
			return 0.0
		case hasSuffix(name, "attn.sink"):
			return 0.25 * next()
		default:
			return synthMatmulFill(name, next)
		}
	})
	m := &Model{Cfg: cfg, manifest: man, raw: raw}

	layout := v41EngramTestLayout(cfg)
	src := &v41EngramMemorySource{
		packed: v41EngramTestPackedRows(int(layout.Rows[0])),
		rows:   int(layout.Rows[0]),
	}
	if err := m.wireV41Engram(layout, []V41EngramRowSource{src}, int64(V41EngramPackedRowBytes)*4); err != nil {
		t.Fatalf("wire rectangular Engram stage: %v", err)
	}
	return m, layout
}

// v41OracleRectangularEngramUpdates independently transcribes the rectangular
// Engram schedule WITHOUT calling v41EngramInject: hash -> gather -> dequant ->
// project -> per-stream gate -> single-stream write-back. It mirrors
// v41OracleFullEngramUpdates but takes dim (EngramHeadDim) independently of H, so
// it is an independent oracle for D != H. Returns expected values as
// [position][stream][dim-hidden].
func v41OracleRectangularEngramUpdates(t *testing.T, m *Model, layout V41EngramLayout, l int, streams [][][]float32, ids []int, eps float32) [][][]float32 {
	t.Helper()
	cfg := m.Cfg
	H := cfg.HiddenSize
	dim := cfg.DeepSeekV41.EngramHeadDim
	hc := 4
	cols := (layout.MaxNgramSize - 1) * layout.HeadsPerNgram

	stage := m.v41EngramStageFor()
	if stage == nil {
		t.Fatal("oracle: Engram stage not wired")
	}
	hash, err := NewV41EngramHashState(layout)
	if err != nil {
		t.Fatal(err)
	}
	allRows, err := hash.Hash(ids, nil)
	if err != nil {
		t.Fatal(err)
	}
	layers := len(layout.Rows)
	cacheIdx := stage.cacheIndex(l)
	layerRows := make([]uint32, 0, len(ids)*cols)
	for tt := range ids {
		base := tt*layers*cols + cacheIdx*cols
		layerRows = append(layerRows, allRows[base:base+cols]...)
	}
	gathered, err := GatherV41EngramRows([]*V41EngramRowCache{stage.caches[cacheIdx]}, layerRows, cols)
	if err != nil {
		t.Fatal(err)
	}

	wKV := cpuOracleTensor(t, m, layerName(l, "engram_kv.weight"))
	qNorm := cpuOracleTensor(t, m, layerName(l, "engram_q_norm.weight"))
	kNorm := cpuOracleTensor(t, m, layerName(l, "engram_k_norm.weight"))

	want := make([][][]float32, len(ids))
	for tt := range ids {
		rowVec := make([]float32, cols*dim)
		for c := 0; c < cols; c++ {
			copy(rowVec[c*dim:], v41OracleEngramDequant(t, gathered[tt*cols+c], dim))
		}
		projected := cpuOracleMatVec(wKV, rowVec, (hc+1)*H, cols*dim)
		value := make([]float32, H)
		for i := 0; i < H; i++ {
			value[i] = v41OracleBF16(projected[hc*H+i])
		}
		want[tt] = make([][]float32, hc)
		for s := 0; s < hc; s++ {
			h := streams[tt][s]
			key := make([]float32, H)
			var h2, k2, dot float64
			for i := 0; i < H; i++ {
				key[i] = v41OracleBF16(projected[s*H+i])
				hh := float64(h[i])
				kk := float64(key[i])
				h2 += hh * hh
				k2 += kk * kk
				dot += hh * float64(qNorm[s*H+i]) * float64(kNorm[s*H+i]) * kk
			}
			dot *= 1 / math.Sqrt(h2/float64(H)+float64(eps))
			dot *= 1 / math.Sqrt(k2/float64(H)+float64(eps))
			dot *= 1 / math.Sqrt(float64(H))
			gate := 1 / (1 + math.Exp(-math.Copysign(math.Sqrt(math.Max(math.Abs(dot), 1e-6)), dot)))
			row := make([]float32, H)
			for i := 0; i < H; i++ {
				delta := float32(float32(gate) * value[i])
				row[i] = v41OracleBF16(float32(h[i] + delta))
			}
			want[tt][s] = row
		}
	}
	return want
}

// TestV41EngramRectangularGeometryContract is the rectangular-geometry acceptance
// witness. It drives the production Engram injector on a D != H fixture and checks
// every stream against an independent scalar oracle, that x stays synchronized with
// stream 0, and that malformed geometry (nonpositive/oversized D, nonpositive H,
// nonpositive columns, an overflowing product) fails closed with
// ErrV41NativeUnsupported before any stream is mutated. On the pre-fix parent the
// D != H call is refused outright, so the oracle-match section fails.
func TestV41EngramRectangularGeometryContract(t *testing.T) {
	const dim = 32 // EngramHeadDim; deliberately != HiddenSize (64)
	m, layout := v41RectangularEngramModel(t, dim)
	cfg := m.Cfg
	H := cfg.HiddenSize
	hc := 4
	eps := float32(cfg.RMSNormEps)
	ids := []int{1, 3, 5}

	if cfg.DeepSeekV41.EngramHeadDim == H {
		t.Fatalf("fixture is not rectangular: EngramHeadDim=%d == HiddenSize=%d", dim, H)
	}

	x, streams := v41FullEngramAsymmetricStreams(ids, H)
	before := make([][][]float32, len(streams))
	for tt := range streams {
		before[tt] = make([][]float32, hc)
		for s := 0; s < hc; s++ {
			before[tt][s] = append([]float32(nil), streams[tt][s]...)
		}
	}

	want := v41OracleRectangularEngramUpdates(t, m, layout, 0, streams, ids, eps)

	if err := m.v41EngramInject(0, x, streams, true, ids, eps); err != nil {
		t.Fatalf("v41EngramInject(rectangular D=%d H=%d) error = %v, want nil", dim, H, err)
	}

	// (1) Every stream row matches the independent rectangular oracle.
	for tt := range ids {
		for s := 0; s < hc; s++ {
			for i := 0; i < H; i++ {
				if d := math.Abs(float64(streams[tt][s][i] - want[tt][s][i])); d > cpuOracleTol {
					t.Fatalf("rect streams[%d][%d][%d] = %v, want %v (|delta| = %.3e > tol %.0e)",
						tt, s, i, streams[tt][s][i], want[tt][s][i], d, cpuOracleTol)
				}
			}
		}
	}

	// (2) x[t] stays synchronized with stream 0 after the call.
	for tt := range ids {
		for i := 0; i < H; i++ {
			if x[tt][i] != streams[tt][0][i] {
				t.Fatalf("x[%d][%d] = %v, want stream 0 value %v", tt, i, x[tt][i], streams[tt][0][i])
			}
		}
	}

	// (3) Non-vacuity: streams 1..3 each moved from their pre-call values.
	for tt := range ids {
		for s := 1; s < hc; s++ {
			moved := false
			for i := 0; i < H; i++ {
				if streams[tt][s][i] != before[tt][s][i] {
					moved = true
					break
				}
			}
			if !moved {
				t.Fatalf("rect streams[%d][%d] is unchanged after injection; stream %d ignored", tt, s, s)
			}
		}
	}

	// (4) Invalid geometry fails closed before any mutation.
	cases := []struct {
		name   string
		mutate func(*Model)
	}{
		{"nonpositive_dim", func(bad *Model) { bad.v41EngramStageFor().headDim = 0 }},
		{"oversized_dim", func(bad *Model) { bad.v41EngramStageFor().headDim = 257 }},
		{"nonpositive_hidden", func(bad *Model) { bad.Cfg.HiddenSize = 0 }},
		{"nonpositive_columns", func(bad *Model) { bad.v41EngramStageFor().columns = 0 }},
		{"overflowing_product", func(bad *Model) { bad.v41EngramStageFor().columns = math.MaxInt / 2 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			bad, _ := v41RectangularEngramModel(t, dim)
			tc.mutate(bad)
			bx, bstreams := v41FullEngramAsymmetricStreams(ids, H)
			snapshot := make([][][]float32, len(bstreams))
			for tt := range bstreams {
				snapshot[tt] = make([][]float32, hc)
				for s := 0; s < hc; s++ {
					snapshot[tt][s] = append([]float32(nil), bstreams[tt][s]...)
				}
			}
			err := bad.v41EngramInject(0, bx, bstreams, true, ids, eps)
			if !errors.Is(err, ErrV41NativeUnsupported) {
				t.Fatalf("%s: v41EngramInject error = %v, want ErrV41NativeUnsupported", tc.name, err)
			}
			for tt := range bstreams {
				for s := 0; s < hc; s++ {
					for i := 0; i < H; i++ {
						if bstreams[tt][s][i] != snapshot[tt][s][i] {
							t.Fatalf("%s: stream[%d][%d][%d] mutated on a refused call", tc.name, tt, s, i)
						}
					}
				}
			}
		})
	}
}

// The positive-clamped zero dot yields gate ~= 0.50025. With value 1/128,
// rounding the delta first lands on an exact half ULP above residual 1, whereas
// the unrounded F32 sum is just above that midpoint and publishes 1.0078125.
// This is an injection-boundary witness, not checkpoint/device parity.
// fak-test:runtime fast est=100ms lane=default
func TestV41EngramFullRoundsResidualSum(t *testing.T) {
	t.Parallel()
	m, _ := v41FullEngramModel(t)
	H := m.Cfg.HiddenSize
	v41WriteTensorF32(t, m, layerName(0, "engram_q_norm.weight"), make([]float32, 4*H))
	projected := make([]float32, 5*H)
	for i := 0; i < 4*H; i++ {
		projected[i] = 1
	}
	for i := 4 * H; i < 5*H; i++ {
		projected[i] = 1.0 / 128
	}
	source := append([]float32(nil), projected...)
	for _, alias := range []bool{false, true} {
		streams := make([][]float32, 4)
		for h := range streams {
			streams[h] = make([]float32, H)
			for i := range streams[h] {
				streams[h][i] = []float32{1, 2, -1, -2}[h]
			}
		}
		x := append([]float32(nil), streams[0]...)
		if alias {
			x = streams[0]
		}
		before := sysFlatten(streams)
		calls := 0
		err := m.v41EngramInjectWithProjection(0, [][]float32{x}, [][][]float32{streams}, true, []int{1}, float32(m.Cfg.RMSNormEps), func(layer int, row []float32, out, in int) ([]float32, v41DenseProjectionOutcome, error) {
			calls++
			if layer != 0 || out != 5*H || len(row) != in {
				t.Fatal("Engram callback lost exact geometry")
			}
			return projected, v41ProjectionHandled, nil
		})
		if err != nil || calls != 1 {
			t.Fatalf("Engram projection calls=%d err=%v", calls, err)
		}
		// Independent scalar gate and BF16 reference: no production gate/mix helpers.
		gate := float32(1 / (1 + math.Exp(-math.Sqrt(1e-6))))
		delta := float32(gate * float32(1.0/128))
		discriminates := false
		for h := range streams {
			for i, got := range streams[h] {
				residual := before[h*H+i]
				want := v41OracleBF16(float32(residual + delta))
				old := residual + v41OracleBF16(delta)
				if got != want || math.Float32bits(got)&0xffff != 0 {
					t.Fatalf("stream%d residual publication=%g want%g", h, got, want)
				}
				discriminates = discriminates || got != old
			}
		}
		if !discriminates || streams[0][0] != 1.0078125 || streams[0][0] == v41OracleBF16(1+v41OracleBF16(delta)) || !reflect.DeepEqual(x, streams[0]) || !reflect.DeepEqual(projected, source) {
			t.Fatal("fixed midpoint, x synchronization, or callback ownership changed")
		}
	}
}
