package model

import (
	"encoding/json"
	"math"
	"reflect"
	"sort"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// deepInferSourceMap transcribes DeepInfer's _source_map: every layer from the
// smallest declared source up to n maps to the nearest source not after it.
func deepInferSourceMap(ids []int, n int) map[int]int {
	sorted := append([]int(nil), ids...)
	sort.Ints(sorted)
	out := make(map[int]int, n)
	for layer := sorted[0]; layer < n; layer++ {
		best := sorted[0]
		for _, s := range sorted {
			if s <= layer && s > best {
				best = s
			}
		}
		out[layer] = best
	}
	return out
}

func deepInferMapped(m map[int]int, layer int) int {
	if src, ok := m[layer]; ok {
		return src
	}
	return -1
}

// DeepInfer derivation rules applied to fak's committed dba1be0a fixture: a
// transcription of DeepInfer's config derivation is compared with fak's V4.1
// plan resolution, rotary policy and derived widths. This is test-only evidence
// for a later wiring decision; it makes no claim about the 1017 config.
// Adapted from DeepInfer at 774ce244; see third_party/deepinfer-v41-parity/NOTICE.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=5ms lane=default
func TestV41ConfigDerivationMatchesDeepInfer(t *testing.T) {
	t.Parallel()
	raw, cfg := readDeepSeekV41Config(t)
	m := cfg.DeepSeekV41
	var fixture struct {
		Text   map[string]json.RawMessage `json:"text_config"`
		Vision map[string]json.RawMessage `json:"vision_config"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	rawInt := func(t *testing.T, block map[string]json.RawMessage, key string) int {
		t.Helper()
		var v int
		if err := json.Unmarshal(block[key], &v); err != nil {
			t.Fatalf("fixture key %q: %v", key, err)
		}
		return v
	}

	if m.DSpark == nil || m.Vision == nil {
		t.Fatal("fixture DSpark/vision markers not retained")
	}
	n := cfg.NumLayers
	total := n + m.DSpark.NextNPredictLayers
	roles := v41AttentionRoles(cfg)
	plans := make([]V41AttentionPlan, total)
	for l := range plans {
		plan, err := v41AttentionPlanFor(cfg, l, roles)
		if err != nil {
			t.Fatalf("layer %d plan: %v", l, err)
		}
		plans[l] = plan
	}
	kv := deepInferSourceMap(m.KVSourceLayerIDs, n)
	index := deepInferSourceMap(m.IndexSourceLayerIDs, n)

	compared, diverged := 0, 0
	same := func(t *testing.T, axis string, layer int, got, want any) {
		t.Helper()
		compared++
		if !reflect.DeepEqual(got, want) {
			diverged++
			t.Errorf("%s layer=%d: fak=%v deepinfer=%v", axis, layer, got, want)
		}
	}

	t.Run("shared source maps", func(t *testing.T) {
		for l, want := range map[int]int{2: 2, 7: 2, 8: 8, 13: 8, 14: 14, 19: 14, 20: 20, 39: 20} {
			if kv[l] != want {
				t.Fatalf("transcribed kv map[%d]=%d want %d", l, kv[l], want)
			}
		}
		for l, want := range map[int]int{7: 2, 8: 8, 19: 14, 20: 20, 23: 20, 24: 24, 27: 24, 28: 28, 32: 32, 36: 36, 39: 36} {
			if index[l] != want {
				t.Fatalf("transcribed index map[%d]=%d want %d", l, index[l], want)
			}
		}
		if len(kv) != n-2 || len(index) != n-2 {
			t.Fatalf("map domains kv=%d index=%d want %d", len(kv), len(index), n-2)
		}
		for l := 0; l < n; l++ {
			// DeepInfer maps ratio-0 layers too, while fak resolves no source for
			// them, so the maps agree only while every mapped layer is compressed.
			if _, mapped := kv[l]; mapped && v41CompressRatioAt(cfg, l) == 0 {
				t.Fatalf("premise: mapped layer %d has ratio 0", l)
			}
			p := plans[l]
			same(t, "kv source", l, p.KVSourceLayer, deepInferMapped(kv, l))
			same(t, "index source", l, p.IndexSourceLayer, deepInferMapped(index, l))
			wantRole := V41AttentionRolePerLayer
			if src, mapped := kv[l]; mapped {
				wantRole = V41AttentionRoleReader
				if src == l {
					wantRole = V41AttentionRoleKVSource
				}
			}
			same(t, "role", l, p.Role, wantRole)
			wantTopK := 0
			if _, mapped := index[l]; mapped {
				wantTopK = 512
			}
			same(t, "index topk", l, p.TopKWidth, wantTopK)
		}
	})

	t.Run("index-K falls back to KV sources", func(t *testing.T) {
		if _, declared := fixture.Text["index_k_source_layer_ids"]; declared {
			t.Fatal("premise: fixture declares index_k_source_layer_ids")
		}
		// fak takes index keys from the KV source (applyDeepSeek41Topology in
		// internal/ggufload/deepseek41.go), so the index-K map has no separate
		// fak field and is compared with KVSourceLayer.
		indexK := deepInferSourceMap(m.KVSourceLayerIDs, n)
		for l := 0; l < n; l++ {
			same(t, "index-K source", l, plans[l].KVSourceLayer, deepInferMapped(indexK, l))
		}
	})

	t.Run("candidate map", func(t *testing.T) {
		candidate := map[int]int{}
		for l := m.CandidateSourceLayerID; l < n; l++ {
			candidate[l] = m.CandidateSourceLayerID
		}
		for l := 0; l < total; l++ {
			want := [3]int{-1, 0, 0}
			if src := deepInferMapped(candidate, l); src >= 0 {
				want = [3]int{src, 2048, 8}
			}
			p := plans[l]
			same(t, "candidate", l, [3]int{p.CandidateSource, p.CandidateTopKBlock, p.CandidateBlockSize}, want)
		}
		wantCandidateCount := n - m.CandidateSourceLayerID
		if len(candidate) != wantCandidateCount || candidate[m.CandidateSourceLayerID] != m.CandidateSourceLayerID || candidate[n-1] != m.CandidateSourceLayerID {
			t.Fatalf("transcribed candidate map=%v", candidate)
		}
	})

	t.Run("attention module list", func(t *testing.T) {
		if len(m.CompressRatios) != total || total != 43 {
			t.Fatalf("compress_ratios=%d layers=%d nextn=%d", len(m.CompressRatios), n, m.DSpark.NextNPredictLayers)
		}
		// DeepInfer's DFlash specs take base theta whatever their ratio; fak
		// reads the ratio, so the two agree only while the nextn ratios are 0.
		if !reflect.DeepEqual(m.CompressRatios[n:], []int{0, 0, 0}) {
			t.Fatalf("premise: nextn ratios=%v", m.CompressRatios[n:])
		}
		for l := 0; l < total; l++ {
			compressed := l < n && m.CompressRatios[l] != 0
			wantTheta := 10000.0
			if compressed {
				wantTheta = 160000
			}
			same(t, "rope theta", l, v41RopeThetaForLayer(cfg, l), wantTheta)
			same(t, "yarn regime", l, v41CompressRatioAt(cfg, l) > 0, compressed)
		}
		for l := n; l < total; l++ {
			p := plans[l]
			same(t, "nextn sources", l, [3]int{p.KVSourceLayer, p.IndexSourceLayer, p.CandidateSource}, [3]int{-1, -1, -1})
		}
		type yarnParams struct {
			Factor   float64 `json:"factor"`
			Orig     int     `json:"original_max_position_embeddings"`
			BetaFast float64 `json:"beta_fast"`
			BetaSlow float64 `json:"beta_slow"`
		}
		var scaling yarnParams
		if err := json.Unmarshal(fixture.Text["rope_scaling"], &scaling); err != nil {
			t.Fatal(err)
		}
		if scaling != (yarnParams{16, 65536, 32, 1}) || cfg.RopeScaling != "yarn" {
			t.Fatalf("rope_variant: fixture rope_scaling=%+v fak RopeScaling=%q", scaling, cfg.RopeScaling)
		}
		rp := cfg.effectiveRopeParameters()
		same(t, "yarn params", -1, yarnParams{rp.Factor, rp.OriginalMaxPositionEmbeddings, rp.BetaFast, rp.BetaSlow}, scaling)
	})

	t.Run("prefill boundary", func(t *testing.T) {
		nPrefill := -1
		for i, ratio := range m.CompressRatios {
			if ratio == 1 {
				nPrefill = i
				break
			}
		}
		if nPrefill != 20 {
			t.Fatalf("n_prefill_layer=%d want 20", nPrefill)
		}
		same(t, "boundary kv map", nPrefill, kv[nPrefill], nPrefill)
		same(t, "boundary role", nPrefill, roles[nPrefill], V41AttentionRoleKVSource)
		same(t, "boundary kv source", nPrefill, plans[nPrefill].KVSourceLayer, nPrefill)
		// Layers at or past the boundary are YOCO consumers; past it they read
		// the boundary's KV stream.
		for l := nPrefill + 1; l < n; l++ {
			same(t, "consumer", l, [2]any{plans[l].Role, plans[l].KVSourceLayer}, [2]any{V41AttentionRoleReader, nPrefill})
		}
		for _, id := range m.EngramLayerIDs {
			same(t, "engram before boundary", id, id < nPrefill, true)
		}
		for _, id := range m.DSpark.TargetLayerIDs {
			same(t, "dspark target in consumers", id, id >= nPrefill, true)
		}
	})

	t.Run("derived widths", func(t *testing.T) {
		nope := cfg.HeadDim - cfg.QKRopeHeadDim
		if nope != 448 || nope+cfg.QKRopeHeadDim != v41KVLoraRank {
			t.Fatalf("DeepInfer kv_lora_rank (NoPE)=%d packed=%d want 448/%d", nope, nope+cfg.QKRopeHeadDim, v41KVLoraRank)
		}
		cos, sin := v41RopeTableForLayer(cfg, 2, 7)
		row := make([]float32, v41KVLoraRank)
		for i := range row {
			row[i] = float32(i%13) + 0.25
		}
		before := append([]float32(nil), row...)
		applyRopeTailInterleaved(row, cos, sin, cfg.QKRopeHeadDim)
		for i := 0; i < nope; i++ {
			if math.Float32bits(row[i]) != math.Float32bits(before[i]) {
				t.Fatalf("NoPE lane %d rotated", i)
			}
		}
		if reflect.DeepEqual(row[nope:], before[nope:]) {
			t.Fatal("rope tail unchanged")
		}

		if compute.V41MHCStreams != m.HCMult {
			t.Fatalf("hc_mult=%d fak streams=%d", m.HCMult, compute.V41MHCStreams)
		}
		streams := make([][]float32, m.HCMult)
		for i := range streams {
			streams[i] = make([]float32, cfg.HiddenSize)
		}
		flat, width, err := v41MHCStreamsRow(streams)
		if err != nil {
			t.Fatal(err)
		}
		if len(flat) != m.HCMult*cfg.HiddenSize || width != cfg.HiddenSize {
			t.Fatalf("hc_norm_group_size: fak row=%d width=%d want %d/%d", len(flat), width, m.HCMult*cfg.HiddenSize, cfg.HiddenSize)
		}

		same(t, "window_size", -1, v41WindowSize, rawInt(t, fixture.Text, "sliding_window"))
	})

	t.Run("vision widths", func(t *testing.T) {
		v := m.Vision
		if got := [4]int{v.HiddenSize, v.NumLayers, v.NumHeads, v.PatchSize}; got != [4]int{1024, 32, 16, 14} {
			t.Fatalf("fak vision=%v want DeepInfer EMBED_DIM/DEPTH/NUM_HEADS/PATCH_SIZE 1024/32/16/14", got)
		}
		mlpRatio := 2.7
		hidden := int(float64(v.HiddenSize) * mlpRatio)
		const multipleOf = 128
		swiglu := multipleOf * ((hidden + multipleOf - 1) / multipleOf)
		if hidden != 2764 || swiglu != 2816 || 2*swiglu != 5632 {
			t.Fatalf("SwiGLU hidden=%d width=%d linear1=%d", hidden, swiglu, 2*swiglu)
		}
		if got := rawInt(t, fixture.Vision, "intermediate_size"); got != swiglu {
			t.Fatalf("intermediate_size=%d want %d", got, swiglu)
		}
		ds := rawInt(t, fixture.Vision, "downsample_ratio")
		if projector := v.HiddenSize * ds * ds; ds != 3 || projector != 9216 {
			t.Fatalf("projector input=%d downsample=%d want 9216/3", projector, ds)
		}
	})

	t.Logf("DeepInfer derivation vs fak: %d divergences over %d comparisons", diverged, compared)
	if diverged != 0 {
		t.Fatalf("diverged=%d", diverged)
	}
}
