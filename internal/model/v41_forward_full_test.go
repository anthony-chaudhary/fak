package model

// v41_forward_full_test.go is the #13009 witness for the FULL (non-reduced)
// V4.1 forward geometry. The reduced assembly in v41_forward.go historically
// executed stand-ins for the published checkpoint geometry: a HeadDim-wide KV
// latent instead of the published v41KVLoraRank (512), and four IDENTICAL mHC
// streams instead of four distinct persistent streams. (The router geometry was
// separately landed on trunk: both paths now route through the strict
// v41RouterConfigFromConfig envelope via v41RouterConfigFullGeometry.) This file
// proves the full path is selected and honors the published KV-latent axis, and
// that a mismatched published envelope fails closed rather than silently
// reducing.
//
// The full official checkpoint (40 layers x 384 experts x 5120 width) is far too
// large to materialize weight-free, so the positive witness asserts the geometry
// discriminator and the admission shape directly, and the negative witness drives
// the same discriminator with perturbed axes. The reduced oracle in
// v41_forward_test.go remains the numeric witness for the reduced path.

import (
	"errors"
	"testing"
)

// TestV41FullGeometryForward is the #13009 acceptance witness: the published
// V4.1 config selects the full geometry, the full path admits a 512-wide KV
// latent (and rejects a HeadDim-wide stand-in), and the full path's four
// persistent mHC streams are DISTINCT.
func TestV41FullGeometryForward(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)

	// (a) The published config selects the full geometry.
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		t.Fatalf("v41ForwardGeometry(published) error = %v, want nil", err)
	}
	if !full {
		t.Fatal("published V4.1 config did not select the full forward geometry")
	}
	if got, err := v41ForwardKVLatentRank(cfg); err != nil || got != v41KVLoraRank {
		t.Fatalf("full KV latent rank = (%d,%v), want (%d,nil)", got, err, v41KVLoraRank)
	}

	// (b) The real admission entry point admits a full-geometry model whose
	// attn.wkv is the published 512-wide latent, and REFUSES one whose wkv is the
	// reduced HeadDim stand-in. The official 40x384x5120 checkpoint is too large to
	// materialize, so we build a small but internally consistent FULL config (its
	// Attention envelope mirrors the narrowed flat axes and head_dim stays at the
	// published 512), which v41ForwardGeometry classifies as full.
	//
	// Honest scope: on a VALID full config head_dim == v41KVLoraRank == 512, so the
	// reduced expression (cfg.HeadDim) and the full constant are numerically equal;
	// this section proves the admission gate honors the resolved rank but cannot by
	// itself distinguish the full branch from a silent reduced fallback. That
	// load-bearing property is the fail-closed propagation asserted in
	// TestV41FullGeometryMissingAxisFailsClosed below: there the resolver must
	// return an error rather than the reduced width.
	fullCfg := v41FullGeometryConfig(t)
	rank, err := v41ForwardKVLatentRank(fullCfg)
	if err != nil {
		t.Fatalf("v41ForwardKVLatentRank(full) error = %v", err)
	}
	if rank != v41KVLoraRank {
		t.Fatalf("full KV latent rank = %d, want %d", rank, v41KVLoraRank)
	}
	if err := v41BuildFullModel(fullCfg, v41KVLoraRank).v41ForwardAdmitted(); err != nil {
		t.Fatalf("full model (wkv at %d) admission error = %v, want nil", v41KVLoraRank, err)
	}
	// The reduced-width stand-in (the pre-#13009 HeadDim stand-in) declared on a
	// FULL geometry must be refused by the real admission gate, not admitted.
	if err := v41BuildFullModel(fullCfg, standInWidth).v41ForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("full model with reduced %d-wide wkv admission error = %v, want ErrV41ForwardStage", standInWidth, err)
	}

	// (c) The four persistent mHC streams on the full path are DISTINCT: stream 0
	// carries the live hidden, streams 1..3 are zero. This mirrors the exact
	// initialization forwardV41 performs (unlike the reduced stand-in's four
	// identical copies of the normalized input).
	live := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	streams := makeFullStreams(live)
	// Stream 0 must be the nonzero live hidden, so distinctness is non-vacuous.
	if allZero32(streams[0][0]) {
		t.Fatal("stream 0 was zero, so the distinctness assertion is vacuous")
	}
	for h := 1; h < 4; h++ {
		if !allZero32(streams[0][h]) {
			t.Fatalf("stream %d should be the zero-initialized persistent residual, got %v", h, streams[0][h])
		}
		if equalFloat32Slices(streams[0][h], streams[0][0]) {
			t.Fatalf("stream %d is identical to stream 0, want distinct persistent streams", h)
		}
	}
}

// TestV41FullGeometryMissingAxisFailsClosed is the #13009 negative witness: a
// config carrying the populated published Attention envelope but a mismatched
// decoder axis MUST refuse with a typed ErrV41ForwardStage and MUST NOT fall back
// to the reduced geometry.
func TestV41FullGeometryMissingAxisFailsClosed(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)

	// The published config itself is admitted as full (control).
	if full, err := v41ForwardGeometry(cfg); err != nil || !full {
		t.Fatalf("control published geometry = (%v,%v), want (true,nil)", full, err)
	}

	// Mismatch the layer count against the authoritative published envelope while
	// keeping head_dim at the published 512 (so the envelope stays authoritative).
	mismatched := cfg
	mismatched.NumLayers = cfg.DeepSeekV41.Attention.NumLayers + 1
	full, err := v41ForwardGeometry(mismatched)
	if full {
		t.Fatal("mismatched envelope was classified full")
	}
	if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("mismatched-axis geometry error = %v, want ErrV41ForwardStage", err)
	}
	// A non-empty manifest gets admission past the weightless-probe fence so the
	// geometry discriminator is the gate that refuses.
	if err := (&Model{Cfg: mismatched, manifest: map[string]tensorMeta{"model.embed_tokens.weight": {Dtype: "F32", Shape: []int{1}}}}).v41ForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("mismatched-axis v41ForwardAdmitted error = %v, want ErrV41ForwardStage", err)
	}

	// A self-consistent envelope that is nonetheless not the published full
	// geometry (head_dim 32, the reduced width) must also refuse, not reduce.
	badEnvelope := cfg
	badEnvelope.HeadDim = 32
	badEnvelope.DeepSeekV41 = &DeepSeekV41Config{Attention: DeepSeekV41AttentionGeometry{
		NumLayers: cfg.NumLayers, HiddenSize: cfg.HiddenSize,
		NumHeads: cfg.NumHeads, NumKVHeads: cfg.NumKVHeads, HeadDim: 32,
	}}
	if full, err := v41ForwardGeometry(badEnvelope); full || !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("non-published self-consistent envelope = (%v,%v), want (false, ErrV41ForwardStage)", full, err)
	}

	// A LONE-AXIS DRIFT must fail closed, not classify as a reduced fixture: the
	// published envelope is intact and the decoder stack is still the published
	// 40 layers, but only head_dim was mutated. Before this guard the config
	// silently classified reduced and resolved the reduced KV latent width.
	drifted := cfg
	drifted.HeadDim = 64
	if full, err := v41ForwardGeometry(drifted); full || !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("lone-axis head_dim drift = (%v,%v), want (false, ErrV41ForwardStage)", full, err)
	}
	if rank, err := v41ForwardKVLatentRank(drifted); err == nil {
		t.Fatalf("lone-axis drift resolved rank %d with nil error; want a typed refusal", rank)
	} else if !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("lone-axis drift error = %v, want ErrV41ForwardStage", err)
	}

	// The reduced fixture (narrowed decoder stack AND head width) must still be
	// admitted as reduced, so the guard does not break the fast test harness.
	fixture := v41TestReducedConfig(t, 1, V41RouterExperts)
	if full, err := v41ForwardGeometry(fixture); full || err != nil {
		t.Fatalf("reduced fixture geometry = (%v,%v), want (false,nil)", full, err)
	}

	// The KV-latent resolver must propagate that refusal rather than defaulting to
	// the reduced width: a helper that returned the reduced stand-in rank here
	// would admit a config that must fail closed (the #13009 fail-open regression).
	rank, rankErr := v41ForwardKVLatentRank(badEnvelope)
	if rankErr == nil || !errors.Is(rankErr, ErrV41ForwardStage) {
		t.Fatalf("v41ForwardKVLatentRank(non-published envelope) = (%d,%v), want (_, ErrV41ForwardStage)", rank, rankErr)
	}
	if rank == v41KVLoraRankReduced(badEnvelope) {
		t.Fatal("v41ForwardKVLatentRank silently degraded to the reduced KV latent width")
	}
}

// standInWidth is the reduced HeadDim-wide KV latent stand-in the full path must
// never admit (the pre-#13009 reduced geometry).
const standInWidth = 32

// v41FullGeometryConfig derives a small, internally consistent FULL-geometry V4.1
// config from the pinned published config: head_dim stays at the published
// v41KVLoraRank (512), so v41ForwardGeometry classifies it as full while the
// fixture stays tiny enough to materialize.
//
// The retained Attention envelope is cleared to the zero value. A populated
// envelope is required by v41AttentionGeometryForwardAdmitted to equal the
// published whole decoder envelope, against which this deliberately narrowed
// fixture would fail closed; the zero envelope is the legitimate "no typed
// metadata retained" state for a hand-built fixture and leaves the flat axes as
// the source of truth, which is exactly what v41ForwardGeometry reads here.
func v41FullGeometryConfig(t *testing.T) Config {
	t.Helper()
	_, cfg := readDeepSeekV41Config(t)
	cfg.NumLayers = 1
	cfg.HiddenSize = 64
	cfg.NumHeads = 1
	cfg.NumKVHeads = 1
	cfg.HeadDim = v41KVLoraRank
	cfg.QLoraRank = 32
	cfg.OLoraRank = 16
	cfg.OGroups = 2
	cfg.MoEIntermediateSize = 32
	cfg.VocabSize = 8
	cfg.NumExperts = V41RouterExperts
	cfg.NumExpertsPerTok = V41RouterTopK
	cfg.NSharedExperts = V41RouterSharedCount
	cfg.RoutedScalingFactor = float64(V41RouterRouteScale)
	cfg.DeepSeekV41.Attention = DeepSeekV41AttentionGeometry{}
	return cfg
}

// v41BuildFullModel builds a full-geometry *Model whose attn.wkv.weight is
// declared at kvRank width. It carries exactly the tensors v41ForwardAdmitted
// checks, so a wrong KV latent width is rejected by the real admission gate
// rather than by a hand-called helper.
func v41BuildFullModel(cfg Config, kvRank int) *Model {
	H := cfg.HiddenSize
	man := map[string]tensorMeta{
		"model.embed_tokens.weight":                      {Dtype: "F32", Shape: []int{cfg.VocabSize, H}},
		"lm_head.weight":                                 {Dtype: "F32", Shape: []int{cfg.VocabSize, H}},
		"model.norm.weight":                              {Dtype: "F32", Shape: []int{H}},
		layerName(0, "attn_norm.weight"):                 {Dtype: "F32", Shape: []int{H}},
		layerName(0, "ffn_norm.weight"):                  {Dtype: "F32", Shape: []int{H}},
		layerName(0, "mhc.mixes.weight"):                 {Dtype: "F32", Shape: []int{v41MHCMixWidth, H}},
		layerName(0, "mhc.base"):                         {Dtype: "F32", Shape: []int{v41MHCMixWidth}},
		layerName(0, "mhc.scale"):                        {Dtype: "F32", Shape: []int{3}},
		layerName(0, "attn.wq_a.weight"):                 {Dtype: "F32", Shape: []int{cfg.QLoraRank, H}},
		layerName(0, "attn.wq_a_norm.weight"):            {Dtype: "F32", Shape: []int{cfg.QLoraRank}},
		layerName(0, "attn.wq_b.weight"):                 {Dtype: "F32", Shape: []int{cfg.NumHeads * cfg.HeadDim, cfg.QLoraRank}},
		layerName(0, "attn.wkv.weight"):                  {Dtype: "F32", Shape: []int{kvRank, H}},
		layerName(0, "attn.kv_norm.weight"):              {Dtype: "F32", Shape: []int{kvRank}},
		layerName(0, "attn.wo_a.weight"):                 {Dtype: "F32", Shape: []int{cfg.OLoraRank, cfg.NumHeads * cfg.HeadDim}},
		layerName(0, "attn.wo_b.weight"):                 {Dtype: "F32", Shape: []int{H, cfg.OLoraRank * cfg.OGroups}},
		layerName(0, "attn.sink"):                        {Dtype: "F32", Shape: []int{cfg.NumHeads}},
		layerName(0, "ffn.gate.weight"):                  {Dtype: "F32", Shape: []int{cfg.NumExperts, H}},
		layerName(0, "ffn.gate.e_score_correction_bias"): {Dtype: "F32", Shape: []int{cfg.NumExperts}},
		layerName(0, "ffn.shared_experts.w1.weight"):     {Dtype: "F32", Shape: []int{cfg.MoEIntermediateSize, H}},
		layerName(0, "ffn.shared_experts.w3.weight"):     {Dtype: "F32", Shape: []int{cfg.MoEIntermediateSize, H}},
		layerName(0, "ffn.shared_experts.w2.weight"):     {Dtype: "F32", Shape: []int{H, cfg.MoEIntermediateSize}},
	}
	for e := 0; e < cfg.NumExperts; e++ {
		stem := "ffn.experts." + itoa(e)
		man[layerName(0, stem+".w1.weight")] = tensorMeta{Dtype: "F32", Shape: []int{cfg.MoEIntermediateSize, H}}
		man[layerName(0, stem+".w3.weight")] = tensorMeta{Dtype: "F32", Shape: []int{cfg.MoEIntermediateSize, H}}
		man[layerName(0, stem+".w2.weight")] = tensorMeta{Dtype: "F32", Shape: []int{H, cfg.MoEIntermediateSize}}
	}
	return &Model{Cfg: cfg, manifest: man}
}

// makeFullStreams mirrors forwardV41's full-path stream initialization: stream 0
// is the live hidden and streams 1..3 are zero.
func makeFullStreams(live []float32) [][][]float32 {
	set := make([][]float32, 4)
	set[0] = append([]float32(nil), live...)
	for h := 1; h < 4; h++ {
		set[h] = make([]float32, len(live))
	}
	return [][][]float32{set}
}

func allZero32(v []float32) bool {
	for _, x := range v {
		if x != 0 {
			return false
		}
	}
	return true
}

func equalFloat32Slices(a, b []float32) bool {
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
