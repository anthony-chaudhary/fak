package model

import (
	"errors"
	"reflect"
	"testing"
)

// v41_mhc_overlap_test.go — witness for the sglang pre-mix hand-off adaptation
// (borrow-port oss-port-sglang-mhc-premix-overlap). The next role layer's
// attention mHC mix stats and pre-collapse are staged during the current role
// layer's FFN window tail and consumed at the next layer's head instead of
// opening that layer's window serially. A pure scheduling change: every value
// pinned below is bit-identical to the serial control; only the event order
// moves, and the total coefficient work across the two-layer span is unchanged.
//
// Design reference (ADAPT): sgl-project/sglang
// python/sglang/srt/models/deepseek_v4_mhc.py @3c6a6c8068658b88e40bcaa5a0bf48b9a94ebdba
// (Apache-2.0) -- "Step 1's pre is consumed by step 2 of the next
// hyper-connection -- the lag that lets step 1 run beside steps 2-3".

// v41MHCOverlapRoleModel builds a two-layer full-geometry V4.1 fixture whose
// consecutive layers are both ratio-2 source/index-source role layers, so the
// pre-mix hand-off has a consumer at layer 1. Plain F32 experts keep the step
// on the host reference path.
func v41MHCOverlapRoleModel(t *testing.T) *Model {
	t.Helper()
	cfg := v41TestReducedConfig(t, 2, V41RouterExperts)
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
	cfg.Window = []int{-1}
	d41 := cfg.DeepSeekV41
	d41.CompressRatios = []int{2, 2}
	d41.KVSourceLayerIDs = nil
	d41.IndexSourceLayerIDs = []int{0, 1}
	d41.EngramLayerIDs = nil
	d41.Attention = DeepSeekV41AttentionGeometry{}
	cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads, cfg.OGroups = v41KVLoraRank, 1, 1, 1
	cfg.IndexNHeads, cfg.IndexHeadDim, cfg.IndexTopK = 1, v41KVLoraRank, 2

	H, I := cfg.HiddenSize, cfg.MoEIntermediateSize
	width := v41CompressorWidth(cfg)
	type ts = synthTensor
	tensors := []ts{
		{"model.embed_tokens.weight", []int{cfg.VocabSize, H}},
		{"lm_head.weight", []int{cfg.VocabSize, H}},
		{"model.norm.weight", []int{H}},
	}
	for l := 0; l < cfg.NumLayers; l++ {
		tensors = append(tensors,
			ts{layerName(l, "attn_norm.weight"), []int{H}},
			ts{layerName(l, "ffn_norm.weight"), []int{H}},
			ts{layerName(l, "mhc.mixes.weight"), []int{4 * H, v41MHCMixWidth}},
			ts{layerName(l, "mhc.ffn_mixes.weight"), []int{4 * H, v41MHCMixWidth}},
			ts{layerName(l, "mhc.base"), []int{v41MHCMixWidth}},
			ts{layerName(l, "mhc.scale"), []int{3}},
			ts{layerName(l, "mhc.ffn_base"), []int{v41MHCMixWidth}},
			ts{layerName(l, "mhc.ffn_scale"), []int{3}},
			ts{layerName(l, "attn.wq_a.weight"), []int{cfg.QLoraRank, H}},
			ts{layerName(l, "attn.wq_a_norm.weight"), []int{cfg.QLoraRank}},
			ts{layerName(l, "attn.wq_b.weight"), []int{cfg.HeadDim, cfg.QLoraRank}},
			ts{layerName(l, "attn.wkv.weight"), []int{v41KVLoraRank, H}},
			ts{layerName(l, "attn.kv_norm.weight"), []int{v41KVLoraRank}},
			ts{layerName(l, "attn.wo_a.weight"), []int{cfg.OLoraRank, cfg.HeadDim}},
			ts{layerName(l, "attn.wo_b.weight"), []int{H, cfg.OLoraRank}},
			ts{layerName(l, "attn.sink"), []int{1}},
			ts{layerName(l, "attn.compressor.wkv.weight"), []int{width, H}},
			ts{layerName(l, "attn.compressor.wgate.weight"), []int{width, H}},
			ts{layerName(l, "attn.compressor.norm.weight"), []int{width}},
			ts{layerName(l, "indexer.wq_b.weight"), []int{cfg.IndexNHeads * cfg.IndexHeadDim, cfg.QLoraRank}},
			ts{layerName(l, "indexer.wk.weight"), []int{cfg.IndexHeadDim, width}},
			ts{layerName(l, "indexer.k_norm.weight"), []int{cfg.IndexHeadDim}},
			ts{layerName(l, "indexer.weights_proj.weight"), []int{cfg.IndexNHeads, H}},
			ts{layerName(l, "ffn.gate.weight"), []int{cfg.NumExperts, H}},
			ts{layerName(l, "ffn.gate.e_score_correction_bias"), []int{cfg.NumExperts}},
			ts{layerName(l, "ffn.shared_experts.w1.weight"), []int{I, H}},
			ts{layerName(l, "ffn.shared_experts.w3.weight"), []int{I, H}},
			ts{layerName(l, "ffn.shared_experts.w2.weight"), []int{H, I}},
		)
		for e := 0; e < cfg.NumExperts; e++ {
			stem := "ffn.experts." + itoa(e)
			tensors = append(tensors,
				ts{layerName(l, stem+".w1.weight"), []int{I, H}},
				ts{layerName(l, stem+".w3.weight"), []int{I, H}},
				ts{layerName(l, stem+".w2.weight"), []int{H, I}},
			)
		}
	}
	man, raw := synthBuildRawResolved(tensors, func(name string) synthElemFill {
		switch {
		case name == "model.norm.weight" || hasSuffix(name, "attn_norm.weight") || hasSuffix(name, "ffn_norm.weight"):
			return synthConstFill(1.0)
		case hasSuffix(name, "mhc.ffn_scale"):
			return synthConstFill(.75)
		case hasSuffix(name, "mhc.scale"):
			return synthConstFill(1.0)
		case hasSuffix(name, "mhc.base") || hasSuffix(name, "mhc.ffn_base"):
			return synthConstFill(0.0)
		case hasSuffix(name, "attn.sink"):
			return synthScaledFill(0.25)
		default:
			return synthMatmulFillFor(name)
		}
	})
	m := &Model{Cfg: cfg, manifest: man, raw: raw}
	if full, err := v41ForwardGeometry(m.Cfg); err != nil || !full {
		t.Fatalf("fixture geometry full=%t: %v", full, err)
	}
	for _, l := range []int{0, 1} {
		plan, err := v41AttentionPlanFor(m.Cfg, l, m.v41AttentionRolesCached())
		if err != nil || (plan.Role == V41AttentionRolePerLayer && plan.Ratio == 0) {
			t.Fatalf("layer %d schedule is not a role (role=%v ratio=%d): %v", l, plan.Role, plan.Ratio, err)
		}
	}
	return m
}

// v41MHCOverlapEventCounts counts the closed mHC observability vocabulary a
// role step emits: the projection fact and the stream-panel facts.
func v41MHCOverlapEventCounts(stages []string) map[string]int {
	counts := map[string]int{}
	for _, stage := range stages {
		if stage == "mhc-projection" || stage == "mhc-streams" {
			counts[stage]++
		}
	}
	return counts
}

func v41MHCOverlapCopyStreams(s [][]float32) [][]float32 {
	out := make([][]float32, len(s))
	for i := range s {
		out[i] = append([]float32(nil), s[i]...)
	}
	return out
}

type v41MHCOverlapOutcome struct {
	x1, xFinal []float32
	s1, sFinal [][]float32
	t0, t1     []string
}

// v41MHCOverlapRunSerial drives the two role layers through the seam with the
// hand-off disabled: layer 1 opens its window with its own projection, split
// and pre-collapse.
func v41MHCOverlapRunSerial(t *testing.T, m *Model, plan0, plan1 V41AttentionPlan, pos int) v41MHCOverlapOutcome {
	t.Helper()
	outcome := v41MHCOverlapOutcome{}
	s := m.NewSession()
	defer s.Close()
	if got := s.Prefill([]int{1, 2}); len(got) == 0 {
		t.Fatal("prefix prefill produced no logits")
	}
	state := s.v41State()
	scratch := &v41ProjScratch{}
	scratch.bindV41StepState(state)
	scratch.mhcCarry = newV41MHCCarry(1)
	x, streams := v41LayerStepInputs(t, m, 3, true)
	s.hostStageObserver = func(stage string) { outcome.t0 = append(outcome.t0, stage) }
	if err := m.v41LayerStepRole(0, plan0, x, streams, pos, state.layers[0], state.attn, scratch); err != nil {
		t.Fatalf("serial layer 0 step: %v", err)
	}
	outcome.x1 = append([]float32(nil), x...)
	outcome.s1 = v41MHCOverlapCopyStreams(streams)
	s.hostStageObserver = func(stage string) { outcome.t1 = append(outcome.t1, stage) }
	if err := m.v41LayerStepRole(1, plan1, x, streams, pos, state.layers[1], state.attn, scratch); err != nil {
		t.Fatalf("serial layer 1 step: %v", err)
	}
	outcome.xFinal = append([]float32(nil), x...)
	outcome.sFinal = v41MHCOverlapCopyStreams(streams)
	return outcome
}

// v41MHCOverlapRunOverlapped drives the same fixed trace with the hand-off
// enabled: layer 0's FFN window tail publishes layer 1's attention mix stats
// and pre-collapse into the carry, and layer 1's head consumes them. It also
// returns the published slot contents and the post-layer-0 panel/pre row the
// publication was computed from.
func v41MHCOverlapRunOverlapped(t *testing.T, m *Model, plan0, plan1 V41AttentionPlan, pos int) (v41MHCOverlapOutcome, v41MHCMix, []float32, [][]float32, []float32) {
	t.Helper()
	outcome := v41MHCOverlapOutcome{}
	s := m.NewSession()
	defer s.Close()
	if got := s.Prefill([]int{1, 2}); len(got) == 0 {
		t.Fatal("prefix prefill produced no logits")
	}
	state := s.v41State()
	scratch := &v41ProjScratch{}
	scratch.bindV41StepState(state)
	scratch.mhcCarry = newV41MHCCarry(1)
	scratch.mhcCarry.EnablePreMixOverlap()
	x, streams := v41LayerStepInputs(t, m, 3, true)
	s.hostStageObserver = func(stage string) { outcome.t0 = append(outcome.t0, stage) }
	if err := m.v41LayerStepRole(0, plan0, x, streams, pos, state.layers[0], state.attn, scratch); err != nil {
		t.Fatalf("overlapped layer 0 step: %v", err)
	}
	carry := scratch.mhcCarry
	if !carry.staged || carry.stagedLayer != 1 || carry.nextLayer != 1 {
		t.Fatalf("layer 0 finish did not publish the pre-mix hand-off (staged=%t layer=%d cursor=%d)", carry.staged, carry.stagedLayer, carry.nextLayer)
	}
	stagedMix, stagedCollapsed := carry.stagedMix, carry.stagedCollapsed
	panel := v41MHCOverlapCopyStreams(streams)
	preRow := append([]float32(nil), carry.pre[0]...)
	outcome.x1 = append([]float32(nil), x...)
	outcome.s1 = v41MHCOverlapCopyStreams(streams)
	s.hostStageObserver = func(stage string) { outcome.t1 = append(outcome.t1, stage) }
	if err := m.v41LayerStepRole(1, plan1, x, streams, pos, state.layers[1], state.attn, scratch); err != nil {
		t.Fatalf("overlapped layer 1 step: %v", err)
	}
	if carry.staged {
		t.Fatal("the hand-off survived its consumer")
	}
	outcome.xFinal = append([]float32(nil), x...)
	outcome.sFinal = v41MHCOverlapCopyStreams(streams)
	return outcome, stagedMix, stagedCollapsed, panel, preRow
}

// fak-test:runtime medium est=5s lane=default
func TestV41MHCOverlapCoefficientParity(t *testing.T) {
	m := v41MHCOverlapRoleModel(t)
	cfg := m.Cfg
	pos := 2 // first decode position after the two-token prefix

	plan0, err := v41AttentionPlanFor(cfg, 0, m.v41AttentionRolesCached())
	if err != nil {
		t.Fatal(err)
	}
	plan1, err := v41AttentionPlanFor(cfg, 1, m.v41AttentionRolesCached())
	if err != nil {
		t.Fatal(err)
	}
	serial := v41MHCOverlapRunSerial(t, m, plan0, plan1, pos)
	overlapped, stagedMix, stagedCollapsed, panel, preRow := v41MHCOverlapRunOverlapped(t, m, plan0, plan1, pos)

	// Serial oracle for layer 1's attention coefficients, straight from the
	// pinned primitives: project the post-layer-0 panel, split, pre-collapse.
	// The same calls and arguments the serial seam would run at layer 1's head.
	H := cfg.HiddenSize
	eps := float32(cfg.RMSNormEps)
	flat, transposed, ok := m.v41MHCWeightLayout(1)
	if !ok {
		t.Fatal("layer 1 mHC mix weight holds no admitted geometry")
	}
	input := make([]float32, 0, 4*H)
	for _, stream := range panel {
		input = append(input, stream...)
	}
	oracleScratch := &v41ProjScratch{}
	projectMHC := m.v41MHCProjectorObserved(1, H, eps, flat, transposed, oracleScratch, nil)
	mixes, err := projectMHC(input)
	if err != nil {
		t.Fatalf("oracle projection: %v", err)
	}
	wantMix, err := v41MHCSplitOnObserved(nil, 1, mixes, m.v41LeafTensor(1, v41LeafMHCScale), m.v41LeafTensor(1, v41LeafMHCBase), 4, hcItersOrDefault(cfg), hcEpsOrDefault(cfg), nil)
	if err != nil {
		t.Fatalf("oracle split: %v", err)
	}
	// The host seam (nil device) keeps the pre-collapse unrounded, exactly as
	// the serial role preparation does; only the device seam rounds here.
	wantCollapsed, err := v41MHCPreObserved(panel, preRow, nil)
	if err != nil {
		t.Fatalf("oracle pre-collapse: %v", err)
	}

	// Coefficient parity: the staged hand-off is bit-identical to the serial
	// arithmetic on the same panel and pre row.
	if !reflect.DeepEqual(stagedMix, wantMix) {
		t.Fatalf("staged coefficients diverge from the serial control: pre=%v post=%v comb=%v", stagedMix.pre, stagedMix.post, stagedMix.comb)
	}
	if !reflect.DeepEqual(stagedCollapsed, wantCollapsed) {
		t.Fatalf("staged pre-collapse diverges from the serial control: got %v want %v", stagedCollapsed[:4], wantCollapsed[:4])
	}

	// Scheduling never changes numerics: layer 0's output and the two-layer
	// end point agree bit-for-bit between regimes.
	if !reflect.DeepEqual(serial.x1, overlapped.x1) || !reflect.DeepEqual(serial.s1, overlapped.s1) {
		t.Fatal("layer 0 output differs between the serial and overlapped schedules")
	}
	if !reflect.DeepEqual(serial.xFinal, overlapped.xFinal) || !reflect.DeepEqual(serial.sFinal, overlapped.sFinal) {
		t.Fatal("two-layer end point differs between the serial and overlapped schedules")
	}

	// Ordered observability: the next sublayer's coefficient work moved from
	// layer 1's window head into layer 0's FFN window tail. Per-step closed
	// counts: the observed seam reports the attention preparation's
	// projection + split + pre-collapse (one projection, two stream facts) and
	// the final post-mix's stream fact; the FFN sublayer's own coefficient
	// work runs through the unobserved seam, so the hand-off adds exactly one
	// projection and two stream facts to layer 0's step and removes the whole
	// attention-preparation block from layer 1's.
	check := func(got, want map[string]int, where string) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s emitted %d tracked facts, want %d", where, len(got), len(want))
		}
		for k, w := range want {
			if got[k] != w {
				t.Fatalf("%s fact %s = %d, want %d", where, k, got[k], w)
			}
		}
	}
	check(v41MHCOverlapEventCounts(serial.t0), map[string]int{"mhc-projection": 1, "mhc-streams": 3}, "serial layer 0 step")
	check(v41MHCOverlapEventCounts(serial.t1), map[string]int{"mhc-projection": 1, "mhc-streams": 3}, "serial layer 1 step")
	check(v41MHCOverlapEventCounts(overlapped.t0), map[string]int{"mhc-projection": 2, "mhc-streams": 5}, "overlapped layer 0 step")
	check(v41MHCOverlapEventCounts(overlapped.t1), map[string]int{"mhc-streams": 1}, "overlapped layer 1 step")

	// Total coefficient work across the span is invariant: overlap reschedules
	// it, it does not add or drop any.
	sum := func(a, b []string) map[string]int {
		total := v41MHCOverlapEventCounts(a)
		for k, v := range v41MHCOverlapEventCounts(b) {
			total[k] += v
		}
		return total
	}
	if !reflect.DeepEqual(sum(serial.t0, serial.t1), sum(overlapped.t0, overlapped.t1)) {
		t.Fatalf("total coefficient work changed: serial=%v overlapped=%v", sum(serial.t0, serial.t1), sum(overlapped.t0, overlapped.t1))
	}

	// Within layer 0's overlapped window the staged projection precedes the
	// staged split and pre-collapse: the last projection sits before the last
	// two stream facts, inside this layer's trace.
	lastProj, lastStream := -1, -1
	for i, stage := range overlapped.t0 {
		if stage == "mhc-projection" {
			lastProj = i
		}
		if stage == "mhc-streams" {
			lastStream = i
		}
	}
	if lastProj < 0 || lastStream <= lastProj {
		t.Fatalf("staged split/pre do not follow the staged projection in layer 0's window: proj=%d stream=%d", lastProj, lastStream)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV41MHCOverlapGuards(t *testing.T) {
	m := v41MHCOverlapRoleModel(t)
	cfg := m.Cfg
	pos := 2

	plan0, err := v41AttentionPlanFor(cfg, 0, m.v41AttentionRolesCached())
	if err != nil {
		t.Fatal(err)
	}

	t.Run("disabled flag publishes nothing", func(t *testing.T) {
		s := m.NewSession()
		defer s.Close()
		if got := s.Prefill([]int{1, 2}); len(got) == 0 {
			t.Fatal("prefix prefill produced no logits")
		}
		state := s.v41State()
		scratch := &v41ProjScratch{}
		scratch.bindV41StepState(state)
		scratch.mhcCarry = newV41MHCCarry(1)
		x, streams := v41LayerStepInputs(t, m, 3, true)
		if err := m.v41LayerStepRole(0, plan0, x, streams, pos, state.layers[0], state.attn, scratch); err != nil {
			t.Fatalf("layer 0 step: %v", err)
		}
		if scratch.mhcCarry.staged {
			t.Fatal("hand-off published with the flag off")
		}
	})

	t.Run("last layer skips staging", func(t *testing.T) {
		scratch := &v41ProjScratch{mhcCarry: newV41MHCCarry(1)}
		scratch.mhcCarry.EnablePreMixOverlap()
		streams := make([][]float32, 4)
		for h := range streams {
			streams[h] = make([]float32, cfg.HiddenSize)
		}
		if err := m.v41MHCStageNextAttention(1, 2, streams, scratch); err != nil {
			t.Fatalf("staging past the last layer: %v", err)
		}
		if scratch.mhcCarry.staged {
			t.Fatal("hand-off published for a nonexistent layer")
		}
	})

	t.Run("plain next layer skips staging", func(t *testing.T) {
		cfg2 := cfg
		d2 := *cfg.DeepSeekV41
		d2.CompressRatios = []int{2, 0}
		d2.IndexSourceLayerIDs = []int{0}
		cfg2.DeepSeekV41 = &d2
		m2 := &Model{Cfg: cfg2, manifest: m.manifest, raw: m.raw}
		scratch := &v41ProjScratch{mhcCarry: newV41MHCCarry(1)}
		scratch.mhcCarry.EnablePreMixOverlap()
		streams := make([][]float32, 4)
		for h := range streams {
			streams[h] = make([]float32, cfg.HiddenSize)
		}
		if err := m2.v41MHCStageNextAttention(0, 1, streams, scratch); err != nil {
			t.Fatalf("staging for a plain next layer: %v", err)
		}
		if scratch.mhcCarry.staged {
			t.Fatal("hand-off published for a plain next layer")
		}
	})

	t.Run("resident panel skips staging", func(t *testing.T) {
		scratch := &v41ProjScratch{mhcCarry: newV41MHCCarry(1)}
		scratch.mhcCarry.EnablePreMixOverlap()
		scratch.mhcCarry.resident = &v41MHCResident{}
		streams := make([][]float32, 4)
		for h := range streams {
			streams[h] = make([]float32, cfg.HiddenSize)
		}
		if err := m.v41MHCStageNextAttention(0, 1, streams, scratch); err != nil {
			t.Fatalf("staging against a resident panel: %v", err)
		}
		if scratch.mhcCarry.staged {
			t.Fatal("hand-off published against a resident panel")
		}
	})

	t.Run("malformed panel refuses", func(t *testing.T) {
		scratch := &v41ProjScratch{mhcCarry: newV41MHCCarry(1)}
		scratch.mhcCarry.EnablePreMixOverlap()
		streams := make([][]float32, 4)
		for h := range streams {
			streams[h] = make([]float32, cfg.HiddenSize)
		}
		err := m.v41MHCStageNextAttention(0, 1, streams[:3], scratch)
		if !errors.Is(err, ErrV41ForwardStage) || scratch.mhcCarry.staged {
			t.Fatalf("short panel escaped: err=%v staged=%t", err, scratch.mhcCarry.staged)
		}
	})

	t.Run("takeStagedAttention contract", func(t *testing.T) {
		c := newV41MHCCarry(1)
		if _, _, ok := c.takeStagedAttention(0); ok {
			t.Fatal("fresh carry reported a pending hand-off")
		}
		mix := v41MHCMix{pre: []float32{1, 0, 0, 0}, post: []float32{2, 2, 2, 2}, comb: make([]float32, 16)}
		c.nextLayer = 1
		c.stageAttention(1, mix, []float32{0.5})
		if _, _, ok := c.takeStagedAttention(0); ok {
			t.Fatal("hand-off accepted for the wrong layer")
		}
		gotMix, gotCollapsed, ok := c.takeStagedAttention(1)
		if !ok || !reflect.DeepEqual(gotMix, mix) || !reflect.DeepEqual(gotCollapsed, []float32{0.5}) {
			t.Fatalf("hand-off refused or altered for its own layer: ok=%t", ok)
		}
		if _, _, ok := c.takeStagedAttention(1); ok {
			t.Fatal("hand-off consumed more than once")
		}
		// Cursor drift: a step advanced past the publication's cursor.
		c.nextLayer = 1
		c.stageAttention(1, mix, []float32{0.5})
		c.nextLayer = 2
		if _, _, ok := c.takeStagedAttention(1); ok {
			t.Fatal("hand-off accepted after the cursor moved on")
		}
		// Malformed shape: the stored coefficients fail the closed contract.
		c.nextLayer = 1
		bad := mix
		bad.comb = bad.comb[:15]
		c.stageAttention(1, bad, []float32{0.5})
		if _, _, ok := c.takeStagedAttention(1); ok {
			t.Fatal("hand-off accepted a malformed coefficient set")
		}
	})
}
