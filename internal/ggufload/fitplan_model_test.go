package ggufload

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fitplan"
)

func fitPlanDenseWeightSource(t *testing.T, untied bool) *WeightSource {
	t.Helper()
	tensors := []TensorInfo{
		{Name: "token_embd.weight", Dims: []uint64{32, 16}, Type: TensorF32},
		{Name: "blk.0.attn_q.weight", Dims: []uint64{32, 32}, Type: TensorF32},
	}
	if untied {
		tensors = append(tensors, TensorInfo{Name: "output.weight", Dims: []uint64{32, 16}, Type: TensorF32})
	}
	f := &File{
		Metadata: map[string]Value{
			"general.architecture":                   {Type: TypeString, Value: "qwen2"},
			"qwen2.context_length":                   {Type: TypeUint64, Value: uint64(16)},
			"qwen2.embedding_length":                 {Type: TypeUint64, Value: uint64(32)},
			"qwen2.block_count":                      {Type: TypeUint64, Value: uint64(2)},
			"qwen2.feed_forward_length":              {Type: TypeUint64, Value: uint64(64)},
			"qwen2.attention.head_count":             {Type: TypeUint64, Value: uint64(4)},
			"qwen2.attention.head_count_kv":          {Type: TypeUint64, Value: uint64(2)},
			"qwen2.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-5)},
			"qwen2.rope.freq_base":                   {Type: TypeFloat32, Value: float32(10000)},
			"tokenizer.ggml.eos_token_id":            {Type: TypeUint32, Value: uint32(2)},
		},
		Tensors: tensors,
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

// fak-test:runtime fast est=1ms lane=default
func TestFitPlanModelDenseUntiedSweepExcludesInputEmbedding(t *testing.T) {
	ws := fitPlanDenseWeightSource(t, true)
	m, err := ws.FitPlanModel(EmbedPayload)
	if err != nil {
		t.Fatal(err)
	}
	payload, err := ws.EstimateLoadBytes()
	if err != nil {
		t.Fatal(err)
	}
	if m.PayloadWeightBytes != payload || m.StreamBytesPerToken != payload-32*16*4 {
		t.Fatalf("payload %d stream %d, want %d and %d", m.PayloadWeightBytes, m.StreamBytesPerToken, payload, payload-32*16*4)
	}
	if want := int64(32*16*2+32*32) - 32*16; m.ActiveParamsPerToken != want {
		t.Fatalf("ActiveParamsPerToken = %d, want %d", m.ActiveParamsPerToken, want)
	}
	if m.GatherPayloadBytes != 32*16*4 || m.GatherResidentBytes != 32*16*4 || m.HeadParams != 32*16 {
		t.Fatalf("gather payload %d resident %d head %d", m.GatherPayloadBytes, m.GatherResidentBytes, m.HeadParams)
	}
	if m.ActiveSetUnknown || m.AttnLayers != 2 {
		t.Fatalf("model = %+v", m)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestFitPlanModelDenseTiedSweepsWholePayload(t *testing.T) {
	m, err := fitPlanDenseWeightSource(t, false).FitPlanModel(EmbedPayload)
	if err != nil {
		t.Fatal(err)
	}
	if m.StreamBytesPerToken != m.PayloadWeightBytes || m.PayloadWeightBytes <= 0 {
		t.Fatalf("tied model stream %d payload %d", m.StreamBytesPerToken, m.PayloadWeightBytes)
	}
	if m.GatherPayloadBytes != 0 || m.GatherResidentBytes != 0 || m.HeadParams != 32*16 {
		t.Fatalf("gather payload %d resident %d head %d", m.GatherPayloadBytes, m.GatherResidentBytes, m.HeadParams)
	}
}

func openGLMFitPlanFixture(t *testing.T, k int) *WeightSource {
	t.Helper()
	raw := glmEmbedGatherGGUF(t, 2, 2, 64, k, 256, 256, TensorQ4_K, TensorQ8_0, TensorQ8_0, true)
	p := filepath.Join(t.TempDir(), "glm.gguf")
	if err := os.WriteFile(p, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	ws, err := OpenWeights(p)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	t.Cleanup(func() { ws.Close() })
	return ws
}

// fak-test:runtime fast est=10ms lane=default
func TestFitPlanModelMoEReusesRoutedActiveSet(t *testing.T) {
	ws := openGLMFitPlanFixture(t, 2)
	as, ok, err := ws.RoutedExpertActiveSet()
	if err != nil || !ok || as.ExpertsUsed <= 0 {
		t.Fatalf("precondition: active set ok=%v err=%v %+v", ok, err, as)
	}
	m, err := ws.FitPlanModel(EmbedPayload)
	if err != nil {
		t.Fatal(err)
	}
	if m.StreamBytesPerToken != as.ActiveBytesPerTokenSwept || m.RoutedExpertBytes != as.RoutedResident ||
		m.ActiveRoutedBytesPerToken != as.ActivePerToken || m.PayloadWeightBytes != as.RoutedResident+as.NonExpertResident {
		t.Fatalf("model %+v does not mirror active set %+v", m, as)
	}
	if m.ActiveSetUnknown || m.StreamBytes(1) != m.StreamBytesPerToken {
		t.Fatalf("model = %+v", m)
	}
	// The untied input table is gathered, not multiplied: its 256 elements leave the active
	// params and the 256-element output.weight is the head.
	if m.ActiveParamsPerToken != as.ActiveParamsPerToken-256 || m.HeadParams != 256 || m.GatherPayloadBytes != as.InputEmbedGather || as.InputEmbedGather <= 0 {
		t.Fatalf("active %d head %d gather %d, active set %+v", m.ActiveParamsPerToken, m.HeadParams, m.GatherPayloadBytes, as)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestFitPlanModelMoEWithoutExpertsUsedIsActiveSetUnknown(t *testing.T) {
	m, err := openGLMFitPlanFixture(t, 0).FitPlanModel(EmbedPayload)
	if err != nil {
		t.Fatal(err)
	}
	if !m.ActiveSetUnknown {
		t.Fatalf("model = %+v, want ActiveSetUnknown", m)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestStreamedResidentDTypeMatchesEstimator(t *testing.T) {
	plan, err := denseStreamedTestWeightSource(t, 256*144).EstimateQ4KLoadMemoryPlan(WithStreamedDenseQ4KWorkingSet(8 * 1024))
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range plan {
		if d.Detail == "gguf-host-dense-streamed" {
			if d.DType != fitplan.StreamedResidentDType {
				t.Fatalf("streamed row DType = %q, want %q", d.DType, fitplan.StreamedResidentDType)
			}
			return
		}
	}
	t.Fatalf("no streamed-dense row in %+v", plan)
}

// fitPlanQ4KWeightSource is a qualified resident-Q4K qwen2 whose Q4_K token table the arm keeps
// as a 256x8192 f32 gather table (8 MiB) against a 1.125 MiB payload.
func fitPlanQ4KWeightSource(t *testing.T, tied bool) *WeightSource {
	t.Helper()
	const h, ff, vocab = 256, 512, 8192
	tensors := []TensorInfo{
		{Name: "token_embd.weight", Dims: []uint64{h, vocab}, Type: TensorQ4_K},
		{Name: "blk.0.attn_q.weight", Dims: []uint64{h, h}, Type: TensorQ4_K},
		{Name: "blk.0.attn_k.weight", Dims: []uint64{h, h}, Type: TensorQ4_K},
		{Name: "blk.0.attn_v.weight", Dims: []uint64{h, h}, Type: TensorQ4_K},
		{Name: "blk.0.attn_output.weight", Dims: []uint64{h, h}, Type: TensorQ4_K},
		{Name: "blk.0.ffn_gate.weight", Dims: []uint64{h, ff}, Type: TensorQ4_K},
		{Name: "blk.0.ffn_up.weight", Dims: []uint64{h, ff}, Type: TensorQ4_K},
		{Name: "blk.0.ffn_down.weight", Dims: []uint64{ff, h}, Type: TensorQ4_K},
	}
	if !tied {
		tensors = append(tensors, TensorInfo{Name: "output.weight", Dims: []uint64{h, vocab}, Type: TensorQ4_K})
	}
	f := &File{
		Metadata: map[string]Value{
			"general.architecture":                   {Type: TypeString, Value: "qwen2"},
			"qwen2.context_length":                   {Type: TypeUint64, Value: uint64(4096)},
			"qwen2.embedding_length":                 {Type: TypeUint64, Value: uint64(h)},
			"qwen2.block_count":                      {Type: TypeUint64, Value: uint64(1)},
			"qwen2.feed_forward_length":              {Type: TypeUint64, Value: uint64(ff)},
			"qwen2.attention.head_count":             {Type: TypeUint64, Value: uint64(4)},
			"qwen2.attention.head_count_kv":          {Type: TypeUint64, Value: uint64(4)},
			"qwen2.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-5)},
			"qwen2.rope.freq_base":                   {Type: TypeFloat32, Value: float32(10000)},
			"tokenizer.ggml.eos_token_id":            {Type: TypeUint32, Value: uint32(2)},
		},
		Tensors: tensors,
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

const fitPlanQ4KEmbedF32 = 256 * 8192 * 4

// fak-test:runtime fast est=5ms lane=default
func TestFitPlanModelGatherResidentFollowsArmStorage(t *testing.T) {
	embPayload := int64(256 * 8192 / 256 * 144) // Q4_K: 144 B per 256-weight super-block
	for _, tc := range []struct {
		tied  bool
		embed EmbedStorage
		want  int64
	}{
		{false, EmbedPayload, embPayload},
		{false, EmbedF32, fitPlanQ4KEmbedF32},
		{false, EmbedQ8, fitPlanQ4KEmbedF32},
		{false, EmbedQ4K, fitPlanQ4KEmbedF32},
		{true, EmbedPayload, 0}, // the table is the swept head itself
		{true, EmbedF32, 0},
		{true, EmbedQ8, fitPlanQ4KEmbedF32}, // f32 gather copy beside a native-Q8 head
		{true, EmbedQ4K, fitPlanQ4KEmbedF32},
	} {
		m, err := fitPlanQ4KWeightSource(t, tc.tied).FitPlanModel(tc.embed)
		if err != nil {
			t.Fatal(err)
		}
		wantPayload := embPayload
		if tc.tied {
			wantPayload = 0
		}
		if m.GatherResidentBytes != tc.want || m.GatherPayloadBytes != wantPayload {
			t.Fatalf("tied=%v embed=%d: gather resident %d payload %d, want %d and %d",
				tc.tied, tc.embed, m.GatherResidentBytes, m.GatherPayloadBytes, tc.want, wantPayload)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestFitPlanQ4KDecodeSweepExcludesGatherTable(t *testing.T) {
	for _, tied := range []bool{false, true} {
		ws := fitPlanQ4KWeightSource(t, tied)
		weights, err := ws.EstimateQ4KLoadMemoryPlan()
		if err != nil {
			t.Fatal(err)
		}
		m, err := ws.FitPlanModel(EmbedQ4K)
		if err != nil {
			t.Fatal(err)
		}
		cfg, err := ws.File.Config()
		if err != nil {
			t.Fatal(err)
		}
		rep := fitplan.Plan(fitplan.Input{
			Model:   m,
			Weights: weights,
			Hardware: fitplan.Hardware{Topology: fitplan.TopologyUnified, PoolUsableBytes: 1 << 34,
				Peaks: fitplan.Peaks{BytesPerSec: 256e9, FLOPsPerSec: 29.7e12}},
			Workload:    fitplan.DefaultWorkload(),
			Calibration: fitplan.GenericEmpirical(),
			Candidates:  fitplan.BuildCandidates([]fitplan.KVTier{{Precision: "f32", Context: cfg.ContextSizeConfig()}}, 2048),
		})
		swept := float64(weights.Total() - fitPlanQ4KEmbedF32) // a tied model keeps its native-Q8 head
		if got := rep.Evidence.ResidentScale * float64(m.StreamBytesPerToken); math.Abs(got-swept) > 1 {
			t.Fatalf("tied=%v: decode sweeps %.0f resident bytes, want %.0f", tied, got, swept)
		}
	}
}
