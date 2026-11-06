package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// native_row_gate_test.go — closed-contract pin for the qwen35 native-row gate after the CUDA
// i-quant advertisement regression (trunk commit 217d2d0f1). That commit added IQ2_XXS/IQ2_XS
// to the CUDA matmul dtype declarations; the serve wiring of the time converted that into
// WithDenseIQResident for every CUDA load, and this gate then admitted GDN in_proj_qkv/z and
// full-attention q/k rows to raw Q4_K residency on a backend whose GDN sequence primitive has
// no raw-row consumer — so GDN warmup failed closed at tensor-preflight ("gdn_in_qkv must be
// resident f32 or q8_0") until the documented per-class revert knobs were set. The restored
// last-good semantics: the gate decision follows the option arm the caller chose, not the
// dtypes the backend advertises. A tiling backend (Vulkan iq*_matvec) passes the row-route arm
// and gets the native rows; a non-tiling backend (CUDA) passes the packed-only arm plus the GDN
// off-switches, and every GDN/q-k tensor keeps the proven f32/Q8 route. These are
// representation decisions pinned by boolean assertions on the gate predicates and the
// estimator's dtype buckets; no performance claim is made here.

// nativeRowGateCfg is the exact Qwen3.8 topology the campaign served: 64 hybrid layers with
// full attention every 4th block, nV != nK so every projection is a real permutation.
func nativeRowGateCfg() model.Config {
	cfg := nativeRowsTestCfg(true)
	cfg.NumLayers = 64
	cfg.LayerTypes = make([]string, cfg.NumLayers)
	for l := range cfg.LayerTypes {
		cfg.LayerTypes[l] = "linear_attention"
		if (l+1)%4 == 0 {
			cfg.LayerTypes[l] = "full_attention"
		}
	}
	return cfg
}

// cudaServeNativeRowOptions mirrors cmd/fak denseKQuantOptions for the live CUDA backend at
// trunk: dense k-quant residency off, the selective Q2_K/Q3_K/Q5_K/Q6_K arms, i-quants retained
// only through the packed arm, and both GDN raw-row off-switches set because CUDA does not tile
// the GDN core.
func cudaServeNativeRowOptions() []Q4KLoadOption {
	return []Q4KLoadOption{
		WithDenseKQuantResident(false),
		WithNativeLinearAttnRows(false),
		WithQwen35NativeSSMOut(false),
		WithDenseQ2KResident(true),
		WithDenseQ6KResident(true),
		WithDenseQ5KResident(true),
		WithDenseQ3KResident(true),
		WithDenseIQPacked(TensorIQ4_XS, TensorIQ2_XXS, TensorIQ2_XS),
	}
}

// TestNativeRowGateStaysOffForPackedOnlyArms pins the restored last-good gate decision: under
// the packed-only arm no qwen35 raw-row route fires, so GDN in-proj, ssm_out and full-attention
// q/k stay on the f32/Q8 route the CUDA GDN sequence primitive's tensor-preflight accepts —
// GDN warmup passes with no revert knobs. // fak-test:runtime fast est=10ms lane=default
func TestNativeRowGateStaysOffForPackedOnlyArms(t *testing.T) {
	cfg := nativeRowGateCfg()
	o := probeQ4KLoadOptions(cudaServeNativeRowOptions())
	gdnQKVShape := []int{512, 256} // convDim x hidden
	gdnZShape := []int{256, 256}   // valueDim x hidden
	outProjShape := []int{256, 256}
	qShape := []int{512, 256}
	kShape := []int{128, 256}
	cases := []struct {
		label string
		got   bool
	}{
		{"GDN in_proj_qkv Q4_K", qwen35NativeRowResident(cfg, "model.layers.0.linear_attn.in_proj_qkv.weight", TensorQ4_K, gdnQKVShape, o)},
		{"GDN in_proj_z IQ2_XXS", qwen35NativeRowResident(cfg, "model.layers.0.linear_attn.in_proj_z.weight", TensorIQ2_XXS, gdnZShape, o)},
		{"GDN ssm_out Q4_K", qwen35TiledOutProjResident(cfg, "model.layers.0.linear_attn.out_proj.weight", TensorQ4_K, outProjShape, o)},
		{"full-attn q_proj Q4_K", qwen35NativeRowResident(cfg, "model.layers.3.self_attn.q_proj.weight", TensorQ4_K, qShape, o)},
		{"full-attn k_proj Q4_K", qwen35NativeRowResident(cfg, "model.layers.3.self_attn.k_proj.weight", TensorQ4_K, kShape, o)},
	}
	for _, tc := range cases {
		if tc.got {
			t.Errorf("%s: gate admitted the raw-row route under the packed-only arm; want the f32/Q8 route", tc.label)
		}
	}
	// Retention itself is intact: the i-quants stay packed, which is what the packed arm is for.
	if !denseKQuantRetained(o, TensorIQ2_XXS) || !denseKQuantRetained(o, TensorIQ4_XS) {
		t.Error("packed-only arm lost i-quant retention; identity-ordered i-quants would expand to Q8")
	}
}

// TestNativeRowGateAdmitsRowRouteArm pins the tiling-backend contract (Vulkan iq*_matvec): the
// row-route arm (WithDenseIQResident) admits exactly the Q4_K/i-quant q/k + GDN in-proj rows
// and the tiled ssm_out. Guards against an over-correction that would strand the composed
// MatMul route the gate exists for. // fak-test:runtime fast est=10ms lane=default
func TestNativeRowGateAdmitsRowRouteArm(t *testing.T) {
	cfg := nativeRowGateCfg()
	o := probeQ4KLoadOptions([]Q4KLoadOption{
		WithDenseKQuantResident(false),
		WithDenseQ2KResident(true),
		WithDenseIQResident(TensorIQ2_S),
	})
	gdnQKVShape := []int{512, 256}
	gdnZShape := []int{256, 256}
	outProjShape := []int{256, 256}
	qShape := []int{512, 256}
	kShape := []int{128, 256}
	cases := []struct {
		label string
		got   bool
	}{
		{"GDN in_proj_qkv Q4_K", qwen35NativeRowResident(cfg, "model.layers.0.linear_attn.in_proj_qkv.weight", TensorQ4_K, gdnQKVShape, o)},
		{"GDN in_proj_z IQ2_S", qwen35NativeRowResident(cfg, "model.layers.0.linear_attn.in_proj_z.weight", TensorIQ2_S, gdnZShape, o)},
		{"GDN ssm_out Q4_K", qwen35TiledOutProjResident(cfg, "model.layers.0.linear_attn.out_proj.weight", TensorQ4_K, outProjShape, o)},
		{"full-attn q_proj Q4_K", qwen35NativeRowResident(cfg, "model.layers.3.self_attn.q_proj.weight", TensorQ4_K, qShape, o)},
		{"full-attn k_proj Q4_K", qwen35NativeRowResident(cfg, "model.layers.3.self_attn.k_proj.weight", TensorQ4_K, kShape, o)},
	}
	for _, tc := range cases {
		if !tc.got {
			t.Errorf("%s: gate declined the raw-row route under the row-route arm; the tiling backend loses its native rows", tc.label)
		}
	}
}

// TestEstimateQ4KLoadMemoryPlanQwen35GDNRowsStayQ8OnPackedArm pins the fit-plan side on an
// i-quant GGUF fixture with Q4_K native rows: under the CUDA serve options the GDN in-proj /
// ssm_out / q-k tensors are charged native-q8-f32-scales (the representation the loader will
// actually store, which the CUDA GDN sequence preflight accepts), while the identity-ordered
// i-quant FFN weight stays packed. A divergence between these buckets is the estimator/loader
// skew the CUDA census test guards against. // fak-test:runtime fast est=10ms lane=default
func TestEstimateQ4KLoadMemoryPlanQwen35GDNRowsStayQ8OnPackedArm(t *testing.T) {
	f := &File{
		Metadata: map[string]Value{
			"general.architecture":                     {Type: TypeString, Value: "qwen3.8"},
			"qwen3.8.context_length":                   {Type: TypeUint64, Value: uint64(16)},
			"qwen3.8.embedding_length":                 {Type: TypeUint64, Value: uint64(256)},
			"qwen3.8.block_count":                      {Type: TypeUint64, Value: uint64(4)},
			"qwen3.8.feed_forward_length":              {Type: TypeUint64, Value: uint64(256)},
			"qwen3.8.attention.head_count":             {Type: TypeUint64, Value: uint64(2)},
			"qwen3.8.attention.head_count_kv":          {Type: TypeUint64, Value: uint64(1)},
			"qwen3.8.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-5)},
			"qwen3.8.full_attention_interval":          {Type: TypeUint64, Value: uint64(4)},
			"qwen3.8.ssm.conv_kernel":                  {Type: TypeUint64, Value: uint64(4)},
			"qwen3.8.ssm.state_size":                   {Type: TypeUint64, Value: uint64(64)},
			"qwen3.8.ssm.group_count":                  {Type: TypeUint64, Value: uint64(2)},
			"qwen3.8.ssm.time_step_rank":               {Type: TypeUint64, Value: uint64(4)},
		},
		// GGUF source names: the retained Qwen artifact ships the GDN in-proj pair as the
		// linear-attention-layer attn_qkv/attn_gate projections, which the qwen35 source-name
		// chain remaps onto model.layers.<L>.linear_attn.in_proj_qkv/in_proj_z.
		Tensors: []TensorInfo{
			{Name: "blk.0.attn_qkv.weight", Dims: []uint64{256, 512}, Type: TensorQ4_K},
			{Name: "blk.0.attn_gate.weight", Dims: []uint64{256, 256}, Type: TensorIQ2_XXS},
			{Name: "blk.0.ssm_out.weight", Dims: []uint64{256, 256}, Type: TensorQ4_K},
			{Name: "blk.3.attn_q.weight", Dims: []uint64{256, 512}, Type: TensorQ4_K},
			{Name: "blk.3.attn_k.weight", Dims: []uint64{256, 128}, Type: TensorQ4_K},
			{Name: "blk.0.ffn_up.weight", Dims: []uint64{256, 256}, Type: TensorIQ2_XXS},
		},
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	plan, err := ws.EstimateQ4KLoadMemoryPlan(cudaServeNativeRowOptions()...)
	if err != nil {
		t.Fatalf("EstimateQ4KLoadMemoryPlan: %v", err)
	}
	nativeQ8 := func(elems int) int64 { return int64(elems + elems/32*4) } // int8 codes + one f32 scale per 32 weights
	wantQ8 := nativeQ8(512*256) + nativeQ8(256*256) + nativeQ8(256*256) + nativeQ8(512*256) + nativeQ8(128*256)
	var gotQ8, gotIQ2XXS int64
	for _, d := range plan {
		switch d.DType {
		case "native-q8-f32-scales":
			gotQ8 += d.Bytes
		case "iq2_xxs":
			gotIQ2XXS += d.Bytes
		default:
			t.Errorf("unexpected device dtype bucket %q (%d B); the GDN/q-k rows leaked off the f32/Q8 route", d.DType, d.Bytes)
		}
	}
	if gotQ8 != wantQ8 {
		t.Errorf("native-q8-f32-scales = %d B, want %d B (all five GDN/q-k rows dequantized to Q8)", gotQ8, wantQ8)
	}
	if gotIQ2XXS <= 0 {
		t.Error("identity-ordered IQ2_XXS FFN weight did not stay packed; the packed arm lost its retention")
	}
}
