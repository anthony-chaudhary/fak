package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// cudaServeCensusOptions is the option set cmd/fak serveDenseKQuantOptions resolves for the CUDA
// backend today: dense k-quant residency off, packed Q2_K, Q3_K, Q5_K and Q6_K kept (declared CUDA matmul
// dtypes with execution probes), and no i-quant type kept by this fixture.
func cudaServeCensusOptions() []Q4KLoadOption {
	return []Q4KLoadOption{WithDenseKQuantResident(false), WithDenseQ2KResident(true), WithDenseQ6KResident(true), WithDenseQ5KResident(true), WithDenseQ3KResident(true)}
}

// TestEstimateQ4KLoadMemoryPlanCUDAServeResidencyPerType pins the per-type device representation
// behind the Qwen3.8-27B UD-Q4_K_M refusal (29.365 GiB planned for a 15.32 GiB payload): Q4_K
// Q3_K and Q5_K matmuls stay packed, the Q4_K token table expands to f32, and IQ4_XS and Q8_0
// matmuls are re-quantized to native Q8 at 1.125 B/weight. A change that keeps one more type packed
// must update this table on purpose. Without the Q3_K arm the Q3_K FFN layer joins the Q8 row
// (the real file's 7 Q3_K FFN tensors: 701,890,560 B as Q8 vs 268,083,200 B packed).
// fak-test:runtime fast est=10ms lane=default
func TestEstimateQ4KLoadMemoryPlanCUDAServeResidencyPerType(t *testing.T) {
	f := &File{
		Metadata: map[string]Value{
			"general.architecture":                   {Type: TypeString, Value: "llama"},
			"llama.embedding_length":                 {Type: TypeUint32, Value: uint32(256)},
			"llama.block_count":                      {Type: TypeUint32, Value: uint32(1)},
			"llama.attention.head_count":             {Type: TypeUint32, Value: uint32(1)},
			"llama.feed_forward_length":              {Type: TypeUint32, Value: uint32(256)},
			"llama.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-5)},
		},
		Tensors: []TensorInfo{
			{Name: "token_embd.weight", Dims: []uint64{256, 8}, Type: TensorQ4_K},
			{Name: "output.weight", Dims: []uint64{256, 8}, Type: TensorQ6_K},
			{Name: "blk.0.ffn_up.weight", Dims: []uint64{256, 256}, Type: TensorQ4_K},
			{Name: "blk.0.ffn_gate.weight", Dims: []uint64{256, 256}, Type: TensorIQ4_XS},
			{Name: "blk.0.attn_v.weight", Dims: []uint64{256, 256}, Type: TensorQ5_K},
			{Name: "blk.0.attn_output.weight", Dims: []uint64{256, 256}, Type: TensorQ8_0},
			{Name: "blk.0.ffn_down.weight", Dims: []uint64{256, 256}, Type: TensorQ3_K},
			{Name: "blk.0.attn_norm.weight", Dims: []uint64{256}, Type: TensorF32},
			{Name: "blk.0.ffn_norm.weight", Dims: []uint64{256}, Type: TensorF32},
			{Name: "output_norm.weight", Dims: []uint64{256}, Type: TensorF32},
		},
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	const (
		matrix   = 256 * 256
		embedF32 = 256 * 8 * 4
		normsF32 = 3 * 256 * 4
		q8       = matrix + matrix/32*4 // int8 codes + one f32 scale per 32 weights
		q4Packed = 256 * 144
		q6Packed = 8 * 210
		q3Packed = 256 * 110
		q5Packed = 256 * 176
	)
	cases := []struct {
		name string
		opts []Q4KLoadOption
		want map[string]int64
	}{
		{name: "cuda-serve", opts: cudaServeCensusOptions(), want: map[string]int64{
			compute.F32.String():   embedF32 + normsF32,
			"native-q8-f32-scales": 2 * q8, // IQ4_XS + Q8_0
			"q3_k":                 q3Packed,
			"q4_k":                 q4Packed,
			"q5_k":                 q5Packed,
			"q6_k":                 q6Packed,
		}},
		{name: "no-q3k-arm", opts: cudaServeCensusOptions()[:4], want: map[string]int64{
			compute.F32.String():   embedF32 + normsF32,
			"native-q8-f32-scales": 3 * q8, // IQ4_XS + Q8_0 + Q3_K
			"q4_k":                 q4Packed,
			"q5_k":                 q5Packed,
			"q6_k":                 q6Packed,
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			censusPlanByDType(t, ws, tc.opts, tc.want)
		})
	}
}

func censusPlanByDType(t *testing.T, ws *WeightSource, opts []Q4KLoadOption, want map[string]int64) {
	t.Helper()
	plan, err := ws.EstimateQ4KLoadMemoryPlan(opts...)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]int64{}
	for _, row := range plan {
		if row.Detail != "gguf-native-q4k-logical-weights" || row.Scope != compute.MemoryScopeDevice || row.Class != compute.MemoryWeights {
			t.Fatalf("unexpected plan row %+v", row)
		}
		got[row.DType] += row.Bytes
	}
	if len(got) != len(want) {
		t.Fatalf("plan dtypes = %v, want %v", got, want)
	}
	for dtype, bytes := range want {
		if got[dtype] != bytes {
			t.Errorf("%s = %d bytes, want %d", dtype, got[dtype], bytes)
		}
	}
}
