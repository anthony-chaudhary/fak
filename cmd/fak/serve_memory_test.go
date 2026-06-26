package main

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

type serveCapBackend struct {
	compute.Backend
	total, free         int64
	hostTotal, hostFree int64
	known, hostKnown    bool
}

func (b serveCapBackend) Caps() compute.Caps {
	return compute.Caps{CapacityProbe: true, HostCapacityProbe: b.hostKnown}
}
func (b serveCapBackend) DeviceMemory() (int64, int64, bool) { return b.total, b.free, b.known }
func (b serveCapBackend) HostMemory() (int64, int64, bool) {
	return b.hostTotal, b.hostFree, b.hostKnown
}

func serveSynthWeightSource(t *testing.T) *ggufload.WeightSource {
	t.Helper()
	f := &ggufload.File{
		Tensors: []ggufload.TensorInfo{
			{Name: "a", Dims: []uint64{256 * 1024}, Type: ggufload.TensorF32},
			{Name: "b", Dims: []uint64{256 * 4096}, Type: ggufload.TensorQ4_K},
		},
	}
	ws, err := ggufload.NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

func serveSynthConfiguredWeightSource(t *testing.T) *ggufload.WeightSource {
	t.Helper()
	f := &ggufload.File{
		Metadata: map[string]ggufload.Value{
			"general.architecture":                   {Type: ggufload.TypeString, Value: "qwen2"},
			"qwen2.context_length":                   {Type: ggufload.TypeUint64, Value: uint64(16)},
			"qwen2.embedding_length":                 {Type: ggufload.TypeUint64, Value: uint64(32)},
			"qwen2.block_count":                      {Type: ggufload.TypeUint64, Value: uint64(2)},
			"qwen2.feed_forward_length":              {Type: ggufload.TypeUint64, Value: uint64(64)},
			"qwen2.attention.head_count":             {Type: ggufload.TypeUint64, Value: uint64(4)},
			"qwen2.attention.head_count_kv":          {Type: ggufload.TypeUint64, Value: uint64(2)},
			"qwen2.attention.layer_norm_rms_epsilon": {Type: ggufload.TypeFloat32, Value: float32(1e-5)},
			"qwen2.rope.freq_base":                   {Type: ggufload.TypeFloat32, Value: float32(10000)},
			"tokenizer.ggml.eos_token_id":            {Type: ggufload.TypeUint32, Value: uint32(2)},
		},
		Tensors: []ggufload.TensorInfo{
			{Name: "a", Dims: []uint64{256 * 1024}, Type: ggufload.TensorF32},
			{Name: "b", Dims: []uint64{256 * 4096}, Type: ggufload.TensorQ4_K},
		},
	}
	ws, err := ggufload.NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

func serveSynthOffloadWeightSource(t *testing.T) *ggufload.WeightSource {
	t.Helper()
	f := &ggufload.File{
		Metadata: map[string]ggufload.Value{
			"general.architecture":                     {Type: ggufload.TypeString, Value: "glm-dsa"},
			"glm-dsa.context_length":                   {Type: ggufload.TypeUint64, Value: uint64(16)},
			"glm-dsa.embedding_length":                 {Type: ggufload.TypeUint64, Value: uint64(32)},
			"glm-dsa.block_count":                      {Type: ggufload.TypeUint64, Value: uint64(2)},
			"glm-dsa.feed_forward_length":              {Type: ggufload.TypeUint64, Value: uint64(64)},
			"glm-dsa.attention.head_count":             {Type: ggufload.TypeUint64, Value: uint64(4)},
			"glm-dsa.attention.head_count_kv":          {Type: ggufload.TypeUint64, Value: uint64(2)},
			"glm-dsa.attention.layer_norm_rms_epsilon": {Type: ggufload.TypeFloat32, Value: float32(1e-5)},
			"glm-dsa.rope.freq_base":                   {Type: ggufload.TypeFloat32, Value: float32(10000)},
			"tokenizer.ggml.eos_token_id":              {Type: ggufload.TypeUint32, Value: uint32(2)},
		},
		Tensors: []ggufload.TensorInfo{
			{Name: "token_embd.weight", Dims: []uint64{256}, Type: ggufload.TensorF32},
			{Name: "blk.0.ffn_gate_inp.weight", Dims: []uint64{128}, Type: ggufload.TensorF32},
			{Name: "blk.0.attn_k_b.weight", Dims: []uint64{64}, Type: ggufload.TensorF32},
			{Name: "blk.0.ffn_gate_shexp.weight", Dims: []uint64{512}, Type: ggufload.TensorF32},
			{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{1024}, Type: ggufload.TensorF32},
		},
	}
	ws, err := ggufload.NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

func TestFitServeGGUFOnDeviceUsesF32ResidentPlan(t *testing.T) {
	ws := serveSynthWeightSource(t)
	be := serveCapBackend{Backend: compute.Default(), total: 5 << 20, free: 5 << 20, known: true}

	err := fitServeGGUFOnDevice(ws, be, true, 0)
	if err == nil {
		t.Fatal("f32-resident device load should be refused once serve headroom is reserved")
	}
	var fe *compute.FitError
	if !errors.As(err, &fe) {
		t.Fatalf("want *compute.FitError, got %T (%v)", err, err)
	}
	if fe.Want != 5<<20 {
		t.Fatalf("fit plan Want=%d, want 5 MiB f32-resident estimate", fe.Want)
	}
	if len(fe.Demands) != 1 || fe.Demands[0].Class != compute.MemoryWeights || fe.Demands[0].Detail != "gguf-f32-load" {
		t.Fatalf("FitError demands = %+v, want classed gguf-f32-load weights demand", fe.Demands)
	}
}

func TestServeGGUFMemoryPlanIncludesQ8WeightsAndKVCache(t *testing.T) {
	ws := serveSynthConfiguredWeightSource(t)
	plan, err := serveGGUFMemoryPlan(ws, false, 8)
	if err != nil {
		t.Fatalf("serveGGUFMemoryPlan: %v", err)
	}
	by := plan.ByClass()
	if got, want := by[compute.MemoryWeights], int64(1048576+589824); got != want {
		t.Fatalf("weights demand = %d, want raw/lean Q8 proxy %d", got, want)
	}
	// context budget override: 2 layers * 8 positions * 2 kv heads * 8 dims *
	// 3 rows (Kraw,K,V) * 4-byte f32.
	if got, want := by[compute.MemoryKVCache], int64(3072); got != want {
		t.Fatalf("kv_cache demand = %d, want %d", got, want)
	}
	if got, want := by[compute.MemoryActivation], int64(128); got != want {
		t.Fatalf("activation demand = %d, want %d", got, want)
	}
	if got, want := by[compute.MemoryScratchpad], int64(3584); got != want {
		t.Fatalf("scratchpad demand = %d, want %d", got, want)
	}
}

func TestGatewayLoadProfileCarriesServeMemoryPlanAndCapacity(t *testing.T) {
	ws := serveSynthConfiguredWeightSource(t)
	plan, err := serveGGUFMemoryPlan(ws, false, 8)
	if err != nil {
		t.Fatalf("serveGGUFMemoryPlan: %v", err)
	}
	be := serveCapBackend{
		Backend:   compute.Default(),
		total:     8 << 30,
		free:      compute.FreeUnknown,
		known:     true,
		hostTotal: 64 << 30,
		hostFree:  48 << 30,
		hostKnown: true,
	}
	profile := withServeGGUFMemoryProfile(toGatewayLoadProfile(&ggufload.LoadProfile{
		Mode:        "gguf-lean-q8-device",
		Source:      "qwen.gguf",
		TotalNanos:  2_000_000_000,
		TensorCount: 2,
		Phases: []ggufload.LoadPhaseStat{
			{Phase: "quantize", Nanos: 2_000_000_000, Bytes: 1 << 20, Tensors: 2},
		},
		Bottleneck: "quantize",
	}), plan, be)

	if profile.MemoryHeadroomRatio != serveGGUFDeviceHeadroom {
		t.Fatalf("MemoryHeadroomRatio = %v, want %v", profile.MemoryHeadroomRatio, serveGGUFDeviceHeadroom)
	}
	var weights, kv int64
	seenDType := map[string]bool{}
	for _, row := range profile.MemoryPlan {
		if row.DType != "" {
			seenDType[row.Class+"/"+row.Scope+"/"+row.DType] = true
		}
		switch row.Class {
		case string(compute.MemoryWeights):
			weights += row.Bytes
			if row.Scope != string(compute.MemoryScopeDevice) {
				t.Fatalf("weights scope = %q, want device", row.Scope)
			}
		case string(compute.MemoryKVCache):
			kv += row.Bytes
		}
	}
	if weights == 0 || kv == 0 {
		t.Fatalf("profile memory plan missing weights/kv rows: %+v", profile.MemoryPlan)
	}
	if !seenDType[string(compute.MemoryKVCache)+"/"+string(compute.MemoryScopeDevice)+"/f32"] {
		t.Fatalf("profile memory plan missing f32 KV dtype row: %+v", profile.MemoryPlan)
	}
	if !seenDType[string(compute.MemoryWeights)+"/"+string(compute.MemoryScopeDevice)+"/f32"] && !seenDType[string(compute.MemoryWeights)+"/"+string(compute.MemoryScopeDevice)+"/q4_k"] && !seenDType[string(compute.MemoryWeights)+"/"+string(compute.MemoryScopeDevice)+"/q8_0"] {
		t.Fatalf("profile memory plan missing concrete weight dtype row: %+v", profile.MemoryPlan)
	}
	if len(profile.MemoryCapacities) != 2 {
		t.Fatalf("MemoryCapacities = %+v, want device+host", profile.MemoryCapacities)
	}
	if cap := profile.MemoryCapacities[0]; cap.Scope != "device" || !cap.Known || cap.FreeKnown {
		t.Fatalf("device capacity = %+v, want known total with free unknown", cap)
	}
	if cap := profile.MemoryCapacities[1]; cap.Scope != "host" || !cap.Known || !cap.FreeKnown || cap.FreeBytes != 48<<30 {
		t.Fatalf("host capacity = %+v, want known total/free", cap)
	}
}

func TestServeGGUFCPUOffloadMemoryPlanKeepsExpertsHostScoped(t *testing.T) {
	ws := serveSynthOffloadWeightSource(t)
	plan, err := serveGGUFCPUOffloadMemoryPlan(ws, 8)
	if err != nil {
		t.Fatalf("serveGGUFCPUOffloadMemoryPlan: %v", err)
	}
	by := plan.ByClass()
	if got, want := by[compute.MemoryWeights], int64(1024+512+256); got != want {
		t.Fatalf("device weights = %d, want %d", got, want)
	}
	if got, want := by[compute.MemoryOffload], int64(2048+4096); got != want {
		t.Fatalf("host offload = %d, want %d", got, want)
	}
	if got, want := by[compute.MemoryKVCache], int64(3072); got != want {
		t.Fatalf("kv_cache = %d, want %d", got, want)
	}
	if got, want := by[compute.MemoryActivation], int64(128); got != want {
		t.Fatalf("activation = %d, want %d", got, want)
	}
	if got, want := by[compute.MemoryScratchpad], int64(3584); got != want {
		t.Fatalf("scratchpad = %d, want %d", got, want)
	}
	const deviceWant = int64(1024 + 512 + 256 + 3072 + 128 + 3584)
	if got := plan.DeviceTotal(); got != deviceWant {
		t.Fatalf("DeviceTotal = %d, want dense weights + KV + HAL transient %d", got, deviceWant)
	}
	fitsDeviceSide := serveCapBackend{Backend: compute.Default(), total: 12 << 10, free: 12 << 10, known: true}
	if err := fitServeGGUFCPUOffloadOnDevice(ws, fitsDeviceSide, 8); err != nil {
		t.Fatalf("fit should ignore host expert bytes and accept dense+KV+HAL transient side: %v", err)
	}
	tooSmallForDenseAndKV := serveCapBackend{Backend: compute.Default(), total: 9 << 10, free: 9 << 10, known: true}
	err = fitServeGGUFCPUOffloadOnDevice(ws, tooSmallForDenseAndKV, 8)
	if err == nil {
		t.Fatal("dense+KV+HAL transient side over device capacity must be refused")
	}
	var fe *compute.FitError
	if !errors.As(err, &fe) {
		t.Fatalf("want *compute.FitError, got %T (%v)", err, err)
	}
	if fe.Want != deviceWant {
		t.Fatalf("FitError Want = %d, want device-only dense+KV+HAL transient side", fe.Want)
	}
}

func TestFitServeGGUFOnDeviceRawPlanAndUnknownCapacityFailOpen(t *testing.T) {
	ws := serveSynthWeightSource(t)
	fitsRaw := serveCapBackend{Backend: compute.Default(), total: 2 << 20, free: 2 << 20, known: true}
	if err := fitServeGGUFOnDevice(ws, fitsRaw, false, 0); err != nil {
		t.Fatalf("raw/lean GGUF plan should fit the 2 MiB test backend, got %v", err)
	}
	unknown := serveCapBackend{Backend: compute.Default(), known: false}
	if err := fitServeGGUFOnDevice(ws, unknown, true, 0); err != nil {
		t.Fatalf("unknown-capacity backend must fail open, got %v", err)
	}
	if err := fitServeGGUFOnDevice(ws, nil, true, 0); err != nil {
		t.Fatalf("nil backend must be a no-op, got %v", err)
	}
}
