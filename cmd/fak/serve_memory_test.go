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
	uploadDtype         bool
}

func (b serveCapBackend) Caps() compute.Caps {
	return compute.Caps{CapacityProbe: true, HostCapacityProbe: b.hostKnown, UploadDtype: b.uploadDtype}
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
			"glm-dsa.expert_count":                     {Type: ggufload.TypeUint64, Value: uint64(4)},
			"glm-dsa.expert_used_count":                {Type: ggufload.TypeUint64, Value: uint64(2)},
			"glm-dsa.expert_feed_forward_length":       {Type: ggufload.TypeUint64, Value: uint64(64)},
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
	plan, err := serveGGUFMemoryPlan(ws, false, 8, serveFitBudget{})
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
	plan, err := serveGGUFMemoryPlan(ws, false, 8, serveFitBudget{})
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
	plan, err := serveGGUFCPUOffloadMemoryPlan(ws, 8, serveFitBudget{})
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

func TestServeGGUFExpertParallelMemoryPlanChargesPerRankExpertBand(t *testing.T) {
	ws := serveSynthOffloadWeightSource(t)
	plan, err := serveGGUFExpertParallelMemoryPlan(ws, 4, 8, serveFitBudget{})
	if err != nil {
		t.Fatalf("serveGGUFExpertParallelMemoryPlan: %v", err)
	}
	by := plan.ByClass()
	// EP replicates dense/router/attention/shared-expert tensors and shards only the routed
	// ffn_*_exps blobs. The synthetic routed blob is 4096 B across four experts, so EP-4 charges
	// one 1024 B expert band instead of the full 4096 B blob to each rank.
	if got, want := by[compute.MemoryWeights], int64(1024+512+256+2048+1024); got != want {
		t.Fatalf("EP per-rank weights = %d, want replicated weights + one expert band %d", got, want)
	}
	byDetail := serveMemoryBytesByDetail(plan)
	if got, want := byDetail["gguf-ep-replicated-load"], int64(1024+512+256+2048); got != want {
		t.Fatalf("EP replicated weight detail = %d, want %d; plan=%+v", got, want, plan)
	}
	if got, want := byDetail["gguf-ep-routed-expert-shard"], int64(1024); got != want {
		t.Fatalf("EP routed shard detail = %d, want %d; plan=%+v", got, want, plan)
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
	if got := by[compute.MemoryOffload]; got != 0 {
		t.Fatalf("EP resident plan must not classify routed experts as host offload, got %d", got)
	}
	const deviceWant = int64(1024 + 512 + 256 + 2048 + 1024 + 3072 + 128 + 3584)
	if got := plan.DeviceTotal(); got != deviceWant {
		t.Fatalf("DeviceTotal = %d, want per-rank weights + KV + HAL transient %d", got, deviceWant)
	}
	fitsPerRank := serveCapBackend{Backend: compute.Default(), total: 14 << 10, free: 14 << 10, known: true}
	if err := fitServeGGUFExpertParallelOnDevice(ws, fitsPerRank, 4, 8); err != nil {
		t.Fatalf("EP per-rank plan should fit a backend sized for one expert band: %v", err)
	}
	tooSmallForPerRank := serveCapBackend{Backend: compute.Default(), total: 12 << 10, free: 12 << 10, known: true}
	err = fitServeGGUFExpertParallelOnDevice(ws, tooSmallForPerRank, 4, 8)
	if err == nil {
		t.Fatal("EP per-rank plan over device capacity must be refused")
	}
	var fe *compute.FitError
	if !errors.As(err, &fe) {
		t.Fatalf("want *compute.FitError, got %T (%v)", err, err)
	}
	if fe.Want != deviceWant {
		t.Fatalf("FitError Want = %d, want per-rank device side", fe.Want)
	}
}

func serveMemoryBytesByDetail(plan compute.MemoryPlan) map[string]int64 {
	out := map[string]int64{}
	for _, d := range plan {
		out[d.Detail] += d.Bytes
	}
	return out
}

// #949: a standard-arch device serve with FAK_Q4K must hold raw Q4_K weights RESIDENT ON THE
// DEVICE — the same non-offload, quant-aware plan the Q8 arm uses, but charged at raw Q4_K
// density (~0.56 B/param) instead of Q8 (~1 B/param). The bug was the dispatch ordering, which
// is exercised end-to-end only on a GPU; here we pin the plan-layer invariant that the standard-
// arch Q4_K serve routes to the device-RESIDENT plan (no host expert offload), which is what the
// new arm selects via fitAndPlanServeGGUFPathOnDevice(..., f32Resident=false, ...).
func TestServeQ4KStandardArchUsesDeviceResidentNotOffloadPlan(t *testing.T) {
	ws := serveSynthConfiguredWeightSource(t) // a standard llama-arch (qwen2) source, Q4_K weight tensor
	plan, err := serveGGUFMemoryPlan(ws, false, 8, serveFitBudget{})
	if err != nil {
		t.Fatalf("serveGGUFMemoryPlan: %v", err)
	}
	by := plan.ByClass()
	// Raw Q4_K density: tensor "b" (256*4096 elems) charged 589824 bytes (~0.5625 B/elem), plus
	// the F32 tensor "a" (1048576). This is the resident weight total the new arm fits — NOT the
	// Q8 ~1 B/param total, and NOT split into an offload class.
	if got, want := by[compute.MemoryWeights], int64(1048576+589824); got != want {
		t.Fatalf("device-resident weights = %d, want raw Q4_K density %d", got, want)
	}
	if got := by[compute.MemoryOffload]; got != 0 {
		t.Fatalf("standard-arch no-offload Q4_K serve must charge 0 host-offload bytes, got %d", got)
	}
	// The resident plan must fit a device sized for the Q4_K-resident total (the VRAM win that
	// #949 unlocks): build a backend just large enough for the device side and assert it fits.
	deviceWant := plan.DeviceTotal()
	uploadFits := serveCapBackend{Backend: compute.Default(), total: deviceWant + deviceWant/4, free: compute.FreeUnknown, known: true, uploadDtype: true}
	if err := fitServeGGUFOnDevice(ws, uploadFits, false, 8); err != nil {
		t.Fatalf("device-resident Q4_K plan should fit a backend sized for it: %v", err)
	}
	// And the backend the new arm requires advertises quantized upload — the guard that keeps a
	// non-quant-upload backend on the f32/Q8 fallback arms unchanged.
	if !uploadFits.Caps().UploadDtype {
		t.Fatal("test backend must advertise UploadDtype to reach the resident Q4_K arm")
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

// serveSynthWideWindowWeightSource is a configured source whose declared context window (4096) is
// far larger than the tiny box the #1046 auto-sizer tests size it against, so the sizer must
// derive a context BELOW the full window.
func serveSynthWideWindowWeightSource(t *testing.T) *ggufload.WeightSource {
	t.Helper()
	f := &ggufload.File{
		Metadata: map[string]ggufload.Value{
			"general.architecture":                   {Type: ggufload.TypeString, Value: "qwen2"},
			"qwen2.context_length":                   {Type: ggufload.TypeUint64, Value: uint64(4096)},
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

// #1046: with no explicit --context-budget-tokens AND a known memory ceiling, the serve boot plan
// auto-sizes the KV cache to the largest context that fits the box — strictly below the model's
// full declared window — instead of sizing against MaxPositionEmbeddings (which would overflow and
// refuse). The derived plan must also fit the backend (no FitError).
func TestServeGGUFMemoryPlanAutoSizesContextToFitWhenNoBudget(t *testing.T) {
	ws := serveSynthWideWindowWeightSource(t)
	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	csc := cfg.ContextSizeConfig()
	fullWindowKV := compute.EstimateKVStoreBytes(csc.KV, csc.MaxContext)

	// A device just large enough for the dense weights plus a partial KV window: the full 4096-token
	// window would not fit after headroom, so the sizer must shrink the context.
	be := serveCapBackend{Backend: compute.Default(), total: 3 << 20, free: 3 << 20, known: true}
	plan, err := serveGGUFMemoryPlan(ws, false, 0 /* no budget -> auto-size */, serveDeviceFitBudget(be))
	if err != nil {
		t.Fatalf("serveGGUFMemoryPlan: %v", err)
	}
	kv := plan.ByClass()[compute.MemoryKVCache]
	if kv <= 0 {
		t.Fatalf("auto-sized plan must still carry a KV demand, got %d", kv)
	}
	if kv >= fullWindowKV {
		t.Fatalf("auto-sized KV = %d, must be below the full-window KV %d (sized down, not full window)", kv, fullWindowKV)
	}
	if err := fitServeGGUFOnDevice(ws, be, false, 0); err != nil {
		t.Fatalf("the auto-sized plan must fit the box it was sized against, got %v", err)
	}
}

// #1046: auto-sizing is NOT an escape hatch around the fit check. A box too small to hold even the
// weights still refuses with the typed FitTooBig when no budget is set — the sizer only lowers the
// context the check runs at; it never lets an oversized model load.
func TestFitServeGGUFOnDeviceAutoSizeStillRefusesWeightsOverflow(t *testing.T) {
	ws := serveSynthConfiguredWeightSource(t) // ~1.56 MiB device weights
	tooSmall := serveCapBackend{Backend: compute.Default(), total: 1 << 20, free: 1 << 20, known: true}
	err := fitServeGGUFOnDevice(ws, tooSmall, false, 0 /* no budget -> auto-size */)
	if err == nil {
		t.Fatal("auto-sized boot fit must still refuse a box too small for the weights alone")
	}
	var fe *compute.FitError
	if !errors.As(err, &fe) {
		t.Fatalf("want *compute.FitError, got %T (%v)", err, err)
	}
}
