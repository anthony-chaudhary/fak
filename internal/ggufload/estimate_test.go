package ggufload

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// estimate_test.go — issue #709 acceptance: the device-fit refusal is reachable at the GGUF
// loader seam, the footprint estimate is callable off the header alone (no tensor read, no
// full load), and the cpu-ref floor is left unchanged. The WeightSource is built from a
// SYNTHETIC File (no .gguf on disk, no weights), so this runs under -short and proves the
// estimate is pure header arithmetic.

// capBackend is a minimal compute.Backend stand-in whose ONLY real behavior is reporting a
// known device capacity — the test double for a cuda/metal backend implementing
// compute.DeviceCapacity. The compute primitives (MatMul/Upload/...) are never invoked on the
// fit-check path (FitsOnDevice touches Caps + DeviceMemory only), so they return zero values.
type capBackend struct {
	total, free int64
	known       bool
}

func (capBackend) Name() string                                            { return "cap-test" }
func (capBackend) Tier() string                                            { return "test" }
func (capBackend) Class() compute.CorrectnessClass                         { return compute.Approx }
func (c capBackend) Caps() compute.Caps                                    { return compute.Caps{CapacityProbe: c.known} }
func (c capBackend) DeviceMemory() (int64, int64, bool)                    { return c.total, c.free, c.known }
func (capBackend) Upload(t compute.Tensor, _ compute.Dtype) compute.Tensor { return t }
func (capBackend) Host(compute.Tensor) ([]float32, bool)                   { return nil, false }
func (capBackend) Read(compute.Tensor) []float32                           { return nil }
func (capBackend) Free(compute.Tensor)                                     {}
func (capBackend) NewKV(compute.KVConfig) compute.KVStore                  { return nil }
func (capBackend) MatMul(_, _ compute.Tensor) compute.Tensor               { return compute.Tensor{} }
func (capBackend) BatchedMatMul(_, _ compute.Tensor, _ int) compute.Tensor { return compute.Tensor{} }
func (capBackend) RMSNorm(_, _ compute.Tensor, _ float32) compute.Tensor   { return compute.Tensor{} }
func (capBackend) RoPE(_ compute.Tensor, _, _, _ int, _ float64) compute.Tensor {
	return compute.Tensor{}
}
func (capBackend) SwiGLU(_, _ compute.Tensor) compute.Tensor { return compute.Tensor{} }
func (capBackend) AddInPlace(_, _ compute.Tensor)            {}
func (capBackend) AddBias(_, _ compute.Tensor)               {}
func (capBackend) Attention(_ compute.Tensor, _ compute.KVStore, _ int, _ bool, _ int, _ float32) compute.Tensor {
	return compute.Tensor{}
}
func (capBackend) Argmax(compute.Tensor) int { return 0 }

var _ compute.DeviceCapacity = capBackend{}

// synthWeightSource builds a WeightSource over a hand-rolled File with two known tensors, so
// the expected EstimateLoadBytes is fixed arithmetic (no weights, no disk). NewWeightSource
// only indexes File.Tensors by name — it never touches the reader — so a nil reader / zero
// size is fine for an estimate-only source.
func synthWeightSource(t *testing.T) *WeightSource {
	t.Helper()
	f := &File{
		Tensors: []TensorInfo{
			{Name: "a", Dims: []uint64{256 * 1024}, Type: TensorF32},  // 262144 elems * 4 = 1 MiB
			{Name: "b", Dims: []uint64{256 * 4096}, Type: TensorQ4_K}, // 1048576 elems / 256 * 144 = 589824 B
		},
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

func TestEstimateLoadBytesIsHeaderArithmetic(t *testing.T) {
	ws := synthWeightSource(t)
	// 1 MiB (1048576) + 589824 = 1638400, with no tensor read and no model built.
	got, err := ws.EstimateLoadBytes()
	if err != nil {
		t.Fatalf("EstimateLoadBytes: %v", err)
	}
	if want := int64(1048576 + 589824); got != want {
		t.Fatalf("EstimateLoadBytes = %d, want %d", got, want)
	}
}

func TestEstimateF32LoadBytesCountsResidentExpansion(t *testing.T) {
	ws := synthWeightSource(t)
	got, err := ws.EstimateF32LoadBytes()
	if err != nil {
		t.Fatalf("EstimateF32LoadBytes: %v", err)
	}
	// The Q4_K tensor expands to f32 under LoadModel: (256*4096 elems)*4 = 4 MiB.
	if want := int64((256*1024)*4 + (256*4096)*4); got != want {
		t.Fatalf("EstimateF32LoadBytes = %d, want %d", got, want)
	}
	plan, err := ws.EstimateF32LoadMemoryPlan()
	if err != nil {
		t.Fatalf("EstimateF32LoadMemoryPlan: %v", err)
	}
	if plan.Total() != got || len(plan) != 1 || plan[0].Class != compute.MemoryWeights {
		t.Fatalf("f32 memory plan = %+v, want one weights demand totaling %d", plan, got)
	}
}

func TestFitOnDeviceRefusesOversizeOnKnownCeiling(t *testing.T) {
	ws := synthWeightSource(t) // needs 1638400 B
	// A capacity-reporting backend whose 1 MiB budget is smaller than the 1.56 MiB model.
	small := capBackend{total: 1 << 20, free: 1 << 20, known: true}
	err := ws.FitOnDevice(small, 0)
	if err == nil {
		t.Fatal("oversize model on a known-small ceiling must be refused, got nil")
	}
	fe, ok := err.(*compute.FitError)
	if !ok {
		t.Fatalf("want *compute.FitError, got %T (%v)", err, err)
	}
	if fe.Verdict != compute.FitTooBig {
		t.Fatalf("Verdict = %s, want FitTooBig", fe.Verdict)
	}
	if fe.Want != 1638400 || fe.Avail != 1<<20 {
		t.Fatalf("FitError sizing: Want=%d Avail=%d, want 1638400 / %d", fe.Want, fe.Avail, 1<<20)
	}
	if len(fe.Demands) != 2 {
		t.Fatalf("FitError demands = %+v, want f32 + q4_k weight demands", fe.Demands)
	}
	byDType := map[string]int64{}
	for _, d := range fe.Demands {
		if d.Class != compute.MemoryWeights || d.ScopeOrDefault() != compute.MemoryScopeDevice {
			t.Fatalf("FitError demand = %+v, want device-scoped weights", d)
		}
		byDType[d.DType] += d.Bytes
	}
	if byDType["f32"] != 1048576 || byDType["q4_k"] != 589824 {
		t.Fatalf("FitError demands by dtype = %+v, want f32=1048576 q4_k=589824", byDType)
	}
	for _, want := range []string{"needs", "device has", "MiB", "FitTooBig"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("FitError message %q missing %q", err.Error(), want)
		}
	}
}

func TestFitF32OnDeviceUsesExpandedResidentFootprint(t *testing.T) {
	ws := synthWeightSource(t)
	// The raw/lean estimate (1.56 MiB) would fit; the f32-resident estimate (5 MiB) must not.
	small := capBackend{total: 2 << 20, free: 2 << 20, known: true}
	if err := ws.FitOnDevice(small, 0); err != nil {
		t.Fatalf("raw/lean fit should pass on 2 MiB device, got %v", err)
	}
	err := ws.FitF32OnDevice(small, 0)
	if err == nil {
		t.Fatal("f32-resident fit should refuse on 2 MiB device")
	}
	fe, ok := err.(*compute.FitError)
	if !ok {
		t.Fatalf("want *compute.FitError, got %T (%v)", err, err)
	}
	if fe.Want != 5<<20 {
		t.Fatalf("FitF32OnDevice Want=%d, want %d", fe.Want, 5<<20)
	}
	if len(fe.Demands) != 1 || fe.Demands[0].Detail != "gguf-f32-load" {
		t.Fatalf("FitF32OnDevice demands = %+v", fe.Demands)
	}
}

func TestEstimateCPUOffloadExpertsMemoryPlanSplitsDeviceAndHost(t *testing.T) {
	f := &File{
		Metadata: map[string]Value{
			"general.architecture": {Type: TypeString, Value: "glm-dsa"},
		},
		Tensors: []TensorInfo{
			{Name: "token_embd.weight", Dims: []uint64{256}, Type: TensorF32},               // device, 1024 B
			{Name: "blk.0.ffn_gate_inp.weight", Dims: []uint64{128}, Type: TensorF32},       // router device, 512 B
			{Name: "blk.0.attn_k_b.weight", Dims: []uint64{64}, Type: TensorF32},            // KV-b half device, 256 B
			{Name: "blk.0.ffn_gate_shexp.weight", Dims: []uint64{512}, Type: TensorF32},     // shared expert host, 2048 B
			{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{1024}, Type: TensorF32},     // routed expert blob host, 4096 B
			{Name: "blk.78.nextn.eh_proj.weight", Dims: []uint64{1 << 20}, Type: TensorF32}, // skipped, counts nowhere
		},
	}
	ws, err := NewWeightSource(f, nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	plan, err := ws.EstimateCPUOffloadExpertsMemoryPlan()
	if err != nil {
		t.Fatalf("EstimateCPUOffloadExpertsMemoryPlan: %v", err)
	}
	by := plan.ByClass()
	if got, want := by[compute.MemoryWeights], int64(1024+512+256); got != want {
		t.Fatalf("device weights = %d, want %d", got, want)
	}
	if got, want := by[compute.MemoryOffload], int64(2048+4096); got != want {
		t.Fatalf("host offload = %d, want %d", got, want)
	}
	if got, want := plan.DeviceTotal(), int64(1024+512+256); got != want {
		t.Fatalf("DeviceTotal = %d, want dense/device side only %d", got, want)
	}
	if got, want := plan.Total(), int64(1024+512+256+2048+4096); got != want {
		t.Fatalf("Total = %d, want all bytes %d", got, want)
	}
	fitsDense := capBackend{total: 2 << 10, free: 2 << 10, known: true}
	if err := ws.FitCPUOffloadExpertsOnDevice(fitsDense, 0); err != nil {
		t.Fatalf("fit should ignore host expert bytes and accept dense side: %v", err)
	}
	tooSmallForDense := capBackend{total: 1 << 10, free: 1 << 10, known: true}
	err = ws.FitCPUOffloadExpertsOnDevice(tooSmallForDense, 0)
	if err == nil {
		t.Fatal("dense side over device capacity must be refused")
	}
	fe, ok := err.(*compute.FitError)
	if !ok {
		t.Fatalf("want *compute.FitError, got %T (%v)", err, err)
	}
	if fe.Want != 1024+512+256 {
		t.Fatalf("FitError Want = %d, want device-only dense side", fe.Want)
	}
	if len(fe.Demands) != 2 || fe.Demands[1].Class != compute.MemoryOffload || fe.Demands[1].Scope != compute.MemoryScopeHost {
		t.Fatalf("FitError demands = %+v, want host-scoped offload demand preserved", fe.Demands)
	}
}

func TestFitOnDeviceFailsOpenOnUnknownCapacity(t *testing.T) {
	ws := synthWeightSource(t)
	// cpu-ref-equivalent: a backend that cannot probe must NEVER be refused (acceptance #2 —
	// the load path is unchanged for unknown capacity), and neither must a nil backend.
	for _, be := range []compute.Backend{
		capBackend{known: false}, // reports capacity but unknown
		nil,                      // no backend at all
	} {
		if err := ws.FitOnDevice(be, 0); err != nil {
			t.Fatalf("unknown/nil capacity must fail open (nil error), got %v", err)
		}
	}
	// Sanity: the same model that was refused at 1 MiB is ACCEPTED on a 2 MiB device.
	big := capBackend{total: 2 << 20, free: 2 << 20, known: true}
	if err := ws.FitOnDevice(big, 0); err != nil {
		t.Fatalf("model that fits a known-big device must not be refused, got %v", err)
	}
}
