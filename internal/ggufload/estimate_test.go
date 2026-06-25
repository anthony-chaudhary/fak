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
	for _, want := range []string{"needs", "device has", "MiB", "FitTooBig"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("FitError message %q missing %q", err.Error(), want)
		}
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
