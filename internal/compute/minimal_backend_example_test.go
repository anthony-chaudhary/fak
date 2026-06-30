package compute_test

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// minimalVendorBackend is the worked example linked from the vendor onboarding doc:
// a backend can take ownership of two hot ops while keeping every other method visible
// and delegated to the reference floor until the device implementation is ready.
type minimalVendorBackend struct {
	fallback       compute.Backend
	matmulCalls    int
	attentionCalls int
}

func (b *minimalVendorBackend) Name() string                    { return "vendor-minimal-example" }
func (b *minimalVendorBackend) Tier() string                    { return "example" }
func (b *minimalVendorBackend) Class() compute.CorrectnessClass { return compute.Approx }
func (b *minimalVendorBackend) Caps() compute.Caps              { return compute.Caps{} }
func (b *minimalVendorBackend) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	return b.fallback.Upload(t, as)
}
func (b *minimalVendorBackend) Host(t compute.Tensor) ([]float32, bool) { return b.fallback.Host(t) }
func (b *minimalVendorBackend) Read(t compute.Tensor) []float32         { return b.fallback.Read(t) }
func (b *minimalVendorBackend) Free(t compute.Tensor)                   { b.fallback.Free(t) }
func (b *minimalVendorBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	return b.fallback.NewKV(cfg)
}
func (b *minimalVendorBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmulCalls++
	return b.fallback.MatMul(w, x)
}
func (b *minimalVendorBackend) BatchedMatMul(w, X compute.Tensor, P int) compute.Tensor {
	return b.fallback.BatchedMatMul(w, X, P)
}
func (b *minimalVendorBackend) RMSNorm(x, weight compute.Tensor, eps float32) compute.Tensor {
	return b.fallback.RMSNorm(x, weight, eps)
}
func (b *minimalVendorBackend) RoPE(x compute.Tensor, pos, nHeads, headDim int, theta float64) compute.Tensor {
	return b.fallback.RoPE(x, pos, nHeads, headDim, theta)
}
func (b *minimalVendorBackend) SwiGLU(gate, up compute.Tensor) compute.Tensor {
	return b.fallback.SwiGLU(gate, up)
}
func (b *minimalVendorBackend) AddInPlace(dst, src compute.Tensor) { b.fallback.AddInPlace(dst, src) }
func (b *minimalVendorBackend) AddBias(dst, bias compute.Tensor)   { b.fallback.AddBias(dst, bias) }
func (b *minimalVendorBackend) Attention(q compute.Tensor, kv compute.KVStore, layer int, causal bool, grp int, scale float32) compute.Tensor {
	b.attentionCalls++
	return b.fallback.Attention(q, kv, layer, causal, grp, scale)
}
func (b *minimalVendorBackend) Argmax(logits compute.Tensor) int { return b.fallback.Argmax(logits) }

func TestMinimalVendorBackendExampleCompilesAndRuns(t *testing.T) {
	floor := compute.Default()
	if floor == nil {
		t.Fatal("compute.Default returned nil")
	}
	be := &minimalVendorBackend{fallback: floor}
	if be.Class() != compute.Approx {
		t.Fatalf("minimal backend class = %v, want Approx", be.Class())
	}
	if got := be.Caps(); got != (compute.Caps{}) {
		t.Fatalf("minimal backend advertised caps = %+v, want none", got)
	}

	w := be.Upload(compute.NewF32(floor, []int{2, 2}, []float32{1, 2, 3, 4}), compute.F32)
	x := be.Upload(compute.NewF32(floor, []int{2}, []float32{5, 6}), compute.F32)
	if got, want := be.Read(be.MatMul(w, x)), []float32{17, 39}; !reflect.DeepEqual(got, want) {
		t.Fatalf("MatMul = %v, want %v", got, want)
	}

	kv := be.NewKV(compute.KVConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 2})
	kRaw := be.Upload(compute.NewF32(floor, []int{2}, []float32{1, 0}), compute.F32)
	kRoPE := be.Upload(compute.NewF32(floor, []int{2}, []float32{1, 0}), compute.F32)
	v := be.Upload(compute.NewF32(floor, []int{2}, []float32{0.25, 0.75}), compute.F32)
	kv.AppendKV(0, kRaw, kRoPE, v, 0)

	q := be.Upload(compute.NewF32(floor, []int{2}, []float32{1, 0}), compute.F32)
	if got, want := be.Read(be.Attention(q, kv, 0, true, 1, 1)), []float32{0.25, 0.75}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Attention = %v, want %v", got, want)
	}
	if be.matmulCalls != 1 || be.attentionCalls != 1 {
		t.Fatalf("hot op calls MatMul=%d Attention=%d, want 1/1", be.matmulCalls, be.attentionCalls)
	}
}
