package model

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestKQuantResidentBytesUsesLazyDescriptor(t *testing.T) {
	t.Parallel()
	const payloadBytes = 2 * q6kBlockBytes
	// Header-only accounting must work before any checkpoint reader or raw payload is needed.
	qt := &kQuantTensor{
		out: 2, in: qkK, nblk: 1, kind: kindQ6K,
		lazy: &LazyQ4KRange{Bytes: payloadBytes},
	}
	if got := kQuantResidentBytes(qt); got != int64(payloadBytes) {
		t.Fatalf("lazy K-quant device bytes = %d, want %d", got, payloadBytes)
	}
	if qt.raw != nil {
		t.Fatal("device-byte accounting materialized the lazy host payload")
	}

	// Both activated-set and next-layer hints must decline the raw-only builder.
	w := expertWeight{name: "model.layers.0.mlp.experts.0.gate_proj.weight", kq: qt}
	key, mk, dtype, residentBytes, ok := w.ringStaging()
	if ok || key != "" || mk != nil || dtype != 0 || residentBytes != 0 {
		t.Fatalf("lazy K-quant offered raw-only prefetch staging: key=%q dtype=%v bytes=%d ok=%v", key, dtype, residentBytes, ok)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestLazyQ4KDeclinesRawOnlyPrefetch(t *testing.T) {
	t.Parallel()
	const payloadBytes = 2 * q4kBlockBytes
	qt := &q4kTensor{
		out: 2, in: qkK, nblk: 1,
		lazy: &LazyQ4KRange{Bytes: payloadBytes},
	}
	if got := q4kResidentBytes(qt); got != int64(payloadBytes) {
		t.Fatalf("lazy Q4_K device bytes = %d, want %d", got, payloadBytes)
	}
	w := expertWeight{name: "model.layers.0.mlp.experts.0.gate_proj.weight", q4: qt}
	key, mk, dtype, residentBytes, ok := w.ringStaging()
	if ok || key != "" || mk != nil || dtype != 0 || residentBytes != 0 {
		t.Fatalf("lazy Q4_K offered raw-only prefetch staging: key=%q dtype=%v bytes=%d ok=%v", key, dtype, residentBytes, ok)
	}
	if qt.raw != nil {
		t.Fatal("speculative staging materialized the lazy Q4_K host payload")
	}
}
