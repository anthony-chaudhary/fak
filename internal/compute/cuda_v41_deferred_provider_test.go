//go:build cuda && cgo && fak_cuda_v41_deferred_read_qualification && fak_cuda_v41_routed_mmid

package compute

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestCUDAV41DeferredModelProviderReadOnly(t *testing.T) {
	c := &cudaBackend{}
	if _, ok := any(c).(V41DeferredStatusBackend); ok {
		t.Fatal("ordinary CUDA gained deferred methods")
	}
	op, ok := ResolveV41DeferredStatus(c)
	q, separate := op.(*cudaV41DeferredQualification)
	if !ok || !separate || q.owner != c || !q.readOnly {
		t.Fatal("qualification provider lost original owner")
	}
	if _, ok := any(q).(Backend); ok {
		t.Fatal("adapter became a backend")
	}
	if q.SupportsV41DeferredStatus(1) {
		t.Fatal("read-only provider advertised deferred producers")
	}
	if out, st, err := q.V41RoutedSwiGLUDeferred(Tensor{}, Tensor{}, 0, 1); err == nil || out.buf != nil || st.Record.buf != nil {
		t.Fatal("read-only provider issued SwiGLU")
	}
	if out, st, err := q.V41BF16PublishDeferred(Tensor{}, 0); err == nil || out.buf != nil || st.Record.buf != nil {
		t.Fatal("read-only provider issued publication")
	}
}
