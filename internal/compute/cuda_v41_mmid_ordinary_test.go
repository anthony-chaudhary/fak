//go:build cuda && cgo && fak_cuda_v41_routed_mmid

package compute

import (
	"sync/atomic"
	"testing"
	"unsafe"
)

// fak-test:runtime fast est=1ms lane=default
// No CUDA math/transport is simulated: this checks only the closed qualification
// entry and the real result helper's issuance/readiness metadata.
func TestCUDAV41MMIDOrdinaryQualification(t *testing.T) {
	c := &cudaBackend{}
	storage := make([]byte, 12)
	input := makeTensor(c, F32, RowMajor, []int{3}, nil, &cudaBuf{ptr: unsafe.Pointer(&storage[0]), n: len(storage)})
	if out, st, err := c.cudaV41MMIDRoutedSwiGLUDeferred(input, input, 10, 1, false); err == nil || out.buf != nil || st.Record.buf != nil {
		t.Fatal("unqualified ordinary SwiGLU exposed output/status")
	}
	if out, st, err := c.cudaV41MMIDBF16PublishDeferred(input, V41BF16PublishRoutedDown, false); err == nil || out.buf != nil || st.Record.buf != nil {
		t.Fatal("unqualified ordinary BF16 exposed output/status")
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	saved := cudaV41MMIDIssuedStatuses
	cudaV41MMIDIssuedStatuses = make(map[*cudaBuf]cudaV41MMIDIssued)
	defer func() { cudaV41MMIDIssuedStatuses = saved }()
	statusStorage := make([]byte, 16)
	output := input.buf.(*cudaBuf)
	record := &cudaBuf{ptr: unsafe.Pointer(&statusStorage[0]), n: len(statusStorage)}
	atomic.StoreUint64(&c.fenceGen, 7)
	out, st := c.cudaV41MMIDOrdinaryResult(output, record, 3, V41DeferredBF16Publish, V41BF16PublishRoutedDown)
	issued, ok := cudaV41MMIDIssuedStatuses[record]
	if !ok || issued.owner != c || issued.ptr != record.ptr || issued.kind != st.Kind || issued.count != 3 || issued.records != 1 ||
		st.Record.buf != record || st.Site != V41BF16PublishRoutedDown || out.buf != output || out.Ready() || st.Record.Ready() {
		t.Fatal("enqueue metadata was not fresh, owned, and pending completion")
	}
	// This is a metadata observation, not evidence that any CUDA fence ran.
	atomic.AddUint64(&c.fenceGen, 1)
	if !out.Ready() || !st.Record.Ready() {
		t.Fatal("result helper did not preserve the checked-fence generation contract")
	}
}
