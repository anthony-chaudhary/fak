//go:build cuda && cgo

package compute

import (
	"sync/atomic"
	"testing"
	"unsafe"
)

type cudaOwnedReadWrapper struct {
	*cudaBackend
	values []float32
	reads  int
}

func (w *cudaOwnedReadWrapper) Read(Tensor) []float32 {
	w.reads++
	return w.values
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDAOwnedNormRead(t *testing.T) {
	// Exercise canonical Read with its existing checked transport injection.
	// Nonparallel: installation and Read serialize through cudaMu.
	c := &cudaBackend{faultLatch: NewDeviceFaultLatch("cuda", 3)}
	storage := []float32{3, 4}
	tensor := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{2}, be: c,
		buf: &cudaBuf{ptr: unsafe.Pointer(&storage[0]), n: 8, be: c}}
	var destination *float32
	calls, status := 0, 0
	cudaMu.Lock()
	prior := cudaReadNative
	cudaReadNative.copy = func(_ int, dst, _ unsafe.Pointer, _ uintptr) int {
		calls++
		destination = (*float32)(dst)
		copy(unsafe.Slice(destination, 2), storage)
		return status
	}
	cudaMu.Unlock()
	t.Cleanup(func() { cudaMu.Lock(); cudaReadNative = prior; cudaMu.Unlock() })

	values, valid := ReadV41RMSNormOutput(c, tensor, 2)
	if !valid || len(values) != 2 || &values[0] != destination || calls != 1 {
		t.Fatal("canonical checked read was copied again or not read exactly once")
	}
	if atomic.LoadUint64(&c.fenceGen) != 1 {
		t.Fatal("canonical read lost its successful device fence")
	}
	values[0] = 9
	if storage[0] != 3 {
		t.Fatal("device result aliases source")
	}

	host := tensor
	host.buf = &hostBuf{f32: storage}
	values, valid = ReadV41RMSNormOutput(c, host, 2)
	if !valid || &values[0] == &storage[0] || calls != 1 {
		t.Fatal("host alias escaped or host read performed device transport")
	}
	wrapper := &cudaOwnedReadWrapper{cudaBackend: c, values: storage}
	values, valid = ReadV41RMSNormOutput(wrapper, tensor, 2)
	if !valid || wrapper.reads != 1 || &values[0] == &storage[0] || calls != 1 {
		t.Fatal("promoted reader bypassed wrapper Read or lent wrapper storage")
	}

	status = 1007
	values = nil
	var failure any
	func() {
		defer func() { failure = recover() }()
		values, _ = ReadV41RMSNormOutput(c, tensor, 2)
	}()
	if _, ok := failure.(*CUDAOpError); !ok || values != nil || calls != 2 || atomic.LoadUint64(&c.fenceGen) != 1 {
		t.Fatal("failed transport published partial output, retried, or advanced the fence")
	}
}
