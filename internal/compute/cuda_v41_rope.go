//go:build cuda && cgo && fak_cuda_v41_tail_rope

package compute

/*
#include <stdint.h>
#include "cuda_backend.h"

// The separately built native library requires -DFAK_CUDA_V41_TAIL_ROPE.
// These candidate-only declarations do not extend the public fcuda_* ABI.
int fak_v41_tail_rope_state(void);
int fak_v41_tail_rope_input(const uint32_t *dInput, uint64_t count);
int fak_v41_tail_rope_qk(const uint32_t *dQ, const uint32_t *dKV,
    const uint32_t *dTable, uint32_t *dQOut, uint32_t *dKVOut,
    int rows, int headDim, int rotaryDim);
int fak_v41_tail_rope_rows(const uint32_t *dQ, const uint32_t *dTable,
    uint32_t *dQOut, int rows, int headDim, int rotaryDim);
int fak_v41_tail_rope_readback(void *hOutput, const uint32_t *dInput, uint64_t count);
uint64_t fak_v41_tail_rope_counter(int kind);
int fak_v41_tail_rope_test_fault(int stage);
int fak_v41_tail_rope_test_free(int onDevice);
int fak_v41_tail_rope_bf16_state(void);
*/
import "C"

import (
	"errors"
	"math"
	"runtime"
	"sync/atomic"
	"unsafe"
)

var (
	_ V41TailRoPEBackend      = (*cudaBackend)(nil)
	_ V41TailRoPERowsBackend  = (*cudaBackend)(nil)
	_ V41TailRoPEExactBackend = (*cudaBackend)(nil)
)

// Ordinary capability describes the implemented candidate, not session health.
// A selected failure must propagate instead of becoming a CPU-replay decision.
func (c *cudaBackend) SupportsV41TailRoPE() bool       { return c != nil }
func (c *cudaBackend) V41TailRoPERowsAvailable() bool  { return c != nil }
func (c *cudaBackend) V41TailRoPEExactAvailable() bool { return false }

func cudaV41RoPEError(op, stage, message string, code int, cause error) error {
	return &CUDAOpError{Op: op, Site: "v41-tail-rope-" + stage, Msg: message, Code: code, Err: cause}
}

func (c *cudaBackend) cudaV41RoPEStatusError(op, stage, message string, status int) error {
	var cause error
	if status > 0 {
		if fault := c.faultLatch.Observe(DeviceFaultExecution, "v41-tail-rope-"+stage, status); fault != nil {
			cause = fault
		}
	}
	return cudaV41RoPEError(op, stage, message, status, cause)
}

// Caller holds cudaMu. Tensor metadata is checked separately before this storage
// check; native exact-base membership independently verifies the Go capacity.
func (c *cudaBackend) cudaV41RoPEBuffer(t Tensor, bytes int, op, name string) (*cudaBuf, error) {
	b, ok := t.buf.(*cudaBuf)
	if t.Backend() != c || !ok || b == nil || b.ptr == nil || b.n < bytes ||
		b.device != 0 || b.managed || b.scales != nil || b.scalesN != 0 ||
		(b.be != nil && b.be != c) {
		return nil, cudaV41RoPEError(op, "residency", name+" must be live device-only F32 storage owned by this backend", 0, nil)
	}
	if err := b.invalidStateError(op + " " + name); err != nil {
		return nil, cudaV41RoPEError(op, "residency", name+" has invalid CUDA state", 0, err)
	}
	return b, nil
}

// Caller holds cudaMu. The native state check never consumes an unknown previous
// CUDA error. No allocator or kernel is entered while either candidate is closed.
func (c *cudaBackend) cudaV41RoPEAdmit(op string) error {
	if c.capturing {
		return cudaV41RoPEError(op, "state", "synchronous tail RoPE cannot enter capture", 0, nil)
	}
	if err := c.faultLatch.Admit("v41-tail-rope"); err != nil {
		return cudaV41RoPEError(op, "state", "CUDA fault latch refuses tail RoPE", 0, err)
	}
	if status := int(C.fak_v41_tail_rope_state()); status != 0 {
		return c.cudaV41RoPEStatusError(op, "state", "native shared-stream state refuses tail RoPE", status)
	}
	return nil
}

func (c *cudaBackend) V41TailRoPEQK(q, kv, sinCos Tensor, heads, headDim, rotaryDim int) (Tensor, Tensor, error) {
	return c.cudaV41TailRoPE(q, kv, sinCos, heads, headDim, rotaryDim, true)
}

func (c *cudaBackend) V41TailRoPERows(x, sinCos Tensor, rows, headDim, rotaryDim int) (Tensor, error) {
	out, _, err := c.cudaV41TailRoPE(x, Tensor{}, sinCos, rows, headDim, rotaryDim, false)
	return out, err
}

// Both forms use the same native arithmetic. Rows has no synthetic KV read or
// allocation. Outputs never enter the transient recycler and publish only after
// checked completion; all inputs remain immutable.
func (c *cudaBackend) cudaV41TailRoPE(q, kv, sinCos Tensor, rows, headDim, rotaryDim int, paired bool) (qOut, kvOut Tensor, err error) {
	op := "V41TailRoPERows"
	if paired {
		op = "V41TailRoPEQK"
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var qo, ko *cudaBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			cause, _ := ConvertCUDAPanic(recovered, op, "v41-tail-rope")
			err = cudaV41RoPEError(op, "panic", "CUDA operation panicked; outputs withheld", 0, cause)
		}
		if err != nil {
			// Native quarantine guards both free entry points before either can
			// reach pool reuse or the unknown-pointer cudaFree fallback.
			for _, b := range []*cudaBuf{qo, ko} {
				if b != nil && b.ptr != nil {
					C.fcuda_free(b.ptr)
					b.ptr, b.n = nil, 0
				}
			}
			qOut, kvOut = Tensor{}, Tensor{}
		}
	}()
	if c == nil {
		return Tensor{}, Tensor{}, cudaV41RoPEError(op, "preflight", "nil CUDA backend", 0, nil)
	}
	var qBytes, kvBytes, tableBytes int
	if paired {
		qBytes, kvBytes, tableBytes, err = validateV41TailRoPE(q, kv, sinCos, rows, headDim, rotaryDim)
	} else {
		qBytes, tableBytes, err = validateV41TailRoPERows(q, sinCos, rows, headDim, rotaryDim)
	}
	if err != nil {
		return Tensor{}, Tensor{}, cudaV41RoPEError(op, "geometry", "invalid tail-adjacent RoPE geometry", 0, err)
	}
	qb, err := c.cudaV41RoPEBuffer(q, qBytes, op, "Q/rows")
	if err != nil {
		return Tensor{}, Tensor{}, err
	}
	tb, err := c.cudaV41RoPEBuffer(sinCos, tableBytes, op, "sinCos")
	if err != nil {
		return Tensor{}, Tensor{}, err
	}
	var kb *cudaBuf
	if paired {
		kb, err = c.cudaV41RoPEBuffer(kv, kvBytes, op, "KV")
		if err != nil {
			return Tensor{}, Tensor{}, err
		}
	}
	if err := c.cudaV41RoPEAdmit(op); err != nil {
		return Tensor{}, Tensor{}, err
	}
	for _, operand := range []struct {
		buf   *cudaBuf
		bytes int
	}{{qb, qBytes}, {tb, tableBytes}, {kb, kvBytes}} {
		if operand.buf != nil {
			if status := int(C.fak_v41_tail_rope_input((*C.uint32_t)(operand.buf.ptr), C.uint64_t(operand.bytes/4))); status != 0 {
				return Tensor{}, Tensor{}, c.cudaV41RoPEStatusError(op, "residency", "native live allocation or extent check failed", status)
			}
		}
	}
	qo, err = c.dallocDeviceOnlyClass(qBytes, MemoryActivation, "v41-tail-rope-q-output")
	if err != nil {
		cause := errors.Join(err, &DeviceAllocError{Bytes: qBytes, Site: "v41-tail-rope-q-output", Class: MemoryActivation})
		return Tensor{}, Tensor{}, cudaV41RoPEError(op, "allocation", "device-only row output allocation failed", 0, cause)
	}
	var status int
	if paired {
		ko, err = c.dallocDeviceOnlyClass(kvBytes, MemoryActivation, "v41-tail-rope-kv-output")
		if err != nil {
			cause := errors.Join(err, &DeviceAllocError{Bytes: kvBytes, Site: "v41-tail-rope-kv-output", Class: MemoryActivation})
			return Tensor{}, Tensor{}, cudaV41RoPEError(op, "allocation", "device-only KV output allocation failed", 0, cause)
		}
		status = int(C.fak_v41_tail_rope_qk((*C.uint32_t)(qb.ptr), (*C.uint32_t)(kb.ptr),
			(*C.uint32_t)(tb.ptr), (*C.uint32_t)(qo.ptr), (*C.uint32_t)(ko.ptr),
			C.int(rows), C.int(headDim), C.int(rotaryDim)))
	} else {
		status = int(C.fak_v41_tail_rope_rows((*C.uint32_t)(qb.ptr), (*C.uint32_t)(tb.ptr),
			(*C.uint32_t)(qo.ptr), C.int(rows), C.int(headDim), C.int(rotaryDim)))
	}
	if status != 0 {
		return Tensor{}, Tensor{}, c.cudaV41RoPEStatusError(op, "execution", "checked tail RoPE failed; outputs withheld", status)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	qOut = makeTensor(c, F32, RowMajor, []int{rows, headDim}, nil, qo)
	if paired {
		kvOut = makeTensor(c, F32, RowMajor, []int{headDim}, nil, ko)
	}
	return qOut, kvOut, nil
}

// Checked observation only, never an execution fallback. The ordinary Read path
// logs transfer errors; qualification must instead receive checked failure.
func (c *cudaBackend) cudaV41TailRoPEReadback(input Tensor) (values []float32, err error) {
	const op = "V41TailRoPEReadback"
	cudaMu.Lock()
	defer cudaMu.Unlock()
	defer func() {
		if recovered := recover(); recovered != nil {
			cause, _ := ConvertCUDAPanic(recovered, op, "v41-tail-rope-readback")
			err = cudaV41RoPEError(op, "readback", "checked readback panicked", 0, cause)
		}
		if err != nil {
			values = nil
		}
	}()
	if c == nil || input.Dtype != F32 || input.Layout != RowMajor || input.Quant != nil || len(input.Shape) == 0 {
		return nil, cudaV41RoPEError(op, "geometry", "expected nonempty row-major F32 metadata", 0, nil)
	}
	count := uint64(1)
	for _, dim := range input.Shape {
		if dim <= 0 || uint64(dim) > math.MaxUint32/count || uint64(dim) > uint64(math.MaxInt/4)/count {
			return nil, cudaV41RoPEError(op, "geometry", "readback shape exceeds checked count/byte domain", 0, nil)
		}
		count *= uint64(dim)
	}
	b, err := c.cudaV41RoPEBuffer(input, int(count)*4, op, "input")
	if err != nil {
		return nil, err
	}
	if err := c.cudaV41RoPEAdmit(op); err != nil {
		return nil, err
	}
	if status := int(C.fak_v41_tail_rope_input((*C.uint32_t)(b.ptr), C.uint64_t(count))); status != 0 {
		return nil, c.cudaV41RoPEStatusError(op, "residency", "native readback extent check failed", status)
	}
	values = make([]float32, int(count))
	status := int(C.fak_v41_tail_rope_readback(unsafe.Pointer(&values[0]), (*C.uint32_t)(b.ptr), C.uint64_t(count)))
	runtime.KeepAlive(values)
	if status != 0 {
		return nil, c.cudaV41RoPEStatusError(op, "readback", "checked host transfer failed", status)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	return values, nil
}

// Private witness seams. Injection/free probes additionally require the offline
// FAK_CUDA_V41_TAIL_ROPE_TEST define; ordinary candidate binaries refuse them.
func (c *cudaBackend) cudaV41TailRoPESnapshot() [7]uint64 {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var out [7]uint64
	for i := range out {
		out[i] = uint64(C.fak_v41_tail_rope_counter(C.int(i)))
	}
	return out
}

func (c *cudaBackend) cudaV41TailRoPEInject(stage int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if status := int(C.fak_v41_tail_rope_test_fault(C.int(stage))); status != 0 {
		return cudaV41RoPEError("V41TailRoPETest", "injection", "test-only native fault gate refused", status, nil)
	}
	return nil
}

func (c *cudaBackend) cudaV41TailRoPEProbe(q, kv, table Tensor, kind int) int {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	qb, qok := q.buf.(*cudaBuf)
	kb, kok := kv.buf.(*cudaBuf)
	tb, tok := table.buf.(*cudaBuf)
	if !qok || !kok || !tok || qb == nil || kb == nil || tb == nil {
		return -1
	}
	switch kind {
	case 0: // Native extent rejects a forged larger Go capacity.
		return int(C.fak_v41_tail_rope_input((*C.uint32_t)(qb.ptr), C.uint64_t(qb.n/4+1)))
	case 1: // Native exact-base membership rejects an interior pointer.
		return int(C.fak_v41_tail_rope_input((*C.uint32_t)(unsafe.Add(qb.ptr, 4)), 1))
	case 2: // Native output/input alias refusal before enqueue.
		return int(C.fak_v41_tail_rope_qk((*C.uint32_t)(qb.ptr), (*C.uint32_t)(kb.ptr), (*C.uint32_t)(tb.ptr),
			(*C.uint32_t)(qb.ptr), (*C.uint32_t)(kb.ptr), C.int(q.Shape[0]), C.int(q.Shape[1]), C.int(table.Shape[0]*2)))
	case 3, 4: // Closed-state ownership probe, no arbitrary frees permitted.
		return int(C.fak_v41_tail_rope_test_free(C.int(kind - 3)))
	case 5:
		return int(C.fak_v41_tail_rope_bf16_state())
	case 6: // Distinct allocation, but intentionally identical output pointers.
		output, err := c.dallocDeviceOnlyClass(qb.n, MemoryActivation, "v41-tail-rope-alias-probe")
		if err != nil {
			return -1
		}
		defer C.fcuda_free(output.ptr)
		return int(C.fak_v41_tail_rope_qk((*C.uint32_t)(qb.ptr), (*C.uint32_t)(kb.ptr), (*C.uint32_t)(tb.ptr),
			(*C.uint32_t)(output.ptr), (*C.uint32_t)(output.ptr), C.int(q.Shape[0]), C.int(q.Shape[1]), C.int(table.Shape[0]*2)))
	default:
		return -1
	}
}

func (c *cudaBackend) cudaV41TailRoPEAllocationFault(after int) error {
	// Confirm the native witness build before touching the existing allocator
	// injection. That helper acquires cudaMu itself; do not hold it here.
	if err := c.cudaV41TailRoPEInject(0); err != nil {
		return err
	}
	qwen35GDNInjectAllocationFailureForTest(after)
	return nil
}

func (c *cudaBackend) cudaV41TailRoPEBadStorage(input Tensor, kind int) Tensor {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	b, ok := input.buf.(*cudaBuf)
	if !ok || b == nil {
		return Tensor{}
	}
	copy := *b
	switch kind {
	case 0:
		copy.n = 0
	case 1:
		copy.ptr = nil
	case 2:
		copy.managed = true
	case 3:
		copy.device = 1
	case 4:
		copy.be = &cudaBackend{}
	case 5:
		copy.invalid = 1
	case 6:
		copy.scales = copy.ptr
		copy.scalesN = 4
	default:
		return Tensor{}
	}
	input.buf = &copy
	return input
}
