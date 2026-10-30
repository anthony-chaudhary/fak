//go:build cuda && cgo && cudaarchspecific && fak_cuda_flashmla_prefill

package compute

/*
#include "cuda_flashmla_sparse_prefill.h"
*/
import "C"

import (
	"runtime"
	"unsafe"
)

// Availability is a physical-acceptance boundary, not a build-success flag.
func (c *cudaBackend) SupportsV41SparsePrefill() bool { return false }
func (c *cudaBackend) V41SparsePrefill(x V41SparsePrefillInput) (V41SparsePrefillOutput, error) {
	return c.cudaV41SparsePrefill(x, false)
}

// physical is accessible only to package-local opt-in acceptance witnesses.
// The selected call never retries through a host implementation.
func (c *cudaBackend) cudaV41SparsePrefill(x V41SparsePrefillInput, physical bool) (V41SparsePrefillOutput, error) {
	var zero V41SparsePrefillOutput
	if err := validateV41SparsePrefill(x); err != nil {
		return zero, err
	}
	if c == nil || !physical {
		return zero, ErrV41SparsePrefillUnavailable
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if c.capturing {
		return zero, &CUDAOpError{Op: "V41SparsePrefill", Site: "capture"}
	}
	if err := c.faultLatch.Admit("flashmla-prefill"); err != nil {
		return zero, err
	}
	out := V41SparsePrefillOutput{Values: make([]float32, x.Queries*32768), MaxLogits: make([]float32, x.Queries*64), LSE: make([]float32, x.Queries*64)}
	code := int(C.fak_flashmla_prefill_host(
		(*C.uint16_t)(unsafe.Pointer(&x.Q[0])), (*C.uint16_t)(unsafe.Pointer(&x.KV[0])),
		(*C.int32_t)(unsafe.Pointer(&x.Indices[0])), (*C.int32_t)(unsafe.Pointer(&x.Lengths[0])),
		(*C.uint32_t)(unsafe.Pointer(&x.Positions[0])), (*C.float)(unsafe.Pointer(&x.Sink[0])), (*C.float)(unsafe.Pointer(&x.CosSin[0])),
		C.int(x.Queries), C.int(x.Keys), C.int(x.TopK), C.int(x.TableRows), C.float(x.Scale),
		(*C.float)(unsafe.Pointer(&out.Values[0])), (*C.float)(unsafe.Pointer(&out.MaxLogits[0])), (*C.float)(unsafe.Pointer(&out.LSE[0]))))
	runtime.KeepAlive(x)
	if code != 0 {
		var cause error
		if code > 0 {
			cause = c.faultLatch.Observe(DeviceFaultExecution, "flashmla-prefill", code)
		}
		return zero, &CUDAOpError{Op: "V41SparsePrefill", Site: "native", Code: code, Err: cause}
	}
	// Unknown panics and Goexit propagate normally through the unlock defer; no
	// recover path can convert interrupted work into a successful empty result.
	return out, nil
}

// Package-local hardware witness state; this never changes production admission.
func cudaFlashMLAPhysicalState() (map[string]any, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var i C.fak_flashmla_identity
	var live C.uint64_t
	var uncertain C.int
	state := int(C.fak_flashmla_prefill_observe(&live, &uncertain))
	identity := int(C.fak_flashmla_prefill_kernel_info(&i))
	out := map[string]any{"device": int(i.device), "major": int(i.major), "minor": int(i.minor), "sms": int(i.sms), "driver": int(i.driver), "runtime": int(i.runtime), "binary_version": int(i.binary_version), "ptx_version": int(i.ptx_version), "name": C.GoString(&i.name[0]), "uuid": C.GoBytes(unsafe.Pointer(&i.uuid[0]), 16), "live_allocations": uint64(live), "uncertain": int(uncertain), "stream_status": state, "identity_status": identity}
	if state != 0 {
		return out, &CUDAOpError{Op: "FlashMLAWitness", Site: "stream", Code: state}
	}
	if identity != 0 {
		return out, &CUDAOpError{Op: "FlashMLAWitness", Site: "identity", Code: identity}
	}
	return out, nil
}
