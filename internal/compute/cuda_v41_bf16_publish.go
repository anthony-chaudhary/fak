//go:build cuda && cgo && fak_cuda_v41_bf16_publish

package compute

/*
#include <stdint.h>
#include "cuda_backend.h"

// The offline CUDA library must also use -DFAK_CUDA_V41_BF16_PUBLISH.
// This private candidate ABI is deliberately not an fcuda_* header extension.
int fak_v41_bf16_publish_state(void);
int fak_v41_bf16_publish_input(const uint32_t *dInput, int count);
int fak_v41_bf16_publish_f32(
    const uint32_t *dInput, uint32_t *dOutput, uint32_t *dStatus,
    void *hStatus, int count, uint32_t site);
*/
import "C"

import (
	"math"
	"sync/atomic"
	"unsafe"
)

var _ V41BF16PublishBackend = (*cudaBackend)(nil)

// SupportsV41BF16Publish describes this explicitly selected candidate and its
// integer/grid-stride count domain, not transient stream or session health.
// Neither ordinary CUDA builds nor device-only model admission are changed.
func (c *cudaBackend) SupportsV41BF16Publish(count int) bool {
	return c != nil && count > 0 && count <= math.MaxInt32 && count <= math.MaxInt/4
}

// V41BF16Publish publishes finite widened BF16 into a fresh caller-owned F32
// allocation. It synchronizes the existing CUDA stream, but never takes over a
// capture. Only the checked 16-byte status crosses to the host, never the input
// or output. A selected failure has no capability-decline or host-replay route.
func (c *cudaBackend) V41BF16Publish(input Tensor, site uint32) (out Tensor, err error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var output, statusBuf *cudaBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			err, _ = ConvertCUDAPanic(recovered, "V41BF16Publish", "v41-bf16-publish")
		}
		// Native quarantine prevents either buffer from re-entering the pool
		// when a submitted operation could not be proven complete. They never
		// enter c.transient, so Recycle cannot bypass this ownership boundary.
		for _, b := range []*cudaBuf{statusBuf, output} {
			if b != nil && b.ptr != nil && (b == statusBuf || err != nil) {
				C.fcuda_free(b.ptr)
				b.ptr, b.n = nil, 0
			}
		}
		if err != nil {
			out = Tensor{}
		}
	}()
	fail := func(stage, message string, code int, cause error) error {
		return &CUDAOpError{Op: "V41BF16Publish", Site: "v41-bf16-publish-" + stage, Msg: message, Code: code, Err: cause}
	}
	if c == nil {
		return Tensor{}, fail("preflight", "nil CUDA backend", 0, nil)
	}
	bytes, invalid := validateV41BF16Publish(input, site)
	if invalid != nil {
		return Tensor{}, fail("geometry", "invalid ordinary BF16 publication geometry or site", 0, invalid)
	}
	source, ok := input.buf.(*cudaBuf)
	if input.Backend() != c || !ok || source == nil || source.ptr == nil ||
		source.device != 0 || source.managed || source.n < bytes || source.scales != nil || source.scalesN != 0 ||
		(source.be != nil && source.be != c) {
		return Tensor{}, fail("residency", "input must be live unquantized device-only F32 storage owned by this CUDA backend", 0, nil)
	}
	if invalid := source.invalidStateError("V41BF16Publish input"); invalid != nil {
		return Tensor{}, fail("residency", "input has invalid CUDA state", 0, invalid)
	}
	if c.capturing {
		return Tensor{}, fail("state", "synchronous publication cannot enter an active capture", 0, nil)
	}
	if err := c.faultLatch.Admit("v41-bf16-publish"); err != nil {
		return Tensor{}, err
	}
	if status := int(C.fak_v41_bf16_publish_state()); status != 0 {
		return Tensor{}, fail("state", "native submission state refuses publication", status, nil)
	}
	if status := int(C.fak_v41_bf16_publish_input((*C.uint32_t)(source.ptr), C.int(input.Shape[0]))); status != 0 {
		return Tensor{}, fail("residency", "native allocation ownership refuses input", status, nil)
	}
	// This allocator never falls back to managed memory. Keeping allocations
	// outside the transient list makes the successful result caller-owned.
	output, err = c.dallocDeviceOnlyClass(bytes, MemoryActivation, "v41-bf16-publish-output")
	if err != nil {
		return Tensor{}, fail("allocation", "device-only output allocation failed", 0, err)
	}
	statusBuf, err = c.dallocDeviceOnlyClass(v41BF16CheckedStatusBytes, MemoryScratchpad, "v41-bf16-publish-status")
	if err != nil {
		return Tensor{}, fail("allocation", "device-only status allocation failed", 0, err)
	}
	var raw [v41BF16CheckedStatusBytes]byte
	status := int(C.fak_v41_bf16_publish_f32(
		(*C.uint32_t)(source.ptr), (*C.uint32_t)(output.ptr), (*C.uint32_t)(statusBuf.ptr),
		unsafe.Pointer(&raw[0]), C.int(input.Shape[0]), C.uint32_t(site)))
	if status != 0 {
		if status > 0 {
			c.faultLatch.Observe(DeviceFaultExecution, "v41-bf16-publish", status)
		}
		return Tensor{}, fail("execution", "checked publication failed; output withheld", status, nil)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	if statusErr := decodeV41BF16PublishStatus(raw[:], input.Shape[0], site); statusErr != nil {
		return Tensor{}, fail("status", "checked publication rejected its result", 0, statusErr)
	}
	return makeTensor(c, F32, RowMajor, []int{input.Shape[0]}, nil, output), nil
}
