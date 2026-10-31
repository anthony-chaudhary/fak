//go:build cuda && cgo && fak_cuda_v41_embed_gather && fak_cuda_v41_engram_iq4nl

package compute

/*
#include <stdint.h>
#include "cuda_backend.h"
// Candidate-local ABI; the native archive needs both
// FAK_CUDA_V41_EMBED_GATHER and FAK_CUDA_V41_EMBED_IQ4NL.
unsigned fak_v41_embed_abi(void);
unsigned fak_v41_embed_iq4nl_rows_abi(void);
int fak_v41_embed_state(void);
int fak_v41_embed_finish(void);
int fak_v41_embed_iq4nl_upload(void *,const void *,uint64_t);
int fak_v41_embed_iq4nl_gather_rows(const void *,float *,uint32_t *,const intptr_t *,int,int,int);
int fak_v41_embed_iq4nl_row_poison(float *,const uint32_t *,int,int,int);
*/
import "C"

import (
	"fmt"
	"math"
	"runtime"
	"sync/atomic"
	"unsafe"
)

var (
	_ V41EngramRowDecodeBackend   = (*cudaBackend)(nil)
	_ V41EngramRowEncodingBackend = (*cudaBackend)(nil)
)

// The checked native caller is available only to the in-package device witness
// until exact-module/device qualification and authenticated model staging land.
// Compiling these tags alone never advertises a serving capability.
func (c *cudaBackend) SupportsV41EngramRowDecode() bool { return false }
func (c *cudaBackend) SupportsV41EngramRowEncoding(encoding V41EngramRowEncoding) bool {
	return encoding == V41EngramRowIQ4NL && c.SupportsV41EngramRowDecode()
}

func (c *cudaBackend) cudaV41EngramRowError(site, detail string, code int, cause error) error {
	if c != nil && code > 0 {
		if fault := c.faultLatch.Observe(DeviceFaultExecution, site, code); cause == nil && fault != nil {
			cause = fault
		}
	}
	return &BackendError{Backend: "cuda", Site: site, Class: "cuda-engram-row-failed",
		Message: fmt.Sprintf("%s (status %d)", detail, code), Err: cause}
}

// Caller holds cudaMu. witness is never supplied by the model or public method.
func (c *cudaBackend) cudaV41EngramRowState(site string, witness bool) error {
	if c == nil || c != cudaDev || c.capturing || !witness {
		return c.cudaV41EngramRowError(site, "unqualified route or invalid backend/capture", -3, nil)
	}
	if err := c.faultLatch.Admit(site); err != nil {
		return c.cudaV41EngramRowError(site, "sticky CUDA fault", -3, err)
	}
	if uint32(C.fak_v41_embed_abi()) != 2 || uint32(C.fak_v41_embed_iq4nl_rows_abi()) != 1 {
		return c.cudaV41EngramRowError(site, "exact IQ4_NL row ABI unavailable", -3, nil)
	}
	if code := int(C.fak_v41_embed_state()); code != 0 {
		return c.cudaV41EngramRowError(site, "native admission refused", code, nil)
	}
	return nil
}

func (c *cudaBackend) V41EngramRowDecode(encoded []byte, encoding V41EngramRowEncoding, rows, dim int) (Tensor, Tensor, error) {
	return c.cudaV41EngramRowDecode(encoded, encoding, rows, dim, false)
}

// The caller supplies only already-gathered, authenticated rows, not a table or
// global IDs. Local identity IDs preserve that order without whole-table upload.
// This adapter does no host arithmetic and never retries a selected error.
func (c *cudaBackend) cudaV41EngramRowDecode(encoded []byte, encoding V41EngramRowEncoding, rows, dim int, witness bool) (panel, status Tensor, err error) {
	const site = "V41EngramRowDecode"
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var source, output, words *cudaBuf
	var ids []C.intptr_t
	admitted := false
	defer func() {
		if recovered := recover(); recovered != nil {
			cause, _ := ConvertCUDAPanic(recovered, site, "engram row")
			err = c.cudaV41EngramRowError(site, "candidate panicked", -1, cause)
		}
		if err != nil && admitted {
			if code := int(C.fak_v41_embed_finish()); code != 0 {
				err = c.cudaV41EngramRowError(site, "dependency drain failed; native allocations retained", code, err)
			}
		}
		for _, b := range []*cudaBuf{source, output, words} {
			if b != nil && b.ptr != nil && (b == source || err != nil) {
				// A terminal native stream retains storage instead of recycling it.
				// Invalidate the Go handle even when native quarantine keeps bytes.
				atomic.StoreUint32(&b.invalid, 1)
				C.fcuda_free(b.ptr)
				b.ptr, b.n = nil, 0
				// Retirement may surface a pool eviction failure after the checked
				// gather fence. Keep that fence generation, but publish no outputs.
				if err == nil {
					if code := int(C.fak_v41_embed_state()); code != 0 {
						var cause error
						if code < 0 {
							cause = c.faultLatch.Observe(DeviceFaultExecution, site, code)
						}
						err = c.cudaV41EngramRowError(site, "source retirement failed; native allocations retained", code, cause)
					}
				}
			}
		}
		if err != nil {
			panel, status = Tensor{}, Tensor{}
		}
		runtime.KeepAlive(encoded)
		runtime.KeepAlive(ids)
	}()
	if encoding != V41EngramRowIQ4NL {
		return Tensor{}, Tensor{}, c.cudaV41EngramRowError(site, "only IQ4_NL rows are supported", -1, nil)
	}
	if invalid := validateV41EngramRowDecode(encoded, encoding, rows, dim); invalid != nil {
		return Tensor{}, Tensor{}, c.cudaV41EngramRowError(site, "invalid stored rows", -1, invalid)
	}
	// The native allocator/status ABI bounds bytes, not just shader elements.
	if rows > math.MaxInt32/(4*dim) {
		return Tensor{}, Tensor{}, c.cudaV41EngramRowError(site, "panel byte extent exceeds int32", -1, nil)
	}
	if err = c.cudaV41EngramRowState(site, witness); err != nil {
		return Tensor{}, Tensor{}, err
	}
	admitted = true
	if source, err = c.dallocDeviceOnlyClass(len(encoded), MemoryActivation, "v41-engram-iq4nl-encoded"); err != nil {
		return Tensor{}, Tensor{}, err
	}
	if output, err = c.dallocDeviceOnlyClass(rows*dim*4, MemoryActivation, "v41-engram-iq4nl-panel"); err != nil {
		return Tensor{}, Tensor{}, err
	}
	if words, err = c.dallocDeviceOnlyClass(rows*4, MemoryScratchpad, "v41-engram-iq4nl-status"); err != nil {
		return Tensor{}, Tensor{}, err
	}
	ids = make([]C.intptr_t, rows)
	for r := range ids {
		ids[r] = C.intptr_t(r)
	}
	if code := int(C.fak_v41_embed_iq4nl_upload(source.ptr, unsafe.Pointer(&encoded[0]), C.uint64_t(len(encoded)))); code != 0 {
		return Tensor{}, Tensor{}, c.cudaV41EngramRowError(site, "checked encoded-row upload failed", code, nil)
	}
	code := int(C.fak_v41_embed_iq4nl_gather_rows(source.ptr, (*C.float)(output.ptr), (*C.uint32_t)(words.ptr), &ids[0], C.int(rows), C.int(dim), C.int(rows)))
	if code != 0 {
		return Tensor{}, Tensor{}, c.cudaV41EngramRowError(site, "checked row decode failed", code, nil)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	return makeTensor(c, F32, RowMajor, []int{rows, dim}, nil, output),
		makeTensor(c, F32, RowMajor, []int{rows}, nil, words), nil
}

func (c *cudaBackend) V41EngramRowDecodePoison(target, status Tensor, rows, index int) error {
	return c.cudaV41EngramRowDecodePoison(target, status, rows, index, false)
}

func (c *cudaBackend) cudaV41EngramRowDecodePoison(target, status Tensor, rows, index int, witness bool) (err error) {
	const site = "V41EngramRowDecodePoison"
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var buffers [2]*cudaBuf
	submitted := false
	defer func() {
		if recovered := recover(); recovered != nil {
			cause, _ := ConvertCUDAPanic(recovered, site, "engram row poison")
			err = c.cudaV41EngramRowError(site, "candidate panicked", -1, cause)
		}
		if err != nil && submitted {
			atomic.StoreUint32(&buffers[0].invalid, 1)
			if code := int(C.fak_v41_embed_finish()); code != 0 {
				err = c.cudaV41EngramRowError(site, "dependency drain failed; target retained", code, err)
			}
		}
		runtime.KeepAlive(target)
		runtime.KeepAlive(status)
	}()
	var counts [2]int
	for i, t := range []Tensor{target, status} {
		count := 1
		for _, dim := range t.Shape {
			if dim <= 0 || dim > math.MaxInt32/(4*count) {
				return c.cudaV41EngramRowError(site, "tensor byte extent exceeds int32", -1, nil)
			}
			count *= dim
		}
		counts[i] = count
		b, ok := t.buf.(*cudaBuf)
		if t.Dtype != F32 || t.Layout != RowMajor || t.Quant != nil || t.be != c || !ok || b == nil || b.ptr == nil ||
			b.n < count*4 || b.managed || b.device != 0 || b.scales != nil || b.scalesN != 0 || (b.be != nil && b.be != c) {
			return c.cudaV41EngramRowError(site, "expected owned device-only F32 storage", -2, nil)
		}
		if invalid := b.invalidStateError(site); invalid != nil {
			return c.cudaV41EngramRowError(site, "invalid storage lifetime", -2, invalid)
		}
		buffers[i] = b
	}
	if invalid := validateV41EngramRowPoison(target, status, rows, index); invalid != nil {
		return c.cudaV41EngramRowError(site, "invalid poison geometry", -1, invalid)
	}
	if buffers[0].ptr == buffers[1].ptr {
		return c.cudaV41EngramRowError(site, "target aliases status", -2, nil)
	}
	if err = c.cudaV41EngramRowState(site, witness); err != nil {
		return err
	}
	submitted = true
	code := int(C.fak_v41_embed_iq4nl_row_poison((*C.float)(buffers[0].ptr), (*C.uint32_t)(buffers[1].ptr), C.int(rows), C.int(counts[0]), C.int(index)))
	if code != 0 {
		return c.cudaV41EngramRowError(site, "checked poison failed", code, nil)
	}
	atomic.AddUint64(&c.fenceGen, 1)
	return nil
}
