//go:build cuda && cgo && fak_cuda_v41_routed_mmid

package compute

/*
#include <stdint.h>
int fak_v41_mmid_ordinary_compiled(int);
int fak_v41_mmid_ordinary_swiglu(const float *, const float *, float *, unsigned *, int, float, float);
int fak_v41_mmid_ordinary_bf16(const uint32_t *, uint32_t *, uint32_t *, int, uint32_t);
*/
import "C"

import (
	"math"
	"runtime"
	"sync/atomic"
)

// Private qualification only. No exported deferred methods or support query
// are supplied: the existing numerical availability gate remains false.
// witness=true is an explicit in-package qualification call, not serving policy.
func (c *cudaBackend) cudaV41MMIDRoutedSwiGLUDeferred(gate, up Tensor, limit, weight float32, witness bool) (out Tensor, st V41DeferredStatus, err error) {
	const site = "V41RoutedSwiGLUDeferred"
	count, invalid := validateV41RoutedSwiGLU(gate, up, limit, weight)
	if invalid != nil || count > math.MaxInt/4 {
		return Tensor{}, V41DeferredStatus{}, cudaV41MMIDError(site, "invalid ordinary routed geometry", 2, invalid)
	}
	if !witness {
		return Tensor{}, V41DeferredStatus{}, cudaV41MMIDError(site, "private numerical qualification not requested", -3, nil)
	}
	err = c.cudaV41MMIDCall(site, func(alloc func(int, string) (*cudaBuf, error)) error {
		if code := int(C.fak_v41_mmid_ordinary_compiled(1)); code != 0 {
			return c.cudaV41MMIDStatus(site, code)
		}
		gb, invalid := c.cudaV41MMIDOperand(site, "gate", gate, count*4)
		if invalid != nil {
			return invalid
		}
		ub, invalid := c.cudaV41MMIDOperand(site, "up", up, count*4)
		if invalid != nil {
			return invalid
		}
		output, invalid := alloc(count*4, "v41-mmid-ordinary-swiglu-output")
		if invalid != nil {
			return invalid
		}
		record, invalid := alloc(v41RoutedSwiGLUStatusBytes, "v41-mmid-ordinary-swiglu-status")
		if invalid != nil {
			return invalid
		}
		code := int(C.fak_v41_mmid_ordinary_swiglu((*C.float)(gb.ptr), (*C.float)(ub.ptr),
			(*C.float)(output.ptr), (*C.uint)(record.ptr), C.int(count), C.float(limit), C.float(weight)))
		if invalid := c.cudaV41MMIDStatus(site, code); invalid != nil {
			return invalid
		}
		out, st = c.cudaV41MMIDOrdinaryResult(output, record, count, V41DeferredRoutedSwiGLU, 0)
		return nil
	})
	runtime.KeepAlive(gate)
	runtime.KeepAlive(up)
	if err != nil {
		out, st = Tensor{}, V41DeferredStatus{}
	}
	return out, st, err
}

func (c *cudaBackend) cudaV41MMIDBF16PublishDeferred(input Tensor, publishSite uint32, witness bool) (out Tensor, st V41DeferredStatus, err error) {
	const site = "V41BF16PublishDeferred"
	bytes, invalid := validateV41BF16Publish(input, publishSite)
	if invalid != nil {
		return Tensor{}, V41DeferredStatus{}, cudaV41MMIDError(site, "invalid ordinary publication geometry", 2, invalid)
	}
	if !witness {
		return Tensor{}, V41DeferredStatus{}, cudaV41MMIDError(site, "private qualification not requested", -3, nil)
	}
	err = c.cudaV41MMIDCall(site, func(alloc func(int, string) (*cudaBuf, error)) error {
		if code := int(C.fak_v41_mmid_ordinary_compiled(2)); code != 0 {
			return c.cudaV41MMIDStatus(site, code)
		}
		ib, invalid := c.cudaV41MMIDOperand(site, "input", input, bytes)
		if invalid != nil {
			return invalid
		}
		output, invalid := alloc(bytes, "v41-mmid-ordinary-bf16-output")
		if invalid != nil {
			return invalid
		}
		record, invalid := alloc(v41BF16CheckedStatusBytes, "v41-mmid-ordinary-bf16-status")
		if invalid != nil {
			return invalid
		}
		code := int(C.fak_v41_mmid_ordinary_bf16((*C.uint32_t)(ib.ptr), (*C.uint32_t)(output.ptr),
			(*C.uint32_t)(record.ptr), C.int(bytes/4), C.uint32_t(publishSite)))
		if invalid := c.cudaV41MMIDStatus(site, code); invalid != nil {
			return invalid
		}
		out, st = c.cudaV41MMIDOrdinaryResult(output, record, bytes/4, V41DeferredBF16Publish, publishSite)
		return nil
	})
	runtime.KeepAlive(input)
	if err != nil {
		out, st = Tensor{}, V41DeferredStatus{}
	}
	return out, st, err
}

// Caller holds cudaMu. Successful enqueue is not a host-ready output: only the
// shared reader's checked completion may advance this generation. The caller
// owns output; the reader consumes the separate freshly issued status record.
func (c *cudaBackend) cudaV41MMIDOrdinaryResult(output, record *cudaBuf, count int, kind V41DeferredStatusKind, site uint32) (Tensor, V41DeferredStatus) {
	generation := atomic.LoadUint64(&c.fenceGen)
	for _, b := range []*cudaBuf{output, record} {
		b.be, b.bornGen = c, generation
	}
	out := makeTensor(c, F32, RowMajor, []int{count}, nil, output)
	st := V41DeferredStatus{Kind: kind, Count: count, Site: site, Records: 1,
		Record: makeTensor(c, F32, RowMajor, []int{4}, nil, record)}
	c.cudaV41MMIDIssue(st)
	return out, st
}
