//go:build cuda && cgo && fak_cuda_v41_routed_mmid

package compute

/*
#include <stddef.h>
#include <stdint.h>
#include "cuda_backend.h"
// Candidate-local ABI from cuda_v41_routed_mmid.cu (always compiled; this tag is the opt-in).
int fak_v41_mmid_state(void);
unsigned long long fak_v41_mmid_launch_count(void);
int fak_v41_mmid_gemv(int, const void *const *, int, const float *, int, const unsigned *, float *, int, int, int);
int fak_v41_mmid_gate_up_swiglu(int, const void *const *, const void *const *, int, const float *, int,
    const unsigned *, float *, unsigned *, int, int, int, float);
int fak_v41_mmid_down_sum(int, const void *const *, int, const float *, const unsigned *, float *, unsigned *,
    int, int, int, unsigned);
int fak_v41_mmid_swiglu_id(const float *, const float *, float *, const unsigned *, unsigned *, int, int, float);
int fak_v41_mmid_publish_sum(const float *, float *, const unsigned *, unsigned *, int, int, unsigned);
int fak_v41_mmid_read(void *, const void *, size_t);
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

var (
	_ V41RoutedIDBackend    = (*cudaBackend)(nil)
	_ V41RoutedFusedBackend = (*cudaBackend)(nil)
)

// The CUDA routed-expert id and fused paths (llama.cpp mul_mat_vec_q ids +
// gate/GLU fusion ported in cuda_v41_routed_mmid.cu). Compiling the tag is the
// opt-in. The model's batched routed layer also needs V41DeferredStatusBackend,
// which CUDA does not implement yet, so selecting this tag alone does not route
// a served layer through it.
func (c *cudaBackend) V41RoutedIDAvailable() bool    { return c.cudaV41MMIDAvailable() }
func (c *cudaBackend) V41RoutedFusedAvailable() bool { return c.cudaV41MMIDAvailable() }

func (c *cudaBackend) SupportsV41RoutedIDWeight(dt Dtype) bool { return dt == Q2_K || dt == Q3_K }
func (c *cudaBackend) SupportsV41RoutedFusedGateUp(gate, up Dtype) bool {
	return gate == up && (gate == Q2_K || gate == Q3_K)
}
func (c *cudaBackend) SupportsV41RoutedFusedDown(down Dtype) bool {
	return down == Q2_K || down == Q3_K
}

func (c *cudaBackend) cudaV41MMIDAvailable() bool {
	if c == nil {
		return false
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return c.cudaV41MMIDState("V41RoutedIDAvailable") == nil
}

func cudaV41MMIDError(site, message string, status int, cause error) error {
	return &BackendError{Backend: "cuda", Site: site, Class: "cuda-v41-routed-mmid-failed",
		Message: fmt.Sprintf("%s (status %d)", message, status), Err: cause, Recovered: cause}
}

// Caller holds cudaMu.
func (c *cudaBackend) cudaV41MMIDState(site string) error {
	if c == nil || c.capturing {
		return cudaV41MMIDError(site, "nil backend or active capture", -3, nil)
	}
	if err := c.faultLatch.Admit(site); err != nil {
		return cudaV41MMIDError(site, "sticky CUDA fault", -3, err)
	}
	if status := int(C.fak_v41_mmid_state()); status != 0 {
		return cudaV41MMIDError(site, "native admission refused", status, nil)
	}
	return nil
}

// cudaV41MMIDOperand admits only live device-0, non-managed, unscaled storage
// owned by c of at least bytes. Caller holds cudaMu.
func (c *cudaBackend) cudaV41MMIDOperand(site, name string, t Tensor, bytes int) (*cudaBuf, error) {
	b, ok := t.buf.(*cudaBuf)
	if t.be != c || !ok || b == nil || b.ptr == nil || b.n < bytes || b.managed || b.device != 0 ||
		b.scales != nil || b.scalesN != 0 || (b.be != nil && b.be != c) {
		return nil, cudaV41MMIDError(site, name+" is not owned device-only storage", -2, nil)
	}
	if invalid := b.invalidStateError(site + " " + name); invalid != nil {
		return nil, cudaV41MMIDError(site, "invalid storage lifetime", -2, invalid)
	}
	return b, nil
}

// cudaV41MMIDSlots returns the slot table's device pointers, each holding at
// least rows k-quant rows of in values. Caller holds cudaMu.
func (c *cudaBackend) cudaV41MMIDSlots(site, name string, weights []Tensor, rows, in int) ([]unsafe.Pointer, int, error) {
	if len(weights) == 0 || len(weights) > V41RoutedIDMaxPicks {
		return nil, 0, cudaV41MMIDError(site, name+" slot table size", 2, nil)
	}
	dtype, blockBytes := 0, 84
	if weights[0].Dtype == Q3_K {
		dtype, blockBytes = 1, 110
	} else if weights[0].Dtype != Q2_K {
		return nil, 0, cudaV41MMIDError(site, name+" dtype is neither Q2_K nor Q3_K", 2, nil)
	}
	bytes, ok := checkedV41RoutedFusedBytes(rows, (in/256)*blockBytes, uint64(^uint(0)>>1))
	if !ok {
		return nil, 0, cudaV41MMIDError(site, name+" weight bytes exceed the host domain", 2, nil)
	}
	ptrs := make([]unsafe.Pointer, len(weights))
	for i, w := range weights {
		b, ok := w.buf.(*cudaBuf)
		if w.be != c || !ok || b == nil || b.ptr == nil || b.n < bytes || b.device != 0 || b.scales != nil ||
			w.Dtype != weights[0].Dtype {
			return nil, 0, cudaV41MMIDError(site, name+" slot is not resident k-quant storage owned by this backend", 2, nil)
		}
		if invalid := b.invalidStateError(site + " " + name); invalid != nil {
			return nil, 0, cudaV41MMIDError(site, "invalid weight lifetime", -2, invalid)
		}
		ptrs[i] = b.ptr
	}
	return ptrs, dtype, nil
}

// cudaV41MMIDCall runs fn under cudaMu with the shared admission, panic and
// cleanup discipline; outputs listed in owned are freed when fn fails.
func (c *cudaBackend) cudaV41MMIDCall(site string, fn func(alloc func(int, string) (*cudaBuf, error)) error) (err error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var owned []*cudaBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			cause, _ := ConvertCUDAPanic(recovered, site, "routed mmid")
			err = cudaV41MMIDError(site, "candidate panicked", -1, cause)
		}
		if err != nil {
			for _, b := range owned {
				if b != nil && b.ptr != nil {
					C.fcuda_free(b.ptr)
					b.ptr, b.n = nil, 0
				}
			}
		}
	}()
	if err := c.cudaV41MMIDState(site); err != nil {
		return err
	}
	alloc := func(bytes int, what string) (*cudaBuf, error) {
		b, err := c.dallocDeviceOnlyClass(bytes, MemoryActivation, what)
		if err != nil {
			return nil, cudaV41MMIDError(site, "device allocation", -2, err)
		}
		owned = append(owned, b)
		return b, nil
	}
	return fn(alloc)
}

func (c *cudaBackend) cudaV41MMIDStatus(site string, status int) error {
	if status == 0 {
		return nil
	}
	if status > 2 && c != nil {
		c.faultLatch.Observe(DeviceFaultExecution, site, status)
	}
	return cudaV41MMIDError(site, "native routed mmid refused or failed", status, nil)
}

func (c *cudaBackend) V41RoutedIDs(slots []int, weights []float32, order []int) (out Tensor, err error) {
	const site = "V41RoutedIDs"
	words, invalid := V41RoutedIDWords(slots, weights, order)
	if invalid != nil {
		return Tensor{}, cudaV41MMIDError(site, "invalid ids", 2, invalid)
	}
	err = c.cudaV41MMIDCall(site, func(alloc func(int, string) (*cudaBuf, error)) error {
		buf, err := alloc(len(words)*4, "v41-routed-ids")
		if err != nil {
			return err
		}
		C.fcuda_h2d(buf.ptr, unsafe.Pointer(&words[0]), C.size_t(len(words)*4))
		out = makeTensor(c, F32, RowMajor, []int{len(words)}, nil, buf)
		return nil
	})
	runtime.KeepAlive(words)
	return out, err
}

func (c *cudaBackend) V41MatVecID(weights []Tensor, ids Tensor, picks int, x Tensor) (out Tensor, err error) {
	const site = "V41MatVecID"
	rows, in, xStride, invalid := validateV41MatVecID(weights, ids, picks, x)
	if invalid != nil {
		return Tensor{}, cudaV41MMIDError(site, "invalid geometry", 2, invalid)
	}
	err = c.cudaV41MMIDCall(site, func(alloc func(int, string) (*cudaBuf, error)) error {
		ptrs, dtype, err := c.cudaV41MMIDSlots(site, "weight", weights, rows, in)
		if err != nil {
			return err
		}
		xb, err := c.cudaV41MMIDOperand(site, "activation", x, x.Shape[0]*4)
		if err != nil {
			return err
		}
		ib, err := c.cudaV41MMIDOperand(site, "ids", ids, picks*12)
		if err != nil {
			return err
		}
		output, err := alloc(picks*rows*4, "v41-routed-id-gemv")
		if err != nil {
			return err
		}
		status := int(C.fak_v41_mmid_gemv(C.int(dtype), (*unsafe.Pointer)(unsafe.Pointer(&ptrs[0])), C.int(len(ptrs)),
			(*C.float)(xb.ptr), C.int(xStride), (*C.uint)(ib.ptr), (*C.float)(output.ptr), C.int(rows), C.int(in), C.int(picks)))
		runtime.KeepAlive(ptrs)
		if err := c.cudaV41MMIDStatus(site, status); err != nil {
			return err
		}
		out = makeTensor(c, F32, RowMajor, []int{picks * rows}, nil, output)
		return nil
	})
	return out, err
}

func (c *cudaBackend) V41RoutedSwiGLUIDDeferred(gate, up, ids Tensor, picks int, limit float32) (out Tensor, st V41DeferredStatus, err error) {
	const site = "V41RoutedSwiGLU"
	count, invalid := validateV41RoutedRows("gate", gate, picks)
	if invalid == nil {
		if upCount, upInvalid := validateV41RoutedRows("up", up, picks); upInvalid != nil {
			invalid = upInvalid
		} else if upCount != count {
			invalid = errors.New("compute: V4.1 routed id gate and up widths differ")
		}
	}
	if invalid == nil {
		invalid = validateV41RoutedIDsTensor(ids, picks)
	}
	if invalid == nil && !finite32(limit) {
		invalid = errors.New("compute: V4.1 routed id limit is not finite")
	}
	if invalid != nil {
		return Tensor{}, V41DeferredStatus{}, cudaV41MMIDError(site, "invalid geometry", 2, invalid)
	}
	err = c.cudaV41MMIDCall(site, func(alloc func(int, string) (*cudaBuf, error)) error {
		gb, err := c.cudaV41MMIDOperand(site, "gate", gate, picks*count*4)
		if err != nil {
			return err
		}
		ub, err := c.cudaV41MMIDOperand(site, "up", up, picks*count*4)
		if err != nil {
			return err
		}
		ib, err := c.cudaV41MMIDOperand(site, "ids", ids, picks*12)
		if err != nil {
			return err
		}
		output, err := alloc(picks*count*4, "v41-routed-swiglu-id")
		if err != nil {
			return err
		}
		record, err := alloc(picks*v41RoutedSwiGLUStatusBytes, "v41-routed-swiglu-id-status")
		if err != nil {
			return err
		}
		status := int(C.fak_v41_mmid_swiglu_id((*C.float)(gb.ptr), (*C.float)(ub.ptr), (*C.float)(output.ptr),
			(*C.uint)(ib.ptr), (*C.uint)(record.ptr), C.int(count), C.int(picks), C.float(limit)))
		if err := c.cudaV41MMIDStatus(site, status); err != nil {
			return err
		}
		out = makeTensor(c, F32, RowMajor, []int{picks * count}, nil, output)
		st = V41DeferredStatus{Kind: V41DeferredRoutedSwiGLU, Count: count, Records: picks,
			Record: makeTensor(c, F32, RowMajor, []int{picks * v41RoutedSwiGLUStatusBytes / 4}, nil, record)}
		return nil
	})
	return out, st, err
}

func (c *cudaBackend) V41RoutedPublishSumDeferred(down, ids Tensor, picks int, publishSite uint32) (out Tensor, st V41DeferredStatus, err error) {
	const site = "V41BF16Publish"
	count, invalid := validateV41RoutedRows("down", down, picks)
	if invalid == nil {
		invalid = validateV41RoutedIDsTensor(ids, picks)
	}
	if invalid == nil && !V41BF16PublishSite(publishSite) {
		invalid = errors.New("compute: invalid ordinary publication site")
	}
	if invalid != nil {
		return Tensor{}, V41DeferredStatus{}, cudaV41MMIDError(site, "invalid geometry", 2, invalid)
	}
	err = c.cudaV41MMIDCall(site, func(alloc func(int, string) (*cudaBuf, error)) error {
		db, err := c.cudaV41MMIDOperand(site, "down", down, picks*count*4)
		if err != nil {
			return err
		}
		ib, err := c.cudaV41MMIDOperand(site, "ids", ids, picks*12)
		if err != nil {
			return err
		}
		output, err := alloc(count*4, "v41-routed-publish-sum")
		if err != nil {
			return err
		}
		record, err := alloc(picks*v41BF16CheckedStatusBytes, "v41-routed-publish-sum-status")
		if err != nil {
			return err
		}
		status := int(C.fak_v41_mmid_publish_sum((*C.float)(db.ptr), (*C.float)(output.ptr), (*C.uint)(ib.ptr),
			(*C.uint)(record.ptr), C.int(count), C.int(picks), C.uint(publishSite)))
		if err := c.cudaV41MMIDStatus(site, status); err != nil {
			return err
		}
		out = makeTensor(c, F32, RowMajor, []int{count}, nil, output)
		st = V41DeferredStatus{Kind: V41DeferredBF16Publish, Count: count, Site: publishSite, Records: picks,
			Record: makeTensor(c, F32, RowMajor, []int{picks * v41BF16CheckedStatusBytes / 4}, nil, record)}
		return nil
	})
	return out, st, err
}

func (c *cudaBackend) V41RoutedGateUpSwiGLUDeferred(gates, ups []Tensor, ids Tensor, x Tensor, picks int, limit float32) (out Tensor, st V41DeferredStatus, err error) {
	const site = "V41RoutedFusedGateUp"
	supported := func(dt Dtype) bool { return c.SupportsV41RoutedFusedGateUp(dt, dt) }
	rows, in, invalid := validateV41RoutedFusedWeights("gate", gates, picks, supported)
	var xStride int
	if invalid == nil {
		if upRows, upIn, upInvalid := validateV41RoutedFusedWeights("up", ups, picks, supported); upInvalid != nil {
			invalid = upInvalid
		} else if upIn != in || upRows != rows || ups[0].Dtype != gates[0].Dtype {
			invalid = errors.New("compute: V4.1 routed fused gate and up slots differ in shape or dtype")
		}
	}
	if invalid == nil {
		invalid = validateV41RoutedIDsTensor(ids, picks)
	}
	if invalid == nil {
		xStride, invalid = validateV41RoutedFusedActivation(x, picks, in)
	}
	if invalid == nil && !finite32(limit) {
		invalid = errors.New("compute: V4.1 routed fused limit is not finite")
	}
	if invalid != nil {
		return Tensor{}, V41DeferredStatus{}, cudaV41MMIDError(site, "invalid geometry", 2, invalid)
	}
	err = c.cudaV41MMIDCall(site, func(alloc func(int, string) (*cudaBuf, error)) error {
		gatePtrs, dtype, err := c.cudaV41MMIDSlots(site, "gate", gates, rows, in)
		if err != nil {
			return err
		}
		upPtrs, _, err := c.cudaV41MMIDSlots(site, "up", ups, rows, in)
		if err != nil {
			return err
		}
		xb, err := c.cudaV41MMIDOperand(site, "activation", x, x.Shape[0]*4)
		if err != nil {
			return err
		}
		ib, err := c.cudaV41MMIDOperand(site, "ids", ids, picks*12)
		if err != nil {
			return err
		}
		output, err := alloc(picks*rows*4, "v41-routed-fused-mid")
		if err != nil {
			return err
		}
		record, err := alloc(picks*v41RoutedSwiGLUStatusBytes, "v41-routed-fused-gate-up-status")
		if err != nil {
			return err
		}
		status := int(C.fak_v41_mmid_gate_up_swiglu(C.int(dtype), (*unsafe.Pointer)(unsafe.Pointer(&gatePtrs[0])),
			(*unsafe.Pointer)(unsafe.Pointer(&upPtrs[0])), C.int(len(gatePtrs)), (*C.float)(xb.ptr), C.int(xStride),
			(*C.uint)(ib.ptr), (*C.float)(output.ptr), (*C.uint)(record.ptr), C.int(rows), C.int(in), C.int(picks), C.float(limit)))
		runtime.KeepAlive(gatePtrs)
		runtime.KeepAlive(upPtrs)
		if err := c.cudaV41MMIDStatus(site, status); err != nil {
			return err
		}
		out = makeTensor(c, F32, RowMajor, []int{picks * rows}, nil, output)
		st = V41DeferredStatus{Kind: V41DeferredRoutedSwiGLU, Count: rows, Records: picks,
			Record: makeTensor(c, F32, RowMajor, []int{picks * v41RoutedSwiGLUStatusBytes / 4}, nil, record)}
		return nil
	})
	return out, st, err
}

func (c *cudaBackend) V41RoutedDownSumDeferred(downs []Tensor, mid, ids Tensor, picks int, publishSite uint32) (out Tensor, st V41DeferredStatus, err error) {
	const site = "V41BF16Publish"
	rows, in, invalid := validateV41RoutedFusedWeights("down", downs, picks, c.SupportsV41RoutedFusedDown)
	if invalid == nil {
		invalid = validateV41RoutedIDsTensor(ids, picks)
	}
	if invalid == nil && !V41BF16PublishSite(publishSite) {
		invalid = errors.New("compute: invalid ordinary publication site")
	}
	if invalid == nil && (mid.Dtype != F32 || mid.Layout != RowMajor || mid.Quant != nil || len(mid.Shape) != 1 ||
		mid.Shape[0] != picks*in) {
		invalid = errors.New("compute: V4.1 routed fused mid must be the unquantized row-major F32 [picks*inter] gate/up output")
	}
	if invalid != nil {
		return Tensor{}, V41DeferredStatus{}, cudaV41MMIDError(site, "invalid geometry", 2, invalid)
	}
	err = c.cudaV41MMIDCall(site, func(alloc func(int, string) (*cudaBuf, error)) error {
		ptrs, dtype, err := c.cudaV41MMIDSlots(site, "down", downs, rows, in)
		if err != nil {
			return err
		}
		mb, err := c.cudaV41MMIDOperand(site, "mid", mid, picks*in*4)
		if err != nil {
			return err
		}
		ib, err := c.cudaV41MMIDOperand(site, "ids", ids, picks*12)
		if err != nil {
			return err
		}
		output, err := alloc(rows*4, "v41-routed-fused-sum")
		if err != nil {
			return err
		}
		record, err := alloc(picks*v41BF16CheckedStatusBytes, "v41-routed-fused-down-status")
		if err != nil {
			return err
		}
		status := int(C.fak_v41_mmid_down_sum(C.int(dtype), (*unsafe.Pointer)(unsafe.Pointer(&ptrs[0])), C.int(len(ptrs)),
			(*C.float)(mb.ptr), (*C.uint)(ib.ptr), (*C.float)(output.ptr), (*C.uint)(record.ptr), C.int(rows), C.int(in),
			C.int(picks), C.uint(publishSite)))
		runtime.KeepAlive(ptrs)
		if err := c.cudaV41MMIDStatus(site, status); err != nil {
			return err
		}
		out = makeTensor(c, F32, RowMajor, []int{rows}, nil, output)
		st = V41DeferredStatus{Kind: V41DeferredBF16Publish, Count: rows, Site: publishSite, Records: picks,
			Record: makeTensor(c, F32, RowMajor, []int{picks * v41BF16CheckedStatusBytes / 4}, nil, record)}
		return nil
	})
	return out, st, err
}

// cudaV41MMIDVerdicts reads one deferred status behind a checked completion,
// frees its record, and returns one verdict per pick record. This is the
// reader the CUDA V41DeferredStatusBackend (not implemented yet) would share.
func (c *cudaBackend) cudaV41MMIDVerdicts(st V41DeferredStatus) (verdicts []error, err error) {
	const site = "V41ReadDeferred"
	records := V41DeferredRecords(st)
	raw := make([]byte, records*16)
	err = c.cudaV41MMIDCall(site, func(func(int, string) (*cudaBuf, error)) error {
		b, err := c.cudaV41MMIDOperand(site, "status", st.Record, len(raw))
		if err != nil {
			return err
		}
		status := int(C.fak_v41_mmid_read(unsafe.Pointer(&raw[0]), b.ptr, C.size_t(len(raw))))
		if err := c.cudaV41MMIDStatus(site, status); err != nil {
			return err
		}
		C.fcuda_free(b.ptr)
		b.ptr, b.n = nil, 0
		return nil
	})
	if err != nil {
		return nil, err
	}
	verdicts = make([]error, records)
	for r := range verdicts {
		verdicts[r] = decodeV41DeferredStatus(st, raw[16*r:16*r+16])
	}
	return verdicts, nil
}

// cudaV41MMIDLaunchesForTest counts successful candidate kernel enqueues.
func cudaV41MMIDLaunchesForTest() uint64 { return uint64(C.fak_v41_mmid_launch_count()) }
