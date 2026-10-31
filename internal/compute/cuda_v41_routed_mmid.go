//go:build cuda && cgo && fak_cuda_v41_routed_mmid

package compute

/*
#include <stddef.h>
#include <stdint.h>
#include "cuda_backend.h"
// Candidate-local ABI from cuda_v41_routed_mmid.cu. The kernels compile in every
// CUDA profile; this tag gates only the binding.
int fak_v41_mmid_state(void);
int fak_v41_mmid_uncertain(void);
unsigned long long fak_v41_mmid_launch_count(void);
int fak_v41_mmid_gemv(int, const void *const *, int, const float *, int, const unsigned *, float *, int, int, int);
int fak_v41_mmid_gate_up_swiglu(int, const void *const *, const void *const *, int, const float *, int,
    const unsigned *, float *, unsigned *, int, int, int, float);
int fak_v41_mmid_down_sum(int, const void *const *, int, const float *, const unsigned *, float *, unsigned *,
    int, int, int, unsigned);
int fak_v41_mmid_swiglu_id(const float *, const float *, float *, const unsigned *, unsigned *, int, int, float);
int fak_v41_mmid_publish_sum(const float *, float *, const unsigned *, unsigned *, int, int, unsigned);
int fak_v41_mmid_read_batch(void *, size_t, const void *const *, const size_t *, const size_t *, int, int *);
int fak_v41_mmid_retire_status(void *);
void fak_v41_mmid_hold_uncertain(void);
*/
import "C"

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync/atomic"
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
// cleanup discipline; outputs listed in owned are freed when fn fails, unless
// the failure left the stream uncertain (a 5xxxx drain status): kernels may
// still write them, so they stay retained until process teardown.
func (c *cudaBackend) cudaV41MMIDCall(site string, fn func(alloc func(int, string) (*cudaBuf, error)) error) (err error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var owned []*cudaBuf
	defer func() {
		if recovered := recover(); recovered != nil {
			cause, _ := ConvertCUDAPanic(recovered, site, "routed mmid")
			err = cudaV41MMIDError(site, "candidate panicked", -1, cause)
		}
		if err != nil && C.fak_v41_mmid_uncertain() == 0 {
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
		c.cudaV41MMIDIssue(st)
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
		c.cudaV41MMIDIssue(st)
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
		c.cudaV41MMIDIssue(st)
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
		c.cudaV41MMIDIssue(st)
		return nil
	})
	return out, st, err
}

// cudaV41MMIDVerdicts preserves the component-test entry point while sharing
// the owned batched reader; it is not a public deferred capability grant.
func (c *cudaBackend) cudaV41MMIDVerdicts(st V41DeferredStatus) ([]error, error) {
	verdicts, _, err := c.cudaV41MMIDReadBatch([]V41DeferredStatus{st}, nil)
	return verdicts, err
}

// cudaV41MMIDLaunchesForTest counts successful candidate kernel enqueues.
func cudaV41MMIDLaunchesForTest() uint64 { return uint64(C.fak_v41_mmid_launch_count()) }

// Issuance is bound to the opaque Go buffer lifetime, not just a recyclable
// device address. Free invalidates that same cudaBuf by clearing ptr. The map
// holds only outstanding status metadata; issue/read sweep entries freed by the
// caller. All access, including the producer calls above, holds cudaMu.
type cudaV41MMIDIssued struct {
	owner   *cudaBackend
	ptr     unsafe.Pointer
	kind    V41DeferredStatusKind
	count   int
	site    uint32
	records int
}

var cudaV41MMIDIssuedStatuses = make(map[*cudaBuf]cudaV41MMIDIssued)

func cudaV41MMIDSweepIssued() {
	for b, issued := range cudaV41MMIDIssuedStatuses {
		if b.ptr == nil || b.ptr != issued.ptr {
			delete(cudaV41MMIDIssuedStatuses, b)
		}
	}
}

func (c *cudaBackend) cudaV41MMIDIssue(st V41DeferredStatus) {
	cudaV41MMIDSweepIssued()
	b := st.Record.buf.(*cudaBuf)
	cudaV41MMIDIssuedStatuses[b] = cudaV41MMIDIssued{
		owner: c, ptr: b.ptr, kind: st.Kind, count: st.Count, site: st.Site, records: st.Records,
	}
}

// A narrow actual-source test seam, not a backend extension. Native owns all
// asynchronous staging; read never lets CUDA retain host or Go slice pointers.
type cudaV41MMIDBatchOperations struct {
	admit  func(*cudaBackend) error
	read   func([]byte, []unsafe.Pointer, []int, []int) (int, bool)
	retire func(unsafe.Pointer) int
	hold   func()
}

var cudaV41MMIDBatchNative = cudaV41MMIDBatchOperations{
	admit: func(c *cudaBackend) error { return c.cudaV41MMIDState("V41ReadDeferred") },
	read: func(host []byte, ptrs []unsafe.Pointer, sizes, offsets []int) (int, bool) {
		var complete C.int
		if len(ptrs) == 0 {
			status := C.fak_v41_mmid_read_batch(nil, 0, nil, nil, nil, 0, &complete)
			return int(status), complete != 0
		}
		ns, no := make([]C.size_t, len(sizes)), make([]C.size_t, len(offsets))
		for i := range sizes {
			ns[i], no[i] = C.size_t(sizes[i]), C.size_t(offsets[i])
		}
		status := C.fak_v41_mmid_read_batch(unsafe.Pointer(&host[0]), C.size_t(len(host)),
			(*unsafe.Pointer)(unsafe.Pointer(&ptrs[0])), &ns[0], &no[0], C.int(len(ptrs)), &complete)
		runtime.KeepAlive(host)
		runtime.KeepAlive(ptrs)
		return int(status), complete != 0
	},
	retire: func(ptr unsafe.Pointer) int { return int(C.fak_v41_mmid_retire_status(ptr)) },
	hold:   func() { C.fak_v41_mmid_hold_uncertain() },
}

type cudaV41MMIDAccepted struct {
	buf    *cudaBuf
	issued cudaV41MMIDIssued
	retain bool // malformed row aliases must not pool caller-visible storage
}

func cudaV41MMIDRangesOverlap(a unsafe.Pointer, an int, b unsafe.Pointer, bn int) bool {
	if a == nil || b == nil {
		return false
	}
	x, y := uintptr(a), uintptr(b)
	// An invalid extent has no proven disjointness: conservatively retain.
	if an <= 0 || bn <= 0 || uintptr(an) > ^uintptr(0)-x || uintptr(bn) > ^uintptr(0)-y {
		return true
	}
	return x < y+uintptr(bn) && y < x+uintptr(an)
}

func cudaV41MMIDStatusDomain(st V41DeferredStatus) bool {
	if st.Count <= 0 || st.Count > math.MaxInt32 || st.Count > math.MaxInt/4 ||
		st.Records < 0 || st.Records > V41RoutedIDMaxPicks {
		return false
	}
	switch st.Kind {
	case V41DeferredRoutedSwiGLU:
		return st.Site == 0
	case V41DeferredBF16Publish:
		return V41BF16PublishSite(st.Site)
	case V41DeferredF32Finite:
		return st.Records <= 1 && v41F32FiniteSite(st.Site)
	}
	return false
}

// Private qualification reader only. Neither this method nor issuing status
// metadata implements V41DeferredStatusBackend or changes model selection.
func (c *cudaBackend) cudaV41MMIDReadBatch(statuses []V41DeferredStatus, rows []Tensor) (verdicts []error, values [][]float32, err error) {
	return c.cudaV41MMIDConsume(statuses, rows, false, nil)
}

// cudaV41MMIDCancel settles previously issued statuses after an early abort,
// including a sticky admission/runtime error in a later producer. It does not
// submit a read, decode status, free caller outputs or turn the original error
// into success. Only descriptors issued by this backend are accepted; foreign,
// unissued and already-consumed handles are ignored, so cleanup is idempotent.
// The first error is preserved; cleanup failures also poison/retain through the
// same native terminal owner as the reader. No public cancellation ABI is added.
func (c *cudaBackend) cudaV41MMIDCancel(statuses []V41DeferredStatus, primary error) error {
	_, _, err := c.cudaV41MMIDConsume(statuses, nil, true, primary)
	return err
}

// One ownership core for reading and aborting; cancellation differs only in
// skipping admission/copies/decoding after collecting every issued owner.
func (c *cudaBackend) cudaV41MMIDConsume(statuses []V41DeferredStatus, rows []Tensor, cancel bool, primary error) ([]error, [][]float32, error) {
	verdicts, values, _, err := c.cudaV41MMIDConsumeChecked(statuses, rows, cancel, false, primary)
	return verdicts, values, err
}

// cudaV41MMIDCancelCompletion additionally establishes a checked stream drain
// when no status was issued: caller outputs can precede the first status. The
// old cancellation wrapper keeps its original no-owner/idempotence behavior.
func (c *cudaBackend) cudaV41MMIDCancelCompletion(statuses []V41DeferredStatus, primary error) (bool, error) {
	_, _, completed, err := c.cudaV41MMIDConsumeChecked(statuses, nil, true, true, primary)
	return completed, err
}

func (c *cudaBackend) cudaV41MMIDConsumeChecked(statuses []V41DeferredStatus, rows []Tensor, cancel, forceDrain bool, primary error) (verdicts []error, values [][]float32, completed bool, err error) {
	const site = "V41ReadDeferred"
	err = primary
	cudaMu.Lock()
	defer cudaMu.Unlock()
	ops := cudaV41MMIDBatchNative
	var accepted []cudaV41MMIDAccepted
	fenced, complete := false, false
	cleanupOK := true
	fail := func(message string) error { return cudaV41MMIDError(site, message, 2, nil) }
	setStatus := func(status int) {
		if status != 0 {
			cleanupOK = false
			failure := c.cudaV41MMIDStatus(site, status)
			if err == nil {
				err = failure
			}
		}
	}
	defer func() {
		if p := recover(); p != nil {
			cleanupOK = false
			if err == nil {
				cause, _ := ConvertCUDAPanic(p, site, "batched mmid read")
				err = cudaV41MMIDError(site, "batched reader panicked", -1, cause)
			}
		}
		// Even a late metadata rejection follows producer work. Establish a
		// completion disposition before returning any accepted owner to a pool.
		if (len(accepted) != 0 || (forceDrain && c != nil)) && !fenced {
			func() {
				defer func() {
					if p := recover(); p != nil {
						cleanupOK = false
						if err == nil {
							err = fail("cleanup completion panicked; records retained")
						}
					}
				}()
				status, done := ops.read(nil, nil, nil, nil)
				fenced, complete = true, done
				setStatus(status)
			}()
		}
		if !complete && (len(accepted) != 0 || fenced || (forceDrain && c != nil)) {
			ops.hold()
		}
		for _, a := range accepted {
			// Logical consumption occurs once, including unknown completion.
			// Native g_live and the shared uncertainty latch retain physical
			// allocations that cannot safely re-enter the pool.
			delete(cudaV41MMIDIssuedStatuses, a.buf)
			a.buf.ptr, a.buf.n = nil, 0
			if !complete || a.retain {
				continue
			}
			func() {
				defer func() {
					if p := recover(); p != nil {
						cleanupOK = false
						if err == nil {
							err = fail("status retirement panicked")
						}
					}
				}()
				// Native 1 means retained by a shared lifetime guard, not a
				// second ownership grant. Negative values are transport errors.
				if code := ops.retire(a.issued.ptr); code < 0 {
					setStatus(code)
				}
			}()
		}
		if err != nil {
			verdicts, values = nil, nil
		} else if fenced && complete {
			atomic.AddUint64(&c.fenceGen, 1)
		}
		completed = complete && cleanupOK
		runtime.KeepAlive(statuses)
		runtime.KeepAlive(rows)
	}()
	cudaV41MMIDSweepIssued()
	// Collect ALL issued owners before checking any descriptor or later row.
	// A forged foreign/non-issued tensor never enters this retirement list.
	for _, st := range statuses {
		b, ok := st.Record.buf.(*cudaBuf)
		if !ok || b == nil || st.Record.be != c || c == nil {
			continue
		}
		issued, ok := cudaV41MMIDIssuedStatuses[b]
		if !ok || issued.owner != c || b.ptr == nil || b.ptr != issued.ptr {
			continue
		}
		duplicate := false
		for _, a := range accepted {
			if a.buf == b || a.issued.ptr == issued.ptr {
				duplicate = true
				break
			}
		}
		if !duplicate {
			accepted = append(accepted, cudaV41MMIDAccepted{buf: b, issued: issued})
		}
	}
	if cancel {
		// No metadata/shape/capability gate can strand an already accepted
		// owner. In particular, do not call ops.admit: the fault which caused
		// this abort may have poisoned the Go latch. Deferred cleanup performs
		// checked native completion independently and applies its disposition.
		if c == nil && err == nil {
			err = fail("nil backend cannot cancel deferred statuses")
		}
		return nil, nil, false, err
	}
	// Account aliases even when the row itself later fails dtype/owner/extent
	// admission. No malformed row can make its storage a status pool return.
	for i := range accepted {
		for _, row := range rows {
			if b, ok := row.buf.(*cudaBuf); ok && b != nil &&
				cudaV41MMIDRangesOverlap(accepted[i].issued.ptr, accepted[i].buf.n, b.ptr, b.n) {
				accepted[i].retain = true
			}
		}
	}
	if c == nil || len(statuses)+len(rows) == 0 || len(statuses) > 64 || len(rows) > 32 || len(accepted) != len(statuses) {
		return nil, nil, false, fail("unissued, duplicate or oversized deferred batch")
	}
	if err = ops.admit(c); err != nil {
		return nil, nil, false, err
	}
	ptrs := make([]unsafe.Pointer, 0, len(statuses)+len(rows))
	sizes := make([]int, 0, cap(ptrs))
	offsets := make([]int, 0, cap(ptrs))
	total, verdictCount := 0, 0
	appendRegion := func(b *cudaBuf, size int) bool {
		if size <= 0 || total > math.MaxInt-15 {
			return false
		}
		offset := (total + 15) &^ 15
		if size > math.MaxInt-offset {
			return false
		}
		for _, p := range ptrs {
			if p == b.ptr {
				return false
			}
		}
		ptrs, sizes, offsets = append(ptrs, b.ptr), append(sizes, size), append(offsets, offset)
		total = offset + size
		return true
	}
	for i, st := range statuses {
		a := accepted[i]
		issued := a.issued
		if a.retain || !cudaV41MMIDStatusDomain(st) || st.Kind != issued.kind || st.Count != issued.count ||
			st.Site != issued.site || st.Records != issued.records || st.Record.Dtype != F32 ||
			st.Record.Layout != RowMajor || st.Record.Quant != nil || len(st.Record.Shape) != 1 {
			return nil, nil, false, fail("deferred status does not match its fresh issuance")
		}
		n := V41DeferredRecords(st)
		if st.Record.Shape[0] != n*4 {
			return nil, nil, false, fail("deferred status shape does not match its records")
		}
		b, invalid := c.cudaV41MMIDOperand(site, "status", st.Record, n*16)
		if invalid != nil {
			return nil, nil, false, invalid
		}
		if !appendRegion(b, n*16) {
			return nil, nil, false, fail("deferred status image extent or alias")
		}
		verdictCount += n // at most 64 descriptors * 8 records
	}
	for _, row := range rows {
		elements, ok := cudaV41MMIDReadElements(row)
		if !ok {
			return nil, nil, false, fail("deferred row must be a bounded contiguous F32 row or panel")
		}
		n := elements * 4
		b, invalid := c.cudaV41MMIDOperand(site, "row", row, n)
		if invalid != nil {
			return nil, nil, false, invalid
		}
		if !appendRegion(b, n) {
			return nil, nil, false, fail("deferred row image extent or alias")
		}
	}
	raw := make([]byte, total)
	status, done := ops.read(raw, ptrs, sizes, offsets)
	fenced, complete = true, done
	setStatus(status)
	if err != nil || !complete {
		if err == nil {
			err = fail("native batch returned without known completion")
		}
		return nil, nil, false, err
	}
	verdicts = make([]error, 0, verdictCount)
	for i, st := range statuses {
		for r := 0; r < V41DeferredRecords(st); r++ {
			at := offsets[i] + r*16
			cause := decodeV41DeferredStatus(st, raw[at:at+16])
			var routed *V41RoutedSwiGLUFault
			if cause != nil && !errors.As(cause, &routed) {
				op := "V41RoutedSwiGLU"
				if st.Kind == V41DeferredBF16Publish {
					op = "V41BF16Publish"
				} else if st.Kind == V41DeferredF32Finite {
					op = "V41F32Finite"
				}
				cause = &BackendError{Backend: "cuda", Site: op, Class: "cuda-v41-deferred-status-failed", Err: cause, Recovered: cause, Message: cause.Error()}
			}
			verdicts = append(verdicts, cause)
		}
	}
	values = make([][]float32, len(rows))
	for i := range rows {
		// Return one flattened host slice per input tensor, not per panel row.
		values[i] = make([]float32, sizes[len(statuses)+i]/4)
		at := offsets[len(statuses)+i]
		for j := range values[i] {
			values[i][j] = math.Float32frombits(binary.LittleEndian.Uint32(raw[at+j*4:]))
		}
	}
	return verdicts, values, false, nil
}

// cudaV41MMIDReadElements admits only the contiguous row-major domains used
// by paired RoPE vectors and target-feature [tokens, hidden] panels. Tensor has
// no stride metadata; arbitrary ranks/layouts are deliberately not admitted.
// Bound multiplication before evaluating it; Numel may overflow for forged
// metadata. Storage ownership/capacity and status aliases remain the reader's
// responsibility, after collecting every issued status for safe settlement.
func cudaV41MMIDReadElements(row Tensor) (int, bool) {
	if row.Dtype != F32 || row.Layout != RowMajor || row.Quant != nil ||
		(len(row.Shape) != 1 && len(row.Shape) != 2) {
		return 0, false
	}
	limit := min(math.MaxInt32, math.MaxInt/4)
	elements := 1
	for _, dim := range row.Shape {
		if dim <= 0 || dim > limit/elements {
			return 0, false
		}
		elements *= dim
	}
	return elements, true
}
