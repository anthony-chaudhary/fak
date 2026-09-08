//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import (
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"
)

type cudaQwen4ExpPLEHashPlan struct {
	mu      sync.Mutex
	backend *cudaBackend

	inputIDs      unsafe.Pointer
	context       unsafe.Pointer
	cuSeqLens     unsafe.Pointer
	multipliers   unsafe.Pointer
	vocabSizes    unsafe.Pointer
	offsets       unsafe.Pointer
	rows          unsafe.Pointer
	tokens        int
	requests      int
	contextLen    int
	ngramSize     int
	headsPerNGram int
	heads         int
	boundary      int64

	dispatched bool
	inFlight   bool
	closed     bool
}

// PrepareQwen4ExpPLEHash validates and uploads one stable-pointer PLE hash
// plan. No prepared pointer is host-addressable, and Dispatch performs no
// allocation or transfer. Callers can therefore capture/replay Dispatch while
// keeping preparation and result inspection outside the forward interval.
func (c *cudaBackend) PrepareQwen4ExpPLEHash(batch Qwen4ExpPLEHashBatch, spec Qwen4ExpPLEHashSpec) (Qwen4ExpPLEHashPlan, error) {
	geometry, err := validateQwen4ExpPLEHash(batch, spec)
	if err != nil {
		return nil, err
	}
	if c.faultLatch != nil {
		if err := c.faultLatch.Admit("qwen4exp-ple-hash-prepare"); err != nil {
			return nil, err
		}
	}
	plan := &cudaQwen4ExpPLEHashPlan{
		backend: c, tokens: geometry.tokens, requests: geometry.requests,
		contextLen: geometry.contextLen, ngramSize: spec.NGramSize,
		headsPerNGram: spec.HeadsPerNGram, heads: geometry.heads,
		boundary: spec.BoundaryToken,
	}
	if geometry.tokens == 0 {
		return plan, nil
	}

	cudaMu.Lock()
	defer cudaMu.Unlock()
	allocate := func(elements, width int, site string) (unsafe.Pointer, error) {
		if elements <= 0 || width <= 0 || elements > int(^uint(0)>>1)/width {
			return nil, &Qwen4ExpPLEHashError{Stage: "prepare", Reason: site + " byte size overflow"}
		}
		bytes := elements * width
		ptr := C.fcuda_malloc(C.size_t(bytes))
		if ptr == nil {
			return nil, &Qwen4ExpPLEHashError{Stage: "prepare", Reason: fmt.Sprintf("%s CUDA allocation of %d bytes failed", site, bytes)}
		}
		return unsafe.Pointer(ptr), nil
	}
	alloc := func(dst *unsafe.Pointer, elements, width int, site string) error {
		ptr, err := allocate(elements, width, site)
		if err != nil {
			return err
		}
		*dst = ptr
		return nil
	}
	allocations := []struct {
		dst      *unsafe.Pointer
		elements int
		width    int
		site     string
	}{
		{&plan.inputIDs, len(batch.InputIDs), 8, "input_ids"},
		{&plan.context, len(batch.NGramContext), 8, "ngram_context"},
		{&plan.cuSeqLens, len(geometry.cuSeqLens), 4, "cu_seqlens"},
		{&plan.multipliers, len(spec.Multipliers), 8, "multipliers"},
		{&plan.vocabSizes, len(spec.HeadVocabSizes), 8, "vocab_sizes"},
		{&plan.offsets, len(spec.HeadOffsets), 8, "offsets"},
		{&plan.rows, geometry.tokens * geometry.heads, 8, "rows"},
	}
	for _, allocation := range allocations {
		if err := alloc(allocation.dst, allocation.elements, allocation.width, allocation.site); err != nil {
			plan.freeLocked()
			return nil, err
		}
	}

	copyI64 := func(dst unsafe.Pointer, values []int64) {
		C.fcuda_h2d(dst, unsafe.Pointer(unsafe.SliceData(values)), C.size_t(len(values)*8))
	}
	copyI64(plan.inputIDs, batch.InputIDs)
	copyI64(plan.context, batch.NGramContext)
	C.fcuda_h2d(plan.cuSeqLens, unsafe.Pointer(unsafe.SliceData(geometry.cuSeqLens)), C.size_t(len(geometry.cuSeqLens)*4))
	copyI64(plan.multipliers, spec.Multipliers)
	copyI64(plan.vocabSizes, spec.HeadVocabSizes)
	copyI64(plan.offsets, spec.HeadOffsets)
	return plan, nil
}

func (p *cudaQwen4ExpPLEHashPlan) Dispatch() (Qwen4ExpPLEHashDispatchReceipt, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return Qwen4ExpPLEHashDispatchReceipt{}, &Qwen4ExpPLEHashError{Stage: "dispatch", Reason: "plan is closed"}
	}
	if p.backend.faultLatch != nil {
		if err := p.backend.faultLatch.Admit("qwen4exp-ple-hash-dispatch"); err != nil {
			return Qwen4ExpPLEHashDispatchReceipt{}, err
		}
	}
	if p.tokens == 0 {
		p.dispatched = true
		return Qwen4ExpPLEHashDispatchReceipt{}, nil
	}

	cudaMu.Lock()
	defer cudaMu.Unlock()
	beforeLaunches := uint64(C.fcuda_qwen4exp_ple_hash_launches())
	beforeH2D := uint64(C.fcuda_h2dxfer_bytes())
	beforeD2H := uint64(C.fcuda_hostxfer_bytes())
	status := int(C.fcuda_qwen4exp_ple_hash_i64(
		(*C.int64_t)(p.inputIDs), (*C.int64_t)(p.context), (*C.int32_t)(p.cuSeqLens),
		(*C.int64_t)(p.multipliers), (*C.int64_t)(p.vocabSizes), (*C.int64_t)(p.offsets),
		(*C.int64_t)(p.rows), C.int(p.tokens), C.int(p.requests), C.int(p.contextLen),
		C.int(p.ngramSize), C.int(p.headsPerNGram), C.int(p.heads), C.int64_t(p.boundary),
	))
	receipt := Qwen4ExpPLEHashDispatchReceipt{
		KernelLaunches:    uint64(C.fcuda_qwen4exp_ple_hash_launches()) - beforeLaunches,
		HostToDeviceBytes: uint64(C.fcuda_h2dxfer_bytes()) - beforeH2D,
		DeviceToHostBytes: uint64(C.fcuda_hostxfer_bytes()) - beforeD2H,
	}
	if status != 0 {
		err := &Qwen4ExpPLEHashError{Stage: "dispatch", Reason: fmt.Sprintf("CUDA status %d", status)}
		if p.backend.faultLatch != nil {
			p.backend.faultLatch.ObserveError(err, "qwen4exp-ple-hash-dispatch")
		}
		return receipt, err
	}
	p.dispatched = true
	p.inFlight = true
	return receipt, nil
}

func (p *cudaQwen4ExpPLEHashPlan) ReadRows() ([]int64, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil, &Qwen4ExpPLEHashError{Stage: "read", Reason: "plan is closed"}
	}
	if !p.dispatched {
		return nil, &Qwen4ExpPLEHashError{Stage: "read", Reason: "plan has not been dispatched"}
	}
	if p.tokens == 0 {
		return []int64{}, nil
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if status := int(C.fcuda_qwen4exp_ple_hash_sync()); status != 0 {
		err := &Qwen4ExpPLEHashError{Stage: "read", Reason: fmt.Sprintf("CUDA stream status %d", status)}
		if p.backend.faultLatch != nil {
			p.backend.faultLatch.ObserveError(err, "qwen4exp-ple-hash-read")
		}
		return nil, err
	}
	rows := make([]int64, p.tokens*p.heads)
	C.fcuda_d2h(unsafe.Pointer(unsafe.SliceData(rows)), p.rows, C.size_t(len(rows)*8))
	atomic.AddUint64(&p.backend.fenceGen, 1)
	p.inFlight = false
	return rows, nil
}

func (p *cudaQwen4ExpPLEHashPlan) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var closeErr error
	if p.inFlight {
		if status := int(C.fcuda_qwen4exp_ple_hash_sync()); status != 0 {
			closeErr = &Qwen4ExpPLEHashError{Stage: "close", Reason: fmt.Sprintf("CUDA stream status %d", status)}
		}
	}
	p.freeLocked()
	p.closed = true
	p.inFlight = false
	return closeErr
}

// freeLocked returns every raw integer buffer to the CUDA pool. cudaMu is held.
func (p *cudaQwen4ExpPLEHashPlan) freeLocked() {
	for _, ptr := range []*unsafe.Pointer{
		&p.inputIDs, &p.context, &p.cuSeqLens, &p.multipliers,
		&p.vocabSizes, &p.offsets, &p.rows,
	} {
		if *ptr != nil {
			C.fcuda_free(*ptr)
			*ptr = nil
		}
	}
}

var _ Qwen4ExpPLEHashPreparer = (*cudaBackend)(nil)
var _ Qwen4ExpPLEHashPlan = (*cudaQwen4ExpPLEHashPlan)(nil)
