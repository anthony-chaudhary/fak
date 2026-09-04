package model

import (
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const (
	CacheRebuildBusy           = "CACHE_REBUILD_BUSY"
	CacheRebuildInvalidBudget  = "CACHE_REBUILD_INVALID_BUDGET"
	CacheRebuildFailed         = "CACHE_REBUILD_FAILED"
	CacheRebuildRollbackFailed = "CACHE_REBUILD_ROLLBACK_FAILED"
)

// CacheGeometry is the native session-owned memory split that can be rebuilt
// without reloading model weights. RecurrentBytes is derived from the model and
// includes the fixed recurrent matrices and complete short-convolution windows.
type CacheGeometry struct {
	ExpertRingBytes  int64
	KVCapacityTokens int
	KVBytes          int64
	RecurrentBytes   int64
	TotalBytes       int64
}

// CacheGeometryRequest describes one explicit idle maintenance transaction.
type CacheGeometryRequest struct {
	ExpertRingBytes   int64
	KVCapacityTokens  int
	DeviceBudgetBytes int64
}

// CacheRebuildReport is the witnessed terminal result of a rebuild.
type CacheRebuildReport struct {
	Old, New           CacheGeometry
	ReclaimedBytes     int64
	AllocatedBytes     int64
	InvalidatedEntries int
	Elapsed            time.Duration
	Status             string
}

// CacheRebuildError carries a stable operator-facing reason token.
type CacheRebuildError struct {
	Reason string
	Err    error
}

func (e *CacheRebuildError) Error() string {
	if e.Err == nil {
		return e.Reason
	}
	return e.Reason + ": " + e.Err.Error()
}
func (e *CacheRebuildError) Unwrap() error { return e.Err }

// cacheGeometryAllocator is replaceable only by same-package tests to witness
// allocation and rollback failures deterministically.
type allocatedCacheGeometry struct {
	cache *KVCache
	ring  *pagedRing
	halKV compute.KVStore
	qwen  *qwen35HALState
}

type cacheGeometryAlloc func(*Session, CacheGeometry) (allocatedCacheGeometry, error)

var cacheGeometryAllocator cacheGeometryAlloc = allocateCacheGeometry

func planCacheGeometry(cfg Config, req CacheGeometryRequest) (CacheGeometry, error) {
	if req.ExpertRingBytes < 0 || req.KVCapacityTokens < 1 || req.DeviceBudgetBytes < 1 {
		return CacheGeometry{}, &CacheRebuildError{Reason: CacheRebuildInvalidBudget, Err: errors.New("expert bytes must be non-negative and KV capacity and device budget must be positive")}
	}
	stride := int64(cfg.NumKVHeads) * int64(cfg.HeadDim)
	kv, ok := checkedMul(int64(cfg.NumLayers), int64(req.KVCapacityTokens), stride, 3, 4)
	if !ok {
		return CacheGeometry{}, &CacheRebuildError{Reason: CacheRebuildInvalidBudget, Err: errors.New("KV geometry overflows int64")}
	}
	recurrent, ok := recurrentGeometryBytes(cfg)
	if !ok {
		return CacheGeometry{}, &CacheRebuildError{Reason: CacheRebuildInvalidBudget, Err: errors.New("recurrent geometry overflows int64")}
	}
	total, ok := checkedAdd(req.ExpertRingBytes, kv, recurrent)
	if !ok || total > req.DeviceBudgetBytes {
		return CacheGeometry{}, &CacheRebuildError{Reason: CacheRebuildInvalidBudget, Err: fmt.Errorf("requested %d bytes exceeds device budget %d", total, req.DeviceBudgetBytes)}
	}
	return CacheGeometry{ExpertRingBytes: req.ExpertRingBytes, KVCapacityTokens: req.KVCapacityTokens, KVBytes: kv, RecurrentBytes: recurrent, TotalBytes: total}, nil
}

func checkedMul(v ...int64) (int64, bool) {
	r := int64(1)
	for _, n := range v {
		if n < 0 || (n != 0 && r > math.MaxInt64/n) {
			return 0, false
		}
		r *= n
	}
	return r, true
}
func checkedAdd(v ...int64) (int64, bool) {
	var r int64
	for _, n := range v {
		if n < 0 || r > math.MaxInt64-n {
			return 0, false
		}
		r += n
	}
	return r, true
}

func recurrentGeometryBytes(cfg Config) (int64, bool) {
	if !cfg.IsQwen35Hybrid() {
		return 0, true
	}
	layers := 0
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isLinearAttnLayer(l) {
			layers++
		}
	}
	recurrent, ok := checkedMul(int64(layers), int64(cfg.LinearNumValueHeads), int64(cfg.LinearKeyHeadDim), int64(cfg.LinearValueHeadDim), 4)
	if !ok {
		return 0, false
	}
	_, _, _, _, keyDim, valDim, convDim := cfg.linearAttnDims()
	_ = keyDim
	_ = valDim
	window, ok := checkedMul(int64(layers), int64(max(0, cfg.LinearConvKernelDim-1)), int64(convDim), 4)
	if !ok {
		return 0, false
	}
	return checkedAdd(recurrent, window)
}

func allocateCacheGeometry(s *Session, g CacheGeometry) (out allocatedCacheGeometry, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("allocate cache geometry: %v", recovered)
		}
	}()
	c := NewKVCache(s.M.Cfg)
	c.Reserve(g.KVCapacityTokens)
	if c.linear != nil {
		_, _, _, _, _, _, convDim := s.M.Cfg.linearAttnDims()
		rows := max(0, s.M.Cfg.LinearConvKernelDim-1)
		for i := range c.linear.layers {
			if len(c.linear.layers[i].recurrent) == 0 {
				continue
			}
			c.linear.layers[i].conv = make([][]float32, rows)
			for j := range c.linear.layers[i].conv {
				c.linear.layers[i].conv[j] = make([]float32, convDim)
			}
		}
	}
	out.cache = c
	out.ring = newPagedRing(s.Backend, g.ExpertRingBytes)
	if s.Backend != nil {
		out.halKV = newHALKVStore(s.Backend, s.M.Cfg)
		if s.M.Cfg.IsQwen35Hybrid() {
			staging := &Session{M: s.M, Backend: s.Backend}
			staging.initQwen35HALState(s.Backend.(Qwen35GDNBackend))
			out.qwen = staging.qwen35HAL
		}
	}
	return out, nil
}

func (s *Session) currentCacheGeometry() CacheGeometry {
	expert := s.ExpertRingBytes
	if s.expertRing != nil {
		expert = s.expertRing.budget()
	}
	capTokens := 0
	if s.Cache != nil && len(s.Cache.K) > 0 && s.Cache.kvStride() > 0 {
		capTokens = cap(s.Cache.K[0]) / s.Cache.kvStride()
	}
	g, err := planCacheGeometry(s.M.Cfg, CacheGeometryRequest{ExpertRingBytes: expert, KVCapacityTokens: max(1, capTokens), DeviceBudgetBytes: math.MaxInt64})
	if err != nil {
		return CacheGeometry{ExpertRingBytes: expert, KVCapacityTokens: capTokens}
	}
	g.KVCapacityTokens = capTokens
	g.KVBytes = int64(s.M.Cfg.NumLayers) * int64(capTokens) * int64(s.Cache.kvStride()) * 3 * 4
	g.TotalBytes = g.ExpertRingBytes + g.KVBytes + g.RecurrentBytes
	return g
}

// RebuildCacheGeometry atomically replaces native expert, KV, recurrent, and
// window pools while no forward operation is active. Model weights are untouched.
// Existing context is deliberately invalidated because allocator identities change.
func (s *Session) RebuildCacheGeometry(req CacheGeometryRequest) (report CacheRebuildReport, err error) {
	started := time.Now()
	if !s.cacheGeometryMu.TryLock() {
		return report, &CacheRebuildError{Reason: CacheRebuildBusy}
	}
	defer s.cacheGeometryMu.Unlock()
	if s.cacheGeometryFailed {
		return report, &CacheRebuildError{Reason: CacheRebuildRollbackFailed, Err: errors.New("session is latched in failed maintenance")}
	}
	target, err := planCacheGeometry(s.M.Cfg, req)
	if err != nil {
		return report, err
	}
	report.Old, report.New = s.currentCacheGeometry(), target
	report.InvalidatedEntries = s.Cache.Len()
	report.ReclaimedBytes = report.Old.TotalBytes
	report.AllocatedBytes = target.TotalBytes
	oldReq := CacheGeometryRequest{ExpertRingBytes: report.Old.ExpertRingBytes, KVCapacityTokens: max(1, report.Old.KVCapacityTokens), DeviceBudgetBytes: math.MaxInt64}
	oldTarget, _ := planCacheGeometry(s.M.Cfg, oldReq)

	if s.expertRing != nil {
		s.expertRing.freeAll()
	}
	if s.halKV != nil {
		s.halKV.Free()
		s.halKV = nil
	}
	if s.Backend != nil {
		if gr, ok := s.Backend.(interface{ GraphReset() }); ok {
			gr.GraphReset()
		}
	}
	s.halLogitsWarm = false
	s.closeQwen35HALState()
	s.Cache, s.expertRing = nil, nil

	allocated, allocErr := cacheGeometryAllocator(s, target)
	if allocErr != nil {
		rollback, rollbackErr := cacheGeometryAllocator(s, oldTarget)
		if rollbackErr != nil {
			s.cacheGeometryFailed = true
			report.Status = CacheRebuildRollbackFailed
			report.Elapsed = time.Since(started)
			return report, &CacheRebuildError{Reason: CacheRebuildRollbackFailed, Err: errors.Join(allocErr, rollbackErr)}
		}
		s.installCacheGeometry(rollback, oldTarget.ExpertRingBytes)
		report.Status = CacheRebuildFailed
		report.Elapsed = time.Since(started)
		return report, &CacheRebuildError{Reason: CacheRebuildFailed, Err: allocErr}
	}
	s.installCacheGeometry(allocated, target.ExpertRingBytes)
	s.cacheGeometryEpoch++
	report.Status = "OK"
	report.Elapsed = time.Since(started)
	return report, nil
}

func (s *Session) installCacheGeometry(allocated allocatedCacheGeometry, expertBytes int64) {
	s.Cache = allocated.cache
	s.expertRing = allocated.ring
	s.halKV = allocated.halKV
	s.qwen35HAL = allocated.qwen
	s.ExpertRingBytes = expertBytes
}
