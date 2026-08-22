package model

import (
	"errors"
	"testing"
)

func geometryTestConfig() Config {
	return Config{NumLayers: 2, NumKVHeads: 2, HeadDim: 4}
}

func TestPlanCacheGeometryRejectsOvercommit(t *testing.T) {
	_, err := planCacheGeometry(geometryTestConfig(), CacheGeometryRequest{ExpertRingBytes: 100, KVCapacityTokens: 4, DeviceBudgetBytes: 200})
	var rebuildErr *CacheRebuildError
	if !errors.As(err, &rebuildErr) || rebuildErr.Reason != CacheRebuildInvalidBudget {
		t.Fatalf("reason = %v, want %s", err, CacheRebuildInvalidBudget)
	}
}

func TestRebuildCacheGeometrySuccessInvalidatesContext(t *testing.T) {
	s := &Session{M: &Model{Cfg: geometryTestConfig()}, Cache: NewKVCache(geometryTestConfig())}
	s.Cache.pos = []int{0, 1}
	snapshot, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	oldModel := s.M
	report, err := s.RebuildCacheGeometry(CacheGeometryRequest{ExpertRingBytes: 64, KVCapacityTokens: 8, DeviceBudgetBytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	if s.M != oldModel {
		t.Fatal("rebuild reloaded model")
	}
	if err := snapshot.Restore(s); err == nil {
		t.Fatal("stale snapshot restored after rebuild")
	}
	if s.Cache.Len() != 0 || report.InvalidatedEntries != 2 {
		t.Fatalf("context len=%d invalidated=%d", s.Cache.Len(), report.InvalidatedEntries)
	}
	if got := cap(s.Cache.K[0]) / s.Cache.kvStride(); got != 8 {
		t.Fatalf("KV capacity=%d, want 8", got)
	}
	if s.expertRing.budget() != 64 || report.Status != "OK" {
		t.Fatalf("ring=%d status=%q", s.expertRing.budget(), report.Status)
	}
}

func TestRebuildCacheGeometryRejectsBusy(t *testing.T) {
	s := &Session{M: &Model{Cfg: geometryTestConfig()}, Cache: NewKVCache(geometryTestConfig())}
	s.cacheGeometryMu.RLock()
	defer s.cacheGeometryMu.RUnlock()
	_, err := s.RebuildCacheGeometry(CacheGeometryRequest{KVCapacityTokens: 1, DeviceBudgetBytes: 1 << 20})
	var rebuildErr *CacheRebuildError
	if !errors.As(err, &rebuildErr) || rebuildErr.Reason != CacheRebuildBusy {
		t.Fatalf("reason=%v", err)
	}
}

func TestRebuildCacheGeometryRollsBackAllocationFailure(t *testing.T) {
	cfg := geometryTestConfig()
	s := &Session{M: &Model{Cfg: cfg}, Cache: NewKVCache(cfg), ExpertRingBytes: 32, expertRing: newPagedRing(nil, 32)}
	s.Cache.Reserve(3)
	oldAllocator := cacheGeometryAllocator
	calls := 0
	cacheGeometryAllocator = func(s *Session, g CacheGeometry) (allocatedCacheGeometry, error) {
		calls++
		if calls == 1 {
			return allocatedCacheGeometry{}, errors.New("injected allocation failure")
		}
		return allocateCacheGeometry(s, g)
	}
	defer func() { cacheGeometryAllocator = oldAllocator }()
	report, err := s.RebuildCacheGeometry(CacheGeometryRequest{ExpertRingBytes: 64, KVCapacityTokens: 8, DeviceBudgetBytes: 1 << 20})
	var rebuildErr *CacheRebuildError
	if !errors.As(err, &rebuildErr) || rebuildErr.Reason != CacheRebuildFailed {
		t.Fatalf("reason=%v", err)
	}
	if s.expertRing.budget() != 32 || cap(s.Cache.K[0])/s.Cache.kvStride() != 3 {
		t.Fatalf("rollback geometry ring=%d kv=%d", s.expertRing.budget(), cap(s.Cache.K[0])/s.Cache.kvStride())
	}
	if report.Status != CacheRebuildFailed || s.cacheGeometryFailed {
		t.Fatalf("status=%q latched=%v", report.Status, s.cacheGeometryFailed)
	}
}

func TestRebuildCacheGeometryLatchesRollbackFailure(t *testing.T) {
	cfg := geometryTestConfig()
	s := &Session{M: &Model{Cfg: cfg}, Cache: NewKVCache(cfg)}
	oldAllocator := cacheGeometryAllocator
	cacheGeometryAllocator = func(*Session, CacheGeometry) (allocatedCacheGeometry, error) {
		return allocatedCacheGeometry{}, errors.New("injected failure")
	}
	defer func() { cacheGeometryAllocator = oldAllocator }()
	_, err := s.RebuildCacheGeometry(CacheGeometryRequest{KVCapacityTokens: 8, DeviceBudgetBytes: 1 << 20})
	var rebuildErr *CacheRebuildError
	if !errors.As(err, &rebuildErr) || rebuildErr.Reason != CacheRebuildRollbackFailed || !s.cacheGeometryFailed {
		t.Fatalf("reason=%v latched=%v", err, s.cacheGeometryFailed)
	}
}

func TestRebuildCacheGeometryNativeBackendPreservesOutputAndWeights(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	prompt := []int{3, 7, 11, 5}

	baseline, err := m.NewBackendSessionChecked(newRecordingQwen35Backend(m))
	if err != nil {
		t.Fatal(err)
	}
	before := baseline.Prefill(prompt)
	baseline.Close()

	s, err := m.NewBackendSessionChecked(newRecordingQwen35Backend(m))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	loaded := s.M
	report, err := s.RebuildCacheGeometry(CacheGeometryRequest{
		KVCapacityTokens:  16,
		DeviceBudgetBytes: 1 << 30,
	})
	if err != nil {
		t.Fatal(err)
	}
	after := s.Prefill(prompt)
	if cosine := cosineF32(t, before, after); cosine < Qwen35GDNParityCosineMin {
		t.Fatalf("native backend output cosine %.9f < %.3f after idle cache rebuild", cosine, Qwen35GDNParityCosineMin)
	}
	if s.M != loaded {
		t.Fatal("cache rebuild reloaded model weights")
	}
	if report.Status != "OK" || report.New.KVCapacityTokens != 16 {
		t.Fatalf("report = %+v", report)
	}
}
