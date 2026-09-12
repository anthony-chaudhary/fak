package model

import (
	"errors"
	"math"
	"testing"
)

func geometryTestConfig() Config {
	return Config{NumLayers: 2, NumKVHeads: 2, HeadDim: 4}
}

func TestMemoryComponentPlanConservesBytes(t *testing.T) {
	components := []MemoryComponent{
		{Kind: MemoryComponentKV, Bytes: 96},
		{Kind: MemoryComponentRecurrent, Bytes: 32},
		{Kind: MemoryComponentExpert, Bytes: 16},
		{Kind: MemoryComponentScratch, Bytes: 8},
	}
	plan, err := PlanMemoryComponents(components, 152)
	if err != nil {
		t.Fatal(err)
	}
	if plan.TotalBytes != 152 || len(plan.Components) != len(components) {
		t.Fatalf("plan = %+v, want four components totaling 152 bytes", plan)
	}
	for i, want := range components {
		if plan.Components[i] != want {
			t.Fatalf("component %d = %+v, want %+v", i, plan.Components[i], want)
		}
	}
	components[0].Bytes = 1
	if plan.Components[0].Bytes != 96 {
		t.Fatal("plan retained the caller's mutable component slice")
	}
	empty, err := PlanMemoryComponents(nil, 0)
	if err != nil || empty.TotalBytes != 0 || len(empty.Components) != 0 {
		t.Fatalf("empty plan = %+v, err=%v", empty, err)
	}

	for _, tc := range []struct {
		name       string
		components []MemoryComponent
		budget     int64
	}{
		{"duplicate zero-byte kind", []MemoryComponent{{Kind: MemoryComponentKV}, {Kind: MemoryComponentKV}}, 1},
		{"over budget", []MemoryComponent{{Kind: MemoryComponentKV, Bytes: 2}}, 1},
		{"overflow", []MemoryComponent{{Kind: MemoryComponentKV, Bytes: math.MaxInt64}, {Kind: MemoryComponentScratch, Bytes: 1}}, math.MaxInt64},
		{"invalid budget", nil, -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := PlanMemoryComponents(tc.components, tc.budget)
			var rebuildErr *CacheRebuildError
			if !errors.As(err, &rebuildErr) || rebuildErr.Reason != CacheRebuildInvalidBudget {
				t.Fatalf("error = %v, want %s", err, CacheRebuildInvalidBudget)
			}
		})
	}

	cfg := geometryTestConfig()
	legacy, err := planCacheGeometry(cfg, CacheGeometryRequest{
		ExpertRingBytes:   17,
		KVCapacityTokens:  3,
		DeviceBudgetBytes: 1 << 20,
	})
	if err != nil {
		t.Fatal(err)
	}
	compat, err := PlanMemoryComponents([]MemoryComponent{
		{Kind: MemoryComponentKV, Bytes: legacy.KVBytes},
		{Kind: MemoryComponentRecurrent, Bytes: legacy.RecurrentBytes},
		{Kind: MemoryComponentExpert, Bytes: legacy.ExpertRingBytes},
	}, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if compat.TotalBytes != legacy.TotalBytes || legacy.TotalBytes != 17+576 {
		t.Fatalf("legacy total=%d component total=%d, want %d", legacy.TotalBytes, compat.TotalBytes, 17+576)
	}
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
