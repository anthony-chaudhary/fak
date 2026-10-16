package model

import "testing"

// fak-test:runtime fast est=5s
func TestSessionKVEvictOversizedRange(t *testing.T) {
	cfg := pagedEvictCfg()
	m := NewSynthetic(cfg)
	all := []int{3, 17, 5}
	a, b := m.NewSession(), m.NewSession()
	defer a.Close()
	defer b.Close()
	a.Prefill(all)
	b.Prefill(all[:1])
	if got := a.evictKV(1, int(^uint(0)>>1)); got != len(all)-1 {
		t.Fatalf("removed=%d", got)
	}
	if a.Cache.Len() != b.Cache.Len() {
		t.Fatal("survivor length differs")
	}
	assertFloat32BitsEqual(t, "post-eviction actual session step", b.Step(23), a.Step(23))
}

// fak-test:runtime fast est=5s
func TestPagedKVEvictOversizedRange(t *testing.T) {
	cfg := pagedEvictCfg()
	m := NewSynthetic(cfg)
	s := m.NewSession()
	defer s.Close()
	all := []int{3, 17, 5}
	s.Prefill(all)
	pool := NewPagedKVPoolWithRaw(cfg, 2)
	seq := snapshotCacheToPaged(pool, s.Cache)
	defer seq.Free()
	if got := seq.Evict(1, int(^uint(0)>>1), cfg); got != len(all)-1 {
		t.Fatalf("removed=%d", got)
	}
	s.Cache.Evict(1, len(all)-1)
	if seq.Len() != s.Cache.Len() {
		t.Fatal("paged survivor length differs")
	}
	for l := 0; l < cfg.NumLayers; l++ {
		assertFloat32BitsEqual(t, "K", s.Cache.K[l], seq.GatherK(l))
		assertFloat32BitsEqual(t, "Kraw", s.Cache.Kraw[l], seq.GatherKraw(l))
		assertFloat32BitsEqual(t, "V", s.Cache.V[l], seq.GatherV(l))
	}
}

// fak-test:runtime fast est=5s
func TestPagedGLMDsaEvictOversizedRange(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensors(t, false)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatal(err)
	}
	s := m.NewSession()
	defer s.Close()
	all := []int{3, 17, 5}
	s.PrefillNoLogits(all)
	paged, err := GLMDsaKVCacheToPaged(s.Cache, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got := paged.Evict(1, int(^uint(0)>>1)); got != len(all)-1 {
		t.Fatalf("removed=%d", got)
	}
	s.Cache.Evict(1, len(all)-1)
	assertGLMDsaCachesEqual(t, "oversized range", s.Cache, paged.ToKVCache())
}
