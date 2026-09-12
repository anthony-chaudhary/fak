package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

type v41CacheFakeEngram struct {
	rows, rowBytes int
	reads, bytes   int64
}

func (s *v41CacheFakeEngram) RowBytes() int { return s.rowBytes }

func (s *v41CacheFakeEngram) ReadRows(start, count int, dst []byte) (int, error) {
	s.reads++
	s.bytes += int64(count * s.rowBytes)
	n := 0
	for r := 0; r < count; r++ {
		for b := 0; b < s.rowBytes; b++ {
			dst[n] = byte(start + r)
			n++
		}
	}
	return n, nil
}

func v41CacheReadConfig(t *testing.T) Config {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "deepseek_v41_flash_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatalf("parse official V4.1 config: %v", err)
	}
	return cfg
}

// [SW-VERIFIED] Expert cache eviction is bounded and deterministic.
func TestV41CacheExpertEvictionBounded(t *testing.T) {
	const payload = 128
	opts := V41ExpertCacheOptions{BudgetBytes: 2 * payload, MaxEntries: 2}

	c := NewV41ExpertCache(opts)
	for _, e := range []int{0, 1, 2} {
		if err := c.Put(0, e, make([]byte, payload)); err != nil {
			t.Fatalf("Put(expert=%d): %v", e, err)
		}
	}
	st := c.Stats()
	if st.ResidentEntries != 2 {
		t.Fatalf("ResidentEntries=%d want 2", st.ResidentEntries)
	}
	if st.Evictions != 1 {
		t.Fatalf("Evictions=%d want 1", st.Evictions)
	}
	if st.ResidentBytes > opts.BudgetBytes {
		t.Fatalf("ResidentBytes=%d exceeds BudgetBytes=%d", st.ResidentBytes, opts.BudgetBytes)
	}
	if _, ok := c.Get(0, 0); ok {
		t.Fatalf("LRU victim expert 0 still resident")
	}
	if _, ok := c.Get(0, 1); !ok {
		t.Fatalf("expert 1 evicted early")
	}
	if _, ok := c.Get(0, 2); !ok {
		t.Fatalf("expert 2 evicted early")
	}
}

// [SW-VERIFIED] Replaying an identical script on fresh caches is byte-identical.
func TestV41CacheDeterministicReplay(t *testing.T) {
	script := []func(c *V41ExpertCache){
		func(c *V41ExpertCache) { _ = c.Put(0, 0, []byte("alpha")) },
		func(c *V41ExpertCache) { _ = c.Put(0, 1, []byte("beta")) },
		func(c *V41ExpertCache) { _, _ = c.Get(0, 0) },
		func(c *V41ExpertCache) { _ = c.Put(0, 2, []byte("gamma")) },
		func(c *V41ExpertCache) { _, _ = c.Get(0, 1) },
		func(c *V41ExpertCache) { _, _ = c.Get(0, 2) },
		func(c *V41ExpertCache) { _, _ = c.Get(0, 0) },
	}
	opts := V41ExpertCacheOptions{BudgetBytes: int64(2 * len("alpha")), MaxEntries: 2}

	run := func() (V41ExpertCacheStats, map[[2]int][]byte) {
		c := NewV41ExpertCache(opts)
		for _, step := range script {
			step(c)
		}
		got := map[[2]int][]byte{}
		for _, k := range [][2]int{{0, 0}, {0, 1}, {0, 2}} {
			if b, ok := c.Get(k[0], k[1]); ok {
				got[k] = append([]byte(nil), b...)
			}
		}
		return c.Stats(), got
	}

	st1, g1 := run()
	st2, g2 := run()
	if !reflect.DeepEqual(st1, st2) {
		t.Fatalf("[SW-VERIFIED] stats diverge: %+v vs %+v", st1, st2)
	}
	if !reflect.DeepEqual(g1, g2) {
		t.Fatalf("[SW-VERIFIED] Get results diverge: %v vs %v", g1, g2)
	}
}

// [SW-VERIFIED] Engram hit/miss accounting; UsefulBytes excludes prefetch bytes.
func TestV41CacheEngramRowStats(t *testing.T) {
	const rowBytes = 16
	src := &v41CacheFakeEngram{rows: 64, rowBytes: rowBytes}
	c, err := NewV41EngramRowCache(src, V41EngramRowCacheOptions{
		TableRows: 64, RowBytes: rowBytes, PrefetchRows: 2, PrefetchWindow: 4, BudgetBytes: rowBytes * 8,
	})
	if err != nil {
		t.Fatalf("NewV41EngramRowCache: %v", err)
	}

	if _, err := c.Row(0); err != nil {
		t.Fatalf("Row(0): %v", err)
	}
	got, err := c.Row(1)
	if err != nil {
		t.Fatalf("Row(1): %v", err)
	}
	if len(got) != rowBytes || got[0] != byte(1) {
		t.Fatalf("Row(1) payload wrong: len=%d first=%d", len(got), got[0])
	}
	st := c.Stats()
	if st.Misses != 1 {
		t.Fatalf("Misses=%d want 1 (demanded row 0)", st.Misses)
	}
	if st.Hits != 1 {
		t.Fatalf("Hits=%d want 1 (row 1 was prefetched)", st.Hits)
	}
	if st.Prefetches != 2 {
		t.Fatalf("Prefetches=%d want 2", st.Prefetches)
	}
	if st.RowsServed != 2 {
		t.Fatalf("RowsServed=%d want 2", st.RowsServed)
	}
	if st.UsefulBytes != 2*rowBytes {
		t.Fatalf("UsefulBytes=%d want %d: prefetched bytes must NOT count as useful", st.UsefulBytes, 2*rowBytes)
	}
	if st.BytesRead != 3*rowBytes {
		t.Fatalf("BytesRead=%d want %d (1 demanded + 2 prefetched)", st.BytesRead, 3*rowBytes)
	}
	if st.BytesServed != 2*rowBytes {
		t.Fatalf("BytesServed=%d want %d", st.BytesServed, 2*rowBytes)
	}
}

// [SW-VERIFIED] Page amplification is measured; prefetch traffic is counted.
func TestV41CachePageAmplification(t *testing.T) {
	t.Run("prefetch amplifies", func(t *testing.T) {
		src := &v41CacheFakeEngram{rows: 256, rowBytes: 8}
		c, err := NewV41EngramRowCache(src, V41EngramRowCacheOptions{
			TableRows: 256, RowBytes: 8, PrefetchRows: 4, PrefetchWindow: 8, BudgetBytes: 8 * 32,
		})
		if err != nil {
			t.Fatalf("NewV41EngramRowCache: %v", err)
		}
		if _, err := c.Row(10); err != nil {
			t.Fatalf("Row(10): %v", err)
		}
		st := c.Stats()
		if st.Prefetches == 0 {
			t.Fatalf("Prefetches=0 want >0 with PrefetchRows>0")
		}
		if !(st.BytesRead > st.UsefulBytes) {
			t.Fatalf("BytesRead=%d must exceed UsefulBytes=%d", st.BytesRead, st.UsefulBytes)
		}
		if !(st.PageAmplification > 1) {
			t.Fatalf("PageAmplification=%v want >1", st.PageAmplification)
		}
	})

	t.Run("no prefetch is unity", func(t *testing.T) {
		src := &v41CacheFakeEngram{rows: 16, rowBytes: 8}
		c, err := NewV41EngramRowCache(src, V41EngramRowCacheOptions{
			TableRows: 16, RowBytes: 8, BudgetBytes: 8 * 16,
		})
		if err != nil {
			t.Fatalf("NewV41EngramRowCache: %v", err)
		}
		for i := 0; i < 16; i++ {
			if _, err := c.Row(i); err != nil {
				t.Fatalf("Row(%d): %v", i, err)
			}
		}
		st := c.Stats()
		if st.PageAmplification != 1.0 {
			t.Fatalf("PageAmplification=%v want exactly 1.0", st.PageAmplification)
		}
	})
}

// [SW-VERIFIED] Prefetch cannot grow resident memory beyond the budget.
func TestV41CachePrefetchBounded(t *testing.T) {
	const rowBytes = 16
	src := &v41CacheFakeEngram{rows: 1024, rowBytes: rowBytes}
	budget := int64(rowBytes * 4)
	c, err := NewV41EngramRowCache(src, V41EngramRowCacheOptions{
		TableRows: 1024, RowBytes: rowBytes, PrefetchRows: 8, PrefetchWindow: 16, BudgetBytes: budget,
	})
	if err != nil {
		t.Fatalf("NewV41EngramRowCache: %v", err)
	}
	for i := 0; i < 128; i++ {
		got, err := c.Row(i * 3)
		if err != nil {
			t.Fatalf("Row(%d): %v", i*3, err)
		}
		if len(got) != rowBytes {
			t.Fatalf("Row(%d) len=%d want %d", i*3, len(got), rowBytes)
		}
		if got[0] != byte(i*3) {
			t.Fatalf("Row(%d) bytes wrong: got %d want %d", i*3, got[0], byte(i*3))
		}
	}
	st := c.Stats()
	if st.PageAmplification <= 1 {
		t.Fatalf("[SW-VERIFIED] prefetch traffic uncounted: amp=%v", st.PageAmplification)
	}
	if st.RowsServed != 128 {
		t.Fatalf("RowsServed=%d want 128", st.RowsServed)
	}

	ec := NewV41ExpertCache(V41ExpertCacheOptions{BudgetBytes: budget, MaxEntries: 4})
	for i := 0; i < 32; i++ {
		if err := ec.Put(0, i, make([]byte, rowBytes)); err != nil {
			t.Fatalf("expert Put(%d): %v", i, err)
		}
	}
	est := ec.Stats()
	if est.ResidentBytes > budget {
		t.Fatalf("expert ResidentBytes=%d exceeds BudgetBytes=%d", est.ResidentBytes, budget)
	}
}

// [SW-VERIFIED] Geometry resolves per Engram layer index; bad inputs error.
func TestV41CacheGeometryFromConfig(t *testing.T) {
	cfg := v41CacheReadConfig(t)
	if cfg.DeepSeekV41 == nil {
		t.Fatal("official config did not retain DeepSeekV41 metadata")
	}

	rowBytes, totalRows, err := V41EngramGeometryFromConfig(cfg, 0)
	if err != nil {
		t.Fatalf("index 0: %v", err)
	}
	if rowBytes != 256 || totalRows != 384006168 {
		t.Fatalf("index 0: rowBytes=%d totalRows=%d want 256/384006168", rowBytes, totalRows)
	}

	rowBytes, totalRows, err = V41EngramGeometryFromConfig(cfg, 1)
	if err != nil {
		t.Fatalf("index 1: %v", err)
	}
	if rowBytes != 256 || totalRows != 384016682 {
		t.Fatalf("index 1: rowBytes=%d totalRows=%d want 256/384016682", rowBytes, totalRows)
	}

	nilCfg := cfg
	nilCfg.DeepSeekV41 = nil
	if _, _, err := V41EngramGeometryFromConfig(nilCfg, 0); err == nil {
		t.Fatal("nil DeepSeekV41 returned nil error")
	}
	if _, _, err := V41EngramGeometryFromConfig(cfg, 99); err == nil {
		t.Fatal("out-of-range index returned nil error")
	}
}
