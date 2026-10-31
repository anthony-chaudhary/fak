package model

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"testing"
)

// TestV41EngramClockPolicy pins #14143's opt-in replacement and its real typed
// attachment. It is a CPU contract witness, not device or performance evidence.
// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
func TestV41EngramClockPolicy(t *testing.T) {
	t.Parallel()
	t.Run("bounded second chance and payload lifetime", func(t *testing.T) {
		src := &v41CacheFakeEngram{rows: 16, rowBytes: 8}
		c, err := NewV41EngramRowCache(src, V41EngramRowCacheOptions{
			TableRows: src.rows, RowBytes: src.rowBytes, BudgetBytes: 3*8 + 7,
			CachePolicy: V41EngramCacheCLOCK,
		})
		if err != nil {
			t.Fatal(err)
		}
		if c.inner.clockRowBytes != 8 || c.inner.maxEntries != 3 || len(c.inner.order) != 0 {
			t.Fatal("CLOCK constructor did not select lazy, budget-rounded row capacity")
		}
		for _, row := range []int{0, 1, 2} {
			if _, err := c.Row(row); err != nil {
				t.Fatal(err)
			}
		}
		held, err := c.Row(0) // only row 0 earns a second chance
		if err != nil {
			t.Fatal(err)
		}
		if _, err := c.Row(3); err != nil {
			t.Fatal(err)
		}
		key := func(row int) v41ExpertKey { return v41ExpertKey{expert: row} }
		if _, ok := c.inner.entries[key(1)]; ok || c.inner.clockHand != 2 {
			t.Fatal("CLOCK failed to skip referenced row 0 and replace unreferenced row 1")
		}
		for _, row := range []int{0, 2, 3} {
			if _, ok := c.inner.Get(0, row); !ok {
				t.Fatalf("row %d missing before full-reference sweep", row)
			}
		}
		// All N slots referenced: N clears plus victim selection, not an
		// unbounded search or a claim of worst-case constant-time eviction.
		if _, err := c.Row(4); err != nil {
			t.Fatal(err)
		}
		if _, ok := c.inner.entries[key(2)]; ok || c.inner.clockHand != 0 {
			t.Fatal("all-referenced sweep did not wrap to the original hand")
		}
		for _, ref := range c.inner.cached {
			if ref != 0 {
				t.Fatal("full sweep left a referenced slot")
			}
		}
		for _, row := range []int{5, 6, 7} {
			if _, err := c.Row(row); err != nil {
				t.Fatal(err)
			}
		}
		if !bytes.Equal(held, make([]byte, 8)) {
			t.Fatal("cache eviction reused storage still borrowed by Row's caller")
		}
		st := c.inner.Stats()
		if st.ResidentEntries != 3 || st.ResidentBytes != 24 || len(c.inner.order) != 3 || len(c.inner.cached) != 3 {
			t.Fatalf("resident accounting exceeded row/byte bound: %+v", st)
		}
		// A replacement also owns its copy; malformed-width insertion must
		// refuse before changing accounting, residency, or the hand.
		payload := bytes.Repeat([]byte{7}, 8)
		if err := c.inner.Put(0, 7, payload); err != nil {
			t.Fatal(err)
		}
		payload[0] = 99
		if c.inner.entries[key(7)][0] != 7 {
			t.Fatal("replacement borrowed the caller's payload")
		}
		st, hand := c.inner.Stats(), c.inner.clockHand
		if err := c.inner.Put(0, 8, []byte{8}); err == nil || c.inner.Stats() != st || c.inner.clockHand != hand {
			t.Fatal("malformed CLOCK row changed cache state")
		}
		c.invalidateRows()
		if len(c.inner.entries) != 0 || len(c.inner.cached) != 0 || len(c.inner.order) != 0 || c.inner.resident != 0 || c.inner.clockHand != 0 || c.inner.clockRowBytes != 8 {
			t.Fatal("invalidation did not clear CLOCK residency/hand while retaining policy")
		}
		reads := src.reads
		if row, err := c.Row(7); err != nil || len(row) != 8 || src.reads != reads+1 {
			t.Fatal("invalidated CLOCK row did not re-fault")
		}
	})

	t.Run("legacy and invalid policies", func(t *testing.T) {
		for _, policy := range []V41EngramCachePolicy{"", V41EngramCacheLRU} {
			src := &v41CacheFakeEngram{rows: 8, rowBytes: 8}
			c, err := NewV41EngramRowCache(src, V41EngramRowCacheOptions{
				TableRows: 8, RowBytes: 8, BudgetBytes: 24, CachePolicy: policy,
			})
			if err != nil || c.inner.clockRowBytes != 0 {
				t.Fatalf("default/LRU changed: %v", err)
			}
		}
		generic := NewV41ExpertCache(V41ExpertCacheOptions{BudgetBytes: 24})
		for _, row := range []int{0, 1, 2} {
			if err := generic.Put(0, row, make([]byte, 8)); err != nil {
				t.Fatal(err)
			}
		}
		generic.Get(0, 0)
		if err := generic.Put(0, 3, make([]byte, 8)); err != nil {
			t.Fatal(err)
		}
		if _, ok := generic.Get(0, 1); ok || generic.clockRowBytes != 0 {
			t.Fatal("generic expert LRU victim changed")
		}
		src := &v41CacheFakeEngram{rows: 8, rowBytes: 8}
		if c, err := NewV41EngramRowCache(src, V41EngramRowCacheOptions{TableRows: 8, RowBytes: 8, BudgetBytes: 24, CachePolicy: "unknown"}); err == nil || c != nil || src.reads != 0 {
			t.Fatal("unknown row-cache policy was not refused before I/O")
		}
		m := &Model{}
		if err := m.AttachV41Engram(V41EngramAttachSpec{CachePolicy: "unknown"}, V41EngramLayout{}, nil, 24); !errors.Is(err, ErrV41NativeUnsupported) || m.Cfg.DeepSeekV41 != nil || m.V41EngramAttached() {
			t.Fatal("unknown attachment policy mutated the model")
		}
	})

	t.Run("typed attachment reaches real forward gather", func(t *testing.T) {
		var logits [][][]float32
		for _, policy := range []V41EngramCachePolicy{V41EngramCacheLRU, V41EngramCacheCLOCK} {
			m, layout := v41ReducedEngramModel(t)
			t.Cleanup(func() { m.CloseWeights() })
			packed := v41EngramTestPackedRows(int(layout.Rows[0]))
			binding := V41EngramArtifactBinding{Shard: "clock-fixture", Size: int64(len(packed)), Rows: int(layout.Rows[0]), RowBytes: V41EngramPackedRowBytes, Digest: v41EngramTestDigest(packed)}
			src, err := NewV41VerifiedPackedEngramRowSource(bytes.NewReader(packed), V41EngramShard{ID: binding.Shard, ExpectedID: binding.Shard, Size: binding.Size, Rows: binding.Rows}, binding)
			if err != nil {
				t.Fatal(err)
			}
			d := m.Cfg.DeepSeekV41
			spec := V41EngramAttachSpec{LayerIDs: d.EngramLayerIDs, MaxNgramSize: d.EngramMaxNgramSize, VocabSize: d.EngramVocabSize,
				NHeads: d.EngramNHeads, HeadDim: d.EngramHeadDim, PadTokenID: d.EngramPadTokenID, CompressedVocabSize: d.EngramCompressedVocabSize, CachePolicy: policy}
			if err := m.AttachV41Engram(spec, layout, []V41EngramRowSource{src}, 4*V41EngramPackedRowBytes); err != nil {
				t.Fatal(err)
			}
			cache := m.v41EngramStageFor().caches[0]
			if !m.V41EngramAttached() || (cache.inner.clockRowBytes > 0) != (policy == V41EngramCacheCLOCK) {
				t.Fatal("typed attachment failed to select the actual stage cache")
			}
			got := m.Forward([]int{1, 3, 5})
			if got == nil || len(got.Logits) != 3 || cache.Stats().RowsServed == 0 {
				t.Fatal("real forward did not consume the attached row cache")
			}
			logits = append(logits, got.Logits)
		}
		if !reflect.DeepEqual(logits[0], logits[1]) {
			t.Fatal("CLOCK changed the reduced native forward's logits")
		}
	})
}

// BenchmarkV41EngramClock compares actual Row calls at fixed width/capacity and
// deterministic traces. It includes allocation/read/copy overhead; it is not a
// device or whole-model timing. Run explicitly with -run '^$' -bench
// '^BenchmarkV41EngramClock$' -benchtime=100ms -count=3, recording Go/CPU/revision.
// cycle: key=i%(N+1); hot: even i->0, odd i->1+(i/2)%(N+1).
// fak-test:runtime medium est=5s lane=optin
func BenchmarkV41EngramClock(b *testing.B) {
	for _, capacity := range []int{4, 256, 4096} {
		for _, trace := range []string{"cycle", "hot"} {
			for _, policy := range []V41EngramCachePolicy{V41EngramCacheLRU, V41EngramCacheCLOCK} {
				b.Run(fmt.Sprintf("rows=%d/trace=%s/%s", capacity, trace, policy), func(b *testing.B) {
					src := &v41CacheFakeEngram{rows: capacity + 2, rowBytes: V41EngramPackedRowBytes}
					c, err := NewV41EngramRowCache(src, V41EngramRowCacheOptions{TableRows: src.rows, RowBytes: src.rowBytes,
						BudgetBytes: int64(capacity * src.rowBytes), CachePolicy: policy})
					if err != nil {
						b.Fatal(err)
					}
					for i := 0; i < capacity; i++ {
						if _, err := c.Row(i); err != nil {
							b.Fatal(err)
						}
					}
					before := c.Stats()
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						key := i % (capacity + 1)
						if trace == "hot" {
							key = 0
							if i%2 != 0 {
								key = 1 + (i/2)%(capacity+1)
							}
						}
						if _, err := c.Row(key); err != nil {
							b.Fatal(err)
						}
					}
					b.StopTimer()
					after := c.Stats()
					b.ReportMetric(float64(after.Hits-before.Hits)/float64(b.N), "hit/op")
					b.ReportMetric(float64(after.BytesRead-before.BytesRead)/float64(b.N), "read-B/op")
				})
			}
		}
	}
}
