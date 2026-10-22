package model

import (
	"bytes"
	"reflect"
	"sync"
	"testing"
	"time"
)

// prefetchOverlap counts reads across several checkpoint readers and records the peak in flight.
type prefetchOverlap struct {
	mu       sync.Mutex
	inflight int
	peak     int
}

type prefetchOverlapReader struct {
	r    *bytes.Reader
	live *prefetchOverlap
}

func (o *prefetchOverlapReader) ReadAt(p []byte, off int64) (int, error) {
	o.live.mu.Lock()
	o.live.inflight++
	o.live.peak = max(o.live.peak, o.live.inflight)
	o.live.mu.Unlock()
	time.Sleep(5 * time.Millisecond)
	n, err := o.r.ReadAt(p, off)
	o.live.mu.Lock()
	o.live.inflight--
	o.live.mu.Unlock()
	return n, err
}

// A joined prefetch returns only after every read it started has settled, and its handoffs
// still serve the demand faults that follow.
// fak-test:runtime fast est=200ms lane=default
func TestExpertCheckpointPrefetchJoinedSettlesItsReads(t *testing.T) {
	tier, src, _, _ := prefetchTestTier(t)
	names := prefetchTestNames(1, 3)
	n, join := tier.prefetchJoined(names)
	if n != len(names) {
		t.Fatalf("prefetchJoined started %d reads, want %d", n, len(names))
	}
	join()
	join()
	tier.mu.Lock()
	loading, ready := len(tier.loading), len(tier.ready)
	tier.mu.Unlock()
	if loading != 0 || ready != len(names) {
		t.Fatalf("after join: loading=%d ready=%d, want 0 and %d", loading, ready, len(names))
	}
	if src.peak < 2 {
		t.Fatalf("peak concurrent checkpoint reads = %d, want overlapped reads", src.peak)
	}
	for _, name := range names {
		if _, err := tier.fault(name); err != nil {
			t.Fatal(err)
		}
	}
	if st := tier.Stats(); st.Reads != len(names) || st.PrefetchHits != len(names) {
		t.Fatalf("reads=%d prefetch hits=%d, want %d of each", st.Reads, st.PrefetchHits, len(names))
	}
}

// FAK_V41_ROUTED_PREFETCH=1 reads a batched layer's checkpoint misses concurrently, serves every
// demand from those reads, leaves none in flight when the layer returns, and keeps the routed sum
// bit-identical to demand-only page-in.
// fak-test:runtime fast est=500ms lane=default
func TestV41RoutedPrefetchJoinsLayerReads(t *testing.T) {
	t.Setenv(v41LayerExpertBatchEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	var oracle []float32
	for _, arm := range []string{"", "1"} {
		t.Run("prefetch="+arm, func(t *testing.T) {
			t.Setenv(v41RoutedPrefetchEnv, arm)
			m := v41IncrementalExpertFixture(t, true, false)
			tier := NewExpertCheckpointTier(0)
			live := &prefetchOverlap{}
			names := func(e int) [3]string {
				stem := "ffn.experts." + itoa(e)
				return [3]string{layerName(0, stem+".w1.weight"), layerName(0, stem+".w3.weight"), layerName(0, stem+".w2.weight")}
			}
			var tripleBytes int64
			for p, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
				w := m.kqw[names(0)[p]]
				var blob []byte
				for e := 0; e < 3; e++ {
					blob = append(blob, m.kqw[names(e)[p]].raw...)
					delete(m.kqw, names(e)[p])
				}
				quant := ExpertCheckpointQ2K
				if p == 2 {
					quant = ExpertCheckpointQ3K
				}
				r := &prefetchOverlapReader{r: bytes.NewReader(blob), live: live}
				if err := tier.AddShard(r, int64(len(blob)), []FusedExpertTensor{{Name: proj, Layer: 0, Proj: proj, Arch: "deepseek41", Quant: quant, Experts: 3, Rows: w.out, Cols: w.in}}); err != nil {
					t.Fatal(err)
				}
				tripleBytes += int64(len(w.raw))
			}
			m.SetExpertCheckpoint(tier)
			s, err := m.NewBackendSessionChecked(newV41RoutedLayerTestBackend())
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			s.ExpertRingBytes = 2 * tripleBytes
			picks := []routePick{{expert: 0, weight: .25}, {expert: 2, weight: .5}}
			_, x := v41RoutedLayerFaultPicks(m)
			layer := s.v41RoutedLayerFunc()
			if layer == nil {
				t.Fatal("layer did not bind")
			}
			result, outcome, err := layer(0, picks, x)
			if outcome != v41RoutedExpertHandled || err != nil || s.BackendSessionClosed() {
				t.Fatalf("outcome=%v err=%v closed=%v, want handled", outcome, err, s.BackendSessionClosed())
			}
			tier.mu.Lock()
			loading := len(tier.loading)
			tier.mu.Unlock()
			st := tier.Stats()
			want := 3 * len(picks)
			if arm == "" {
				oracle = append([]float32(nil), result...)
				if st.Prefetched != 0 || live.peak != 1 {
					t.Fatalf("demand-only arm: prefetched=%d peak=%d, want 0 and serial reads", st.Prefetched, live.peak)
				}
				return
			}
			if !reflect.DeepEqual(result, oracle) {
				t.Fatal("prefetch changed the routed sum")
			}
			if st.Prefetched != want || st.Reads != want || st.PrefetchHits+st.Coalesced != want {
				t.Fatalf("prefetched=%d reads=%d hits=%d coalesced=%d, want %d reads all served by the prefetch",
					st.Prefetched, st.Reads, st.PrefetchHits, st.Coalesced, want)
			}
			if loading != 0 || live.peak < 2 {
				t.Fatalf("after the layer: loading=%d peak=%d, want joined and overlapped reads", loading, live.peak)
			}
			if s.expertRing != nil && (len(s.expertRing.holds) != 0 || s.ringDepth != 0) {
				t.Fatal("prefetch left demand holds behind")
			}
		})
	}
}
