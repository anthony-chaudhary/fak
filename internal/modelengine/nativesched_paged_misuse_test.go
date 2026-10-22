package modelengine

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=2s lane=default
func TestNativePreemptionMetricsPagedMisuse(t *testing.T) {
	const counter = "fak_sched_preempt_paged_misuse_releases_total"

	t.Run("swap_round_trip_counts_zero", func(t *testing.T) {
		s := NewNativeScheduler(model.NewSynthetic(SyntheticConfig()))
		s.SetKVPreemptionPolicy(NativePreemptionPolicy{
			Mode:        NativePreemptSwap,
			MaxBlocks:   1,
			BlockTokens: 128,
		})
		defer s.Close()

		_ = drainIssue31Requests(t, s, issue31Calls())
		st := s.KVPreemptionStats()
		if st.SwapPreemptions != 1 || st.Readmitted != 1 {
			t.Fatalf("swap path not exercised: SwapPreemptions=%d Readmitted=%d, want 1 and 1", st.SwapPreemptions, st.Readmitted)
		}
		if st.PagedMisuseReleases != 0 {
			t.Fatalf("PagedMisuseReleases = %d, want 0", st.PagedMisuseReleases)
		}
		var b strings.Builder
		s.WriteKVPreemptionMetrics(&b)
		out := b.String()
		for _, want := range []string{
			"# TYPE " + counter + " counter\n",
			counter + " 0\n",
		} {
			if !strings.Contains(out, want) {
				t.Fatalf("metrics missing %q", want)
			}
		}
	})

	t.Run("accrues_each_swap_pool", func(t *testing.T) {
		m := model.NewSynthetic(SyntheticConfig())
		s := NewNativeScheduler(m)
		defer s.Close()

		first := model.NewPagedKVPoolWithRaw(m.Cfg, 16)
		id := first.Alloc()
		first.Release(id)
		first.Release(id)
		first.Release(-1)
		second := model.NewPagedKVPoolWithRaw(m.Cfg, 16)
		second.Release(second.TotalBlocks())

		s.mu.Lock()
		s.accruePagedMisuseLocked(first)
		s.accruePagedMisuseLocked(second)
		s.mu.Unlock()

		if got := s.KVPreemptionStats().PagedMisuseReleases; got != 3 {
			t.Fatalf("PagedMisuseReleases = %d, want 3", got)
		}
		var b strings.Builder
		s.WriteKVPreemptionMetrics(&b)
		if want := counter + " 3\n"; !strings.Contains(b.String(), want) {
			t.Fatalf("metrics missing %q", want)
		}
	})
}
