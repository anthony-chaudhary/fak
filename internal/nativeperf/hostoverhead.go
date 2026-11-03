package nativeperf

import (
	"fmt"
	"math"
	"time"
)

// HasHostOverhead reports whether the repetition records any v3 host-overhead
// decomposition field. Absent fields stay absent: this is "unknown", never a
// measured zero.
func (rep Repetition) HasHostOverhead() bool {
	return rep.HostPhaseUs != nil || rep.KernelLaunchCount != nil || rep.InterKernelGapSumUs != nil ||
		rep.H2DCopyBytes != nil || rep.SyncRoundtrips != nil
}

// hostOverheadProblems checks one recorded decomposition value set: times must
// be finite and non-negative. uint64 counts and bytes cannot be negative.
func (rep Repetition) hostOverheadProblems() error {
	for _, check := range []struct {
		name string
		v    *float64
	}{{"host_phase_us", rep.HostPhaseUs}, {"inter_kernel_gap_sum_us", rep.InterKernelGapSumUs}} {
		if check.v == nil {
			continue
		}
		if math.IsNaN(*check.v) || math.IsInf(*check.v, 0) || *check.v < 0 {
			return fmt.Errorf("%s must be finite and non-negative", check.name)
		}
	}
	return nil
}

func hostOverheadPresent(reps []Repetition) bool {
	for _, rep := range reps {
		if rep.HasHostOverhead() {
			return true
		}
	}
	return false
}

// HostOverheadSummary aggregates the per-repetition v3 decomposition across
// one receipt. Each mean covers only the repetitions that recorded the field;
// a field no repetition recorded stays nil — unknown, never zero.
type HostOverheadSummary struct {
	RecordedInRepetitions int      // repetitions carrying at least one decomposition field
	HostPhaseUs           *float64 // mean host-phase µs over recording repetitions
	KernelLaunchCount     *float64 // mean kernel launches over recording repetitions
	InterKernelGapSumUs   *float64 // mean inter-kernel gap sum µs over recording repetitions
	H2DCopyBytes          *float64 // mean host->device bytes over recording repetitions
	SyncRoundtrips        *float64 // mean synchronous round trips over recording repetitions
}

// HostOverheadSummary returns the receipt-level view of the v3 decomposition.
// A receipt without any decomposition data yields an all-nil summary.
func (r ExperimentReceipt) HostOverheadSummary() HostOverheadSummary {
	var s HostOverheadSummary
	sums := [5]float64{}
	n := [5]int{}
	for _, rep := range r.Repetitions {
		if rep.HasHostOverhead() {
			s.RecordedInRepetitions++
		}
		add := func(i int, v float64) { sums[i] += v; n[i]++ }
		if rep.HostPhaseUs != nil {
			add(0, *rep.HostPhaseUs)
		}
		if rep.KernelLaunchCount != nil {
			add(1, float64(*rep.KernelLaunchCount))
		}
		if rep.InterKernelGapSumUs != nil {
			add(2, *rep.InterKernelGapSumUs)
		}
		if rep.H2DCopyBytes != nil {
			add(3, float64(*rep.H2DCopyBytes))
		}
		if rep.SyncRoundtrips != nil {
			add(4, float64(*rep.SyncRoundtrips))
		}
	}
	means := [5]*float64{}
	for i := range means {
		if n[i] > 0 {
			m := sums[i] / float64(n[i])
			means[i] = &m
		}
	}
	s.HostPhaseUs, s.KernelLaunchCount, s.InterKernelGapSumUs, s.H2DCopyBytes, s.SyncRoundtrips =
		means[0], means[1], means[2], means[3], means[4]
	return s
}

// HostShareFromDecomposition returns the mean host share of the decode window
// over the repetitions that recorded both host_phase_us and
// decode_milliseconds: host_phase_us divided by the decode window wall in
// microseconds. ok is false when no repetition carries the pair — an absent
// measurement, never a fabricated zero. The share may exceed 1 on
// multi-threaded hosts, matching DecodeHostWorkTotals.HostShare semantics.
func (r ExperimentReceipt) HostShareFromDecomposition() (float64, bool) {
	var sum, n float64
	for _, rep := range r.Repetitions {
		if rep.HostPhaseUs == nil || rep.DecodeMilliseconds <= 0 {
			continue
		}
		share := *rep.HostPhaseUs / (rep.DecodeMilliseconds * 1e3)
		if math.IsNaN(share) || math.IsInf(share, 0) || share < 0 {
			continue
		}
		sum += share
		n++
	}
	if n == 0 {
		return 0, false
	}
	return sum / n, true
}

// FillHostOverhead projects one request's folded decode windows into the
// repetition's v3 decomposition, setting only what the collector actually
// measured: host_phase_us from process CPU inside the decode loop,
// kernel_launch_count from the kernel profile's launch census, and
// sync_roundtrips from observed host waits on device completion. Fields the
// source did not measure — including h2d_copy_bytes and
// inter_kernel_gap_sum_us, which this collector does not observe — stay
// absent rather than being filled with zeros.
func (rep *Repetition) FillHostOverhead(t DecodeHostWorkTotals) {
	if rep == nil {
		return
	}
	if t.CPU != nil {
		us := float64(*t.CPU) / float64(time.Microsecond)
		rep.HostPhaseUs = &us
	}
	if t.Kernels != nil {
		var launches uint64
		for _, row := range t.Kernels.Rows {
			launches += row.Calls
		}
		rep.KernelLaunchCount = &launches
	}
	if t.SyncsObserved {
		var syncs uint64
		for _, s := range t.SyncSites {
			syncs += s.Syncs
		}
		rep.SyncRoundtrips = &syncs
	}
}
