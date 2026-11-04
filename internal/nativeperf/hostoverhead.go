package nativeperf

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// Host-overhead gap reasons: the closed machine-readable vocabulary that
// Repetition.HostOverheadGaps entries draw from. Each entry names exactly one
// decomposition field that stayed absent and why the collector could not
// observe it; an absent field with no gap entry still reads as unknown, never
// as a fabricated zero.
const (
	// HostOverheadGapDisabled: the source counter was never kept in this
	// process (kernel timestamp profile off, FAK_H2D_CENSUS unset).
	HostOverheadGapDisabled = "disabled"
	// HostOverheadGapUnobserved: the source exists but did not cover every
	// decode window of the request (overlap, counter reset, partial sync
	// coverage, inverted kernel wall).
	HostOverheadGapUnobserved = "unobserved"
	// HostOverheadGapUnsupported: no backend running in this process implements
	// the source (no capture-once graph backend for the replay witness).
	HostOverheadGapUnsupported = "unsupported"
)

// hostOverheadGapFields is the closed set of decomposition field names a gap
// entry may name, spelled exactly as the receipt's JSON keys.
var hostOverheadGapFields = map[string]bool{
	"host_phase_us":                true,
	"kernel_launch_count":          true,
	"inter_kernel_gap_sum_us":      true,
	"h2d_copy_bytes":               true,
	"sync_roundtrips":              true,
	"graph_replay_engaged_tokens":  true,
	"graph_replay_degraded_tokens": true,
}

// HasHostOverhead reports whether the repetition records any v3 host-overhead
// decomposition field or gap reason. Absent fields stay absent: this is
// "unknown", never a measured zero.
func (rep Repetition) HasHostOverhead() bool {
	return rep.HostPhaseUs != nil || rep.KernelLaunchCount != nil || rep.InterKernelGapSumUs != nil ||
		rep.H2DCopyBytes != nil || rep.SyncRoundtrips != nil ||
		rep.GraphReplayEngagedTokens != nil || rep.GraphReplayDegradedTokens != nil ||
		len(rep.HostOverheadGaps) > 0
}

// hostOverheadProblems checks one recorded decomposition value set: times must
// be finite and non-negative. uint64 counts and bytes cannot be negative. Gap
// entries must match the closed "<field>=<reason>" grammar and may never name a
// field the repetition actually records.
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
	recorded := map[string]bool{
		"host_phase_us":                rep.HostPhaseUs != nil,
		"kernel_launch_count":          rep.KernelLaunchCount != nil,
		"inter_kernel_gap_sum_us":      rep.InterKernelGapSumUs != nil,
		"h2d_copy_bytes":               rep.H2DCopyBytes != nil,
		"sync_roundtrips":              rep.SyncRoundtrips != nil,
		"graph_replay_engaged_tokens":  rep.GraphReplayEngagedTokens != nil,
		"graph_replay_degraded_tokens": rep.GraphReplayDegradedTokens != nil,
	}
	for _, gap := range rep.HostOverheadGaps {
		field, reason, ok := strings.Cut(gap, "=")
		validReason := reason == HostOverheadGapDisabled || reason == HostOverheadGapUnobserved || reason == HostOverheadGapUnsupported
		if !ok || !hostOverheadGapFields[field] || !validReason {
			return fmt.Errorf("host_overhead_gaps entry %q violates the closed field/reason grammar", gap)
		}
		if recorded[field] {
			return fmt.Errorf("host_overhead_gaps names recorded field %q", field)
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
	GraphReplayEngaged    *float64 // mean graph-replayed tokens over recording repetitions
	GraphReplayDegraded   *float64 // mean degraded-to-per-layer tokens over recording repetitions
}

// HostOverheadSummary returns the receipt-level view of the v3 decomposition.
// A receipt without any decomposition data yields an all-nil summary.
func (r ExperimentReceipt) HostOverheadSummary() HostOverheadSummary {
	var s HostOverheadSummary
	sums := [7]float64{}
	n := [7]int{}
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
		if rep.GraphReplayEngagedTokens != nil {
			add(5, float64(*rep.GraphReplayEngagedTokens))
		}
		if rep.GraphReplayDegradedTokens != nil {
			add(6, float64(*rep.GraphReplayDegradedTokens))
		}
	}
	means := [7]*float64{}
	for i := range means {
		if n[i] > 0 {
			m := sums[i] / float64(n[i])
			means[i] = &m
		}
	}
	s.HostPhaseUs, s.KernelLaunchCount, s.InterKernelGapSumUs, s.H2DCopyBytes, s.SyncRoundtrips,
		s.GraphReplayEngaged, s.GraphReplayDegraded =
		means[0], means[1], means[2], means[3], means[4], means[5], means[6]
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
// repetition's v3 decomposition, setting every field the collector actually
// measured and recording a closed-vocabulary gap reason for every field it
// could not observe — an unobserved field stays absent, never a fabricated
// zero:
//
//   - host_phase_us from process CPU inside the decode loop;
//   - kernel_launch_count from the kernel profile's launch census;
//   - inter_kernel_gap_sum_us from the kernel profile's window wall minus its
//     GPU-busy sum (the timed region's time outside device work, where host
//     launch/setup overhead shows up);
//   - h2d_copy_bytes from the FAK_H2D_CENSUS counters diffed around the
//     windows;
//   - sync_roundtrips from observed host waits on device completion;
//   - graph_replay_engaged_tokens / graph_replay_degraded_tokens from the
//     graph-replay engagement witness rolled up over the windows.
func (rep *Repetition) FillHostOverhead(t DecodeHostWorkTotals) {
	if rep == nil {
		return
	}
	var gaps []string
	if t.CPU != nil {
		us := float64(*t.CPU) / float64(time.Microsecond)
		rep.HostPhaseUs = &us
	} else {
		gaps = append(gaps, "host_phase_us="+HostOverheadGapUnobserved)
	}
	if t.Kernels != nil {
		var launches uint64
		for _, row := range t.Kernels.Rows {
			launches += row.Calls
		}
		rep.KernelLaunchCount = &launches
		if hostOverheadFinite(t.Kernels.GPUNS) && hostOverheadFinite(t.Kernels.WallNS) &&
			t.Kernels.GPUNS > 0 && t.Kernels.WallNS >= t.Kernels.GPUNS {
			gapUs := (t.Kernels.WallNS - t.Kernels.GPUNS) / 1000
			rep.InterKernelGapSumUs = &gapUs
		} else {
			gaps = append(gaps, "inter_kernel_gap_sum_us="+HostOverheadGapUnobserved)
		}
	} else {
		gaps = append(gaps,
			"kernel_launch_count="+HostOverheadGapDisabled,
			"inter_kernel_gap_sum_us="+HostOverheadGapDisabled)
	}
	if t.SyncsObserved {
		var syncs uint64
		for _, s := range t.SyncSites {
			syncs += s.Syncs
		}
		rep.SyncRoundtrips = &syncs
	} else {
		gaps = append(gaps, "sync_roundtrips="+HostOverheadGapUnobserved)
	}
	if t.H2DCopyBytes != nil {
		rep.H2DCopyBytes = t.H2DCopyBytes
	} else if t.H2DCensusDisabled {
		gaps = append(gaps, "h2d_copy_bytes="+HostOverheadGapDisabled)
	} else {
		gaps = append(gaps, "h2d_copy_bytes="+HostOverheadGapUnobserved)
	}
	if t.GraphReplayEngaged != nil && t.GraphReplayDegraded != nil {
		rep.GraphReplayEngagedTokens = t.GraphReplayEngaged
		rep.GraphReplayDegradedTokens = t.GraphReplayDegraded
	} else if !t.GraphBackendSupported {
		gaps = append(gaps,
			"graph_replay_engaged_tokens="+HostOverheadGapUnsupported,
			"graph_replay_degraded_tokens="+HostOverheadGapUnsupported)
	} else {
		gaps = append(gaps,
			"graph_replay_engaged_tokens="+HostOverheadGapUnobserved,
			"graph_replay_degraded_tokens="+HostOverheadGapUnobserved)
	}
	rep.HostOverheadGaps = gaps
}

func hostOverheadFinite(v float64) bool { return v >= 0 && !math.IsNaN(v) && !math.IsInf(v, 0) }
