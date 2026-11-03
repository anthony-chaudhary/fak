// Package regime classifies decode-step performance profiles into an
// overhead regime from multi-signal inputs: inter-kernel gap sum vs
// kernel-execution sum, device-busy fraction, and memory-traffic counters
// when present. It replaces single-reading prose verdicts (one lesson note
// per model, e.g. PHL-127 "GPU busy 71.9/75.2 ms/tok") with a reproducible,
// deterministic classification that can be run over every perf receipt.
//
// Signals that are absent from a profile are reported as missing, never
// guessed: a sparse profile classifies as UNKNOWN rather than forcing a
// verdict from partial data.
package regime

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

// Regime is the decode-step overhead regime for one profile.
type Regime string

const (
	// RegimeLaunchBound: the host cannot keep the launch queue fed; the
	// device idles between kernels waiting on per-layer host setup.
	RegimeLaunchBound Regime = "LAUNCH_BOUND"
	// RegimeComputeBound: the device is busy most of the step; kernel
	// execution dominates (PHL-127 shape: 71.9/75.2 ms/tok busy).
	RegimeComputeBound Regime = "COMPUTE_BOUND"
	// RegimeMemBound: observed memory traffic saturates device bandwidth.
	RegimeMemBound Regime = "MEM_BOUND"
	// RegimeUnknown: signals are missing or contradictory; no verdict.
	RegimeUnknown Regime = "UNKNOWN"
)

// Tunable thresholds. They are constants by design: the classifier has no
// user-facing flags beyond --fail-on, and every threshold it applies is
// named in the emitted evidence so a reading is auditable against these.
const (
	// LaunchGapShareMin: inter-kernel gaps occupy at least this share of
	// (gap + execution) time -> LAUNCH_BOUND. At small batch sizes the
	// launch path is the suspect, so half-or-more idle-between-kernels is
	// already decisive.
	LaunchGapShareMin = 0.5
	// DeviceBusyFractionMin: device busy at least this fraction of the
	// step -> COMPUTE_BOUND. PHL-127 measured 71.9/75.2 = 0.956.
	DeviceBusyFractionMin = 0.8
	// DeviceIdleFractionMax: device busy below this fraction of the step
	// -> LAUNCH_BOUND (host overhead dominates the decode step). Between
	// this and DeviceBusyFractionMin the profile is mixed -> UNKNOWN.
	DeviceIdleFractionMax = 0.5
	// MemBWUtilizationMin: observed memory-bandwidth utilization at or
	// above this fraction of device peak -> MEM_BOUND.
	MemBWUtilizationMin = 0.8
)

// Profile carries the multi-signal inputs for one decode step. All times are
// microseconds; zero means "not reported" (never "zero work"). Fractions use
// -1 as the absent sentinel via NewProfile.
type Profile struct {
	BusyUsPerStep    float64 // device-busy time per decode step
	TotalUsPerStep   float64 // wall-clock time per decode step
	GapSumUs         float64 // sum of inter-kernel (host-visible) gaps per step
	ExecSumUs        float64 // sum of kernel execution time per step
	MemBytesPerStep  uint64  // memory traffic bytes per step
	PeakMemBWBytesPS float64 // device peak memory bandwidth, bytes/sec
	BusyFraction     float64 // direct device-busy fraction 0..1 (-1 = absent)
	MemBWUtilization float64 // direct mem-bandwidth utilization 0..1 (-1 = absent)
}

// NewProfile returns a Profile with all signals marked absent.
func NewProfile() Profile {
	return Profile{BusyFraction: -1, MemBWUtilization: -1}
}

// Verdict is the classification result plus the exact evidence used.
type Verdict struct {
	Regime   Regime
	Evidence []string // deterministic, threshold-named evidence lines
	Missing  []string // signal names absent from the profile
}

// usField routes a time-valued alias to its Profile field with a scale to us.
type usField struct {
	field string
	scale float64
}

// usAliases maps accepted JSON keys (lowercase) to their target field and the
// multiplier converting the reported unit to microseconds.
var usAliases = map[string]usField{
	"device_busy_ms_per_token": {"busy", 1e3},
	"gpu_busy_ms_per_token":    {"busy", 1e3},
	"busy_ms_per_token":        {"busy", 1e3},
	"device_busy_us_per_token": {"busy", 1},
	"device_busy_ns_per_token": {"busy", 1e-3},
	"total_ms_per_token":       {"total", 1e3},
	"wall_ms_per_token":        {"total", 1e3},
	"ms_per_token":             {"total", 1e3},
	"step_wall_ms":             {"total", 1e3},
	"total_us_per_token":       {"total", 1},
	"step_wall_us":             {"total", 1},
	"inter_kernel_gap_sum_us":  {"gap", 1},
	"kernel_gap_sum_us":        {"gap", 1},
	"gap_sum_us":               {"gap", 1},
	"host_gap_sum_us":          {"gap", 1},
	"inter_kernel_gap_sum_ms":  {"gap", 1e3},
	"kernel_execution_sum_us":  {"exec", 1},
	"kernel_exec_sum_us":       {"exec", 1},
	"exec_sum_us":              {"exec", 1},
	"device_work_sum_us":       {"exec", 1},
}

// byteAliases maps memory-traffic aliases to a bytes-per-step scale.
var byteAliases = map[string]float64{
	"mem_traffic_bytes_per_step":    1,
	"memory_traffic_bytes_per_step": 1,
	"mem_bytes_per_step":            1,
	"h2d_d2h_copy_bytes_per_step":   1,
	"mem_traffic_mib_per_step":      1 << 20,
	"memory_traffic_mib_per_step":   1 << 20,
}

// bwAliases maps peak-bandwidth aliases to a bytes/sec scale. GB/S is decimal
// (1e9), matching how device datasheets quote it.
var bwAliases = map[string]float64{
	"device_peak_bandwidth_gbps":    1e9,
	"peak_mem_bw_gbps":              1e9,
	"peak_memory_bandwidth_gbps":    1e9,
	"device_peak_bandwidth_bytes_s": 1,
	"peak_mem_bw_bytes_s":           1,
}

// fracAliases maps direct-fraction aliases to their Profile field.
var fracAliases = map[string]string{
	"device_busy_fraction":         "busyfrac",
	"gpu_busy_fraction":            "busyfrac",
	"busy_fraction":                "busyfrac",
	"mem_bw_utilization":           "memutil",
	"memory_bandwidth_utilization": "memutil",
}

// DecodeProfile extracts every regime signal present in a JSON receipt or
// profile document. It walks the whole tree (first occurrence of each alias
// wins) so receipts nest their numbers wherever they like. Unknown fields are
// ignored; absent signals stay absent. A v2 native-performance receipt's
// repetitions[].decode_milliseconds mean fills TotalUsPerStep when no total
// alias is present. Malformed JSON is an error; a valid-but-sparse document
// yields an all-absent Profile, which Classify reports as UNKNOWN.
func DecodeProfile(data []byte) (Profile, error) {
	var root any
	if err := json.Unmarshal(data, &root); err != nil {
		return NewProfile(), fmt.Errorf("decode profile: %w", err)
	}
	p := NewProfile()
	walkJSON(root, "", func(path string, key string, value any) {
		applyAlias(&p, key, value)
	})
	if p.TotalUsPerStep <= 0 {
		if m, ok := nestedValue(root, "repetitions"); ok {
			reps, _ := m.([]any)
			var sum, n float64
			for _, r := range reps {
				rm, ok := r.(map[string]any)
				if !ok {
					continue
				}
				if f, ok := finiteFloat(rm["decode_milliseconds"]); ok {
					sum += f
					n++
				}
			}
			if n > 0 {
				p.TotalUsPerStep = sum / n * 1e3
			}
		}
	}
	return p, nil
}

func applyAlias(p *Profile, key string, value any) {
	f, ok := finiteFloat(value)
	if !ok {
		return
	}
	if spec, ok := usAliases[strings.ToLower(key)]; ok {
		v := f * spec.scale
		if v < 0 || math.IsInf(v, 0) || math.IsNaN(v) {
			return
		}
		switch spec.field {
		case "busy":
			p.BusyUsPerStep = v
		case "total":
			p.TotalUsPerStep = v
		case "gap":
			p.GapSumUs = v
		case "exec":
			p.ExecSumUs = v
		}
		return
	}
	if scale, ok := byteAliases[strings.ToLower(key)]; ok && f >= 0 {
		p.MemBytesPerStep = uint64(f * scale)
		return
	}
	if scale, ok := bwAliases[strings.ToLower(key)]; ok && f > 0 {
		p.PeakMemBWBytesPS = f * scale
		return
	}
	if field, ok := fracAliases[strings.ToLower(key)]; ok && f >= 0 && f <= 1 {
		switch field {
		case "busyfrac":
			p.BusyFraction = f
		case "memutil":
			p.MemBWUtilization = f
		}
	}
}

// walkJSON visits every object key in the tree depth-first; arrays are
// descended element-wise. The visitor sees the full dotted path for context.
func walkJSON(node any, prefix string, visit func(path, key string, value any)) {
	switch v := node.(type) {
	case map[string]any:
		for k, child := range v {
			path := k
			if prefix != "" {
				path = prefix + "." + k
			}
			visit(path, k, child)
			walkJSON(child, path, visit)
		}
	case []any:
		for i, child := range v {
			walkJSON(child, fmt.Sprintf("%s[%d]", prefix, i), visit)
		}
	}
}

func nestedValue(root any, key string) (any, bool) {
	m, ok := root.(map[string]any)
	if !ok {
		return nil, false
	}
	v, ok := m[key]
	return v, ok
}

func finiteFloat(v any) (float64, bool) {
	f, ok := v.(float64)
	if !ok || math.IsNaN(f) || math.IsInf(f, 0) {
		return 0, false
	}
	return f, true
}

// Classify deterministically labels the profile. Signal priority is physical,
// not arbitrary: a launch-starved device cannot simultaneously saturate
// memory bandwidth, so the gap signal decides first, then bandwidth, then the
// coarse busy fraction. Conflicting coarser signals never override a finer
// one; anything undecided is UNKNOWN.
func Classify(p Profile) Verdict {
	v := Verdict{}

	gapShare := math.NaN()
	if p.GapSumUs > 0 && p.ExecSumUs > 0 {
		gapShare = p.GapSumUs / (p.GapSumUs + p.ExecSumUs)
	} else {
		if p.GapSumUs <= 0 {
			v.Missing = append(v.Missing, "kernel_gap_sum")
		}
		if p.ExecSumUs <= 0 {
			v.Missing = append(v.Missing, "kernel_exec_sum")
		}
	}

	busyFrac := math.NaN()
	switch {
	case p.BusyFraction >= 0:
		busyFrac = p.BusyFraction
	case p.BusyUsPerStep > 0 && p.TotalUsPerStep > 0:
		busyFrac = p.BusyUsPerStep / p.TotalUsPerStep
	default:
		if p.BusyUsPerStep <= 0 {
			v.Missing = append(v.Missing, "device_busy")
		}
		if p.TotalUsPerStep <= 0 {
			v.Missing = append(v.Missing, "step_total_time")
		}
	}

	memUtil := math.NaN()
	switch {
	case p.MemBWUtilization >= 0:
		memUtil = p.MemBWUtilization
	case p.MemBytesPerStep > 0 && p.PeakMemBWBytesPS > 0 && p.TotalUsPerStep > 0:
		stepSec := p.TotalUsPerStep / 1e6
		memUtil = float64(p.MemBytesPerStep) / (p.PeakMemBWBytesPS * stepSec)
	default:
		if p.MemBytesPerStep == 0 {
			v.Missing = append(v.Missing, "mem_traffic")
		}
		if p.PeakMemBWBytesPS <= 0 {
			v.Missing = append(v.Missing, "peak_mem_bw")
		}
	}

	switch {
	case !math.IsNaN(gapShare) && gapShare >= LaunchGapShareMin:
		v.Regime = RegimeLaunchBound
		v.Evidence = append(v.Evidence, fmt.Sprintf(
			"gap_share=%.3f (gap_sum=%.1fus exec_sum=%.1fus) >= LaunchGapShareMin(%.3f)",
			gapShare, p.GapSumUs, p.ExecSumUs, LaunchGapShareMin))
	case !math.IsNaN(memUtil) && memUtil >= MemBWUtilizationMin:
		v.Regime = RegimeMemBound
		if p.MemBWUtilization >= 0 {
			v.Evidence = append(v.Evidence, fmt.Sprintf(
				"mem_bw_utilization=%.3f (reported directly) >= MemBWUtilizationMin(%.3f)",
				memUtil, MemBWUtilizationMin))
		} else {
			v.Evidence = append(v.Evidence, fmt.Sprintf(
				"mem_bw_utilization=%.3f (traffic=%dB step=%.1fus peak=%.1fGB/s) >= MemBWUtilizationMin(%.3f)",
				memUtil, p.MemBytesPerStep, p.TotalUsPerStep, p.PeakMemBWBytesPS/1e9, MemBWUtilizationMin))
		}
	case !math.IsNaN(busyFrac):
		switch {
		case busyFrac >= DeviceBusyFractionMin:
			v.Regime = RegimeComputeBound
			v.Evidence = append(v.Evidence, fmt.Sprintf(
				"busy_fraction=%.3f (busy=%.1fus total=%.1fus) >= DeviceBusyFractionMin(%.3f)",
				busyFrac, p.BusyUsPerStep, p.TotalUsPerStep, DeviceBusyFractionMin))
		case busyFrac < DeviceIdleFractionMax:
			v.Regime = RegimeLaunchBound
			v.Evidence = append(v.Evidence, fmt.Sprintf(
				"busy_fraction=%.3f (busy=%.1fus total=%.1fus) < DeviceIdleFractionMax(%.3f): device idle most of the step",
				busyFrac, p.BusyUsPerStep, p.TotalUsPerStep, DeviceIdleFractionMax))
		default:
			v.Regime = RegimeUnknown
			v.Evidence = append(v.Evidence, fmt.Sprintf(
				"busy_fraction=%.3f is between DeviceIdleFractionMax(%.3f) and DeviceBusyFractionMin(%.3f): mixed regime",
				busyFrac, DeviceIdleFractionMax, DeviceBusyFractionMin))
		}
	default:
		v.Regime = RegimeUnknown
		v.Evidence = append(v.Evidence, "no decisive regime signal present; refusing to guess")
	}
	return v
}

// ParseRegime normalizes a regime name for --fail-on matching.
func ParseRegime(s string) (Regime, bool) {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case string(RegimeLaunchBound):
		return RegimeLaunchBound, true
	case string(RegimeComputeBound):
		return RegimeComputeBound, true
	case string(RegimeMemBound):
		return RegimeMemBound, true
	case string(RegimeUnknown):
		return RegimeUnknown, true
	}
	return RegimeUnknown, false
}
