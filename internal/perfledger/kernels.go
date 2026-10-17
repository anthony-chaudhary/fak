package perfledger

import (
	"fmt"
	"math"
	"sort"
)

const (
	// KernelSourceStructured marks a kernel table read from the backend's
	// timestamp fold through its C ABI, never parsed from stderr.
	KernelSourceStructured = "structured"
	// KernelPipeliningDisabledByProfile records that the timestamp profile
	// serializes batch submission, so kernel ms is exact but decode tok/s on a
	// profiled row is not the unprofiled token time.
	KernelPipeliningDisabledByProfile = "disabled-by-profile"
	// KernelOther is the folded row for every kernel past the selection.
	KernelOther = "other"
	// SelectedCoverageTarget is the GPU-time share the selection must reach.
	SelectedCoverageTarget = 0.95
	// MaxSelectedKernels caps the selection; with a flat profile the cap can
	// leave SelectedCoverage below the target, which a receipt must refuse.
	MaxSelectedKernels = 24
)

// KernelSample is one (kernel, groups) shape's GPU time inside a decode window.
type KernelSample struct {
	Kernel, Op string
	GX, GY     uint32
	Calls      uint64
	NS         float64
}

// KernelWindow is the decode-window input NewNativeKernels folds.
type KernelWindow struct {
	Backend         string
	Steps           int
	GPUNS           float64
	WindowWallNS    float64
	Truncated       uint64
	UntimedCompute  uint64
	UntimedTransfer uint64
	BarriersElided  uint64
	Samples         []KernelSample
}

// SelectedKernel is one selected kernel shape's per-token device time. Share
// is its fraction of the window's timed GPU time.
type SelectedKernel struct {
	Kernel        string  `json:"kernel"`
	Op            string  `json:"op"`
	Groups        string  `json:"groups"`
	CallsPerToken float64 `json:"calls_per_token"`
	GPUMSPerToken float64 `json:"gpu_ms_per_token"`
	USPerCall     float64 `json:"us_per_call"`
	Share         float64 `json:"share"`
}

// NativeKernels is the per-token GPU time of the kernels a native decode
// selected. SelectedKernels holds the smallest prefix, by GPU time, whose
// share reaches SelectedCoverageTarget (capped at MaxSelectedKernels), then one
// KernelOther row for the rest. SelectedCoverage is the prefix's share and
// excludes the KernelOther row. DeviceBusy is timed GPU time over the decode
// window's wall. Complete is false when any compute ran untimed, the timestamp
// pool truncated or a timed batch was lost, or barriers were elided; untimed
// transfers are reported only.
type NativeKernels struct {
	Source           string           `json:"source"`
	Backend          string           `json:"backend"`
	DecodeSteps      int              `json:"decode_steps"`
	GPUMSPerToken    float64          `json:"gpu_ms_per_token"`
	WallMSPerToken   float64          `json:"wall_ms_per_token"`
	DeviceBusy       float64          `json:"device_busy"`
	SelectedCoverage float64          `json:"selected_coverage"`
	Truncated        uint64           `json:"truncated"`
	UntimedCompute   uint64           `json:"untimed_compute"`
	UntimedTransfer  uint64           `json:"untimed_transfer"`
	BarriersElided   uint64           `json:"barriers_elided"`
	Complete         bool             `json:"complete"`
	Pipelining       string           `json:"pipelining"`
	SelectedKernels  []SelectedKernel `json:"selected_kernels"`
}

func finitePositive(v float64) bool {
	return v > 0 && !math.IsNaN(v) && !math.IsInf(v, 0)
}

// NewNativeKernels folds one request's decode-window kernel samples. It is nil
// when steps, GPU time or window wall are not positive and finite, when any
// sample is negative, NaN or infinite, or when no sample has a call.
func NewNativeKernels(w KernelWindow) *NativeKernels {
	if w.Steps <= 0 || !finitePositive(w.GPUNS) || !finitePositive(w.WindowWallNS) {
		return nil
	}
	samples := make([]KernelSample, 0, len(w.Samples))
	var total float64
	for _, s := range w.Samples {
		if s.NS < 0 || math.IsNaN(s.NS) || math.IsInf(s.NS, 0) {
			return nil
		}
		if s.Calls == 0 {
			continue
		}
		samples = append(samples, s)
		total += s.NS
	}
	if !finitePositive(total) {
		return nil
	}
	sort.SliceStable(samples, func(i, j int) bool {
		a, b := samples[i], samples[j]
		if a.NS != b.NS {
			return a.NS > b.NS
		}
		if a.Kernel != b.Kernel {
			return a.Kernel < b.Kernel
		}
		if a.GX != b.GX {
			return a.GX < b.GX
		}
		return a.GY < b.GY
	})
	steps := float64(w.Steps)
	n := &NativeKernels{
		Source:          KernelSourceStructured,
		Backend:         w.Backend,
		DecodeSteps:     w.Steps,
		GPUMSPerToken:   roundTo(w.GPUNS/steps/1e6, 1e6),
		WallMSPerToken:  roundTo(w.WindowWallNS/steps/1e6, 1e6),
		DeviceBusy:      roundTo(w.GPUNS/w.WindowWallNS, 1e6),
		Truncated:       w.Truncated,
		UntimedCompute:  w.UntimedCompute,
		UntimedTransfer: w.UntimedTransfer,
		BarriersElided:  w.BarriersElided,
		Complete:        w.Truncated == 0 && w.UntimedCompute == 0 && w.BarriersElided == 0,
		Pipelining:      KernelPipeliningDisabledByProfile,
	}
	row := func(kernel, op, groups string, calls uint64, ns float64) SelectedKernel {
		return SelectedKernel{
			Kernel:        kernel,
			Op:            op,
			Groups:        groups,
			CallsPerToken: roundTo(float64(calls)/steps, 1e6),
			GPUMSPerToken: roundTo(ns/steps/1e6, 1e6),
			USPerCall:     roundTo(ns/float64(calls)/1e3, 1e6),
			Share:         roundTo(ns/total, 1e9),
		}
	}
	var selectedNS float64
	i := 0
	for ; i < len(samples) && i < MaxSelectedKernels; i++ {
		if selectedNS/total >= SelectedCoverageTarget {
			break
		}
		s := samples[i]
		n.SelectedKernels = append(n.SelectedKernels, row(s.Kernel, s.Op, fmt.Sprintf("%dx%d", s.GX, s.GY), s.Calls, s.NS))
		selectedNS += s.NS
	}
	n.SelectedCoverage = roundTo(selectedNS/total, 1e9)
	if i < len(samples) {
		var calls uint64
		var ns float64
		for _, s := range samples[i:] {
			calls += s.Calls
			ns += s.NS
		}
		n.SelectedKernels = append(n.SelectedKernels, row(KernelOther, KernelOther, "", calls, ns))
	}
	return n
}

// WithKernels returns a copy carrying the decode kernel table; nil leaves the
// timing unchanged.
func (n *NativeTiming) WithKernels(k *NativeKernels) *NativeTiming {
	if n == nil || k == nil {
		return n
	}
	out := *n
	out.Kernels = k
	return &out
}

// KernelShare is one kernel's mean GPU-time share across the measured rows.
type KernelShare struct {
	Kernel string  `json:"kernel"`
	Share  float64 `json:"share"`
}

const topKernelsShown = 3

type kernelFold struct {
	gpuMSPerToken, coverage, deviceBusy []float64
	complete                            int
	share                               map[string]float64
}

func (f *kernelFold) add(k *NativeKernels) bool {
	if k == nil || !finitePositive(k.GPUMSPerToken) || k.DecodeSteps <= 0 {
		return false
	}
	f.gpuMSPerToken = append(f.gpuMSPerToken, k.GPUMSPerToken)
	f.coverage = append(f.coverage, k.SelectedCoverage)
	f.deviceBusy = append(f.deviceBusy, k.DeviceBusy)
	if k.Complete {
		f.complete++
	}
	if f.share == nil {
		f.share = map[string]float64{}
	}
	for _, s := range k.SelectedKernels {
		if s.Kernel != KernelOther {
			f.share[s.Kernel] += s.Share
		}
	}
	return true
}

func (f *kernelFold) top(measured int) []KernelShare {
	if measured <= 0 || len(f.share) == 0 {
		return nil
	}
	out := make([]KernelShare, 0, len(f.share))
	for kernel, share := range f.share {
		out = append(out, KernelShare{Kernel: kernel, Share: roundTo(share/float64(measured), 10000)})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Share != out[j].Share {
			return out[i].Share > out[j].Share
		}
		return out[i].Kernel < out[j].Kernel
	})
	if len(out) > topKernelsShown {
		out = out[:topKernelsShown]
	}
	return out
}
