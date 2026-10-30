package main

import (
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	computestrix "github.com/anthony-chaudhary/fak/internal/compute/strix"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

type serveSizingRing struct {
	RequestedContextTokens int                    `json:"requested_context_tokens"`
	ResolvedContextTokens  int                    `json:"resolved_context_tokens"`
	DeviceTotalBytes       int64                  `json:"device_total_bytes"`
	DeviceFreeBytes        int64                  `json:"device_free_bytes"`
	HostTotalBytes         int64                  `json:"host_total_bytes"`
	CarveoutBytes          int64                  `json:"carveout_bytes"`
	CarveoutKnown          bool                   `json:"carveout_known"`
	HostLoadPeak           *ggufload.HostLoadPeak `json:"host_load_peak,omitempty"`
	Fit                    string                 `json:"fit"`
	Reason                 string                 `json:"reason,omitempty"`
	CombinedFit            string                 `json:"combined_fit"`
	TargetCgroup           string                 `json:"target_cgroup"`
	Runtime                string                 `json:"runtime"`
	Limitations            []string               `json:"limitations"`
	Mode                   serveExpertRingMode    `json:"mode"`
	RingBytes              int64                  `json:"ring_bytes"`
	FloorBytes             int64                  `json:"floor_bytes"`
	CeilingBytes           int64                  `json:"ceiling_bytes"`
	DeviceBoundBytes       int64                  `json:"device_bound_bytes"`
	DeviceBoundKnown       bool                   `json:"device_bound_known"`
	HostBoundBytes         int64                  `json:"host_bound_bytes"`
	HostBoundKnown         bool                   `json:"host_bound_known"`
	DeviceAvailableBytes   int64                  `json:"device_available_bytes"`
	HostAvailableBytes     int64                  `json:"host_available_bytes"`
	HostMarginBytes        int64                  `json:"host_margin_bytes"`
	NonRingDeviceBytes     int64                  `json:"non_ring_device_bytes"`
	NonRingHostBytes       int64                  `json:"non_ring_host_bytes"`
	FixedHostPeakBytes     int64                  `json:"fixed_host_peak_bytes"`
	FixedHostPeakKnown     bool                   `json:"fixed_host_peak_known"`
	DenseHostWorkingBytes  int64                  `json:"dense_host_working_set_bytes"`
	HostPeakBytes          int64                  `json:"host_peak_bytes"`
	HostPeakKnown          bool                   `json:"host_peak_known"`
	HostPeakMarginBytes    int64                  `json:"host_peak_margin_bytes"`
	HostExpertRetention    int64                  `json:"host_expert_retention_bytes"`
	SharedPool             bool                   `json:"shared_pool"`
	PoolCapBytes           int64                  `json:"pool_cap_bytes"`
	GuardTrimBytes         int64                  `json:"guard_trim_bytes"`
	Demands                []serveSizingDemandRow `json:"demands"`
}

type serveSizingSnapshot struct {
	compute.Backend
	deviceTotal, deviceFree int64
	deviceKnown             bool
	ceiling                 int64
	ceilingKnown            bool
	weightBudget            int64
	weightBudgetEnabled     bool
	host                    serveExpertRingHost
}

func (s serveSizingSnapshot) DeviceMemory() (int64, int64, bool) {
	return s.deviceTotal, s.deviceFree, s.deviceKnown
}
func (s serveSizingSnapshot) HostMemory() (int64, int64, bool) {
	return s.host.Total, s.host.Avail, s.host.Known
}
func (s serveSizingSnapshot) DeviceLocalCeiling() (int64, bool) { return s.ceiling, s.ceilingKnown }
func (s serveSizingSnapshot) DeviceWeightBudget() (int64, bool) {
	return s.weightBudget, s.weightBudgetEnabled
}
func (s serveSizingSnapshot) Caps() compute.Caps {
	caps := s.Backend.Caps()
	caps.HostCapacityProbe = true
	return caps
}
func (s serveSizingSnapshot) DedicatedVRAMCarveout() (int64, bool) {
	return s.host.Carveout, s.host.CarveoutKnown
}

func newServeSizingRing() *serveSizingRing {
	return &serveSizingRing{Fit: "UNKNOWN", CombinedFit: "UNKNOWN", TargetCgroup: "UNKNOWN", Runtime: "UNRUN", Limitations: []string{
		"Header-derived planning snapshot; no tensor payloads loaded or listener started.",
		"Target service cgroup and all ancestor constraints require external evidence; this process membership is not the future service membership.",
		"Physical arena reservation, page-cache residency and additional runtime RAM caches are unmeasured; cgroup counters may omit GPU allocations on shared-memory devices.",
	}, Demands: []serveSizingDemandRow{}}
}

func buildServeSizingRingForFlags(ws *ggufload.WeightSource, be compute.Backend, sf *serveFlags) *serveSizingRing {
	r := newServeSizingRing()
	r.RequestedContextTokens = *sf.nativeContextTokens
	size, err := resolveServeExpertRingSize(derefString(sf.expertRingBytes), os.Getenv)
	if err != nil {
		r.Fit, r.Reason = "REFUSED", err.Error()
		return r
	}
	r.Mode = size.Mode
	if be == nil || *sf.cpuOffloadExperts || *sf.expertParallel != 1 || *sf.tensorParallel != 1 || !serveDeviceResidentQ4K(be) {
		r.Reason = "activated ring requires a single-rank resident quantized load route without host expert offload"
		return r
	}
	policy, err := resolveServeExecutionPolicy(derefString(sf.executionPolicy), derefString(sf.model), os.Getenv(ServeBringupDedicatedEnv), nil)
	if err != nil {
		r.Fit, r.Reason = "REFUSED", err.Error()
		return r
	}
	halo, _, err := computestrix.DetectPhysicalGFX1151("", "")
	if err != nil {
		r.Reason = err.Error()
		return r
	}
	if err := policy.admit(validateServeHaloCPUOffload(halo, *sf.cpuOffloadExperts, *sf.nCPUMoE, os.Getenv(agent.ExpertSpillEnv))); err != nil || !serveExpertRingEligible(be, policy.requireDeviceExecution(halo, be)) {
		r.Reason = "activated ring route not eligible: needs a device-only serve or a discrete CUDA backend with probeable capacity"
		return r
	}
	artifact := ggufload.ClassifyTensorQuant(ws.File.Tensors)
	opts := serveResidentQ4KLoadOptions(be, *sf.ggufPath, true, artifact)
	host := serveExpertRingHostProbe(*sf.ggufPath, be, opts, compute.HostSystemMemoryInfo)
	size = size.forHostTotal(host.Total, host.Known)
	return buildServeSizingRing(ws, be, *sf.ggufPath, *sf.nativeContextTokens, size, host, host.RSSAnonPeak > 0, opts)
}

func buildServeSizingRing(ws *ggufload.WeightSource, be compute.Backend, path string, tokens int, size serveExpertRingSize, host serveExpertRingHost, fixedKnown bool, opts []ggufload.Q4KLoadOption) *serveSizingRing {
	r := newServeSizingRing()
	r.RequestedContextTokens = tokens
	r.Mode, r.FixedHostPeakKnown = size.Mode, fixedKnown
	dt, df, dk := compute.DeviceMemoryInfo(be)
	r.DeviceTotalBytes, r.DeviceFreeBytes, r.HostTotalBytes = dt, df, host.Total
	r.HostAvailableBytes, r.CarveoutBytes, r.CarveoutKnown = host.Avail, host.Carveout, host.CarveoutKnown
	if !dk || !host.Known || !fixedKnown {
		r.Reason = "one or more capacity or fixed host-peak inputs are unavailable"
		return r
	}
	ceiling, ceilingKnown := compute.DeviceCeilingInfo(be)
	snapshot := serveSizingSnapshot{Backend: be, deviceTotal: dt, deviceFree: df, deviceKnown: dk, ceiling: ceiling, ceilingKnown: ceilingKnown, host: host}
	if budgeter, ok := be.(deviceWeightBudgetBackend); ok {
		snapshot.weightBudget, snapshot.weightBudgetEnabled = budgeter.DeviceWeightBudget()
	}
	weights, fit, err := serveNativeContextSizingInputs(ws, snapshot, false, false, 1, 0, path)
	if err != nil {
		r.Reason = err.Error()
		return r
	}
	if ringWeights, ok := serveExpertRingContextWeights(ws, snapshot, true, size, weights, fit); ok {
		weights = ringWeights
	}
	resolution, _, err := resolveServeNativeContext(ws, snapshot, weights, fit, tokens)
	if err != nil {
		r.Reason = err.Error()
		return r
	}
	r.RequestedContextTokens, r.ResolvedContextTokens = tokens, resolution.ResolvedTokens
	tokens = resolution.ResolvedTokens
	arm := resolveDeviceServeLoadArm(ws, snapshot, false, false)
	full, err := serveGGUFMemoryPlanForArm(ws, arm, tokens, fit, serveQ4KFitOptions(path, ws, snapshot, arm, fit)...)
	if err != nil {
		r.Reason = err.Error()
		return r
	}
	full = applyDeviceWeightBudget(full, snapshot)
	_, fullErr := refuseIfTooBigOnDevice(full, nil, snapshot, &fit)
	if !serveFullPlanFitTooBig(fullErr) {
		r.Reason = "full resident plan does not select the activated-expert fallback"
		return r
	}
	ring, selected, err := serveActivatedExpertRingPlacementSized(ws, snapshot, true, fullErr, tokens, fit, size, host)
	if err != nil || !selected {
		r.Fit, r.Reason = "REFUSED", fmt.Sprint(err)
		return r
	}
	s := ring.Sizing
	r.RingBytes, r.FloorBytes, r.CeilingBytes = ring.RingBytes, s.Floor, s.Ceiling
	r.DeviceBoundBytes, r.DeviceBoundKnown = s.DeviceBound, s.DeviceBoundKnown
	r.HostBoundBytes, r.HostBoundKnown = s.HostBound, s.HostBoundKnown
	r.DeviceAvailableBytes, r.HostAvailableBytes = s.DeviceAvail, s.HostAvail
	r.HostMarginBytes, r.NonRingDeviceBytes, r.NonRingHostBytes = s.HostMargin, s.NonRingDevice, s.NonRingHost
	r.FixedHostPeakBytes, r.SharedPool, r.PoolCapBytes, r.GuardTrimBytes = host.RSSAnonPeak, s.SharedPool, s.PoolCapBytes, s.GuardTrimBytes
	for _, d := range ring.Plan {
		r.Demands = append(r.Demands, serveSizingDemandRow{Class: string(d.Class), Bytes: d.Bytes, Detail: d.Detail, Scope: string(d.ScopeOrDefault()), DType: d.DType})
	}
	opts = append(append([]ggufload.Q4KLoadOption(nil), opts...), serveActivatedExpertRingLoadOptions(ring)...)
	opts = append(opts, serveActivatedExpertRingDenseOptionForHost(path, opts, serveHostFitBudgetFromReported(host.Total, host.Avail, host.Known, nil), os.Getenv, func() (int64, int64, bool) { return host.Total, host.Avail, host.Known }))
	effects := ggufload.ApplyQ4KLoadOptions(opts)
	r.DenseHostWorkingBytes, r.HostExpertRetention = effects.StreamedDenseBytes, effects.StreamedExpertBytes
	peak, peakErr := estimateServeNativeHostLoadPeak(path, false, opts)
	if peakErr == nil {
		r.HostLoadPeak = &peak
	}
	r.HostPeakBytes, r.HostPeakKnown, r.HostPeakMarginBytes = peak.PeakAnonBytes, peakErr == nil, serveHostPeakMarginBytes(os.Getenv)
	if peakErr == nil && host.Known {
		if err := ggufload.AdmitHostLoadPeak(peak, host.Avail, true, r.HostPeakMarginBytes); err != nil {
			r.Fit, r.Reason = "REFUSED", err.Error()
			return r
		}
	}
	if !dk || !host.Known || !fixedKnown || peakErr != nil {
		r.Reason = "one or more capacity or host-peak inputs are unavailable"
		return r
	}
	r.Fit = "PLANNED"
	return r
}
