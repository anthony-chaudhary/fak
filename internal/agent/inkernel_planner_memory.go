package agent

import (
	"errors"
	"log"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func (p *InKernelPlanner) refuseOversizeRequest(promptTokens, maxNew int) error {
	if p == nil || p.backend == nil || p.m == nil {
		return nil
	}
	plan, extrasErr := p.requestMemoryPlanWithExtras(promptTokens, maxNew, p.requestMTPHistoryEligible(maxNew))
	if extrasErr != nil {
		// A required runtime-extras bound is missing (unknown panel width or history
		// lifetime). Refuse before allocation with a typed capacity-unknown error rather
		// than trusting a silently cheap plan.
		return &InKernelCapacityError{
			Class:  compute.MemoryUnknown,
			Scope:  compute.MemoryScopeDevice,
			Site:   inKernelRuntimeExtrasUnknownSite,
			Detail: extrasErr.Error(),
			Cause:  extrasErr,
		}
	}
	if len(plan) == 0 {
		return nil
	}
	p.recordRequestMemoryPlan(promptTokens, maxNew, plan)
	if err := compute.RefuseMemoryPlanIfTooBig(p.backend, plan, inKernelRequestDeviceHeadroom); err != nil {
		var fe *compute.FitError
		if errors.As(err, &fe) {
			if p.maybeTrimRequestPressure(plan, "capacity_precheck") {
				p.recordRequestMemoryPlan(promptTokens, maxNew, plan)
				if retryErr := compute.RefuseMemoryPlanIfTooBig(p.backend, plan, inKernelRequestDeviceHeadroom); retryErr == nil {
					p.recordRequestPressureTrimResolved(plan, "capacity_precheck")
					return nil
				} else if errors.As(retryErr, &fe) {
					err = retryErr
				} else {
					return retryErr
				}
			}
			return p.capacityErrorFromFit(fe)
		}
		return err
	}
	if p.maybeTrimRequestPressure(plan, "low_margin") {
		p.recordRequestMemoryPlan(promptTokens, maxNew, plan)
	}
	return nil
}

type requestPressureFit struct {
	scope     compute.MemoryScope
	class     compute.MemoryClass
	want      int64
	budget    int64
	margin    int64
	freeKnown bool
}

func (p *InKernelPlanner) maybeTrimRequestPressure(plan compute.MemoryPlan, reason string) bool {
	fit, ok := p.requestDevicePressureFit(plan)
	if !ok || !shouldTrimRequestPressure(fit) {
		return false
	}
	trimmed := p.trimBackendIdlePools()
	p.recordRequestPressureTrim(fit, reason, trimmed, false)
	if trimmed {
		log.Printf("inkernel_chat pressure-trim model=%s backend=%s scope=%s class=%s reason=%s want=%d budget=%d margin=%d action=trim-idle-pools",
			p.modelID, p.backend.Name(), fit.scope, fit.class, reason, fit.want, fit.budget, fit.margin)
	}
	return trimmed
}

func (p *InKernelPlanner) recordRequestPressureTrimResolved(plan compute.MemoryPlan, reason string) {
	fit, ok := p.requestDevicePressureFit(plan)
	if !ok {
		return
	}
	p.recordRequestPressureTrim(fit, reason, false, true)
}

func (p *InKernelPlanner) requestDevicePressureFit(plan compute.MemoryPlan) (requestPressureFit, bool) {
	if p == nil || p.backend == nil {
		return requestPressureFit{}, false
	}
	total, free, known := compute.DeviceMemoryInfo(p.backend)
	if !known || total <= 0 || free < 0 {
		return requestPressureFit{}, false
	}
	want := plan.DeviceTotal()
	if want <= 0 {
		return requestPressureFit{}, false
	}
	budget := ApplyByteHeadroom(free, inKernelRequestDeviceHeadroom)
	return requestPressureFit{
		scope:     compute.MemoryScopeDevice,
		class:     primaryDemandClass(plan, compute.MemoryScopeDevice),
		want:      want,
		budget:    budget,
		margin:    budget - want,
		freeKnown: true,
	}, true
}

func shouldTrimRequestPressure(fit requestPressureFit) bool {
	if !fit.freeKnown || fit.want <= 0 {
		return false
	}
	if fit.margin < 0 {
		return true
	}
	return fit.margin <= requestPressureTrimMarginThreshold(fit.budget)
}

func requestPressureTrimMarginThreshold(budget int64) int64 {
	if budget <= 0 {
		return 0
	}
	threshold := int64(float64(budget) * inKernelRequestPressureTrimMarginRatio)
	if threshold < inKernelRequestPressureTrimMinMarginBytes {
		threshold = inKernelRequestPressureTrimMinMarginBytes
	}
	return threshold
}

func (p *InKernelPlanner) recordRequestPressureTrim(fit requestPressureFit, reason string, trimmed, resolved bool) {
	if p == nil {
		return
	}
	scope := strings.TrimSpace(string(fit.scope))
	if scope == "" {
		scope = string(compute.MemoryScopeDevice)
	}
	class := strings.TrimSpace(string(fit.class))
	if class == "" {
		class = string(compute.MemoryUnknown)
	}
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "unknown"
	}
	p.pressureTrimMu.Lock()
	if p.pressureTrim == nil {
		p.pressureTrim = map[requestPressureTrimKey]*requestPressureTrimStats{}
	}
	key := requestPressureTrimKey{scope: scope, class: class, reason: reason}
	st := p.pressureTrim[key]
	if st == nil {
		st = &requestPressureTrimStats{}
		p.pressureTrim[key] = st
	}
	if resolved {
		st.resolved++
	} else {
		st.attempts++
		if trimmed {
			st.trimmed++
		} else {
			st.noHooks++
		}
	}
	st.lastWantBytes = positiveInt64ToUint64(fit.want)
	st.lastBudgetBytes = positiveInt64ToUint64(fit.budget)
	st.lastMarginBytes = fit.margin
	p.pressureTrimMu.Unlock()
}

func positiveInt64ToUint64(v int64) uint64 {
	if v <= 0 {
		return 0
	}
	return uint64(v)
}

func (p *InKernelPlanner) recordRequestMemoryPlan(promptTokens, maxNew int, plan compute.MemoryPlan) {
	if p == nil || p.backend == nil {
		return
	}
	plannedTokens := promptTokens + maxNew
	if plannedTokens < promptTokens {
		plannedTokens = promptTokens
	}
	deviceTotal, deviceFree, deviceKnown := compute.DeviceMemoryInfo(p.backend)
	hostTotal, hostFree, hostKnown := compute.HostMemoryInfo(p.backend)
	stats := RequestMemoryStats{
		Observed:      len(plan) > 0,
		Backend:       p.backend.Name(),
		PromptTokens:  promptTokens,
		MaxNewTokens:  maxNew,
		PlannedTokens: plannedTokens,
		HeadroomRatio: inKernelRequestDeviceHeadroom,
		MemoryPlan:    requestMemoryDemands(plan),
		Capacities: []RequestMemoryCapacity{
			requestMemoryCapacity(string(compute.MemoryScopeDevice), deviceTotal, deviceFree, deviceKnown),
			requestMemoryCapacity(string(compute.MemoryScopeHost), hostTotal, hostFree, hostKnown),
		},
	}
	p.reqMemMu.Lock()
	p.lastReqMemory = stats
	p.reqMemMu.Unlock()
}

func requestMemoryDemands(plan compute.MemoryPlan) []RequestMemoryDemand {
	if len(plan) == 0 {
		return nil
	}
	out := make([]RequestMemoryDemand, 0, len(plan))
	for _, d := range plan {
		if d.Bytes <= 0 {
			continue
		}
		class := d.Class
		if class == "" {
			class = compute.MemoryUnknown
		}
		out = append(out, RequestMemoryDemand{
			Class:  string(class),
			Scope:  string(d.ScopeOrDefault()),
			DType:  d.DType,
			Bytes:  d.Bytes,
			Detail: d.Detail,
		})
	}
	return out
}

func requestMemoryCapacity(scope string, total, free int64, known bool) RequestMemoryCapacity {
	cap := RequestMemoryCapacity{
		Scope:      scope,
		TotalBytes: total,
		Known:      known,
		FreeKnown:  known && free >= 0,
	}
	if !known {
		cap.TotalBytes = 0
		return cap
	}
	if cap.FreeKnown {
		cap.FreeBytes = free
	}
	return cap
}

func (p *InKernelPlanner) requestMemoryPlan(promptTokens, maxNew int) compute.MemoryPlan {
	plan, _ := p.requestMemoryPlanWithExtras(promptTokens, maxNew, false)
	return plan
}

// requestMemoryPlanWithExtras composes the base context plan (#1049) with the Qwen3.8
// runtime-extras reservation (#13315): the simultaneous Vulkan prefill-panel peak and the
// optional retained-MTP target-hidden history. retainMTPHistory is REQUEST-LOCAL — it must be
// derived from configured support PLUS this request's sampling eligibility known before
// admission, never from planner-wide VulkanMTPEnabled() alone (a downgraded or non-decode
// request must not reserve retained-history bytes).
//
// It returns a typed RuntimeExtraCapacityUnknownError when the enabled extras cannot be
// bounded (missing panel width or history lifetime inputs); the caller refuses the request
// before allocation rather than trusting a silently cheap plan.
func (p *InKernelPlanner) requestMemoryPlanWithExtras(promptTokens, maxNew int, retainMTPHistory bool) (compute.MemoryPlan, error) {
	if p == nil || p.m == nil {
		return nil, nil
	}
	if promptTokens < 0 {
		promptTokens = 0
	}
	if maxNew < 0 {
		maxNew = 0
	}
	plannedTokens := promptTokens + maxNew
	if plannedTokens < promptTokens {
		plannedTokens = promptTokens
	}
	// Delegate to the single context auto-sizer (#1049) — the same function the serve boot
	// path uses — so boot and per-request build a byte-identical KV+scratch plan for the
	// same (model, tokens). The per-request count is exact, so it is the explicit override
	// (>=0); resident weights (below) stay this path's own demand.
	_, plan := compute.AutoSizeContextPlan(p.m.Cfg.ContextSizeConfigWithPrecision(computeKVPrecisionFor(p.kvPrecision)), nil, compute.FreeUnknown, plannedTokens)
	if p.backend != nil && p.includeResidentWeightsInRequestFit() {
		if r := p.m.ResidentReport(); r != nil && r.TotalResidentBytes > 0 {
			plan = append(compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: r.TotalResidentBytes, Detail: "resident-weights", DType: "mixed"}}, plan...)
		}
	}
	extras, err := p.qwen35RuntimeExtraDemands(plannedTokens, retainMTPHistory, plan)
	if err != nil {
		return nil, err
	}
	if len(extras) > 0 {
		plan = append(plan, extras...)
	}
	return plan, nil
}

// qwen35RuntimeExtraDemands prices the panel and retained-history terms via the pure #13315
// estimator. It charges only the panel peak ABOVE the HAL transient scratch already present in
// basePlan (device scope) so integration cannot double count, and only reserves retained history
// when retainMTPHistory is set for THIS request.
//
// The panel width is the gate: when the request is not on the priced Qwen3.8 prefill-panel path
// (nativeInferencePrefillChunkTokens()==0) the extras estimator is not invoked and the base plan
// is preserved byte-for-byte — the same fail-open contract every other capacity helper uses for
// incomplete geometry. When the panel width IS bound but other required bounds are missing, the
// estimator returns its typed capacity-unknown and the caller refuses before allocation.
func (p *InKernelPlanner) qwen35RuntimeExtraDemands(plannedTokens int, retainMTPHistory bool, basePlan compute.MemoryPlan) (compute.MemoryPlan, error) {
	if p == nil || p.m == nil || p.m.Cfg.HiddenSize <= 0 {
		return nil, nil
	}
	panelTokens := p.nativeInferencePrefillChunkTokens()
	if panelTokens <= 0 || plannedTokens <= 0 {
		return nil, nil
	}
	// The configured chunk is a CEILING on the panel, not its width: a prompt no longer than
	// the chunk prefills in ONE pass of len(prompt) rows (prefillDivergentSuffix), so the
	// simultaneous live set is min(chunk, P) <= min(chunk, P+O). Without this clamp every
	// request whose P+O is below the chunk (512 by default) — including the one-token serve
	// warmup — trips the estimator's panel<=planned invariant and is refused as
	// capacity-unknown, so a Qwen3.8 Vulkan serve never becomes ready.
	if panelTokens > plannedTokens {
		panelTokens = plannedTokens
	}
	cfg := compute.Qwen35RuntimeExtraConfig{
		HiddenWidth:                    p.m.Cfg.HiddenSize,
		PlannedTokens:                  plannedTokens,
		PanelTokens:                    panelTokens,
		RetainedHistory:                retainMTPHistory,
		AlreadyPricedHALTransientBytes: deviceTransientBytesInPlan(basePlan),
		Policy:                         compute.RuntimeExtraPolicyConservative,
	}
	if retainMTPHistory {
		cfg.FullHistoryCopies = p.retainedMTPFullHistoryCopies()
	}
	return compute.EstimateQwen35RuntimeExtraMemoryPlan(cfg)
}

// retainedMTPFullHistoryCopies is the count of coexisting full hidden histories: 3 when a
// full-prompt cache snapshot participates (reuse is active), 2 without it. It is the copy
// lifetime bound the #13315 estimator expects and never infers enablement from tensor presence.
func (p *InKernelPlanner) retainedMTPFullHistoryCopies() int {
	if p.tree != nil && inKernelPlannerPrefixReuseSupported(p.m, p.backend) {
		return 3
	}
	return 2
}

// requestMTPHistoryEligible reports whether THIS request will run the retained-MTP-history
// decode path: planner-wide MTP support must be configured AND the request must be a real
// decode (a positive output budget). It is the request-local predicate the extras reservation
// needs, evaluated before admission.
func (p *InKernelPlanner) requestMTPHistoryEligible(maxNew int) bool {
	return p != nil && maxNew > 0 && p.VulkanMTPEnabled()
}

// deviceTransientBytesInPlan sums the device-scope activation/scratch demands already priced in
// a plan, so the panel term subtracts only the overlapping HAL transient scratch.
func deviceTransientBytesInPlan(plan compute.MemoryPlan) int64 {
	var total int64
	for _, d := range plan {
		if d.ScopeOrDefault() != compute.MemoryScopeDevice {
			continue
		}
		switch d.Class {
		case compute.MemoryActivation, compute.MemoryScratchpad:
			if d.Bytes > 0 {
				total += d.Bytes
			}
		}
	}
	return total
}

func (p *InKernelPlanner) includeResidentWeightsInRequestFit() bool {
	if p == nil || p.backend == nil {
		return false
	}
	_, free, known := compute.DeviceMemoryInfo(p.backend)
	return !known || free < 0
}
