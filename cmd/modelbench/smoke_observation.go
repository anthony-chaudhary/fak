package main

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const smokeV41ModelAttributionSchema = "fak.modelbench.v41-model-attribution/1"

type smokeV41ModelAttribution struct {
	Schema                      string                               `json:"schema"`
	Scope                       string                               `json:"scope"`
	Phase                       string                               `json:"phase"`
	Partial                     bool                                 `json:"partial"`
	ExclusiveRequestAttribution bool                                 `json:"exclusive_request_attribution"`
	Before                      model.V41ExpertFaultAttributionPhase `json:"before"`
	After                       model.V41ExpertFaultAttributionPhase `json:"after"`
}

type smokePrefillObservation struct {
	before         compute.BackendExecutionSnapshot
	v41ModelBefore *model.V41ExpertFaultAttributionPhase
	available      bool
	err            error
}

func beginSmokePrefillObservation(be compute.Backend, m *model.Model) smokePrefillObservation {
	before, available, err := compute.CaptureBackendExecutionSnapshot(be)
	o := smokePrefillObservation{before: before, available: available, err: err}
	if m != nil && m.Cfg.IsDeepSeekV41() {
		before := m.V41ExpertFaultAttribution().Prefill
		o.v41ModelBefore = &before
	}
	return o
}

func (o smokePrefillObservation) end(be compute.Backend, m *model.Model) map[string]any {
	result := o.backendResult(be)
	if result["status"] == "observed" && o.v41ModelBefore != nil && m != nil && m.Cfg.IsDeepSeekV41() {
		result["v41_model_attribution"] = smokeV41ModelAttribution{
			Schema:                      smokeV41ModelAttributionSchema,
			Scope:                       "model_lifetime_snapshots",
			Phase:                       "prefill",
			Partial:                     true,
			ExclusiveRequestAttribution: false,
			Before:                      *o.v41ModelBefore,
			After:                       m.V41ExpertFaultAttribution().Prefill,
		}
	}
	return result
}

func (o smokePrefillObservation) backendResult(be compute.Backend) map[string]any {
	if o.err != nil {
		return map[string]any{"status": "error", "reason": o.err.Error()}
	}
	if !o.available {
		return map[string]any{"status": "unsupported"}
	}
	after, available, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil {
		return map[string]any{"status": "error", "reason": err.Error()}
	}
	if !available {
		return map[string]any{"status": "error", "reason": "backend observation became unavailable"}
	}
	observation, err := compute.BackendExecutionDelta(o.before, after)
	if err != nil {
		return map[string]any{"status": "error", "reason": err.Error()}
	}
	c := observation.Counters
	counters := map[string]any{
		"q4k_stage_calls":          c.Q4KStageCalls,
		"q4k_stage_bytes":          c.Q4KStageBytes,
		"fallbacks":                c.Fallbacks,
		"tensor_home_hits":         c.TensorHomeHits,
		"tensor_home_admissions":   c.TensorHomeAdmissions,
		"tensor_home_bypasses":     c.TensorHomeBypasses,
		"tensor_home_copied_bytes": c.TensorHomeCopiedBytes,
	}
	// The generic snapshot cannot distinguish disabled dispatch profiling from zero.
	dispatchObserved := c.ComputeDispatches > 0 && c.DispatchSubmits > 0
	if dispatchObserved {
		counters["compute_dispatches"] = c.ComputeDispatches
		counters["q4k_matmul_dispatches"] = c.Q4KMatmulDispatches
		counters["other_dispatches"] = c.OtherDispatches
		counters["dispatch_submits"] = c.DispatchSubmits
	}
	if observation.TransferCountersObserved {
		counters["h2d_bytes"] = c.H2DBytes
		counters["h2d_count"] = c.H2DCount
		counters["d2h_bytes"] = c.D2HBytes
		counters["d2h_count"] = c.D2HCount
		counters["d2d_copies"] = c.D2DCopies
		counters["d2d_bytes"] = c.D2DBytes
	}
	return map[string]any{
		"status": "observed",
		"identity": map[string]any{
			"backend": observation.Identity.Backend,
			"device":  observation.Identity.Device,
			"driver":  observation.Identity.Driver,
			"runtime": observation.Identity.Runtime,
		},
		"dispatch_observed":          dispatchObserved,
		"transfer_counters_observed": observation.TransferCountersObserved,
		"counters":                   counters,
	}
}
