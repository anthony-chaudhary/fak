package main

import "github.com/anthony-chaudhary/fak/internal/compute"

type smokePrefillObservation struct {
	before    compute.BackendExecutionSnapshot
	available bool
	err       error
}

func beginSmokePrefillObservation(be compute.Backend) smokePrefillObservation {
	before, available, err := compute.CaptureBackendExecutionSnapshot(be)
	return smokePrefillObservation{before: before, available: available, err: err}
}

func (o smokePrefillObservation) end(be compute.Backend) map[string]any {
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
