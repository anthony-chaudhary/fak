package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	smokeV41ModelAttributionSchema = "fak.modelbench.v41-model-attribution/1"
	smokeCheckpointSchema          = "fak.modelbench.smoke-checkpoint/1"
	smokeCheckpointMaxBytes        = 16 << 10
)

var smokeCheckpointWritten = func(string) {}

type smokeCheckpointWriter struct {
	path     string
	sequence uint64
}

func newSmokeCheckpointWriter(path string) *smokeCheckpointWriter {
	if path == "" {
		return nil
	}
	return &smokeCheckpointWriter{path: path}
}

func (w *smokeCheckpointWriter) write(phase string, backendExecution map[string]any) error {
	if w == nil {
		return nil
	}
	w.sequence++
	report := map[string]any{
		"schema":                        smokeCheckpointSchema,
		"phase":                         phase,
		"sequence":                      w.sequence,
		"partial":                       true,
		"qualifying":                    false,
		"exclusive_request_attribution": false,
		"backend_execution":             checkpointBackendExecution(backendExecution),
	}
	raw, err := benchcli.MarshalReport(report)
	if err != nil {
		return fmt.Errorf("marshal smoke checkpoint: %w", err)
	}
	if len(raw) == 0 || len(raw) > smokeCheckpointMaxBytes {
		return fmt.Errorf("smoke checkpoint size %d exceeds %d-byte bound", len(raw), smokeCheckpointMaxBytes)
	}
	dir := filepath.Dir(w.path)
	tmp, err := os.CreateTemp(dir, ".modelbench-checkpoint-*")
	if err != nil {
		return fmt.Errorf("create smoke checkpoint: %w", err)
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err = tmp.Chmod(0o644); err == nil {
		_, err = tmp.Write(raw)
	}
	if err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err == nil {
		err = os.Rename(tmpPath, w.path)
	}
	if err != nil {
		return fmt.Errorf("write smoke checkpoint: %w", err)
	}
	smokeCheckpointWritten(phase)
	return nil
}

func checkpointBackendExecution(result map[string]any) map[string]any {
	checkpoint := make(map[string]any, len(result))
	for key, value := range result {
		if key != "v41_model_attribution" {
			checkpoint[key] = value
		}
	}
	return checkpoint
}

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

func (o smokePrefillObservation) started() map[string]any {
	if o.err != nil {
		return map[string]any{"status": "error", "reason": o.err.Error()}
	}
	if !o.available {
		return map[string]any{"status": "unsupported"}
	}
	result := map[string]any{
		"status":         "pending",
		"window_status":  "open",
		"counter_status": "unavailable_until_prefill_boundary",
		"identity": map[string]any{
			"backend": o.before.Identity.Backend,
			"device":  o.before.Identity.Device,
			"driver":  o.before.Identity.Driver,
			"runtime": o.before.Identity.Runtime,
		},
	}
	if o.before.DeviceMemoryObserved || o.before.DeviceAllocationObserved {
		result["residency"] = map[string]any{
			"tensor_home_entries":          o.before.TensorHomeEntries,
			"tensor_home_resident_bytes":   o.before.TensorHomeResidentBytes,
			"device_memory_observed":       o.before.DeviceMemoryObserved,
			"device_memory_total_bytes":    o.before.DeviceMemoryTotalBytes,
			"device_memory_free_bytes":     o.before.DeviceMemoryFreeBytes,
			"device_allocation_observed":   o.before.DeviceAllocationObserved,
			"device_allocation_live_bytes": o.before.DeviceAllocationLiveBytes,
		}
	}
	return result
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
