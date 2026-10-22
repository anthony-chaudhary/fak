package main

import "github.com/anthony-chaudhary/fak/internal/model"

// phaseProfileReceipt preserves the original PhaseProfile JSON object while
// adding the bounded DS4.1 model-lifetime ledger observation when applicable.
type phaseProfileReceipt struct {
	*model.PhaseProfile
	V41ModelAttribution *phaseV41ModelAttribution `json:"v41_model_attribution,omitempty"`
}

// phaseV41ModelAttribution deliberately embeds the smoke receipt contract. The
// snapshots are cumulative model-lifetime counters, not an exclusive timing
// decomposition; consumers may difference Before and After for this measured call.
type phaseV41ModelAttribution struct {
	smokeV41ModelAttribution
	WallNanos                 int64  `json:"wall_nanos"`
	AggregateTimerSemantics   string `json:"aggregate_timer_semantics"`
	HostWallNanos             *int64 `json:"host_wall_nanos"`
	HostWallObservationStatus string `json:"host_wall_observation_status"`
	CopyWallNanos             *int64 `json:"copy_wall_nanos"`
	CopyWallObservationStatus string `json:"copy_wall_observation_status"`
	HardwareExecutionEvidence string `json:"hardware_execution_evidence"`
	// ColdStages is the exclusive per-stage partition of this measured call.
	ColdStages *phaseV41ColdStages `json:"v41_cold_stages,omitempty"`
}

const phaseV41ColdStagesSchema = "fak.modelbench.v41-cold-stages/1"

// phaseV41ColdStages is one window of the session's cold-forward stage
// recorder: stage rows plus unassigned sum exactly to total. A family the
// backend did not observe is omitted, never zeroed: counters need
// counters_observed, dispatch_submits and compute_dispatches also need
// dispatch_counts_live (FAK_VULKAN_DISPATCH_PROFILE=1 on Vulkan),
// device_submits needs device_submits_observed (live with the profile off),
// and device_work needs device_work_observed.
type phaseV41ColdStages struct {
	Schema                string              `json:"schema"`
	Semantics             string              `json:"semantics"`
	CountersObserved      bool                `json:"counters_observed"`
	DispatchCountsLive    bool                `json:"dispatch_counts_live"`
	DeviceSubmitsObserved bool                `json:"device_submits_observed"`
	DeviceWorkObserved    bool                `json:"device_work_observed"`
	SubmitWaitCoverage    string              `json:"submit_wait_coverage,omitempty"`
	Stages                []phaseV41ColdStage `json:"stages"`
	Unassigned            phaseV41ColdStage   `json:"unassigned"`
	Total                 phaseV41ColdStage   `json:"total"`
}

type phaseV41ColdStage struct {
	Stage         string                       `json:"stage"`
	Entries       uint64                       `json:"entries"`
	HostWallNanos int64                        `json:"host_wall_nanos"`
	Counters      *phaseV41ColdStageCounters   `json:"counters,omitempty"`
	DeviceSubmits *uint64                      `json:"device_submits,omitempty"`
	DeviceWork    *phaseV41ColdStageDeviceWork `json:"device_work,omitempty"`
}

type phaseV41ColdStageCounters struct {
	DispatchSubmits   *uint64 `json:"dispatch_submits,omitempty"`
	ComputeDispatches *uint64 `json:"compute_dispatches,omitempty"`
	D2HCount          uint64  `json:"d2h_count"`
	D2HBytes          uint64  `json:"d2h_bytes"`
	H2DCount          uint64  `json:"h2d_count"`
	H2DBytes          uint64  `json:"h2d_bytes"`
	D2DCopies         uint64  `json:"d2d_copies"`
	D2DBytes          uint64  `json:"d2d_bytes"`
}

type phaseV41ColdStageDeviceWork struct {
	Dispatches      uint64 `json:"dispatches"`
	SubmitWaitNanos uint64 `json:"submit_wait_ns"`
}

func newPhaseV41ColdStages(window model.V41ColdStageAttribution) *phaseV41ColdStages {
	row := func(stage string, c model.V41ColdStageCounters) phaseV41ColdStage {
		out := phaseV41ColdStage{Stage: stage, Entries: c.Entries, HostWallNanos: c.HostWallNanos}
		if window.CountersObserved {
			k := c.Counters
			out.Counters = &phaseV41ColdStageCounters{
				D2HCount: k.D2HCount, D2HBytes: k.D2HBytes, H2DCount: k.H2DCount, H2DBytes: k.H2DBytes,
				D2DCopies: k.D2DCopies, D2DBytes: k.D2DBytes,
			}
			if window.DispatchCountsLive {
				submits, dispatches := k.DispatchSubmits, k.ComputeDispatches
				out.Counters.DispatchSubmits, out.Counters.ComputeDispatches = &submits, &dispatches
			}
		}
		if window.DeviceSubmitsObserved {
			submits := c.DeviceSubmits
			out.DeviceSubmits = &submits
		}
		if window.DeviceWorkObserved {
			out.DeviceWork = &phaseV41ColdStageDeviceWork{Dispatches: c.DeviceDispatches, SubmitWaitNanos: c.SubmitWaitNanos}
		}
		return out
	}
	out := &phaseV41ColdStages{
		Schema: phaseV41ColdStagesSchema, Semantics: "exclusive_partition",
		CountersObserved: window.CountersObserved, DispatchCountsLive: window.DispatchCountsLive,
		DeviceSubmitsObserved: window.DeviceSubmitsObserved, DeviceWorkObserved: window.DeviceWorkObserved,
		Stages:     make([]phaseV41ColdStage, 0, model.V41ColdStageCount),
		Unassigned: row("unassigned", window.Unassigned),
		Total:      row("total", window.Total()),
	}
	if window.DeviceWorkObserved {
		out.SubmitWaitCoverage = "lower_bound"
		if window.SubmitWaitComplete {
			out.SubmitWaitCoverage = "complete"
		}
	}
	for stage := model.V41ColdStage(0); stage < model.V41ColdStageCount; stage++ {
		out.Stages = append(out.Stages, row(stage.String(), window.Stages[stage]))
	}
	return out
}

type phaseObservation struct {
	phase      string
	before     *model.V41ExpertFaultAttributionPhase
	coldStages model.V41ColdStageAttribution
}

func beginPhaseObservation(s *model.Session, phase string) phaseObservation {
	o := phaseObservation{phase: phase}
	if s == nil || s.M == nil || !s.M.Cfg.IsDeepSeekV41() {
		return o
	}
	snapshot := s.M.V41ExpertFaultAttribution()
	if phase == "decode" {
		before := snapshot.Decode
		o.before = &before
	} else {
		before := snapshot.Prefill
		o.before = &before
	}
	s.EnableV41ColdStageAttribution()
	o.coldStages, _ = s.V41ColdStageAttribution()
	return o
}

func (o phaseObservation) end(s *model.Session, wallNanos int64) *phaseV41ModelAttribution {
	if o.before == nil || s == nil || s.M == nil || !s.M.Cfg.IsDeepSeekV41() {
		return nil
	}
	snapshot := s.M.V41ExpertFaultAttribution()
	after := snapshot.Prefill
	if o.phase == "decode" {
		after = snapshot.Decode
	}
	var coldStages *phaseV41ColdStages
	if closing, ok := s.V41ColdStageAttribution(); ok {
		if window, err := closing.Sub(o.coldStages); err == nil {
			coldStages = newPhaseV41ColdStages(window)
		}
	}
	return &phaseV41ModelAttribution{
		smokeV41ModelAttribution: smokeV41ModelAttribution{
			Schema:                      smokeV41ModelAttributionSchema,
			Scope:                       "model_lifetime_snapshots",
			Phase:                       o.phase,
			Partial:                     true,
			ExclusiveRequestAttribution: false,
			Before:                      *o.before,
			After:                       after,
		},
		WallNanos:                 wallNanos,
		AggregateTimerSemantics:   "may_nest",
		HostWallObservationStatus: "unknown",
		CopyWallObservationStatus: "unknown",
		HardwareExecutionEvidence: "not_established_by_model_counters",
		ColdStages:                coldStages,
	}
}
