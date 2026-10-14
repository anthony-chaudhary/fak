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
}

type phaseObservation struct {
	phase  string
	before *model.V41ExpertFaultAttributionPhase
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
	}
}
