package looptrigger

import "time"

const Schema = "fak-loop-trigger/1"

type Decision string
type Reason string

const (
	Run     Decision = "RUN"
	Skip    Decision = "SKIP"
	Defer   Decision = "DEFER"
	Merge   Decision = "MERGE"
	Reroute Decision = "REROUTE"

	NoDemand        Reason = "NO_DEMAND"
	Cooldown        Reason = "COOLDOWN"
	InputStale      Reason = "INPUT_STALE"
	AlreadyOwned    Reason = "ALREADY_OWNED"
	NoCapacity      Reason = "NO_CAPACITY"
	BelowValueFloor Reason = "BELOW_VALUE_FLOOR"
	Deadline        Reason = "DEADLINE"
	DemandReady     Reason = "DEMAND_READY"
)

type Input struct {
	Loop               string
	ObservedAt         time.Time
	EligibleUnits      int
	OldestAge          time.Duration
	SourceAge          time.Duration
	MaxSourceAge       time.Duration
	OverlapCount       int
	OfferedCapacity    int
	RequiredCapacity   int
	SinceLastRun       time.Duration
	Cooldown           time.Duration
	ServiceWindow      time.Duration
	ExpectedValue      float64
	ValueFloor         float64
	EstimatedWall      time.Duration
	EstimatedAttention time.Duration
	EvidenceRefs       []string
}

type Demand struct {
	EligibleUnits    int   `json:"eligible_units"`
	OldestAgeSeconds int64 `json:"oldest_age_s"`
}
type Freshness struct {
	State            string `json:"state"`
	SourceAgeSeconds int64  `json:"source_age_s"`
	MaxAgeSeconds    int64  `json:"max_age_s"`
}
type Ownership struct {
	State        string `json:"state"`
	OverlapCount int    `json:"overlap_count"`
}
type Capacity struct {
	Offered  int `json:"offered"`
	Required int `json:"required"`
}
type Timing struct {
	State                string `json:"state"`
	SinceLastRunSeconds  int64  `json:"since_last_run_s"`
	CooldownSeconds      int64  `json:"cooldown_s"`
	ServiceWindowSeconds int64  `json:"service_window_s"`
	LatenessSeconds      int64  `json:"lateness_s"`
}
type Cost struct {
	ExpectedValue             float64 `json:"expected_value"`
	ValueFloor                float64 `json:"value_floor"`
	EstimatedWallSeconds      int64   `json:"estimated_wall_s"`
	EstimatedAttentionSeconds int64   `json:"estimated_attention_s"`
}
type Receipt struct {
	Schema       string    `json:"schema"`
	Loop         string    `json:"loop"`
	ObservedAt   string    `json:"observed_at"`
	Decision     Decision  `json:"decision"`
	Reason       Reason    `json:"reason"`
	Demand       Demand    `json:"demand"`
	Freshness    Freshness `json:"freshness"`
	Ownership    Ownership `json:"ownership"`
	Capacity     Capacity  `json:"capacity"`
	Timing       Timing    `json:"timing"`
	Cost         Cost      `json:"cost"`
	EvidenceRefs []string  `json:"evidence_refs,omitempty"`
}

// Evaluate is pure and applies fail-closed precedence: demand, freshness,
// ownership, capacity, cooldown, value, then deadline/readiness.
func Evaluate(in Input) Receipt {
	late := time.Duration(0)
	if in.ServiceWindow > 0 && in.OldestAge > in.ServiceWindow {
		late = in.OldestAge - in.ServiceWindow
	}
	r := Receipt{Schema: Schema, Loop: in.Loop, ObservedAt: in.ObservedAt.UTC().Format(time.RFC3339Nano), Decision: Run, Reason: DemandReady,
		Demand:    Demand{in.EligibleUnits, seconds(in.OldestAge)},
		Freshness: Freshness{"FRESH", seconds(in.SourceAge), seconds(in.MaxSourceAge)},
		Ownership: Ownership{"FREE", in.OverlapCount}, Capacity: Capacity{in.OfferedCapacity, in.RequiredCapacity},
		Timing: Timing{"TIMELY", seconds(in.SinceLastRun), seconds(in.Cooldown), seconds(in.ServiceWindow), seconds(late)},
		Cost:   Cost{in.ExpectedValue, in.ValueFloor, seconds(in.EstimatedWall), seconds(in.EstimatedAttention)}, EvidenceRefs: append([]string(nil), in.EvidenceRefs...)}
	switch {
	case in.EligibleUnits <= 0:
		r.Decision, r.Reason = Skip, NoDemand
	case in.MaxSourceAge > 0 && in.SourceAge > in.MaxSourceAge:
		r.Decision, r.Reason, r.Freshness.State = Defer, InputStale, "STALE"
	case in.OverlapCount > 0:
		r.Decision, r.Reason, r.Ownership.State = Merge, AlreadyOwned, "OWNED"
	case in.RequiredCapacity > in.OfferedCapacity:
		r.Decision, r.Reason = Defer, NoCapacity
	case in.SinceLastRun < in.Cooldown:
		r.Decision, r.Reason, r.Timing.State = Defer, Cooldown, "EARLY"
	case in.ExpectedValue < in.ValueFloor:
		r.Decision, r.Reason = Skip, BelowValueFloor
	case late > 0:
		r.Decision, r.Reason, r.Timing.State = Run, Deadline, "OVERDUE"
	}
	return r
}

func seconds(d time.Duration) int64 { return int64(d / time.Second) }
