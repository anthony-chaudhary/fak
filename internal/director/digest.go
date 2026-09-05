package director

import (
	"github.com/anthony-chaudhary/fak/internal/supervisoragent"
)

// DigestSchema is the canonical A2A wire schema for DirectorDigest.
const DigestSchema = "/a2a/v1/director/digest"

// DigestSchemaV1 is the internal schema identifier.
const DigestSchemaV1 = "fak.director.digest.v1"

// WorkerState is the closed per-worker health class for the director digest.
type WorkerState string

const (
	WorkerHealthy WorkerState = "healthy"
	WorkerStalled WorkerState = "stalled"
	WorkerBlocked WorkerState = "blocked"
	WorkerDone    WorkerState = "done"
)

// Lease modes and lane kinds (closed vocabularies).
const (
	LeaseModeExclusive = "exclusive"
	LeaseModeShared    = "shared"

	LaneKindCluster = "cluster"
	LaneKindKeyword = "keyword"
	LaneKindGlobal  = "global"
)

// DirectorDigest is the structured multi-agent roll-up digest for zero-self-report
// supervisor steering. Strictly excludes all free-text prose and ungrounded self-reports
// by construction.
type DirectorDigest struct {
	Schema           string             `json:"schema"`
	Timestamp        int64              `json:"timestamp"`
	TotalWorkers     int                `json:"total_workers"`
	ActiveWorkers    int                `json:"active_workers"`
	StalledWorkers   int                `json:"stalled_workers"`
	CompletedWorkers int                `json:"completed_workers"`
	FleetVelocity    FleetVelocityScore `json:"fleet_velocity"`
	Workers          []WorkerDigestRow  `json:"workers"`
	Leases           []LeaseSnapshot    `json:"leases"`
	RollupHash       string             `json:"rollup_hash"`
}

// WorkerDigestRow is one worker's verified progress and health facts.
type WorkerDigestRow struct {
	RunID           string      `json:"run_id"`
	Lane            string      `json:"lane"`
	Issue           string      `json:"issue"`
	State           WorkerState `json:"state"`
	StepCount       int         `json:"step_count"`
	VerifiedCommits int         `json:"verified_commits"`
	TreeTouches     int         `json:"tree_touches"`
	VelocityScore   float64     `json:"velocity_score"`
	LastWitnessMs   int64       `json:"last_witness_ms"`
}

// LeaseSnapshot is one active lease's metadata.
type LeaseSnapshot struct {
	Lane     string   `json:"lane"`
	LaneKind string   `json:"lane_kind"`
	Tree     []string `json:"tree"`
	Holder   string   `json:"holder"`
	Mode     string   `json:"mode"`
}

// FleetVelocityScore is the aggregate velocity metrics across the fleet.
type FleetVelocityScore struct {
	TotalCommits   int     `json:"total_commits"`
	CommitsPerHour float64 `json:"commits_per_hour"`
	BlockRate      float64 `json:"block_rate"`
	StallRate      float64 `json:"stall_rate"`
}

// SteeringRecommendation is one closed supervisor action recommendation emitted
// by the director based on digest metrics.
type SteeringRecommendation struct {
	Action           supervisoragent.ActionKind       `json:"action"`
	RunID            string                           `json:"run_id,omitempty"`
	Lane             string                           `json:"lane,omitempty"`
	Issue            string                           `json:"issue,omitempty"`
	Tree             []string                         `json:"tree,omitempty"`
	Reason           string                           `json:"reason"`
	SupervisorAction supervisoragent.SupervisorAction `json:"-"`
}

// ActionKind aliases supervisoragent.ActionKind for closed steering vocabulary.
type ActionKind = supervisoragent.ActionKind

const (
	ActionSpawn      = supervisoragent.ActionSpawn
	ActionReplace    = supervisoragent.ActionReplace
	ActionReplan     = supervisoragent.ActionRedispatch
	ActionRedispatch = supervisoragent.ActionRedispatch
	ActionWiden      = supervisoragent.ActionWiden
	ActionEscalate   = supervisoragent.ActionEscalate
	ActionHold       = supervisoragent.ActionHold
)

// Closed reason tokens for steering recommendations.
const (
	ReasonWorkerStalled      = "WORKER_STALLED"
	ReasonWorkerBlocked      = "WORKER_BLOCKED"
	ReasonWorkerThrashing    = "WORKER_THRASHING"
	ReasonFleetHighStallRate = "FLEET_HIGH_STALL_RATE"
	ReasonFleetHighBlockRate = "FLEET_HIGH_BLOCK_RATE"
	ReasonLaneIdle           = "LANE_IDLE"
	ReasonLaneWiden          = "LANE_WIDEN_REQUESTED"
	ReasonFleetHealthy       = "FLEET_HEALTHY"
	ReasonFleetIdle          = "FLEET_IDLE"
)
