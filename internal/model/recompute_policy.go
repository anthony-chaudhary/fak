package model

import "fmt"

// RecomputeCandidate describes one deterministic intermediate whose stored bytes can
// be replaced by arithmetic. All costs are per accepted token in the same envelope.
type RecomputeCandidate struct {
	Name               string  `json:"name"`
	StoredBytes        int64   `json:"stored_bytes"`
	ReadBytes          int64   `json:"read_bytes"`
	RecomputeFLOPs     int64   `json:"recompute_flops"`
	MemoryJoules       float64 `json:"memory_joules"`
	ComputeJoules      float64 `json:"compute_joules"`
	MemoryNanoseconds  int64   `json:"memory_nanoseconds"`
	ComputeNanoseconds int64   `json:"compute_nanoseconds"`
	Deterministic      bool    `json:"deterministic"`
}

type RecomputeDecision struct {
	Schema                  string  `json:"schema"`
	Engine                  string  `json:"engine"`
	Candidate               string  `json:"candidate"`
	AcceptedTokens          int     `json:"accepted_tokens"`
	Recompute               bool    `json:"recompute"`
	RemovedBytes            int64   `json:"removed_bytes"`
	AddedFLOPs              int64   `json:"added_flops"`
	NetNanoseconds          int64   `json:"net_nanoseconds"`
	NetJoules               float64 `json:"net_joules"`
	BytesRemovedPerAccepted float64 `json:"bytes_removed_per_accepted_token"`
	Prediction              string  `json:"prediction"`
	CounterHypothesis       string  `json:"counter_hypothesis"`
	QualityConstraint       string  `json:"quality_constraint"`
	StopRule                string  `json:"stop_rule"`
	Rollback                string  `json:"rollback"`
}

// DecideRecompute keeps only candidates where both measured time and energy models
// improve. A disagreement rejects the optimization instead of hiding a reversal shape.
func DecideRecompute(c RecomputeCandidate, acceptedTokens int) (RecomputeDecision, error) {
	if c.Name == "" || c.StoredBytes < 0 || c.ReadBytes < 0 || c.RecomputeFLOPs < 0 || c.MemoryJoules < 0 || c.ComputeJoules < 0 || c.MemoryNanoseconds < 0 || c.ComputeNanoseconds < 0 || acceptedTokens < 0 {
		return RecomputeDecision{}, fmt.Errorf("model: invalid recompute envelope")
	}
	d := RecomputeDecision{Schema: "fak-recompute-decision/1", Engine: "fak-native", Candidate: c.Name, AcceptedTokens: acceptedTokens, AddedFLOPs: c.RecomputeFLOPs, NetNanoseconds: c.MemoryNanoseconds - c.ComputeNanoseconds, NetJoules: c.MemoryJoules - c.ComputeJoules, Prediction: "recompute wins only when avoided write+reread exceeds added arithmetic", CounterHypothesis: "compute pressure or occupancy loss dominates saved traffic", QualityConstraint: "bit-identical deterministic intermediate", StopRule: "reject on nonpositive net time, nonpositive net energy, or nondeterminism", Rollback: "store and reread intermediate"}
	d.Recompute = c.Deterministic && d.NetNanoseconds > 0 && d.NetJoules > 0
	if d.Recompute {
		d.RemovedBytes = c.StoredBytes + c.ReadBytes
		if acceptedTokens > 0 {
			d.BytesRemovedPerAccepted = float64(d.RemovedBytes) / float64(acceptedTokens)
		}
	}
	return d, nil
}
