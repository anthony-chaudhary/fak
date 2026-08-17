package valuechain

import (
	"encoding/json"
	"fmt"
	"os"
)

// AgenticPacket converts a graduated AgenticBench result packet's explicit
// value_chain measurements without changing its benchmark-native verdict.
type AgenticPacket struct {
	Schema             string             `json:"schema"`
	Status             string             `json:"status"`
	ResultClaimAllowed bool               `json:"result_claim_allowed"`
	BenchmarkNative    bool               `json:"benchmark_native"`
	SameTaskIDs        bool               `json:"same_task_ids"`
	SameModel          bool               `json:"same_model"`
	SameBudget         bool               `json:"same_budget"`
	OfficialGrader     AgenticGrader      `json:"official_grader"`
	Arms               []AgenticPacketArm `json:"arms"`
	MetricCategories   map[string]bool    `json:"metric_categories"`
	Artifacts          []string           `json:"artifacts"`
	ValueChain         []AgenticArm       `json:"value_chain"`
}
type AgenticGrader struct {
	Available bool `json:"available"`
}
type AgenticPacketArm struct {
	Role string `json:"role"`
}
type AgenticArm struct {
	Role       string             `json:"role"`
	TraceID    string             `json:"trace_id"`
	PairID     string             `json:"pair_id,omitempty"`
	Turns      int64              `json:"turns"`
	CostUSD    *float64           `json:"cost_usd,omitempty"`
	Outcomes   map[string]float64 `json:"outcomes"`
	Provenance string             `json:"provenance"`
}

func ReadAgenticPacket(path, stageID string) (Input, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Input{}, err
	}
	var p AgenticPacket
	if err := json.Unmarshal(b, &p); err != nil {
		return Input{}, err
	}
	if p.Schema != "fak.agentic-benchmark-result-packet.v1" {
		return Input{}, fmt.Errorf("agentic packet schema is not fak.agentic-benchmark-result-packet.v1")
	}
	if err := validateAgenticGraduation(p); err != nil {
		return Input{}, err
	}
	if len(p.ValueChain) == 0 {
		return Input{}, fmt.Errorf("agentic packet has no explicit value_chain measurements")
	}
	in := Input{Schema: Schema}
	for i, a := range p.ValueChain {
		if a.Role == "" || a.TraceID == "" || a.Provenance == "" {
			return Input{}, fmt.Errorf("agentic value_chain arm %d requires role, trace_id, and provenance", i)
		}
		in.Observations = append(in.Observations, Observation{ID: fmt.Sprintf("agentic-%s-%d", a.Role, i+1), TraceID: a.TraceID, PairID: a.PairID, StageID: stageID, Arm: a.Role, Turns: a.Turns, CostUSD: a.CostUSD, Outcomes: a.Outcomes, Provenance: a.Provenance})
	}
	return in, nil
}

func validateAgenticGraduation(p AgenticPacket) error {
	if p.Status != "PASS_RESULT" || !p.ResultClaimAllowed {
		return fmt.Errorf("agentic packet is not graduated: status=%s result_claim_allowed=%t", p.Status, p.ResultClaimAllowed)
	}
	if !p.BenchmarkNative || !p.SameTaskIDs || !p.SameModel || !p.SameBudget || !p.OfficialGrader.Available {
		return fmt.Errorf("agentic packet is not graduated: benchmark parity or official grader evidence is missing")
	}
	roles := map[string]bool{}
	for _, arm := range p.Arms {
		roles[arm.Role] = true
	}
	if !roles["raw"] || !roles["fak"] {
		return fmt.Errorf("agentic packet is not graduated: arms must include raw and fak roles")
	}
	for _, category := range []string{"task_success", "safe_success", "cost_or_token_budget", "latency", "policy_events", "evidence_completeness"} {
		if !p.MetricCategories[category] {
			return fmt.Errorf("agentic packet is not graduated: metric category %s is missing", category)
		}
	}
	if len(p.Artifacts) == 0 {
		return fmt.Errorf("agentic packet is not graduated: artifacts are missing")
	}
	return nil
}
