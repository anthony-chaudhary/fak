package valuechain

import (
	"encoding/json"
	"fmt"
	"os"
)

// AgenticPacket converts a graduated AgenticBench result packet's explicit
// value_chain measurements without changing its benchmark-native verdict.
type AgenticPacket struct {
	Schema             string       `json:"schema"`
	Status             string       `json:"status"`
	ResultClaimAllowed bool         `json:"result_claim_allowed"`
	ValueChain         []AgenticArm `json:"value_chain"`
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
	if p.Status != "PASS_RESULT" || !p.ResultClaimAllowed {
		return Input{}, fmt.Errorf("agentic packet is not graduated: status=%s result_claim_allowed=%t", p.Status, p.ResultClaimAllowed)
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
