// Package toolcallcontrol applies deterministic, pre-execution checks to proposed
// agent tool calls and attributes their long-context cost in an ablation report.
package toolcallcontrol

import (
	"crypto/sha256"
	"encoding/hex"
	"math/big"
	"sort"
	"strings"
)

// Action is the gate's recommended treatment of a proposed tool call.
type Action string

const (
	Allow Action = "allow"
	Reuse Action = "reuse"
	Batch Action = "batch"
	Defer Action = "defer"
)

// Proposal describes why the model believes another tool call is necessary.
type Proposal struct {
	ID                 string `json:"id"`
	Tool               string `json:"tool"`
	Args               string `json:"args"`
	PromptTokens       int64  `json:"context_tokens"`
	EvidenceGap        string `json:"evidence_gap,omitempty"`
	EffectIfNew        string `json:"effect_if_new,omitempty"`
	ExpectedInfoGainBP int    `json:"expected_info_gain_bp,omitempty"`
	ReadOnly           bool   `json:"read_only,omitempty"`
	BatchKey           string `json:"batch_key,omitempty"`
	StateEpoch         string `json:"state_epoch,omitempty"`
}

// Observation is a prior completed call that may satisfy a new proposal.
type Observation struct {
	Tool       string `json:"tool"`
	Args       string `json:"args"`
	StateEpoch string `json:"state_epoch,omitempty"`
	ResultRef  string `json:"result_ref,omitempty"`
}

// Decision records a checkable gate outcome, not a model self-report.
type Verdict struct {
	ID                 string `json:"id"`
	Action             Action `json:"action"`
	Reason             string `json:"reason"`
	Fingerprint        string `json:"fingerprint"`
	ReuseRef           string `json:"reuse_ref,omitempty"`
	BatchKey           string `json:"batch_key,omitempty"`
	ReplayUnitsSaved   int64  `json:"replay_units_saved"`
	ReplaySquaredSaved string `json:"replay_squared_saved"`
}

// Config controls conservative admission. Calls below MinInfoGainBP are deferred
// only when the model also cannot name an evidence gap or changed decision.
type Config struct {
	MinInfoGainBP int
}

// Instruction returns the naive model-side policy. The deterministic gate remains
// necessary because instructions alone are neither enforced nor observable.
func Instruction(contextTokens int64) string {
	urgency := ""
	if contextTokens >= 64_000 {
		urgency = " This is a long-context turn: each continuation replays substantial context, so prefer reuse and one batched read."
	}
	return "Before any tool call, name the missing evidence and the decision that could change. Reuse fresh results; do not repeat an identical call without a state change. Batch independent reads. If existing evidence is sufficient, answer or stop instead of calling a tool." + urgency
}

// Evaluate applies cheap checks before execution. It does not block mutation or
// externally changing calls merely because the model supplied weak rationale.
func Evaluate(cfg Config, p Proposal, prior []Observation, peers []Proposal) Verdict {
	if cfg.MinInfoGainBP <= 0 {
		cfg.MinInfoGainBP = 500
	}
	fp := fingerprint(p.Tool, p.Args, p.StateEpoch)
	d := Verdict{ID: p.ID, Action: Allow, Reason: "novel_or_required", Fingerprint: fp}

	for i := len(prior) - 1; i >= 0; i-- {
		o := prior[i]
		if fingerprint(o.Tool, o.Args, o.StateEpoch) == fp {
			d.Action = Reuse
			d.Reason = "exact_fresh_result"
			d.ReuseRef = o.ResultRef
			return withAvoidance(d, p.PromptTokens)
		}
	}

	if p.ReadOnly && p.BatchKey != "" && hasBatchPeer(p, peers) {
		if isBatchLeader(p, peers) {
			d.Reason = "batch_leader"
			d.BatchKey = p.BatchKey
			return d
		}
		d.Action = Batch
		d.Reason = "merged_into_batch_leader"
		d.BatchKey = p.BatchKey
		return withAvoidance(d, p.PromptTokens)
	}

	if p.ReadOnly && strings.TrimSpace(p.EvidenceGap) == "" &&
		strings.TrimSpace(p.EffectIfNew) == "" && p.ExpectedInfoGainBP < cfg.MinInfoGainBP {
		d.Action = Defer
		d.Reason = "no_actionable_evidence_gap"
		return withAvoidance(d, p.PromptTokens)
	}
	return d
}

func isBatchLeader(p Proposal, peers []Proposal) bool {
	leader := p.ID
	for _, q := range peers {
		if q.ReadOnly && q.BatchKey == p.BatchKey && q.ID < leader {
			leader = q.ID
		}
	}
	return p.ID == leader
}
func hasBatchPeer(p Proposal, peers []Proposal) bool {
	for _, q := range peers {
		if q.ID != p.ID && q.ReadOnly && q.BatchKey == p.BatchKey {
			return true
		}
	}
	return false
}

func withAvoidance(d Verdict, tokens int64) Verdict {
	if tokens < 0 {
		tokens = 0
	}
	d.ReplayUnitsSaved = tokens
	// This is a labeled quadratic exposure proxy, not dollars, latency, FLOPs,
	// or a claim about any provider's implementation.
	d.ReplaySquaredSaved = squareDecimal(tokens)
	return d
}

func squareDecimal(v int64) string {
	if v <= 0 {
		return "0"
	}
	n := big.NewInt(v)
	return new(big.Int).Mul(n, n).String()
}

func fingerprint(tool, args, epoch string) string {
	normalized := strings.Join([]string{strings.TrimSpace(tool), strings.TrimSpace(args), strings.TrimSpace(epoch)}, "\x00")
	sum := sha256.Sum256([]byte(normalized))
	return hex.EncodeToString(sum[:8])
}

// Arm is one turn-control ablation arm.
type Arm struct {
	Name      string             `json:"name"`
	Decisions map[string]Verdict `json:"decisions"`
}

// LabeledProposal supplies independent ground truth for an ablation.
type LabeledProposal struct {
	Proposal
	Needed bool `json:"needed"`
}

// ArmMetrics separates saved context from a quadratic exposure proxy so reports
// cannot conflate either with measured provider cost.
type ArmMetrics struct {
	Name                  string `json:"name"`
	CallsExecuted         int    `json:"calls_executed"`
	UnneededCallsAvoided  int    `json:"unneeded_calls_avoided"`
	NeededCallsSuppressed int    `json:"needed_calls_suppressed"`
	ReplayUnitsSaved      int64  `json:"replay_units_saved"`
	ReplaySquaredSaved    string `json:"replay_squared_saved"`
}

// Ablate scores observed arm decisions against the same labeled trace.
func Ablate(trace []LabeledProposal, arms []Arm) []ArmMetrics {
	out := make([]ArmMetrics, 0, len(arms))
	for _, arm := range arms {
		m := ArmMetrics{Name: arm.Name}
		proxy := new(big.Int)
		for _, item := range trace {
			d, ok := arm.Decisions[item.ID]
			execute := !ok || d.Action == Allow
			if execute {
				m.CallsExecuted++
			}
			if !execute && item.Needed && d.Action == Defer {
				m.NeededCallsSuppressed++
			}
			if !execute && !item.Needed {
				m.UnneededCallsAvoided++
				if item.PromptTokens > 0 {
					m.ReplayUnitsSaved += item.PromptTokens
					t := big.NewInt(item.PromptTokens)
					proxy.Add(proxy, new(big.Int).Mul(t, t))
				}
			}
		}
		m.ReplaySquaredSaved = proxy.String()
		out = append(out, m)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}
