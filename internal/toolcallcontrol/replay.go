package toolcallcontrol

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"sort"
	"strings"
)

// ReplayRow is one independently labeled proposed call. Rows are replayed in
// sequence; calls sharing a turn may inspect peer proposals but never peer outcomes.
type ReplayRow struct {
	ID                 string           `json:"id"`
	Turn               int              `json:"turn"`
	Tool               string           `json:"tool"`
	Args               json.RawMessage  `json:"args"`
	Rationale          string           `json:"rationale,omitempty"`
	EvidenceGap        string           `json:"evidence_gap,omitempty"`
	EffectIfNew        string           `json:"effect_if_new,omitempty"`
	ExpectedInfoGainBP int              `json:"expected_info_gain_bp,omitempty"`
	BatchKey           string           `json:"batch_key,omitempty"`
	ReadOnly           bool             `json:"read_only"`
	StateEpoch         string           `json:"state_epoch"`
	PromptUnits        int64            `json:"prompt_units"`
	Needed             *bool            `json:"needed"`
	ResultID           string           `json:"result_id,omitempty"`
	Succeeded          bool             `json:"succeeded"`
	ControllerUnits    map[string]int64 `json:"controller_units_by_arm,omitempty"`
	RecoveryUnits      int64            `json:"false_suppression_recovery_units,omitempty"`
	CostBasis          string           `json:"cost_basis,omitempty"`
}

type ReplayMetrics struct {
	Proposed                      int    `json:"proposed"`
	CallsExecuted                 int    `json:"calls_executed"`
	UnneededAvoided               int    `json:"unneeded_avoided"`
	NeededSuppressed              int    `json:"needed_suppressed"`
	ReplayUnitsSaved              int64  `json:"replay_units_saved"`
	ReplaySquareProxy             string `json:"replay_square_proxy"`
	ControllerUnits               int64  `json:"controller_units"`
	FalseSuppressionRecoveryUnits int64  `json:"false_suppression_recovery_units"`
	NetReplayValue                int64  `json:"net_replay_value"`
	BreakEven                     bool   `json:"break_even"`
	CostBasis                     string `json:"cost_basis"`
}

type ReplayOutcome struct {
	ID              string `json:"id"`
	Turn            int    `json:"turn"`
	Action          Action `json:"action"`
	Reason          string `json:"reason"`
	Needed          bool   `json:"needed"`
	PromptUnits     int64  `json:"prompt_units"`
	RecordRef       string `json:"record_ref"`
	ControllerUnits int64  `json:"controller_units"`
	RecoveryUnits   int64  `json:"false_suppression_recovery_units"`
}

type ReplayArm struct {
	Name      string          `json:"name"`
	Metrics   ReplayMetrics   `json:"metrics"`
	Buckets   []ReplaySlice   `json:"context_buckets"`
	Reasons   []ReplaySlice   `json:"reasons"`
	Decisions []ReplayOutcome `json:"decisions"`
}

type ReplaySlice struct {
	Name                          string `json:"name"`
	CallsExecuted                 int    `json:"calls_executed"`
	UnneededAvoided               int    `json:"unneeded_avoided"`
	NeededSuppressed              int    `json:"needed_suppressed"`
	ReplayUnitsSaved              int64  `json:"replay_units_saved"`
	ReplaySquareProxy             string `json:"replay_square_proxy"`
	ControllerUnits               int64  `json:"controller_units"`
	FalseSuppressionRecoveryUnits int64  `json:"false_suppression_recovery_units"`
	NetReplayValue                int64  `json:"net_replay_value"`
	BreakEven                     bool   `json:"break_even"`
}

type ReplayCompactArm struct {
	Name    string        `json:"name"`
	Metrics ReplayMetrics `json:"metrics"`
	Buckets []ReplaySlice `json:"context_buckets"`
	Reasons []ReplaySlice `json:"reasons"`
	Records string        `json:"records"`
}

type ReplayCompact struct {
	Schema    string             `json:"schema"`
	TraceRows int                `json:"trace_rows"`
	Arms      []ReplayCompactArm `json:"arms"`
}

type ReplayReport struct {
	Schema    string      `json:"schema"`
	TraceRows int         `json:"trace_rows"`
	Arms      []ReplayArm `json:"arms"`
}

func DecodeReplay(r io.Reader) ([]ReplayRow, error) {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 64*1024), 4*1024*1024)
	var rows []ReplayRow
	for line := 1; s.Scan(); line++ {
		b := bytes.TrimSpace(s.Bytes())
		if len(b) == 0 {
			continue
		}
		var row ReplayRow
		if err := json.Unmarshal(b, &row); err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		if row.ID == "" || row.Tool == "" || row.Needed == nil {
			return nil, fmt.Errorf("line %d: id, tool, and independent needed label are required", line)
		}
		if row.PromptUnits < 0 || row.RecoveryUnits < 0 {
			return nil, fmt.Errorf("line %d: prompt_units and recovery units must be non-negative", line)
		}
		for arm, units := range row.ControllerUnits {
			if units < 0 {
				return nil, fmt.Errorf("line %d: controller units for %s must be non-negative", line, arm)
			}
		}
		if row.CostBasis != "" && row.CostBasis != "observed" && row.CostBasis != "scenario" {
			return nil, fmt.Errorf("line %d: cost_basis must be observed or scenario", line)
		}
		rows = append(rows, row)
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("trace is empty")
	}
	return rows, nil
}

// Replay runs every arm over the same immutable rows. Only observations from
// earlier turns enter prior, preventing same-turn result leakage.
func Replay(rows []ReplayRow) ReplayReport {
	names := []string{"control", "instruction-only", "exact-reuse", "batch", "rationale-prefilter"}
	out := ReplayReport{Schema: "fak-toolcall-control-replay/1", TraceRows: len(rows)}
	for _, name := range names {
		out.Arms = append(out.Arms, replayArm(name, rows))
	}
	return out
}

func replayArm(name string, rows []ReplayRow) ReplayArm {
	arm := ReplayArm{Name: name}
	proxy := new(big.Int)
	prior := []Observation{}
	for i := 0; i < len(rows); {
		j := i + 1
		for j < len(rows) && rows[j].Turn == rows[i].Turn {
			j++
		}
		turnRows := rows[i:j]
		proposals := make([]Proposal, len(turnRows))
		for k, row := range turnRows {
			proposals[k] = replayProposal(row)
		}
		batchCharged := map[string]bool{}
		for k, row := range turnRows {
			verdict := replayVerdict(name, proposals[k], prior, proposals)
			executed := verdict.Action == Allow
			if verdict.Action == Batch {
				key := batchKey(proposals[k])
				executed = !batchCharged[key]
				batchCharged[key] = true
			}
			needed := *row.Needed
			arm.Metrics.Proposed++
			arm.Metrics.ControllerUnits += row.ControllerUnits[name]
			arm.Metrics.CostBasis = mergeCostBasis(arm.Metrics.CostBasis, row.CostBasis)
			if executed {
				arm.Metrics.CallsExecuted++
			} else {
				if needed {
					arm.Metrics.NeededSuppressed++
					arm.Metrics.FalseSuppressionRecoveryUnits += row.RecoveryUnits
				} else {
					arm.Metrics.UnneededAvoided++
				}
				arm.Metrics.ReplayUnitsSaved += row.PromptUnits
				n := big.NewInt(row.PromptUnits)
				proxy.Add(proxy, new(big.Int).Mul(n, n))
			}
			arm.Decisions = append(arm.Decisions, ReplayOutcome{ID: row.ID, Turn: row.Turn, Action: verdict.Action, Reason: verdict.Reason, Needed: needed, PromptUnits: row.PromptUnits, RecordRef: "decisions#" + row.ID, ControllerUnits: row.ControllerUnits[name], RecoveryUnits: row.RecoveryUnits})
		}
		for _, row := range turnRows {
			if row.Succeeded {
				prior = append(prior, Observation{Tool: row.Tool, Args: string(row.Args), StateEpoch: row.StateEpoch, ResultRef: row.ResultID})
			}
		}
		i = j
	}
	arm.Metrics.ReplaySquareProxy = proxy.String()
	arm.Metrics.NetReplayValue = arm.Metrics.ReplayUnitsSaved - arm.Metrics.ControllerUnits - arm.Metrics.FalseSuppressionRecoveryUnits
	arm.Metrics.BreakEven = arm.Metrics.NetReplayValue >= 0
	arm.Buckets = sliceOutcomes(arm.Decisions, func(d ReplayOutcome) string { return contextBucket(d.PromptUnits) })
	arm.Reasons = sliceOutcomes(arm.Decisions, func(d ReplayOutcome) string { return d.Reason })
	return arm
}

func replayVerdict(name string, p Proposal, prior []Observation, peers []Proposal) Verdict {
	switch name {
	case "control", "instruction-only":
		return Verdict{Action: Allow, Reason: name + "_execute"}
	case "exact-reuse":
		p.BatchKey = ""
		p.EvidenceGap = "present"
		return Evaluate(Config{}, p, prior, nil)
	case "batch":
		p.EvidenceGap = "present"
		return Evaluate(Config{}, p, nil, peers)
	case "rationale-prefilter":
		p.BatchKey = ""
		return Evaluate(Config{}, p, nil, nil)
	default:
		return Verdict{Action: Allow, Reason: "unknown_arm_fail_open"}
	}
}

func replayProposal(r ReplayRow) Proposal {
	return Proposal{ID: r.ID, Tool: r.Tool, Args: string(r.Args), EvidenceGap: firstNonempty(r.EvidenceGap, r.Rationale), EffectIfNew: r.EffectIfNew, ExpectedInfoGainBP: r.ExpectedInfoGainBP, ReadOnly: r.ReadOnly, BatchKey: r.BatchKey, StateEpoch: r.StateEpoch, PromptTokens: r.PromptUnits}
}
func batchKey(p Proposal) string { return p.Tool + "\x00" + p.StateEpoch }
func firstNonempty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func (r ReplayReport) Arm(name string) (ReplayArm, bool) {
	i := sort.Search(len(r.Arms), func(i int) bool { return strings.Compare(r.Arms[i].Name, name) >= 0 })
	if i < len(r.Arms) && r.Arms[i].Name == name {
		return r.Arms[i], true
	}
	for _, arm := range r.Arms {
		if arm.Name == name {
			return arm, true
		}
	}
	return ReplayArm{}, false
}

// Compact keeps operator aggregates and a stable link to the full decision array.
func (r ReplayReport) Compact() ReplayCompact {
	out := ReplayCompact{Schema: "fak-toolcall-control-replay-summary/1", TraceRows: r.TraceRows}
	for _, arm := range r.Arms {
		out.Arms = append(out.Arms, ReplayCompactArm{Name: arm.Name, Metrics: arm.Metrics, Buckets: arm.Buckets, Reasons: arm.Reasons, Records: "full-report.json#/arms/" + arm.Name + "/decisions"})
	}
	return out
}

func contextBucket(units int64) string {
	switch {
	case units < 8_000:
		return "lt-8k"
	case units < 32_000:
		return "8k-32k"
	case units < 64_000:
		return "32k-64k"
	case units < 128_000:
		return "64k-128k"
	default:
		return "gte-128k"
	}
}

func sliceOutcomes(rows []ReplayOutcome, key func(ReplayOutcome) string) []ReplaySlice {
	type acc struct {
		slice  ReplaySlice
		square *big.Int
	}
	m := map[string]*acc{}
	for _, row := range rows {
		name := key(row)
		a := m[name]
		if a == nil {
			a = &acc{slice: ReplaySlice{Name: name}, square: new(big.Int)}
			m[name] = a
		}
		a.slice.ControllerUnits += row.ControllerUnits
		if row.Action == Allow {
			a.slice.CallsExecuted++
			continue
		}
		if row.Needed {
			a.slice.NeededSuppressed++
			a.slice.FalseSuppressionRecoveryUnits += row.RecoveryUnits
		} else {
			a.slice.UnneededAvoided++
		}
		a.slice.ReplayUnitsSaved += row.PromptUnits
		n := big.NewInt(row.PromptUnits)
		a.square.Add(a.square, new(big.Int).Mul(n, n))
	}
	keys := make([]string, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make([]ReplaySlice, 0, len(keys))
	for _, key := range keys {
		a := m[key]
		a.slice.ReplaySquareProxy = a.square.String()
		a.slice.NetReplayValue = a.slice.ReplayUnitsSaved - a.slice.ControllerUnits - a.slice.FalseSuppressionRecoveryUnits
		a.slice.BreakEven = a.slice.NetReplayValue >= 0
		out = append(out, a.slice)
	}
	return out
}

func mergeCostBasis(current, next string) string {
	if next == "" {
		return current
	}
	if current == "" || current == next {
		return next
	}
	return "mixed"
}
