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
	ID                 string          `json:"id"`
	Turn               int             `json:"turn"`
	Tool               string          `json:"tool"`
	Args               json.RawMessage `json:"args"`
	Rationale          string          `json:"rationale,omitempty"`
	EvidenceGap        string          `json:"evidence_gap,omitempty"`
	EffectIfNew        string          `json:"effect_if_new,omitempty"`
	ExpectedInfoGainBP int             `json:"expected_info_gain_bp,omitempty"`
	BatchKey           string          `json:"batch_key,omitempty"`
	ReadOnly           bool            `json:"read_only"`
	StateEpoch         string          `json:"state_epoch"`
	PromptUnits        int64           `json:"prompt_units"`
	Needed             *bool           `json:"needed"`
	ResultID           string          `json:"result_id,omitempty"`
	Succeeded          bool            `json:"succeeded"`
}

type ReplayMetrics struct {
	Proposed          int    `json:"proposed"`
	CallsExecuted     int    `json:"calls_executed"`
	UnneededAvoided   int    `json:"unneeded_avoided"`
	NeededSuppressed  int    `json:"needed_suppressed"`
	ReplayUnitsSaved  int64  `json:"replay_units_saved"`
	ReplaySquareProxy string `json:"replay_square_proxy"`
}

type ReplayOutcome struct {
	ID          string `json:"id"`
	Turn        int    `json:"turn"`
	Action      Action `json:"action"`
	Reason      string `json:"reason"`
	Needed      bool   `json:"needed"`
	PromptUnits int64  `json:"prompt_units"`
}

type ReplayArm struct {
	Name      string          `json:"name"`
	Metrics   ReplayMetrics   `json:"metrics"`
	Decisions []ReplayOutcome `json:"decisions"`
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
		if row.PromptUnits < 0 {
			return nil, fmt.Errorf("line %d: prompt_units must be non-negative", line)
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
			if executed {
				arm.Metrics.CallsExecuted++
			} else {
				if needed {
					arm.Metrics.NeededSuppressed++
				} else {
					arm.Metrics.UnneededAvoided++
				}
				arm.Metrics.ReplayUnitsSaved += row.PromptUnits
				n := big.NewInt(row.PromptUnits)
				proxy.Add(proxy, new(big.Int).Mul(n, n))
			}
			arm.Decisions = append(arm.Decisions, ReplayOutcome{ID: row.ID, Turn: row.Turn, Action: verdict.Action, Reason: verdict.Reason, Needed: needed, PromptUnits: row.PromptUnits})
		}
		for _, row := range turnRows {
			if row.Succeeded {
				prior = append(prior, Observation{Tool: row.Tool, Args: string(row.Args), StateEpoch: row.StateEpoch, ResultRef: row.ResultID})
			}
		}
		i = j
	}
	arm.Metrics.ReplaySquareProxy = proxy.String()
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
