// health.go — the Go port of tools/codex_turn_health.py (#5063): the Codex
// turn/compaction health rollup, folded on the EXACT-turn_id lifecycle contract this
// package already enforces instead of the script's boolean in_turn flag.
//
// THE DEFECT THE PORT REMOVES. The Python fold tracked "am I in a turn?" as a
// boolean: a second task_started silently RESET the per-turn tool counter, so an
// abandoned turn's tool/token work leaked into whichever turn was open next, and
// turn_aborted was not handled at all, so an explicitly aborted turn was never
// closed. Here every turn is keyed by its exact turn_id, a superseding start closes
// the abandoned turn with a typed NON-success outcome at an evidence-backed boundary
// (no post-boundary tool delta is attributable to it), and turn_aborted is a
// first-class terminal — the same repair Fold makes, applied to the health stats.
//
// Like the script it replaces, this fold copies NO prompts, tool arguments, tool
// results, diffs, or model text into its report: only structural counts, rates,
// opaque session ids, and the coarse CATEGORY of a zero-tool turn's trailing
// message survive.
package codexlifecycle

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// HealthSchema identifies the report shape. /2 (not the script's /1): the fold is
// keyed by turn_id and reports aborted/superseded/unterminated turns the boolean
// fold could not see.
const HealthSchema = "fak-codex-turn-health/2"

// Classification thresholds (named, not magic) — carried over from the script.
const (
	// CompactBudget is codex model_auto_compact_token_limit for guarded launches
	// (cmd/fak/codex_launcher.go). A compaction near this value is the intended
	// #4253 budget, NOT premature.
	CompactBudget = 96_000
	// PrematureFill: a compaction below this occupancy is genuinely premature —
	// almost always a stuck no-op loop re-accreting tiny context.
	PrematureFill = 40_000
	// RefusalLoopMin: a session is a "guard-refusal loop" once this many of its
	// turns end with every proposed tool call refused.
	RefusalLoopMin = 3
	// InflationMinTurns / InflationZeroRatio flag a session for turn inflation:
	// enough turns, and most call no tool.
	InflationMinTurns  = 5
	InflationZeroRatio = 0.5
)

// Zero-tool turn categories, from the turn's trailing agent message.
const (
	ZeroGuardRefused = "guard_refused"
	ZeroPreambleNoop = "preamble_noop"
	ZeroTalkOnly     = "talk_only"
	ZeroSilent       = "silent"
)

// Markers used to categorise the trailing message of a zero-tool turn. These match
// the fak kernel's refusal envelope and the model's skill-announce preamble;
// substring checks only, never stored.
var refusalMarkers = []string{"refused by the fak kernel", "require_witness", "policy_block",
	"were refused", "proposed tool call"}
var preambleMarkers = []string{"i'll use the", "i’ll use the", "using `", "using the `"}

// ClassifyZeroTool returns the category of a zero-tool turn from its trailing agent
// message.
func ClassifyZeroTool(msg string) string {
	low := strings.ToLower(msg)
	for _, m := range refusalMarkers {
		if strings.Contains(low, m) {
			return ZeroGuardRefused
		}
	}
	if strings.Contains(low, "skill") {
		for _, m := range preambleMarkers {
			if strings.Contains(low, m) {
				return ZeroPreambleNoop
			}
		}
	}
	if strings.TrimSpace(low) != "" {
		return ZeroTalkOnly
	}
	return ZeroSilent
}

// Health-row kinds beyond the lifecycle events.
const (
	kindToolCall  = "function_call"
	kindAgentMsg  = "agent_message"
	kindTokens    = "token_count"
	kindCompacted = "compacted"
	kindModel     = "turn_context"
)

// HealthRow is one health-relevant record read from a rollout: a lifecycle event
// (task_started / task_complete / turn_aborted, keyed by turn_id) or a structural
// delta (tool call, trailing message, token occupancy, compaction, model).
type HealthRow struct {
	Kind        string
	TurnID      string
	Message     string // agent_message text; categorised by FoldHealth, never reported
	Model       string
	InputTokens int
}

// SessionStats is the pure per-session fold: structural counts only.
type SessionStats struct {
	Session   string         `json:"session"`
	Model     string         `json:"model,omitempty"`
	Turns     int            `json:"turns"`
	ToolCalls int            `json:"tool_calls"`
	ZeroTool  map[string]int `json:"zero_tool"` // category -> zero-tool COMPLETED turns
	// Aborted / Superseded / Unterminated are the turns the boolean fold miscounted:
	// none of them is a success, and none is zero-tool-classified as if it completed.
	Aborted      int   `json:"aborted,omitempty"`
	Superseded   int   `json:"superseded,omitempty"`
	Unterminated int   `json:"unterminated,omitempty"`
	Compactions  []int `json:"compactions,omitempty"` // occupancy at each real compaction
}

// ZeroToolTotal is the count of completed turns that called no tool.
func (s SessionStats) ZeroToolTotal() int {
	n := 0
	for _, v := range s.ZeroTool {
		n += v
	}
	return n
}

// FoldHealth folds one session's health rows into structural stats, keyed by exact
// turn_id with the same reconciliation rules as Fold:
//
//   - a new task_started while an older turn is open closes the old turn as
//     SUPERSEDED at that boundary — its tool count freezes, and every later tool
//     call or message is attributed to the new turn, never the abandoned one;
//   - turn_aborted is a terminal (keyed by turn_id; an unkeyed legacy abort binds
//     to the open turn) — the turn is Aborted, never a zero-tool "completion";
//   - a late OBSERVED task_complete for a turn we had synthesized closed repairs the
//     row (observed evidence outranks the inference), still with the frozen,
//     pre-boundary tool count;
//   - only a turn closed by task_complete is zero-tool classified.
//
// Rows must be in rollout order (append-only files already are). Pure: no IO.
func FoldHealth(rows []HealthRow) SessionStats {
	s := SessionStats{ZeroTool: map[string]int{}}
	type turnAcc struct {
		tools      int
		msg        string
		terminated bool
		observed   bool // terminal was read from the rollout, not synthesized
	}
	var turns []*turnAcc
	idx := map[string]int{} // turn_id -> index into turns
	active := -1
	lastInput := 0

	for _, r := range rows {
		switch r.Kind {
		case kindModel:
			if r.Model != "" {
				s.Model = r.Model
			}
		case kindToolCall:
			s.ToolCalls++
			if active >= 0 {
				turns[active].tools++
			}
		case kindAgentMsg:
			if active >= 0 {
				turns[active].msg = r.Message
			}
		case kindTokens:
			if r.InputTokens > 0 {
				lastInput = r.InputTokens
			}
		case kindCompacted:
			if lastInput > 0 {
				s.Compactions = append(s.Compactions, lastInput)
			}
		case KindStarted:
			if r.TurnID == "" {
				continue // unaddressable, same policy as Fold (orphan)
			}
			if _, seen := idx[r.TurnID]; seen {
				continue // reused id, same policy as Fold (never reopened)
			}
			// THE REPAIR: close the still-open turn as superseded — non-success,
			// tool count frozen at this boundary.
			if active >= 0 {
				turns[active].terminated = true
				s.Superseded++
			}
			turns = append(turns, &turnAcc{})
			idx[r.TurnID] = len(turns) - 1
			active = len(turns) - 1
			s.Turns++
		case KindComplete, KindAborted:
			i := -1
			if r.TurnID != "" {
				if j, seen := idx[r.TurnID]; seen {
					i = j
				}
			} else if active >= 0 {
				// Legacy rollouts emit terminals with no turn_id; within one rollout
				// a turn is single-threaded, so it can only bind the open turn.
				i = active
			}
			if i < 0 {
				continue // orphan terminal
			}
			t := turns[i]
			if t.observed {
				continue // multiply terminated: the first observed terminal wins
			}
			if t.terminated {
				// Observed evidence outranks the synthesized supersede: repair.
				s.Superseded--
			}
			t.terminated, t.observed = true, true
			if r.Kind == KindAborted {
				s.Aborted++
			} else if t.tools == 0 {
				s.ZeroTool[ClassifyZeroTool(t.msg)]++
			}
			if active == i {
				active = -1
			}
		}
	}
	if active >= 0 && !turns[active].terminated {
		s.Unterminated++
	}
	return s
}

// ParseHealthRollout reads a Codex rollout JSONL stream into health rows in file
// order, plus the rollout's Meta. Same durability contract as ReadRollout: a torn
// or non-JSON line is skipped, never fatal. The top-level `compacted` record is the
// real compaction marker; the paired event_msg/context_compacted is deliberately
// ignored to avoid double counting.
func ParseHealthRollout(r io.Reader) (Meta, []HealthRow, error) {
	var meta Meta
	var out []HealthRow
	haveMeta := false

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 256*1024), 64*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var rec struct {
			Type    string `json:"type"`
			Payload struct {
				Type          string `json:"type"`
				TurnID        string `json:"turn_id"`
				Message       string `json:"message"`
				Model         string `json:"model"`
				ID            string `json:"id"`
				AltID         string `json:"session_id"`
				ModelProvider string `json:"model_provider"`
				CLIVersion    string `json:"cli_version"`
				CWD           string `json:"cwd"`
				Info          struct {
					LastTokenUsage struct {
						InputTokens int `json:"input_tokens"`
					} `json:"last_token_usage"`
				} `json:"info"`
			} `json:"payload"`
		}
		if json.Unmarshal([]byte(line), &rec) != nil {
			continue
		}
		switch rec.Type {
		case "session_meta":
			if haveMeta {
				continue
			}
			haveMeta = true
			meta = Meta{
				RolloutID:  firstNonEmpty(rec.Payload.ID, rec.Payload.AltID),
				Provider:   strings.TrimSpace(rec.Payload.ModelProvider),
				CLIVersion: strings.TrimSpace(rec.Payload.CLIVersion),
				CWD:        strings.TrimSpace(rec.Payload.CWD),
			}
		case kindModel:
			out = append(out, HealthRow{Kind: kindModel, Model: strings.TrimSpace(rec.Payload.Model)})
		case kindCompacted:
			out = append(out, HealthRow{Kind: kindCompacted})
		case "response_item":
			if rec.Payload.Type == kindToolCall {
				out = append(out, HealthRow{Kind: kindToolCall})
			}
		case "event_msg":
			switch rec.Payload.Type {
			case KindStarted, KindComplete, KindAborted:
				out = append(out, HealthRow{Kind: rec.Payload.Type, TurnID: strings.TrimSpace(rec.Payload.TurnID)})
			case kindAgentMsg:
				out = append(out, HealthRow{Kind: kindAgentMsg, Message: rec.Payload.Message})
			case kindTokens:
				out = append(out, HealthRow{Kind: kindTokens, InputTokens: rec.Payload.Info.LastTokenUsage.InputTokens})
			}
		}
	}
	if err := sc.Err(); err != nil {
		return meta, out, err
	}
	return meta, out, nil
}

// HealthTotals is the corpus-wide structural rollup.
type HealthTotals struct {
	Turns            int     `json:"turns"`
	ToolCalls        int     `json:"tool_calls"`
	ToolCallsPerTurn float64 `json:"tool_calls_per_turn"`
	ZeroToolTurns    int     `json:"zero_tool_turns"`
	ZeroToolRate     float64 `json:"zero_tool_rate"`
	Aborted          int     `json:"aborted,omitempty"`
	Superseded       int     `json:"superseded,omitempty"`
	Unterminated     int     `json:"unterminated,omitempty"`
}

// CompactionStats buckets compaction occupancies against the guarded budget.
type CompactionStats struct {
	Events             int `json:"events"`
	Budget             int `json:"budget"`
	OccupancyP50       int `json:"occupancy_p50"`
	OccupancyP90       int `json:"occupancy_p90"`
	NearBudget96K      int `json:"near_budget_96k"`
	NearWindow200KPlus int `json:"near_window_200k_plus"`
	PrematureLT40K     int `json:"premature_lt40k"`
}

// RefusalLoop is one worst-offender session re-proposing refused tool calls.
type RefusalLoop struct {
	Session      string `json:"session"`
	Model        string `json:"model,omitempty"`
	RefusedTurns int    `json:"refused_turns"`
	Turns        int    `json:"turns"`
}

// Inflation is one worst-offender session whose turns mostly call no tool.
type Inflation struct {
	Session       string `json:"session"`
	Model         string `json:"model,omitempty"`
	Turns         int    `json:"turns"`
	ToolCalls     int    `json:"tool_calls"`
	ZeroToolTurns int    `json:"zero_tool_turns"`
}

// HealthReport is the corpus health report — the script's roll_up, typed.
type HealthReport struct {
	Schema            string          `json:"schema"`
	SessionsWithTurns int             `json:"sessions_with_turns"`
	Totals            HealthTotals    `json:"totals"`
	ZeroToolBreakdown map[string]int  `json:"zero_tool_breakdown"`
	Compaction        CompactionStats `json:"compaction"`
	GuardRefusalLoops []RefusalLoop   `json:"guard_refusal_loops"`
	TurnInflation     []Inflation     `json:"turn_inflation"`
	Flags             []string        `json:"flags"`
	Scanned           int             `json:"scanned_files,omitempty"`
	Unreadable        int             `json:"unreadable,omitempty"`
}

// RollUp aggregates per-session stats into the health report. Sessions with no
// turns are dropped (a rollout that never started a task carries no health signal).
// Pure: no IO, deterministic ordering.
func RollUp(stats []SessionStats, top int) HealthReport {
	kept := make([]SessionStats, 0, len(stats))
	for _, s := range stats {
		if s.Turns > 0 {
			kept = append(kept, s)
		}
	}
	rep := HealthReport{
		Schema:            HealthSchema,
		SessionsWithTurns: len(kept),
		ZeroToolBreakdown: map[string]int{},
		GuardRefusalLoops: []RefusalLoop{},
		TurnInflation:     []Inflation{},
		Flags:             []string{},
	}
	var allFills []int
	for _, s := range kept {
		rep.Totals.Turns += s.Turns
		rep.Totals.ToolCalls += s.ToolCalls
		rep.Totals.Aborted += s.Aborted
		rep.Totals.Superseded += s.Superseded
		rep.Totals.Unterminated += s.Unterminated
		for k, v := range s.ZeroTool {
			rep.ZeroToolBreakdown[k] += v
		}
		allFills = append(allFills, s.Compactions...)
		if refused := s.ZeroTool[ZeroGuardRefused]; refused >= RefusalLoopMin {
			rep.GuardRefusalLoops = append(rep.GuardRefusalLoops, RefusalLoop{
				Session: s.Session, Model: s.Model, RefusedTurns: refused, Turns: s.Turns})
		}
		if z := s.ZeroToolTotal(); s.Turns >= InflationMinTurns &&
			float64(z)/float64(s.Turns) >= InflationZeroRatio {
			rep.TurnInflation = append(rep.TurnInflation, Inflation{
				Session: s.Session, Model: s.Model, Turns: s.Turns,
				ToolCalls: s.ToolCalls, ZeroToolTurns: z})
		}
	}
	zeroTotal := 0
	for _, v := range rep.ZeroToolBreakdown {
		zeroTotal += v
	}
	rep.Totals.ZeroToolTurns = zeroTotal
	if rep.Totals.Turns > 0 {
		rep.Totals.ToolCallsPerTurn = math.Round(float64(rep.Totals.ToolCalls)/float64(rep.Totals.Turns)*10) / 10
		rep.Totals.ZeroToolRate = math.Round(float64(zeroTotal)/float64(rep.Totals.Turns)*1000) / 1000
	}

	rep.Compaction = CompactionStats{
		Events:       len(allFills),
		Budget:       CompactBudget,
		OccupancyP50: percentile(allFills, 50),
		OccupancyP90: percentile(allFills, 90),
	}
	premature := 0
	for _, f := range allFills {
		switch {
		case f < PrematureFill:
			premature++
		case float64(f) >= 0.85*CompactBudget && float64(f) <= 1.10*CompactBudget:
			rep.Compaction.NearBudget96K++
		}
		if f >= 200_000 {
			rep.Compaction.NearWindow200KPlus++
		}
	}
	rep.Compaction.PrematureLT40K = premature

	sort.SliceStable(rep.GuardRefusalLoops, func(i, j int) bool {
		return rep.GuardRefusalLoops[i].RefusedTurns > rep.GuardRefusalLoops[j].RefusedTurns
	})
	sort.SliceStable(rep.TurnInflation, func(i, j int) bool {
		return rep.TurnInflation[i].ZeroToolTurns > rep.TurnInflation[j].ZeroToolTurns
	})
	if top > 0 {
		if len(rep.GuardRefusalLoops) > top {
			rep.GuardRefusalLoops = rep.GuardRefusalLoops[:top]
		}
		if len(rep.TurnInflation) > top {
			rep.TurnInflation = rep.TurnInflation[:top]
		}
	}

	if t := rep.Totals.Turns; t > 0 && float64(zeroTotal)/float64(t) > 0.20 {
		rep.Flags = append(rep.Flags, fmt.Sprintf("HIGH_ZERO_TOOL_RATE: %d/%d turns (%.0f%%) call no tool",
			zeroTotal, t, 100*float64(zeroTotal)/float64(t)))
	}
	if n := len(rep.GuardRefusalLoops); n > 0 {
		rep.Flags = append(rep.Flags, fmt.Sprintf("GUARD_REFUSAL_LOOPS: %d session(s) re-propose refused tool calls", n))
	}
	if premature > 0 {
		rep.Flags = append(rep.Flags, fmt.Sprintf("PREMATURE_COMPACTION: %d compaction(s) fired under %dK occupancy (stuck-loop symptom)",
			premature, PrematureFill/1000))
	}
	return rep
}

// ScanHealth folds every rollout under root into the health report. Same durability
// as ScanCorpus: an unreadable rollout is counted, never fatal. opt.CWD scopes to
// one repository's sessions; opt.Limit caps files scanned, newest first.
func ScanHealth(root string, opt ScanOptions, top int) (HealthReport, error) {
	paths, err := rolloutPaths(root, opt.Limit)
	if err != nil {
		return HealthReport{Schema: HealthSchema}, err
	}
	var stats []SessionStats
	unreadable := 0
	for _, p := range paths {
		fh, openErr := os.Open(p)
		if openErr != nil {
			unreadable++
			continue
		}
		meta, rows, parseErr := ParseHealthRollout(fh)
		_ = fh.Close()
		if parseErr != nil {
			unreadable++
			continue
		}
		if opt.CWD != "" && !sameDir(meta.CWD, opt.CWD) {
			continue
		}
		s := FoldHealth(rows)
		s.Session = strings.TrimSuffix(filepath.Base(p), ".jsonl")
		stats = append(stats, s)
	}
	rep := RollUp(stats, top)
	rep.Scanned = len(paths)
	rep.Unreadable = unreadable
	return rep, nil
}

// percentile is the script's linear-interpolation percentile, truncated to int.
func percentile(vals []int, p float64) int {
	if len(vals) == 0 {
		return 0
	}
	v := append([]int(nil), vals...)
	sort.Ints(v)
	k := float64(len(v)-1) * p / 100.0
	f := int(k)
	if f+1 < len(v) {
		return int(float64(v[f]) + float64(v[f+1]-v[f])*(k-float64(f)))
	}
	return v[f]
}
