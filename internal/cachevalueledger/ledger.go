package cachevalueledger

import (
	"bufio"
	"encoding/json"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
)

const (
	Schema           = "fak-cache-value-ledger/1"
	DefaultLedgerRel = "docs/nightrun/cache-value.jsonl"
)

type Row struct {
	Schema       string         `json:"schema"`
	Date         string         `json:"date"`
	SessionType  string         `json:"session_type"`
	Context      string         `json:"context"`
	PID          int            `json:"pid"`
	UnixMillis   int64          `json:"unix_millis"`
	Turns        uint64         `json:"turns"`
	PromptTokens uint64         `json:"prompt_tokens"`
	ReusedTokens uint64         `json:"reused_tokens"`
	FrozenTurns  uint64         `json:"frozen_turns"`
	PartialTurns uint64         `json:"partial_turns"`
	ColdTurns    uint64         `json:"cold_turns"`
	ReuseRatio   float64        `json:"reuse_ratio"`
	Stats        cacheobs.Stats `json:"stats"`
	GeneratedAt  string         `json:"generated_at"`
}

func ParseLedger(content string) []Row {
	var rows []Row
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row Row
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		if row.Date == "" || row.SessionType == "" {
			continue
		}
		rows = append(rows, row)
	}
	return rows
}

func AppendLedgerLine(row Row) (string, error) {
	b, err := json.Marshal(row)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func NewRow(sessionType, context string, stats cacheobs.Stats, now time.Time) Row {
	return Row{
		Schema:       Schema,
		Date:         now.UTC().Format("2006-01-02"),
		SessionType:  sessionType,
		Context:      context,
		PID:          os.Getpid(),
		UnixMillis:   now.UnixMilli(),
		Turns:        stats.Turns,
		PromptTokens: stats.PromptTokens,
		ReusedTokens: stats.ReusedTokens,
		FrozenTurns:  stats.FrozenTurns,
		PartialTurns: stats.PartialTurns,
		ColdTurns:    stats.ColdTurns,
		ReuseRatio:   stats.ReuseRatio,
		Stats:        stats,
		GeneratedAt:  now.UTC().Format(time.RFC3339),
	}
}

func Append(sessionType, context, ledgerPath string, stats cacheobs.Stats) error {
	now := time.Now()
	row := NewRow(sessionType, context, stats, now)
	line, err := AppendLedgerLine(row)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(ledgerPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.WriteString(line + "\n"); err != nil {
		return err
	}
	return nil
}

func ReadLedgerFile(path string) []Row {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	return ParseLedger(string(b))
}

// PublishableValueFamily is the ONLY cache-value framing #1066's honesty fence permits
// for a published number (the canonical statement lives in
// internal/cachewitness.WarmKVMarginalFamily; this leaf is too low in the import tier to
// depend on it, so the string is mirrored here). The gate publishes the WITNESSED realized
// reuse ratio and NEVER the vs-naive re-prefill multiple (1/(1-reuse)): the honest
// single-session cache value is the marginal over a tuned warm-KV server (~1.0x), which a
// long trajectory already earns; the >1.0x value is cross-worker shared-prefix, not a
// single session's turn-over-turn reuse.
const PublishableValueFamily = "marginal-over-tuned-warm-KV (~1.0x single-session; the vs-naive 1/(1-reuse) re-prefill multiple is excluded per #1066)"

// MinGateTurns is the minimum number of MULTI-TURN turns required before the gate enforces
// its reuse floor. Below it the gate reports INSUFFICIENT and does not fail — a thin corpus
// is not evidence of a regression — mirroring the "falls open, clearly labelled" posture
// `fak vcache score` takes when no live snapshot is present.
const MinGateTurns = 8

// ScoreLedgerResult summarizes the cache-value ledger for the regression gate. The headline
// is RealizedReuseRatio — the WITNESSED in-kernel KV-prefix reuse over sessions where reuse
// is even possible. The pre-#1066 OverallMultiplier (reused/(prompt-reused) = the vs-naive
// re-prefill multiple minus one) is deliberately gone; see PublishableValueFamily.
type ScoreLedgerResult struct {
	TotalSessions      int `json:"total_sessions"`
	MultiTurnSessions  int `json:"multi_turn_sessions"`
	SingleTurnSessions int `json:"single_turn_sessions"`

	TotalTurns   uint64 `json:"total_turns"`
	FrozenTurns  uint64 `json:"frozen_turns"`
	PartialTurns uint64 `json:"partial_turns"`
	ColdTurns    uint64 `json:"cold_turns"`

	TotalPromptTokens uint64 `json:"total_prompt_tokens"`
	TotalReusedTokens uint64 `json:"total_reused_tokens"`

	// Gate inputs: realized reuse over MULTI-TURN sessions only (turns >= 2), where
	// cross-turn KV-prefix reuse can actually happen. A single-turn cold `fak run` has no
	// previous turn to reuse from, so folding it in would manufacture a false regression.
	MultiTurnTurns     uint64  `json:"multi_turn_turns"`
	GatePromptTokens   uint64  `json:"gate_prompt_tokens"`
	GateReusedTokens   uint64  `json:"gate_reused_tokens"`
	RealizedReuseRatio float64 `json:"realized_reuse_ratio"`

	// #1066 honesty fence — see PublishableValueFamily. These are constant self-labels so a
	// downstream reader can never mistake the realized reuse for the forbidden multiple.
	PublishableValueFamily  string  `json:"publishable_value_family"`
	SingleSessionMarginalX  float64 `json:"single_session_marginal_over_warm_kv_x"`
	VsNaiveMultipleExcluded bool    `json:"vs_naive_multiple_excluded"`
}

// HasEnoughData reports whether the multi-turn corpus is thick enough to enforce the floor.
func (r ScoreLedgerResult) HasEnoughData() bool { return r.MultiTurnTurns >= MinGateTurns }

// ScoreLedger reads the ledger and computes the honest realized-reuse summary the regression
// gate checks. RealizedReuseRatio is reused/prompt over sessions with >= 2 turns; the
// vs-naive re-prefill multiple is never computed (#1066).
func ScoreLedger(path string) (ScoreLedgerResult, error) {
	result := ScoreLedgerResult{
		PublishableValueFamily:  PublishableValueFamily,
		SingleSessionMarginalX:  1.0,
		VsNaiveMultipleExcluded: true,
	}
	rows := ReadLedgerFile(path)
	for _, r := range rows {
		if r.Turns == 0 {
			continue
		}
		result.TotalSessions++
		result.TotalTurns += r.Turns
		result.FrozenTurns += r.FrozenTurns
		result.PartialTurns += r.PartialTurns
		result.ColdTurns += r.ColdTurns
		result.TotalPromptTokens += r.PromptTokens
		result.TotalReusedTokens += r.ReusedTokens
		if r.Turns >= 2 {
			result.MultiTurnSessions++
			result.MultiTurnTurns += r.Turns
			result.GatePromptTokens += r.PromptTokens
			result.GateReusedTokens += r.ReusedTokens
		} else {
			result.SingleTurnSessions++
		}
	}
	if result.GatePromptTokens > 0 {
		result.RealizedReuseRatio = float64(result.GateReusedTokens) / float64(result.GatePromptTokens)
	}
	return result, nil
}
