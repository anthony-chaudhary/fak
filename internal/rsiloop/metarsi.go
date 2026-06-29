package rsiloop

// metarsi.go closes the apex "meta-RSI" rung (#1195, part of #1173) the loops
// explainer (docs/explainers/engineering-is-building-loops.md) labels conceptual:
// RSI applied to the keep-GATE itself. shipgate.Gate already trips an ESCALATE
// breaker after K consecutive non-keeps; until now a human read that signal and
// hand-tuned the keep-policy. This fold consumes the breaker's escalation judgment
// (the rsiloop journal) and, when escalations CLUSTER, PROPOSES a bounded keep-
// policy adjustment — propose-only by default; applying one is an explicit, logged,
// human-gated act (Apply with allow=true).
//
// ANTI-GOODHART FENCE (load-bearing). The meta-objective is keep-rate GATED ON
// truth-clean (KeepRateTruthClean: truth-clean keeps / cycles), not raw keep-rate.
// A policy change whose only effect is to admit MORE keeps by dropping the truth
// requirement cannot win: the new keeps are truth-DIRTY, so they do not count in the
// numerator, the meta-metric does not rise, and EvaluateProposal — which reuses the
// SAME non-forgeable keep-bit (shipgate.Evaluate) it is tuning — REVERTS. Meta-RSI
// is RSI applied to the gate, so it reuses the gate's own keep-bit: turtles all the
// way down, every turtle witnessed. This is the FoldCalibrable / IntentionalFloor
// lesson from #1021 — a fence that cannot be won by checking less.
//
// LANDING IS STILL GATED. Fold never mutates anything; Apply with allow=false (the
// default) is a logged no-op. A real loop applies a KEPT proposal in an isolated
// worktree (shipgate.ApplyInWorktree), re-measures the journal under the new policy,
// and only then lands it — exactly the propose -> measure -> keep/revert discipline
// rsiloop.Run already runs one rung down.

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// MetaMetricName labels the meta-objective in any journal/telemetry: the fraction of
// improve cycles that produced a TRUTH-CLEAN keep. Higher is better.
const MetaMetricName = "keep_rate_truth_clean"

// KeepPolicy is the tunable keep-gate state the meta-loop adjusts — the three knobs
// the issue names: the strict-gain bar a candidate must clear, the breaker K
// (consecutive non-keeps before ESCALATE), and a candidate-generation throttle.
type KeepPolicy struct {
	GainThreshold float64 // minimum strict gain a candidate must show to keep
	BreakerK      int     // consecutive non-keeps before ESCALATE (shipgate.Gate.K)
	Throttle      int     // max candidates generated per cycle
}

// Knob names the keep-policy dimension a Proposal adjusts.
type Knob uint8

const (
	KnobGainThreshold Knob = iota // raise the strict-gain bar
	KnobBreakerK                  // widen/narrow the escalation breaker
	KnobThrottle                  // throttle candidate generation
)

// String renders the knob as a stable token.
func (k Knob) String() string {
	switch k {
	case KnobGainThreshold:
		return "gain_threshold"
	case KnobBreakerK:
		return "breaker_k"
	case KnobThrottle:
		return "candidate_throttle"
	}
	return "?"
}

// Proposal is a BOUNDED keep-policy adjustment the fold emits. It is a hypothesis to
// be WITNESSED (EvaluateProposal), never an authority to land — Before/After name the
// one knob's old and new value, capped by MetaConfig so a single fold can never make
// an unbounded swing.
type Proposal struct {
	Knob        Knob
	Before      float64
	After       float64
	Escalations int    // the clustered-escalation count that triggered the proposal
	Window      int    // the number of improve cycles scanned
	Rationale   string // why this bounded move, in one line
}

// MetaConfig parameterizes the fold. The bounds are what make a proposal bounded: a
// single step (GainStep) capped at a ceiling (GainCeiling), so clustered escalation
// can nudge the gate, never slam it.
type MetaConfig struct {
	Window         int     // # of most-recent improve cycles to scan
	MinEscalations int     // escalations within the window that trigger a proposal
	GainStep       float64 // bounded increment to the gain threshold
	GainCeiling    float64 // the gain threshold is never proposed above this
}

// DefaultMetaConfig is the conservative default: scan the last 20 cycles, trigger on
// 2+ escalations, nudge the gain bar by 0.05, never past 0.5.
func DefaultMetaConfig() MetaConfig {
	return MetaConfig{Window: 20, MinEscalations: 2, GainStep: 0.05, GainCeiling: 0.5}
}

// Fold is the PURE meta-RSI fold over the rsiloop journal. It scans the most-recent
// cfg.Window improve cycles; if ESCALATE decisions cluster (>= cfg.MinEscalations) it
// returns a BOUNDED proposal to tighten the keep-gate (raise the strict-gain bar by
// one capped step), else (Proposal{}, false). It NEVER mutates cur and NEVER applies
// anything — the proposal is a hypothesis the witness (EvaluateProposal) then judges.
func Fold(rows []Row, cur KeepPolicy, cfg MetaConfig) (Proposal, bool) {
	d := DefaultMetaConfig()
	if cfg.Window <= 0 {
		cfg.Window = d.Window
	}
	if cfg.MinEscalations <= 0 {
		cfg.MinEscalations = d.MinEscalations
	}
	if cfg.GainStep <= 0 {
		cfg.GainStep = d.GainStep
	}
	if cfg.GainCeiling <= 0 {
		cfg.GainCeiling = d.GainCeiling
	}

	esc, seen := 0, 0
	for i := len(rows) - 1; i >= 0 && seen < cfg.Window; i-- {
		if rows[i].Mode != "improve" {
			continue
		}
		seen++
		if rows[i].Decision == shipgate.ESCALATE.String() {
			esc++
		}
	}
	if esc < cfg.MinEscalations {
		return Proposal{}, false
	}

	next := cur.GainThreshold + cfg.GainStep
	if next > cfg.GainCeiling {
		next = cfg.GainCeiling
	}
	if next <= cur.GainThreshold {
		// Already at the bounded ceiling — a fold cannot propose past it.
		return Proposal{}, false
	}
	return Proposal{
		Knob:        KnobGainThreshold,
		Before:      cur.GainThreshold,
		After:       next,
		Escalations: esc,
		Window:      seen,
		Rationale: fmt.Sprintf(
			"%d escalations clustered in the last %d cycles; raise the strict-gain bar %.3g->%.3g (bounded by ceiling %.3g) and witness whether truth-clean keep-rate improves",
			esc, seen, cur.GainThreshold, next, cfg.GainCeiling),
	}, true
}

// KeepRateTruthClean is the anti-goodhart meta-objective: of the improve cycles in
// rows, the fraction that produced a TRUTH-CLEAN keep (Kept AND TruthClean). A keep
// that is not truth-clean does NOT count — so admitting more keeps by dropping the
// truth requirement cannot raise this number. Returns 0 for an empty journal.
func KeepRateTruthClean(rows []Row) float64 {
	cycles, clean := 0, 0
	for _, r := range rows {
		if r.Mode != "improve" {
			continue
		}
		cycles++
		if r.Kept && r.TruthClean {
			clean++
		}
	}
	if cycles == 0 {
		return 0
	}
	return float64(clean) / float64(cycles)
}

// noTruthDirtyKeep reports whether the journal admitted NO truth-dirty keep — a keep
// with TruthClean=false is a slop-keep only a loosened gate could produce.
func noTruthDirtyKeep(rows []Row) bool {
	for _, r := range rows {
		if r.Mode == "improve" && r.Kept && !r.TruthClean {
			return false
		}
	}
	return true
}

// noSuiteRedKeep reports whether the journal admitted NO keep on a red suite — a keep
// with SuiteGreen=false is the other slop-keep a loosened gate could produce.
func noSuiteRedKeep(rows []Row) bool {
	for _, r := range rows {
		if r.Mode == "improve" && r.Kept && !r.SuiteGreen {
			return false
		}
	}
	return true
}

// EvaluateProposal witnesses a proposed policy change through shipgate's NON-FORGEABLE
// keep-bit. before/after are the journals OBSERVED under the current and the proposed
// policy (in the real loop, after is measured in an isolated worktree). The proposal
// is KEPT only if the truth-clean keep-rate STRICTLY rises AND the after-journal is
// itself clean (no truth-dirty keep, no suite-red keep). Because the metric counts
// only truth-clean keeps, a proposal that raises raw keeps by LOOSENING truth-clean
// mechanically REVERTS — the load-bearing anti-goodhart fence.
func EvaluateProposal(before, after []Row) (shipgate.Decision, shipgate.Witness) {
	w := shipgate.Witness{
		Metric:      MetaMetricName,
		Before:      KeepRateTruthClean(before),
		After:       KeepRateTruthClean(after),
		LowerBetter: false, // a higher truth-clean keep-rate is better
		SuiteGreen:  noSuiteRedKeep(after),
		TruthClean:  noTruthDirtyKeep(after),
	}
	return shipgate.Evaluate(w)
}

// ApplyRecord is the logged outcome of an apply attempt — every apply (and every
// propose-only no-op) leaves a record, so the meta-loop's actions are auditable.
type ApplyRecord struct {
	Proposal Proposal
	Applied  bool
	Policy   KeepPolicy // the resulting policy (== the input policy when not applied)
	Log      string
}

// Apply gates a proposal behind an explicit allow flag (the CLI's --apply). With
// allow=false (the DEFAULT) it is a logged no-op: propose-only. With allow=true it
// returns the adjusted policy and a record naming the change, so every applied
// retune is witnessed in the log. It never mutates cur.
func Apply(cur KeepPolicy, p Proposal, allow bool) ApplyRecord {
	if !allow {
		return ApplyRecord{
			Proposal: p,
			Applied:  false,
			Policy:   cur,
			Log:      fmt.Sprintf("propose-only (default): %s %.3g->%.3g NOT applied; pass --apply to apply", p.Knob, p.Before, p.After),
		}
	}
	next := cur
	switch p.Knob {
	case KnobGainThreshold:
		next.GainThreshold = p.After
	case KnobBreakerK:
		next.BreakerK = int(p.After)
	case KnobThrottle:
		next.Throttle = int(p.After)
	}
	return ApplyRecord{
		Proposal: p,
		Applied:  true,
		Policy:   next,
		Log:      fmt.Sprintf("APPLIED %s: %.3g->%.3g (%s)", p.Knob, p.Before, p.After, p.Rationale),
	}
}

// ReadJournal loads an rsiloop JSONL journal into rows for the fold. It is CORRUPTION-
// TOLERANT (a torn final line from an O_APPEND crash is skipped, never fatal) — the
// same fail-open discipline LastTrack uses — and a missing file is (nil, nil), not an
// error, so the fold degrades to "no history, no proposal" rather than crashing.
func ReadJournal(path string) ([]Row, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	var rows []Row
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var r Row
		if err := json.Unmarshal(line, &r); err != nil {
			continue // skip a torn / non-JSON line rather than fail the whole read
		}
		rows = append(rows, r)
	}
	return rows, nil
}
