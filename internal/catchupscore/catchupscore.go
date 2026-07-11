// Package catchupscore folds the dev system's "how caught up are we?" question
// into one control-pane scorecard. It is the CATCH-UP lens: not "how much
// happened over a window" (that is `fak cadence`), but "how far BEHIND is the
// dev system RIGHT NOW, at each level" -- a backlog/pace read a human glances at
// and a gate ratchets on, and that `fak scoreboard post --from -` ships to Slack.
//
// The dev system is caught up (or behind) at several independent LEVELS:
//
//   - intake       -- issues arriving vs triaged: is the triage queue draining?
//   - measurement  -- is the portfolio scoring ITSELF, or are cards unmeasured?
//   - index        -- does the dev self-index still agree with the tree?
//   - trunk        -- is the trunk green/shippable, or blocked?
//   - loops        -- are the gardening / RSI loops running on cadence?
//
// Each level carries a 0..1 CAUGHT-UP FRACTION (1.0 == fully caught up) AND a
// raw, UNBOUNDED BEHIND count in the level's own unit (untriaged items,
// unmeasured cards, stale index entities, red build blocks, overdue loops). The
// fraction feeds the grade and the pass-line pressure; the behind count feeds
// the one UNBOUNDED headline the fold exposes -- catchup_backlog, the total
// outstanding units across every level -- so a system three times as far behind
// reads three times as heavy instead of saturating a bounded 0..1 bar.
//
// Like ComposeCacheHealth this core is PURE over caller-supplied Facts and imports
// nothing but fmt/sort + pkg/scorecard, so it stays deterministic, unit-testable
// with fixtures, and reusable by the leased `fak score catchup` shell without this
// scoring core reaching up into any cmd/report package. Every LEVEL is nil-able:
// a level with no evidence this run is EXCLUDED from the fold (never scored 0),
// exactly like the nil-able cache families it is modeled on. Each level is also
// INDIVIDUALLY RETIRABLE -- its defect is retired by discharging the REAL backlog
// behind it (triage the queue, measure the cards, refresh the index, green the
// trunk, run the loops), never by weakening the pass line.
package catchupscore

import (
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Schema tags the catch-up SCORE card so a roster/consumer reads it apart from
// every other control-pane card.
const Schema = "fak-catchup-scorecard/1"

// DebtKey is the corpus debt integer this card writes: the count of LEVELS below
// the caught-up pass line. Exported so the CLI names it in the shared
// --json/--markdown/--compare tail and `fak scoreboard post --debt-key` targets it.
const DebtKey = "catchup_debt"

// BacklogKey is the corpus key for the one UNBOUNDED headline: the total
// outstanding (behind) units summed across every measured level. Unlike the 0..1
// fraction, it does not saturate -- twice as far behind reads twice as large.
const BacklogKey = "catchup_backlog"

// PassLine is the 0..1 caught-up fraction at/above which a level is considered
// caught up (not in debt). It is a single conservative floor an operator TIGHTENS
// over time (the ratchet knob), NOT a per-level tuned threshold. The worst-first
// worklist orders EVERY measured level regardless of this floor; the floor only
// decides which levels count as debt.
const PassLine = 0.8

// The canonical, ordered level keys. The order is the deterministic tie-break when
// two levels carry equal caught-up fraction AND equal behind count, and the
// enumeration a consumer iterates to address each level standalone.
const (
	LevelIntake      = "intake"      // issues awaiting triage vs the open queue
	LevelMeasurement = "measurement" // scorecards measured vs errored/unmeasured
	LevelIndex       = "index"       // dev self-index entities in agreement with the tree
	LevelTrunk       = "trunk"       // trunk shippable vs blocked (red build checks)
	LevelLoops       = "loops"       // gardening / RSI loops on cadence vs overdue
)

// Levels is the canonical ordered level set. len == the fixed denominator
// levels_total; a level is folded only when the caller supplies evidence for it.
var Levels = []string{
	LevelIntake,
	LevelMeasurement,
	LevelIndex,
	LevelTrunk,
	LevelLoops,
}

// levelLabels give each level a human phrase for the worklist detail + retire hint.
var levelLabels = map[string]string{
	LevelIntake:      "issue-triage intake",
	LevelMeasurement: "portfolio self-measurement",
	LevelIndex:       "dev self-index freshness",
	LevelTrunk:       "trunk shippability",
	LevelLoops:       "gardening/RSI loop cadence",
}

// levelUnits name the behind count's unit per level, for the worklist detail.
var levelUnits = map[string]string{
	LevelIntake:      "untriaged items",
	LevelMeasurement: "unmeasured cards",
	LevelIndex:       "stale index entities",
	LevelTrunk:       "blocking build checks",
	LevelLoops:       "overdue loops",
}

// levelRetire names the REAL backlog to discharge to retire each level's defect --
// never "lower the pass line".
var levelRetire = map[string]string{
	LevelIntake:      "triage the queue (close/label/route the untriaged items)",
	LevelMeasurement: "fix the errored cards so the portfolio scores itself",
	LevelIndex:       "refresh the self-index (declare the leaf, fix the dead link, catalog the verb)",
	LevelTrunk:       "green the trunk (clear the blocking build checks)",
	LevelLoops:       "run the overdue loops back onto cadence",
}

// Level is one dev-system level's caught-up evidence. Behind is the raw,
// non-negative, UNBOUNDED count of not-caught-up units (the backlog this level
// contributes); Total is the full set those units are drawn from (behind +
// caught-up). The caught-up fraction is (Total-Behind)/Total, clamped, with
// Total==0 meaning "nothing outstanding" -> 1.0. Note is optional extra context
// appended to the worklist detail.
type Level struct {
	Behind int    `json:"behind"`
	Total  int    `json:"total"`
	Note   string `json:"note,omitempty"`
}

// Facts carries one nil-able Level per canonical level. A nil level is "no
// evidence this run": it is EXCLUDED from the fold (never scored 0). The caller
// (the leased `fak score catchup` shell) fills these from the existing local
// sources; this scoring core reads no ledger and imports no shell.
type Facts struct {
	Intake      *Level
	Measurement *Level
	Index       *Level
	Trunk       *Level
	Loops       *Level
}

// level returns the caller-supplied Level for one key, or nil when absent.
func (f Facts) level(name string) *Level {
	switch name {
	case LevelIntake:
		return f.Intake
	case LevelMeasurement:
		return f.Measurement
	case LevelIndex:
		return f.Index
	case LevelTrunk:
		return f.Trunk
	case LevelLoops:
		return f.Loops
	}
	return nil
}

// caughtFraction derives a level's 0..1 caught-up fraction from its behind/total
// counts. Behind is floored at 0; a Total below Behind is treated as == Behind
// (a fully-behind level reads 0.0 rather than a nonsense negative); Total==0 (and
// so Behind==0) means nothing is outstanding -> 1.0.
func caughtFraction(l Level) float64 {
	behind := l.Behind
	if behind < 0 {
		behind = 0
	}
	total := l.Total
	if total < behind {
		total = behind
	}
	if total == 0 {
		return 1
	}
	return clamp01(float64(total-behind) / float64(total))
}

// behindCount is a level's non-negative behind contribution to catchup_backlog.
func behindCount(l Level) int {
	if l.Behind < 0 {
		return 0
	}
	return l.Behind
}

// Row is one level's row in the worst-first worklist: its key, its 0..1 caught-up
// fraction, the raw behind count + total, the pass line, whether it is in debt
// (below the pass line), and a human detail. The worklist is EVERY measured level,
// sorted worst-first (lowest fraction, then most-behind, then canonical order), so
// a human reads the level furthest behind at a glance and a gate ratchets on debt.
type Row struct {
	Level    string  `json:"level"`
	CaughtUp float64 `json:"caught_up"`
	Behind   int     `json:"behind"`
	Total    int     `json:"total"`
	PassLine float64 `json:"pass_line"`
	InDebt   bool    `json:"in_debt"`
	Detail   string  `json:"detail"`
}

// levelDetail renders one level's worklist/KPI detail line.
func levelDetail(level string, fraction float64, behind, total int, note string) string {
	status := "caught up"
	if fraction+gateEps < PassLine {
		status = "BEHIND"
	}
	detail := fmt.Sprintf("%s %.0f%% caught up (%d/%d %s outstanding, pass line %.0f%%, %s)",
		levelLabels[level], fraction*100, behind, total, levelUnits[level], PassLine*100, status)
	if note != "" {
		detail += " -- " + note
	}
	return detail
}

// CatchUp is the pure core: it folds the present levels into the single 0..1
// caught-up number (the mean of the levels that HAVE evidence), the UNBOUNDED
// backlog (sum of behind counts), and the worst-first worklist. present is the
// count of levels folded; when it is 0 the number is 1.0 (nothing is known-behind),
// the backlog 0, and the worklist empty.
func CatchUp(f Facts) (number float64, backlog, present int, worklist []Row) {
	rank := map[string]int{}
	for i, l := range Levels {
		rank[l] = i
	}
	var sum float64
	worklist = make([]Row, 0, len(Levels))
	for _, name := range Levels {
		l := f.level(name)
		if l == nil {
			continue
		}
		fraction := caughtFraction(*l)
		behind := behindCount(*l)
		sum += fraction
		backlog += behind
		present++
		total := l.Total
		if total < behind {
			total = behind
		}
		worklist = append(worklist, Row{
			Level:    name,
			CaughtUp: scorecard.Round3(fraction),
			Behind:   behind,
			Total:    total,
			PassLine: PassLine,
			InDebt:   fraction+gateEps < PassLine,
			Detail:   levelDetail(name, fraction, behind, total, l.Note),
		})
	}
	sort.SliceStable(worklist, func(i, j int) bool {
		if worklist[i].CaughtUp != worklist[j].CaughtUp {
			return worklist[i].CaughtUp < worklist[j].CaughtUp
		}
		if worklist[i].Behind != worklist[j].Behind {
			return worklist[i].Behind > worklist[j].Behind
		}
		return rank[worklist[i].Level] < rank[worklist[j].Level]
	})
	if present == 0 {
		return 1, 0, 0, worklist
	}
	return sum / float64(present), backlog, present, worklist
}

// levelKPI builds one level KPI. Score is 100*fraction so the Fold composite (mean
// of scores) is exactly 100*number; PassLine feeds the unbounded pressure layer
// (deficit below the caught-up bar). A level below the pass line owns exactly one
// Defect, so Fold's debt is the count of behind levels and ok flips iff any level
// is behind.
func levelKPI(name string, l Level) scorecard.KPI {
	fraction := caughtFraction(l)
	behind := behindCount(l)
	total := l.Total
	if total < behind {
		total = behind
	}
	k := scorecard.KPI{
		Key:      name,
		Group:    "catchup",
		Score:    100 * fraction,
		PassLine: 100 * PassLine,
		Detail:   levelDetail(name, fraction, behind, total, l.Note),
	}
	if fraction+gateEps < PassLine {
		k.Defects = []string{fmt.Sprintf(
			"%s: %d %s outstanding -- %.0f%% caught up < %.0f%% pass line; %s (never weaken the floor)",
			name, behind, levelUnits[name], fraction*100, PassLine*100, levelRetire[name])}
	}
	return k
}

// levelKPIs builds one KPI per level that has evidence, in canonical order. With no
// evidence it returns a single caught-up INSUFFICIENT KPI so the fold's value stays
// a coherent 1.0 (nothing is known-behind) instead of collapsing an empty slice to
// a spurious 0/F.
func levelKPIs(f Facts) []scorecard.KPI {
	kpis := make([]scorecard.KPI, 0, len(Levels))
	for _, name := range Levels {
		if l := f.level(name); l != nil {
			kpis = append(kpis, levelKPI(name, *l))
		}
	}
	if len(kpis) == 0 {
		return []scorecard.KPI{{
			Key:      "catchup_evidence",
			Group:    "catchup",
			Score:    100,
			PassLine: 100 * PassLine,
			Detail:   "no dev-system level evidence this run -- nothing to fold (INSUFFICIENT); catchup defaults to 1.0",
		}}
	}
	return kpis
}

// Compose folds the levels into the single catch-up control-pane payload. corpus.value
// is the 0..1 headline (== corpus["catchup"] by construction); corpus[BacklogKey] is
// the UNBOUNDED total outstanding units across levels; corpus["catchup_worklist"] is the
// worst-first level order; corpus[DebtKey] is the count of levels below the pass line;
// ok == (debt == 0). The standard grade curve (GradeStd) is used because this is an
// OPERATIONAL pace card, not a provenance-honesty card.
func Compose(f Facts) scorecard.Payload {
	number, backlog, present, worklist := CatchUp(f)
	if worklist == nil {
		worklist = []Row{}
	}
	return scorecard.Fold(Schema, levelKPIs(f), DebtKey, nil, scorecard.Messages{
		Finding: "the dev system is BEHIND at one or more levels: a level fell below the caught-up pass line -- " +
			"work the worst-first level worklist",
		FindingClean: "the dev system is caught up: every measured level clears the caught-up pass line",
		NextAction: "catch up the worst-first level (discharge its REAL backlog): triage intake / measure the " +
			"unmeasured cards / refresh the self-index / green the trunk / run the overdue loops",
		NextActionClean: "hold the line; keep every level caught up and tighten the ratchet",
		Grade:           scorecard.GradeStd,
		ExtraCorpus: map[string]any{
			"catchup":          scorecard.Round3(number),
			BacklogKey:         backlog,
			"catchup_worklist": worklist,
			"levels_present":   present,
			"levels_total":     len(Levels),
			"pass_line":        PassLine,
		},
	})
}

// gateEps is the tolerance that keeps a fraction exactly on the pass line from
// reading as behind (mirrors pkg/scorecard's own gate epsilon).
const gateEps = 1e-9

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
