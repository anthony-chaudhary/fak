package milestonereport

// ratchet.go gives the milestone CLIMB a REAL CI gate (issue #1442, epic #1436),
// distinct from the report's advisory `--check`. The two witnessed climb KPIs —
// matured_cells (cells at the MATURED floor M4+) and milestone_progress (the M0-M7
// mean-rank %) — are pinned in a committed baseline (docs/milestones/baseline.json),
// and a regression in EITHER reds the gate the way scorecard debt does. This mirrors
// the portfolio control pane's baseline-pin/ratchet pattern (tools/scorecard_baseline.json):
// a tracked file the check compares against, re-pinned on an improvement, overridable
// only by an explicit flag.
//
// Why a ratchet and not just milestone_debt: debt can stay flat while a cell silently
// drops a rung (one cell climbs as another regresses). The ratchet pins the ABSOLUTE
// climb level, so a same-debt rung swap that lowers matured_cells or progress still
// reds. It is a floor under the headline climb, not a re-derivation of the debt.

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// BaselineSchema is the schema id stamped into docs/milestones/baseline.json.
const BaselineSchema = "fak-milestone-climb.baseline/1"

// RatchetSchema is the control-pane schema id for the climb-ratchet card.
const RatchetSchema = "fak-milestone-climb-ratchet/1"

// RatchetDebtKey is the headline integer the control pane folds for the climb
// ratchet: 0 when the climb holds the pinned floor, 1 when it regressed (and was
// not overridden). It is a DISTINCT gate from milestone_debt — a same-debt rung
// swap that lowers matured_cells leaves milestone_debt flat but reds this.
const RatchetDebtKey = "climb_ratchet_debt"

// BaselineRel is the repo-relative path the committed climb baseline lives at.
const BaselineRel = "docs/milestones/baseline.json"

// Baseline is the pinned floor under the climb KPIs. MaturedCells must not fall
// below Baseline.MaturedCells and ProgressPct must not fall below Baseline.ProgressPct
// (within Epsilon) without an explicit override. Commit records where it was pinned so
// the trend is commit-over-commit and legible in `git blame`.
type Baseline struct {
	Schema       string  `json:"schema"`
	MaturedCells int     `json:"matured_cells"`
	ProgressPct  float64 `json:"progress_pct"`
	Commit       string  `json:"commit,omitempty"`
}

// Epsilon is the progress-regression tolerance: ProgressPct is rounded to one
// decimal (round1), so anything within this band is float noise, not a regression.
const Epsilon = 0.05

// LoadBaseline reads a committed climb baseline. A MISSING file is not a hard error
// to the caller's eye — the ratchet treats a nil baseline as "nothing pinned yet"
// (clean, first run pins it) — but the os.ReadFile / json error is surfaced so a typo
// in the path or a corrupt file is visible rather than silently green.
func LoadBaseline(path string) (*Baseline, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var bl Baseline
	if err := json.Unmarshal(b, &bl); err != nil {
		return nil, err
	}
	return &bl, nil
}

// ClimbKPIs are the two ratcheted numbers pulled off a built scorecard payload's
// corpus. They are read back from the SAME map the control pane folds (corpus
// matured_cells / milestone_progress), so the gate and the pane can never disagree.
type ClimbKPIs struct {
	MaturedCells int
	ProgressPct  float64
}

// climbFromPayload extracts the two ratcheted KPIs from a payload corpus. The corpus
// is map[string]any, so a value arrives either as a native Go int/float64 (built
// in-process) or a float64 (round-tripped through JSON); ratchetInt / ratchetFloat
// normalize both. An absent key is an error — a ratchet over a payload that does not
// carry the KPIs is a wiring bug, not a silent pass.
func climbFromPayload(corpus map[string]any) (ClimbKPIs, error) {
	m, okM := ratchetInt(corpus["matured_cells"])
	if !okM {
		return ClimbKPIs{}, fmt.Errorf("payload corpus missing matured_cells")
	}
	p, okP := ratchetFloat(corpus["milestone_progress"])
	if !okP {
		return ClimbKPIs{}, fmt.Errorf("payload corpus missing milestone_progress")
	}
	return ClimbKPIs{MaturedCells: m, ProgressPct: p}, nil
}

func ratchetInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int64:
		return int(n), true
	case float64:
		return int(n), true
	case json.Number:
		i, err := n.Int64()
		if err != nil {
			return 0, false
		}
		return int(i), true
	default:
		return 0, false
	}
}

func ratchetFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case json.Number:
		f, err := n.Float64()
		if err != nil {
			return 0, false
		}
		return f, true
	default:
		return 0, false
	}
}

// RatchetVerdict is the outcome of comparing the current climb against the pinned
// baseline. OK is the one-bit CI gate; Regressions names every KPI that fell (empty on
// pass); Improved is true when the current climb is strictly above the baseline on at
// least one KPI and below on none — the signal to re-pin. FirstRun is true when no
// baseline was pinned yet (clean; the caller should pin).
type RatchetVerdict struct {
	OK          bool
	FirstRun    bool
	Improved    bool
	Regressions []string
	Current     ClimbKPIs
	Baseline    *Baseline
}

// CheckRatchet compares the current scorecard payload's climb KPIs against the pinned
// baseline. A nil baseline (nothing pinned) is a clean FIRST RUN. Otherwise EITHER
// matured_cells falling below the pin OR progress_pct falling below it (beyond Epsilon)
// is a regression that reds the gate — unless `override` is set, in which case the drop
// is recorded in Regressions but OK stays true (the explicit-allow escape hatch the
// issue requires). An improvement on either KPI with no regression sets Improved so the
// caller re-pins.
func CheckRatchet(corpus map[string]any, base *Baseline, override bool) (RatchetVerdict, error) {
	cur, err := climbFromPayload(corpus)
	if err != nil {
		return RatchetVerdict{}, err
	}
	v := RatchetVerdict{Current: cur, Baseline: base}
	if base == nil {
		v.OK = true
		v.FirstRun = true
		return v, nil
	}

	var regress []string
	if cur.MaturedCells < base.MaturedCells {
		regress = append(regress, fmt.Sprintf("matured_cells regressed %d -> %d (pinned floor %d)",
			base.MaturedCells, cur.MaturedCells, base.MaturedCells))
	}
	if cur.ProgressPct < base.ProgressPct-Epsilon {
		regress = append(regress, fmt.Sprintf("milestone_progress regressed %.1f%% -> %.1f%% (pinned floor %.1f%%)",
			base.ProgressPct, cur.ProgressPct, base.ProgressPct))
	}
	sort.Strings(regress)
	v.Regressions = regress

	improvedMatured := cur.MaturedCells > base.MaturedCells
	improvedProgress := cur.ProgressPct > base.ProgressPct+Epsilon
	v.Improved = len(regress) == 0 && (improvedMatured || improvedProgress)

	// The gate reds on any regression unless explicitly overridden.
	v.OK = len(regress) == 0 || override
	return v, nil
}

// PinFrom builds the baseline to write from the current payload corpus + commit. It is
// used both on a clean first run and to re-pin after an improvement, so the floor only
// ever ratchets UP.
func PinFrom(corpus map[string]any, commit string) (*Baseline, error) {
	cur, err := climbFromPayload(corpus)
	if err != nil {
		return nil, err
	}
	return &Baseline{
		Schema:       BaselineSchema,
		MaturedCells: cur.MaturedCells,
		ProgressPct:  cur.ProgressPct,
		Commit:       commit,
	}, nil
}

// WriteBaseline marshals a baseline to path with a trailing newline (so the committed
// file is diff-friendly and matches the repo's JSON-data convention).
func WriteBaseline(path string, b *Baseline) error {
	out, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return err
	}
	out = append(out, '\n')
	return os.WriteFile(path, out, 0o644)
}

// RenderVerdict renders a one-line human summary of the ratchet outcome for the CLI.
func RenderVerdict(v RatchetVerdict, override bool) string {
	switch {
	case v.FirstRun:
		return fmt.Sprintf("milestone climb ratchet: no baseline pinned yet — pin %d matured cell(s) @ %.1f%% with --pin",
			v.Current.MaturedCells, v.Current.ProgressPct)
	case len(v.Regressions) > 0 && override:
		return fmt.Sprintf("milestone climb ratchet: OVERRIDDEN — %d regression(s) allowed: %v",
			len(v.Regressions), v.Regressions)
	case len(v.Regressions) > 0:
		return fmt.Sprintf("milestone climb ratchet: RED — %v (re-pin with --pin only on a real improvement, or --allow-regress to override)",
			v.Regressions)
	case v.Improved:
		return fmt.Sprintf("milestone climb ratchet: GREEN + IMPROVED — %d matured cell(s) @ %.1f%% (floor %d @ %.1f%%); re-pin with --pin",
			v.Current.MaturedCells, v.Current.ProgressPct, v.Baseline.MaturedCells, v.Baseline.ProgressPct)
	default:
		return fmt.Sprintf("milestone climb ratchet: GREEN — %d matured cell(s) @ %.1f%% holds the pinned floor (%d @ %.1f%%)",
			v.Current.MaturedCells, v.Current.ProgressPct, v.Baseline.MaturedCells, v.Baseline.ProgressPct)
	}
}

// BuildRatchetScorecard folds the ratchet verdict into a control-pane Payload so
// tools/scorecard_control_pane.py picks the climb ratchet up next to the other KPIs
// (issue #1442). climb_ratchet_debt is 1 on an un-overridden regression, else 0; the
// single KPI's defects carry the named regressions so the control pane's find_int
// fold reds CI exactly when the climb regresses. A first run (nothing pinned) is clean
// debt 0 — the floor is set on the next --pin.
func BuildRatchetScorecard(v RatchetVerdict, override bool) scorecard.Payload {
	debt := 0
	var defects []string
	if len(v.Regressions) > 0 && !override {
		// ONE defect (the joined regression list) so scorecard.Fold's len(Defects)
		// count yields a BINARY climb_ratchet_debt of 1 — the gate is "regressed or
		// not," not "how many KPIs regressed." The named drops stay in the message.
		debt = 1
		defects = []string{strings.Join(v.Regressions, "; ")}
	}

	score := 100.0
	if debt > 0 {
		score = 0
	}
	kpi := scorecard.KPI{
		Key:     "climb_ratchet",
		Group:   "climb",
		Score:   score,
		Detail:  RenderVerdict(v, override),
		Defects: defects,
	}

	floor := "unpinned"
	if v.Baseline != nil {
		floor = fmt.Sprintf("%d cell(s) @ %.1f%%", v.Baseline.MaturedCells, v.Baseline.ProgressPct)
	}
	finding := fmt.Sprintf("climb_ratchet_debt %d — current %d matured cell(s) @ %.1f%% vs pinned floor %s",
		debt, v.Current.MaturedCells, v.Current.ProgressPct, floor)

	return scorecard.Fold(RatchetSchema, []scorecard.KPI{kpi}, RatchetDebtKey, nil, scorecard.Messages{
		Finding:         finding,
		Reason:          finding,
		FindingClean:    fmt.Sprintf("the climb holds the pinned floor: %d matured cell(s) @ %.1f%%", v.Current.MaturedCells, v.Current.ProgressPct),
		NextAction:      "a climb KPI regressed below the committed baseline — restore the dropped cell(s), re-pin with `fak milestone-scorecard --pin`, or override with --allow-regress",
		NextActionClean: "hold the line: re-pin only on a real climb improvement (`fak milestone-scorecard --pin`)",
		ExtraCorpus: map[string]any{
			"matured_cells":      v.Current.MaturedCells,
			"milestone_progress": v.Current.ProgressPct,
			"first_run":          v.FirstRun,
			"improved":           v.Improved,
			"regressions":        v.Regressions,
		},
	})
}
