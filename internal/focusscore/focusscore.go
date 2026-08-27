// Package focusscore grades the ONE thing the per-objective trajectory-control fold
// cannot see: is the fleet as a whole CONVERGING on its live goal, or fanning out too
// broad — many objectives declared active at once, detours run past budget while their
// parents sit paused, open objectives drifting or stalled instead of moving?
//
// internal/trajctl already classifies a SINGLE objective's witnessed progress curve into
// a closed HEALTHY/STALL/DRIFT/DETOUR_OVERRUN signal (curve.go). What it does not do is
// FOLD THE OBJECTIVE TREE into a single "are we too broad?" number. That aggregate is the
// focus question, and this scorecard answers it — re-derived entirely from the same
// trajctl ledger (docs/nightrun/trajctl.jsonl) the controller reads, so the score moves
// only when the fleet actually converges: close or meet objectives, bring detours back to
// their paused parent, get the witnessed curve rising again.
//
// The headline is focus_debt (unbounded, one per structural breadth/non-convergence
// defect), graded on two axes:
//
//   - CONVERGENCE: are the OPEN objectives moving toward their goal? Each open objective
//     the curve fold marks DRIFT (declining), STALL (flat while busy), or DETOUR_OVERRUN
//     (a child past budget while its parent is paused) is one debt. A fleet of ten
//     healthy-rising objectives is convergent; ten stalled ones are not.
//   - BREADTH: is concurrency bounded? Every objective declared `active` beyond a pinned
//     WIP cap is one debt — declaring six goals live at once is not focus, it is fan-out.
//     A detour whose parent is NOT paused (a side-quest taken without parking the main
//     goal) is a softer breadth signal.
//
// You cannot lower focus_debt by editing a JSON file: the active/paused/met counts and the
// per-objective signals are all re-folded from the witnessed ledger. You lower it by
// converging — the same move the score is asking for.
package focusscore

import (
	"math"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Schema is the control-pane schema id. Same family shape as loopscore/conflationscore so
// the control pane folds it identically.
const Schema = "fak-focusscore-scorecard/1"

// DebtKey is the headline integer the control pane reads (corpus.focus_debt).
const DebtKey = "focus_debt"

// DefaultWIPCap is the pinned work-in-progress ceiling: how many objectives may be
// `active` at once before the fleet is fanning out rather than converging. One live goal
// plus a small margin for a parked-parent detour and a follow-on. Every active objective
// beyond this is one breadth debt. It is a constant (not a knob a caller can loosen to
// game the score) — moving the cap is a code change with a golden-test consequence.
const DefaultWIPCap = 3

// Axis weights. Convergence leads: a bounded but stalled fleet is worse than a slightly
// broad but uniformly rising one — the point of focus is MOTION toward the goal, and
// breadth only matters because it dilutes that motion.
const (
	convergenceWeight = 0.60
	breadthWeight     = 0.40
)

// KPIResult aliases the shared binary criterion shape while preserving this package's API.
type KPIResult = scorecard.BinaryResult

// Evidence is the raw, re-derived-from-disk corpus the KPIs read. Every field is a
// fleet-wide tally over the folded objective tree + its per-objective curve signals.
type Evidence struct {
	Objectives int `json:"objectives"` // every objective ever declared (any status)
	Active     int `json:"active"`     // objectives in the active state (concurrently live)
	Paused     int `json:"paused"`     // objectives parked (a paused parent behind a detour)
	Met        int `json:"met"`        // objectives reached their goal (convergence, banked)
	Abandoned  int `json:"abandoned"`  // objectives explicitly dropped
	Open       int `json:"open"`       // active + paused (the steerable fleet)

	Drift           int                        `json:"drift"`            // open objectives the curve marks DRIFT (declining)
	Stall           int                        `json:"stall"`            // open objectives marked STALL (flat while busy)
	DetourOverrun   int                        `json:"detour_overrun"`   // open objectives marked DETOUR_OVERRUN (past budget, parent paused)
	Healthy         int                        `json:"healthy"`          // open objectives marked HEALTHY (rising/steady)
	CalibrationDebt int                        `json:"calibration_debt"` // one when the worst measured scorer is anti-correlated
	WorstCalibrated *trajctl.ScorerCalibration `json:"worst_calibrated,omitempty"`

	WIPCap      int `json:"wip_cap"`      // the pinned active ceiling this fold graded against
	ExcessWIP   int `json:"excess_wip"`   // max(0, active - wip_cap): active objectives beyond the cap
	LooseDetour int `json:"loose_detour"` // open child objectives whose parent is NOT paused (side-quest without parking)

	LedgerPresent bool `json:"ledger_present"` // the trajctl ledger exists and folded >=1 objective
}

// ScorecardPayload is the rendered card. Mirrors loopscore.ScorecardPayload.
type ScorecardPayload struct {
	Schema      string         `json:"schema"`
	OK          bool           `json:"ok"`
	Verdict     string         `json:"verdict"`
	Finding     string         `json:"finding"`
	Reason      string         `json:"reason"`
	NextAction  string         `json:"next_action"`
	Workspace   string         `json:"workspace"`
	Corpus      map[string]any `json:"corpus"`
	KPIs        []KPIPayload   `json:"kpis"`
	Convergence []KPIResult    `json:"convergence"`
	Breadth     []KPIResult    `json:"breadth"`
	Evidence    Evidence       `json:"evidence"`
}

// KPIPayload aliases the shared stable JSON projection while preserving this package's API.
type KPIPayload = scorecard.BinaryPayload

// Options pins the inputs and clock so the score is deterministic for tests.
type Options struct {
	Root       string
	LedgerPath string
	WIPCap     int

	// rows overrides the disk read for tests; nil means load from LedgerPath.
	rows      []trajctl.Row
	useInputs bool
}

func (o Options) normalize() Options {
	if o.Root == "" {
		o.Root = "."
	}
	if o.LedgerPath == "" {
		o.LedgerPath = filepath.Join(o.Root, filepath.FromSlash(trajctl.DefaultLedgerRel))
	}
	if o.WIPCap <= 0 {
		o.WIPCap = DefaultWIPCap
	}
	return o
}

// ---- evidence gathering (the impure shell, kept thin) -----------------------------

func gatherEvidence(opts Options) Evidence {
	var rows []trajctl.Row
	if opts.useInputs {
		rows = opts.rows
	} else {
		rows = trajctl.ReadLedgerFile(opts.LedgerPath)
	}

	st := trajctl.Fold(rows)
	curves := st.OpenCurves() // active|paused objectives, each with its derived signal

	var ev Evidence
	calibration := st.Calibrate()
	if worst, ok := calibration.WorstCalibrated(); ok {
		ev.WorstCalibrated = &worst
		if worst.Verdict == trajctl.CalibrationMiscalibrated {
			ev.CalibrationDebt = 1
		}
	}
	ev.WIPCap = opts.WIPCap
	ev.LedgerPresent = len(st.Objectives) > 0
	ev.Objectives = len(st.Objectives)

	for _, id := range st.ObjectiveIDs() {
		switch st.Objectives[id].Status {
		case trajctl.StatusActive:
			ev.Active++
		case trajctl.StatusPaused:
			ev.Paused++
		case trajctl.StatusMet:
			ev.Met++
		case trajctl.StatusAbandoned:
			ev.Abandoned++
		}
	}
	ev.Open = ev.Active + ev.Paused

	for _, oc := range curves.Objectives {
		switch oc.Signal {
		case trajctl.SignalDrift:
			ev.Drift++
		case trajctl.SignalStall:
			ev.Stall++
		case trajctl.SignalDetourOverrun:
			ev.DetourOverrun++
		default:
			ev.Healthy++
		}
		// A loose detour is an open child whose parent is NOT paused: a side-quest
		// taken without parking the main goal. DETOUR_OVERRUN already covers the
		// paused-parent-past-budget case, so this counts only the un-parked ones.
		if oc.ParentID != "" && oc.Signal != trajctl.SignalDetourOverrun {
			if parent, ok := st.Objectives[oc.ParentID]; ok && parent.Status != trajctl.StatusPaused {
				ev.LooseDetour++
			}
		}
	}

	if ev.Active > ev.WIPCap {
		ev.ExcessWIP = ev.Active - ev.WIPCap
	}
	return ev
}

// ---- KPI definitions --------------------------------------------------------------

// convergenceResults grade whether the OPEN objectives are moving toward their goal.
func convergenceResults(ev Evidence) []KPIResult {
	const axis = "convergence"
	notDrifting := ev.Drift == 0
	notStalled := ev.Stall == 0
	noOverrun := ev.DetourOverrun == 0
	return []KPIResult{
		result("no_drift", axis, true, 3,
			"no open objective is DRIFTing — none has a declining witnessed progress curve",
			notDrifting,
			strconv.Itoa(ev.Drift)+"/"+strconv.Itoa(ev.Open)+" open objective(s) drifting (declining progress)"),
		result("no_detour_overrun", axis, true, 3,
			"no detour has run past its turn budget while its parent is paused — no unreturned side-quest",
			noOverrun,
			strconv.Itoa(ev.DetourOverrun)+"/"+strconv.Itoa(ev.Open)+" open objective(s) past a detour budget"),
		result("no_stall", axis, true, 2,
			"no open objective is STALLed — none is flat-and-busy (activity without witnessed movement)",
			notStalled,
			strconv.Itoa(ev.Stall)+"/"+strconv.Itoa(ev.Open)+" open objective(s) stalled (flat while active)"),
		result("progress_banked", axis, false, 1,
			"at least one objective has reached MET — convergence is banked, not only promised",
			ev.Open == 0 || ev.Met > 0,
			strconv.Itoa(ev.Met)+" objective(s) met of "+strconv.Itoa(ev.Objectives)+" declared"),
	}
}

// breadthResults grade whether concurrency is bounded — is the fleet converging on a live
// goal, or fanning out across many at once?
func breadthResults(ev Evidence) []KPIResult {
	const axis = "breadth"
	return []KPIResult{
		result("wip_bounded", axis, true, 3,
			"active objectives stay within the WIP cap — the fleet is not declaring more live goals than it can converge",
			ev.ExcessWIP == 0,
			strconv.Itoa(ev.Active)+" active vs cap "+strconv.Itoa(ev.WIPCap)+" ("+strconv.Itoa(ev.ExcessWIP)+" over)"),
		result("detours_park_parent", axis, false, 2,
			"every open detour has a PAUSED parent — a side-quest was taken only after parking the main goal",
			ev.LooseDetour == 0,
			strconv.Itoa(ev.LooseDetour)+" open detour(s) whose parent is not paused"),
		result("ledger_present", axis, true, 1,
			"the fleet declares its objectives to the trajctl ledger at all — focus is measurable, not implicit",
			ev.LedgerPresent,
			boolStr(ev.LedgerPresent, "trajctl ledger folded >=1 objective", "no trajctl objectives found")),
	}
}

// ---- fold -------------------------------------------------------------------------

var axisWeights = map[string]float64{
	"convergence": convergenceWeight,
	"breadth":     breadthWeight,
}

// focusDebt is the unbounded headline: the count of concrete, re-derivable breadth /
// non-convergence defects. It is DELIBERATELY not just the HARD-KPI-fail count (which
// saturates at one per KPI): a fleet six objectives over the WIP cap and with four
// drifting objectives should read as ten debt, not two. So each defect is counted at its
// natural magnitude — one per excess active objective, one per drifting/stalled/overrun
// open objective — while the per-KPI grade (below) stays a clean pass/fail for the letter.
func focusDebt(ev Evidence) int {
	return ev.ExcessWIP + ev.Drift + ev.Stall + ev.DetourOverrun + ev.CalibrationDebt
}

func Build(opts Options) ScorecardPayload {
	opts = opts.normalize()
	root := scorecard.WorkspaceRoot(opts.Root)
	ev := gatherEvidence(opts)

	convergence := convergenceResults(ev)
	breadth := breadthResults(ev)
	all := append(append([]KPIResult{}, convergence...), breadth...)

	cScore := scorecard.BinaryAxisScore(convergence)
	bScore := scorecard.BinaryAxisScore(breadth)
	composite := int(math.Round(convergenceWeight*float64(cScore) + breadthWeight*float64(bScore)))
	grade := scorecard.GradeStd(float64(composite))

	kpis, weights := scorecard.ProjectBinary(all, axisWeights, nil)

	debt := focusDebt(ev)
	ok := debt == 0

	finding := "fleet_converging_on_goal"
	next := "hold the line; re-run after a session — keep active objectives within the WIP cap and every open curve rising"
	var reason string
	if ok {
		reason = "focus-score: convergence value " + scorecard.ScoreValueText(cScore) + ", breadth value " + scorecard.ScoreValueText(bScore) +
			", composite value " + scorecard.ScoreValueText(composite) + " (" + grade + "); " + strconv.Itoa(ev.Active) + " active within cap " +
			strconv.Itoa(ev.WIPCap) + ", " + strconv.Itoa(ev.Healthy) + "/" + strconv.Itoa(ev.Open) + " open objective(s) healthy; zero focus debt"
	} else {
		finding = "focus_debt"
		reason = "focus-score carries " + strconv.Itoa(debt) + " debt (convergence value " + scorecard.ScoreValueText(cScore) +
			", breadth value " + scorecard.ScoreValueText(bScore) + ", composite value " + scorecard.ScoreValueText(composite) + " " + grade + "): " +
			debtBreakdown(ev)
		next = "converge worst-first: " + worstFirstNext(ev)
	}

	p := scorecard.Fold(Schema, kpis, DebtKey, weights, scorecard.Messages{
		Grade:           func(float64) string { return grade },
		Finding:         finding,
		FindingClean:    finding,
		NextAction:      next,
		NextActionClean: next,
		Reason:          reason,
		ExtraCorpus: map[string]any{
			DebtKey:             debt,
			"score":             composite,
			"grade":             grade,
			"convergence_value": scorecard.Round3(scorecard.ValueFromScore(float64(cScore))),
			"convergence_score": cScore,
			"breadth_value":     scorecard.Round3(scorecard.ValueFromScore(float64(bScore))),
			"breadth_score":     bScore,
			"objectives":        ev.Objectives,
			"active":            ev.Active,
			"paused":            ev.Paused,
			"met":               ev.Met,
			"open":              ev.Open,
			"wip_cap":           ev.WIPCap,
			"excess_wip":        ev.ExcessWIP,
			"drift":             ev.Drift,
			"stall":             ev.Stall,
			"detour_overrun":    ev.DetourOverrun,
			"healthy":           ev.Healthy,
			"loose_detour":      ev.LooseDetour,
			"calibration_debt":  ev.CalibrationDebt,
		},
	})

	// focus_debt is magnitude-counted and deliberately decoupled from the kernel's
	// len(Defects) gate (a fleet 4 objectives over the cap is 4 debt, not 1; a missing
	// ledger HARD-fails a KPI but adds ZERO fan-out debt). So the OK/verdict the card
	// reports must follow focus_debt, not the kernel's Defect count — override both here.
	verdict := "OK"
	if !ok {
		verdict = "ACTION"
	}

	return ScorecardPayload{
		Schema:      p.Schema,
		OK:          ok,
		Verdict:     verdict,
		Finding:     p.Finding,
		Reason:      p.Reason,
		NextAction:  p.NextAction,
		Workspace:   root,
		Corpus:      p.Corpus,
		KPIs:        scorecard.BinaryPayloads(all),
		Convergence: convergence,
		Breadth:     breadth,
		Evidence:    ev,
	}
}

// debtBreakdown names each debt source at its magnitude, worst-first, for the reason line.
func debtBreakdown(ev Evidence) string {
	var parts []string
	if ev.DetourOverrun > 0 {
		parts = append(parts, strconv.Itoa(ev.DetourOverrun)+" detour-overrun")
	}
	if ev.Drift > 0 {
		parts = append(parts, strconv.Itoa(ev.Drift)+" drifting")
	}
	if ev.Stall > 0 {
		parts = append(parts, strconv.Itoa(ev.Stall)+" stalled")
	}
	if ev.ExcessWIP > 0 {
		parts = append(parts, strconv.Itoa(ev.ExcessWIP)+" over WIP cap")
	}
	if len(parts) == 0 {
		return "clean"
	}
	return strings.Join(parts, ", ")
}

// worstFirstNext returns the single most-actionable next move, in the same severity order
// the curve fold ranks signals (overrun > drift > stall > breadth).
func worstFirstNext(ev Evidence) string {
	switch {
	case ev.DetourOverrun > 0:
		return "return " + strconv.Itoa(ev.DetourOverrun) + " over-budget detour(s) to their paused parent"
	case ev.Drift > 0:
		return "arrest " + strconv.Itoa(ev.Drift) + " drifting objective(s) — the witnessed progress curve is declining"
	case ev.Stall > 0:
		return "unstick " + strconv.Itoa(ev.Stall) + " stalled objective(s) — activity without witnessed movement"
	case ev.ExcessWIP > 0:
		return "pause or meet " + strconv.Itoa(ev.ExcessWIP) + " active objective(s) to get back under the WIP cap of " + strconv.Itoa(ev.WIPCap)
	default:
		return "converge the open objectives"
	}
}

// ---- render -----------------------------------------------------------------------

func Render(p ScorecardPayload) string {
	c := p.Corpus
	lines := []string{
		"focus-score — " + p.Verdict + " (" + p.Finding + ")",
		"  focus_debt: " + scorecard.MetricText(c[DebtKey]) + "   value " + scorecard.MetricText(c["value"]) +
			" [" + scorecard.MetricText(c["grade"]) + "]   (convergence value " + scorecard.MetricText(c["convergence_value"]) +
			"; breadth value " + scorecard.MetricText(c["breadth_value"]) + ")",
		"  evidence: " + scorecard.MetricText(c["objectives"]) + " objective(s); " + scorecard.MetricText(c["active"]) +
			" active (cap " + scorecard.MetricText(c["wip_cap"]) + "); " + scorecard.MetricText(c["paused"]) + " paused; " +
			scorecard.MetricText(c["met"]) + " met; " + scorecard.MetricText(c["healthy"]) + "/" + scorecard.MetricText(c["open"]) + " open healthy; " +
			scorecard.MetricText(c["drift"]) + " drift; " + scorecard.MetricText(c["stall"]) + " stall; " + scorecard.MetricText(c["detour_overrun"]) + " overrun",
		"",
		"  CONVERGENCE (are the open objectives moving toward their goal?):",
	}
	for _, r := range p.Convergence {
		lines = append(lines, "    "+scorecard.PassMark(r.Passed)+" "+r.Label+"  ["+r.Detail+"]")
	}
	lines = append(lines, "", "  BREADTH (is concurrency bounded, or fanning out too broad?):")
	for _, r := range p.Breadth {
		lines = append(lines, "    "+scorecard.PassMark(r.Passed)+" "+r.Label+"  ["+r.Detail+"]")
	}
	lines = append(lines, "", "  NEXT: "+p.NextAction)
	return strings.Join(lines, "\n")
}

func Markdown(p ScorecardPayload) string {
	c := p.Corpus
	var b strings.Builder
	b.WriteString("---\n")
	b.WriteString(`title: "fak focus scorecard"` + "\n")
	b.WriteString(`description: "Whether the fleet is CONVERGING on its live goal or fanning out too broad — bounded work-in-progress and every open objective's witnessed progress curve rising — re-derived from the trajectory-control ledger fak writes, never a self-report."` + "\n")
	b.WriteString("---\n\n")
	b.WriteString("# fak focus scorecard\n\n")
	b.WriteString("**focus_debt: " + scorecard.MetricText(c[DebtKey]) + "**; value **" + scorecard.MetricText(c["value"]) +
		" (" + scorecard.MetricText(c["grade"]) + ")**; convergence value " + scorecard.MetricText(c["convergence_value"]) +
		"; breadth value " + scorecard.MetricText(c["breadth_value"]) + "\n\n")
	b.WriteString("> " + p.Reason + "\n\n")
	b.WriteString("The question: is the fleet *converging on its live goal*, or fanning out too broad — declaring many objectives active at once, taking detours that run past budget while their parent sits paused, letting open objectives drift or stall instead of moving? `internal/trajctl` already classifies a *single* objective's witnessed progress curve (HEALTHY/STALL/DRIFT/DETOUR_OVERRUN); this scorecard folds the whole objective **tree** into one focus number. Every count is re-derived from the trajectory-control ledger (`" + trajctl.DefaultLedgerRel + "`) fak's own `fak trajctl` tooling writes — so the score moves only when the fleet actually converges: close or meet objectives, return detours to their paused parent, get the witnessed curves rising.\n\n")
	b.WriteString("## Convergence — are the open objectives moving toward their goal?\n\n| ok | criterion | detail |\n|---|---|---|\n")
	for _, r := range p.Convergence {
		b.WriteString("| " + scorecard.PassMark(r.Passed) + " | " + r.Label + " | " + r.Detail + " |\n")
	}
	b.WriteString("\n## Breadth — is concurrency bounded, or fanning out too broad?\n\n| ok | criterion | detail |\n|---|---|---|\n")
	for _, r := range p.Breadth {
		b.WriteString("| " + scorecard.PassMark(r.Passed) + " | " + r.Label + " | " + r.Detail + " |\n")
	}
	b.WriteString("\n## Run it\n\n```bash\ngo run ./cmd/fak focus-score             # score the fleet's convergence on its goal\ngo run ./cmd/fak focus-score --markdown  # regenerate this doc\ngo run ./cmd/fak focus-score --json      # control-pane payload (corpus.focus_debt)\ngo test ./internal/focusscore/...        # prove the fold over a broad vs converged corpus\n```\n\n")
	b.WriteString("## Driving focus_debt down\n\n")
	b.WriteString("focus_debt is unbounded and counted at magnitude: each objective over the WIP cap, each open objective the curve fold marks DRIFT / STALL / DETOUR_OVERRUN is one debt. You cannot lower it by editing a file — the counts are re-folded from the witnessed ledger. You lower it by **converging**:\n\n")
	b.WriteString("1. **Return over-budget detours.** A DETOUR_OVERRUN is a child objective that ran past its declared turn budget while its parent is paused — close it out or hand its remainder back and un-pause the parent.\n")
	b.WriteString("2. **Arrest drift, unstick stalls.** A DRIFTing objective's witnessed curve is *declining* (e.g. a re-score after a commit went dangling); a STALLed one is busy but flat. Both mean the live work is not producing witnessed progress — fix the objective or abandon it honestly.\n")
	b.WriteString("3. **Bound the WIP.** More than " + scorecard.MetricText(c["wip_cap"]) + " active objectives at once is fan-out, not focus. Pause the ones that can wait behind the one live goal so the fleet converges on it first.\n\n")
	b.WriteString("Re-run after a session and `--compare` against a pinned `--json` baseline: the verdict reports the debt retired and whether the fleet converged.\n\n")
	b.WriteString("**Next:** " + p.NextAction + "\n")
	return b.String()
}

func Compare(current ScorecardPayload, baseline map[string]any) string {
	bc, _ := baseline["corpus"].(map[string]any)
	if bc == nil {
		bc = baseline
	}
	bDebt := scorecard.IntValue(bc[DebtKey])
	cDebt := scorecard.IntValue(current.Corpus[DebtKey])
	bScore := scorecard.IntValue(bc["score"])
	cScore := scorecard.IntValue(current.Corpus["score"])
	lines := []string{
		"focus-score compare:",
		"  focus_debt: " + strconv.Itoa(bDebt) + " -> " + strconv.Itoa(cDebt) + "  (retired " + strconv.Itoa(bDebt-cDebt) + ")",
		"  value: " + scorecard.MetricText(bc["value"]) + " -> " + scorecard.MetricText(current.Corpus["value"]) +
			"  grade " + scorecard.MetricText(bc["grade"]) + " -> " + scorecard.MetricText(current.Corpus["grade"]),
	}
	switch {
	case bDebt > 0 && cDebt == 0:
		lines = append(lines, "  VERDICT: fleet converged — all focus debt retired")
	case cDebt < bDebt:
		lines = append(lines, "  VERDICT: converging ("+strconv.Itoa(bDebt)+" -> "+strconv.Itoa(cDebt)+" debt, composite "+strconv.Itoa(bScore)+" -> "+strconv.Itoa(cScore)+")")
	case cDebt > bDebt:
		lines = append(lines, "  VERDICT: fanning out ("+strconv.Itoa(bDebt)+" -> "+strconv.Itoa(cDebt)+" debt)")
	default:
		lines = append(lines, "  VERDICT: no change")
	}
	return strings.Join(lines, "\n")
}

// ---- small helpers (mirror loopscore idiom) ---------------------------------------

func result(key, axis string, hard bool, weight int, label string, passed bool, detail string) KPIResult {
	return KPIResult{Key: key, Label: label, Hard: hard, Weight: weight, Axis: axis, Passed: passed, Detail: detail}
}

func boolStr(ok bool, yes, no string) string {
	if ok {
		return yes
	}
	return no
}
