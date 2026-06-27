package dojo

// dojo.go holds the pure scoring + fold + render + gate surface: the Prediction
// / Outcome / Episode types, Score (one prediction scored against one measured
// outcome), the calibration band + letter grade, and Fold (a run's episodes
// rolled into one control-pane envelope). The durable ledger + trend live in
// ledger.go; the Scenario/Lever runner lives in run.go; the package doc is in
// doc.go.

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// Schema is the stable control-pane schema identifier for the report envelope.
const Schema = "fak-dojo-report/1"

// LedgerSchema tags each durable history row so a reader can validate the line.
const LedgerSchema = "fak-dojo-ledger/1"

// DefaultLedgerRel is the committed, append-only history ledger (one JSONL row
// per dojo tick). It lives under docs/ so it is durable trunk evidence, not a
// regenerable build artifact — the same posture the cadence ledger keeps.
const DefaultLedgerRel = "docs/dojo/history.jsonl"

// MaxCalibErr caps a single episode's normalized residual so a claim near zero
// that reality refutes cannot dominate the fold with an unbounded ratio.
const MaxCalibErr = 2.0

const gradeNA = "n/a"

// Provenance keeps every number honest about whose it is: a WITNESSED value is
// one fak authored and controls; an OBSERVED value is relayed from an upstream
// party (the model provider) and fak does not control it. A bad OBSERVED number
// is not, by itself, a fak fault — the same boundary the exit summary draws.
type Provenance string

const (
	// Witnessed marks a number fak authored and controls.
	Witnessed Provenance = "WITNESSED"
	// Observed marks a number relayed from an upstream party (the provider).
	Observed Provenance = "OBSERVED"
)

// Verdict names for one scored episode.
const (
	// VerdictCalibrated: reality met the claim within the calibration band.
	VerdictCalibrated = "CALIBRATED"
	// VerdictOverClaim: reality fell short of the claim (the theory promised more
	// than billed reality delivered).
	VerdictOverClaim = "OVER_CLAIM"
	// VerdictUnderClaim: reality exceeded the claim (the theory under-promised; a
	// saving the model is not crediting).
	VerdictUnderClaim = "UNDER_CLAIM"
	// VerdictUnmeasured: no ground truth existed to score the claim against.
	VerdictUnmeasured = "UNMEASURED"
)

// Prediction is the THEORY a lever declares for one metric BEFORE billed reality
// is consulted: the Claimed number and the Basis that produced it.
type Prediction struct {
	Lever   string  `json:"lever"`
	Metric  string  `json:"metric"`
	Claimed float64 `json:"claimed"`
	Unit    string  `json:"unit"`
	Basis   string  `json:"basis"`
}

// Outcome is the measured reality for the same metric, lifted from the provider's
// own usage records. Measured is false when no ground truth existed (the episode
// scores UNMEASURED rather than a misleading zero); Sample is how many
// boundaries/turns stand behind the number.
type Outcome struct {
	Realized   float64    `json:"realized"`
	Provenance Provenance `json:"provenance"`
	Source     string     `json:"source"`
	Measured   bool       `json:"measured"`
	Sample     int        `json:"sample"`
}

// Episode is one scored prediction-vs-reality: the gap (Residual), its
// normalized magnitude (CalibErr), the Verdict (did reality meet / fall short of
// / exceed the claim), and the letter Grade.
type Episode struct {
	Scenario   string     `json:"scenario"`
	Lever      string     `json:"lever"`
	Metric     string     `json:"metric"`
	Unit       string     `json:"unit"`
	Claimed    float64    `json:"claimed"`
	Realized   float64    `json:"realized"`
	Residual   float64    `json:"residual"`
	CalibErr   float64    `json:"calib_err"`
	Verdict    string     `json:"verdict"`
	Grade      string     `json:"grade"`
	Provenance Provenance `json:"provenance"`
	Source     string     `json:"source"`
	Sample     int        `json:"sample"`
	Basis      string     `json:"basis"`
}

// CalibBand maps a normalized residual to a verdict and a letter grade. A
// residual at or under CalibratedMax is CALIBRATED (grade A); the rest of the
// ladder grades the miss by magnitude.
type CalibBand struct {
	CalibratedMax float64 `json:"calibrated_max"`
	GradeB        float64 `json:"grade_b"`
	GradeC        float64 `json:"grade_c"`
	GradeD        float64 `json:"grade_d"`
}

// DefaultCalibBand is the conservative scoring band: within 10% of the claim is
// calibrated (A), then B/C/D widen out to a 60% miss, beyond which is F.
func DefaultCalibBand() CalibBand {
	return CalibBand{CalibratedMax: 0.10, GradeB: 0.20, GradeC: 0.35, GradeD: 0.60}
}

func (b CalibBand) grade(ce float64) string {
	switch {
	case ce <= b.CalibratedMax:
		return "A"
	case ce <= b.GradeB:
		return "B"
	case ce <= b.GradeC:
		return "C"
	case ce <= b.GradeD:
		return "D"
	default:
		return "F"
	}
}

// Score folds one prediction and its measured outcome into a scored episode. It
// is pure and total: an unmeasured outcome yields an UNMEASURED episode (never a
// scored zero), and a claim of "nothing" that reality confirms scores as
// perfectly calibrated rather than as an unbounded relative error.
func Score(scenario string, p Prediction, o Outcome, band CalibBand) Episode {
	e := Episode{
		Scenario:   scenario,
		Lever:      p.Lever,
		Metric:     p.Metric,
		Unit:       p.Unit,
		Claimed:    p.Claimed,
		Realized:   o.Realized,
		Provenance: o.Provenance,
		Source:     o.Source,
		Sample:     o.Sample,
		Basis:      p.Basis,
	}
	if !o.Measured {
		e.Verdict = VerdictUnmeasured
		e.Grade = gradeNA
		return e
	}
	e.Residual = o.Realized - p.Claimed
	e.CalibErr = calibErr(p.Claimed, o.Realized)
	switch {
	case e.CalibErr <= band.CalibratedMax:
		e.Verdict = VerdictCalibrated
	case e.Residual < 0:
		e.Verdict = VerdictOverClaim
	default:
		e.Verdict = VerdictUnderClaim
	}
	e.Grade = band.grade(e.CalibErr)
	return e
}

// calibErr is the normalized magnitude of the residual, capped at MaxCalibErr.
// It divides by the claim, falling back to the realized magnitude when the claim
// is ~0, and returns 0 when both are ~0 (the claim of "nothing" held).
func calibErr(claimed, realized float64) float64 {
	resid := math.Abs(realized - claimed)
	denom := math.Abs(claimed)
	if denom < 1e-9 {
		denom = math.Abs(realized)
	}
	if denom < 1e-9 {
		return 0
	}
	ce := resid / denom
	if ce > MaxCalibErr {
		return MaxCalibErr
	}
	return ce
}

// FoldOpts carries the ambient context the fold stamps onto the envelope.
type FoldOpts struct {
	Workspace   string
	Commit      string
	GeneratedAt string
	Date        string
}

// Report is one folded dojo control-pane envelope: the schema/ok/verdict/finding/
// reason/next_action header, the per-run aggregate (lever/episode/calibrated
// counts + the mean calibration error + overall grade), the scored episodes, and
// the optional per-tick trend.
type Report struct {
	Schema      string `json:"schema"`
	OK          bool   `json:"ok"`
	Verdict     string `json:"verdict"`
	Finding     string `json:"finding"`
	Reason      string `json:"reason"`
	NextAction  string `json:"next_action"`
	Workspace   string `json:"workspace"`
	Commit      string `json:"commit"`
	GeneratedAt string `json:"generated_at"`
	Date        string `json:"date"`

	LeverCount   int     `json:"lever_count"`
	EpisodeCount int     `json:"episode_count"`
	Measured     int     `json:"measured"`
	Unmeasured   int     `json:"unmeasured"`
	Calibrated   int     `json:"calibrated"`
	MeanCalibErr float64 `json:"mean_calib_err"`
	Grade        string  `json:"grade"`

	Episodes []Episode `json:"episodes"`
	Trend    *Trend    `json:"trend,omitempty"`

	// gate fields, set only for the --check --json envelope.
	GateExit    *int   `json:"gate_exit,omitempty"`
	GateMessage string `json:"gate_message,omitempty"`
}

// Fold rolls a run's episodes into one control-pane envelope. The verdict ladder
// is a REPORT contract, not a second quality gate: it is ACTION only when the run
// could not be MEASURED (no episodes, or none with ground truth) and OK
// otherwise, surfacing any over-claim as an advisory line — the same advisory
// posture the cadence report keeps, so the gym measures without double-gating.
func Fold(episodes []Episode, opts FoldOpts) Report {
	r := Report{
		Schema:       Schema,
		Workspace:    opts.Workspace,
		Commit:       opts.Commit,
		GeneratedAt:  opts.GeneratedAt,
		Date:         opts.Date,
		Episodes:     episodes,
		EpisodeCount: len(episodes),
	}

	levers := map[string]struct{}{}
	var sumCE float64
	var overs, unders []string
	for _, e := range episodes {
		levers[e.Lever] = struct{}{}
		if e.Verdict == VerdictUnmeasured {
			r.Unmeasured++
			continue
		}
		r.Measured++
		sumCE += e.CalibErr
		switch e.Verdict {
		case VerdictCalibrated:
			r.Calibrated++
		case VerdictOverClaim:
			overs = append(overs, e.Lever+"/"+e.Metric)
		case VerdictUnderClaim:
			unders = append(unders, e.Lever+"/"+e.Metric)
		}
	}
	r.LeverCount = len(levers)
	if r.Measured > 0 {
		r.MeanCalibErr = sumCE / float64(r.Measured)
	}
	r.Grade = DefaultCalibBand().grade(r.MeanCalibErr)

	summary := fmt.Sprintf("%d lever(s), %d episode(s): %d calibrated, %d over-claim, %d under-claim, %d unmeasured; mean calib-err %.3f (grade %s)",
		r.LeverCount, r.EpisodeCount, r.Calibrated, len(overs), len(unders), r.Unmeasured, r.MeanCalibErr, r.Grade)

	switch {
	case r.EpisodeCount == 0:
		r.OK, r.Verdict, r.Finding = false, "ACTION", "dojo_empty"
		r.Reason = "no episodes — no lever produced a prediction over the scenario(s)"
		r.NextAction = "register a lever and a scenario with a real corpus, then re-run `fak dojo run`"
	case r.Measured == 0:
		r.OK, r.Verdict, r.Finding = false, "ACTION", "dojo_unmeasured"
		r.Reason = "dojo run incomplete — no episode had ground truth to score against (" + summary + ")"
		r.NextAction = "point the scenario at a corpus with billed usage records so the predictions can be scored"
	default:
		r.OK, r.Verdict, r.Finding = true, "OK", "dojo_recorded"
		r.Reason = "dojo recorded; " + summary
		if len(overs) > 0 {
			r.Reason += " — advisory: over-claim(s) at " + strings.Join(overs, ", ") + " (a theory promising more than billed reality delivered)"
		}
		r.NextAction = nextAction(overs, unders)
	}
	return r
}

func nextAction(overs, unders []string) string {
	switch {
	case len(overs) > 0:
		return "recalibrate the over-claiming lever(s) so the theory matches billed reality; the dojo tick keeps trending the gap"
	case len(unders) > 0:
		return "harvest the under-claimed saving(s) — billed reality beat the theory, so there is free savings the model is not crediting"
	default:
		return "hold the line; the scheduled dojo tick keeps the calibration trended"
	}
}

// Render produces the human snapshot: the header, the per-run aggregate, one row
// per scored episode, the optional trend, and the next action.
func Render(r Report) string {
	lines := []string{
		fmt.Sprintf("dojo report — %s (%s)  grade %s  @%s  %s", r.Verdict, r.Finding, r.Grade, shortCommit(r.Commit), r.Date),
		"",
		fmt.Sprintf("  %d lever(s) · %d episode(s) · %d measured · %d calibrated · mean calib-err %.3f",
			r.LeverCount, r.EpisodeCount, r.Measured, r.Calibrated, r.MeanCalibErr),
		"",
		fmt.Sprintf("  %-15s %-26s %10s %10s %9s  %-11s %-3s %-9s %s",
			"lever", "metric", "claimed", "realized", "calib_err", "verdict", "grd", "prov", "n"),
	}
	for _, e := range r.Episodes {
		lines = append(lines, fmt.Sprintf("  %-15s %-26s %10.3f %10.3f %9.3f  %-11s %-3s %-9s %d",
			truncate(e.Lever, 15), truncate(e.Metric, 26), e.Claimed, e.Realized, e.CalibErr, e.Verdict, e.Grade, e.Provenance, e.Sample))
	}
	if r.Trend != nil {
		lines = append(lines, "", "  trend: "+r.Trend.Summary)
	}
	lines = append(lines, "", "  -> "+r.NextAction)
	return strings.Join(lines, "\n")
}

// CheckGate is the advisory CI gate over a folded report: it fails ONLY when the
// run could not be measured (empty or no-ground-truth), never on an over-claim —
// the dojo is a measurement mirror, not a second quality gate.
//
//	0  dojo recorded (clear or over-claim advisory)
//	1  the run could not be measured (incomplete)
func CheckGate(r Report) (int, string) {
	if r.Finding == "dojo_empty" || r.Finding == "dojo_unmeasured" {
		return 1, "DOJO INCOMPLETE: " + r.Reason
	}
	return 0, "DOJO OK: " + r.Reason
}

// WithGate returns a copy reconciled to a CheckGate decision, for --check --json.
func (r Report) WithGate(code int, message string) Report {
	q := r
	q.OK = code == 0
	if code == 0 {
		q.Verdict = "OK"
	} else {
		q.Verdict = "ACTION"
	}
	c := code
	q.GateExit = &c
	q.GateMessage = message
	return q
}

// SortEpisodes orders episodes for a stable, worst-first render: by descending
// calibration error, then lever, then metric. UNMEASURED episodes (calib-err 0)
// sort to the end of the scored block by their zero magnitude.
func SortEpisodes(eps []Episode) {
	sort.SliceStable(eps, func(i, j int) bool {
		if eps[i].CalibErr != eps[j].CalibErr {
			return eps[i].CalibErr > eps[j].CalibErr
		}
		if eps[i].Lever != eps[j].Lever {
			return eps[i].Lever < eps[j].Lever
		}
		return eps[i].Metric < eps[j].Metric
	})
}

func shortCommit(c string) string {
	if c == "" {
		return "unknown"
	}
	if len(c) > 12 {
		return c[:12]
	}
	return c
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if n <= 1 {
		return s[:n]
	}
	return s[:n-1] + "…"
}
