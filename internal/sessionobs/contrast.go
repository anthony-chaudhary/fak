package sessionobs

// contrast.go -- the LEARN rung made concrete: the value-vs-waste behavior CONTRAST
// that turns the scrubbed corpus from a graded pile of records into something a loop
// can ACT on. The scorecard in sessionobs.go grades how far up the ladder the pipeline
// has climbed; this is what the top rung (learn) actually DOES once it is built.
//
// THE QUESTION IT ANSWERS. Cost data says what a session spent; the outcome link says
// whether it shipped or stalled. The contrast asks the next question -- WHICH behaviors
// separate the sessions that shipped from the sessions that stalled? Hold the value-side
// cohort (Shipped + Claimed) against the waste-side cohort (Stopped) and, for each
// behavior feature, compare its mean across the two. A feature whose mean differs
// sharply between the cohorts is a candidate early-warning the agent can act on; a
// feature that is flat across both teaches nothing.
//
// THE NON-CIRCULARITY LAW. A contrast is only learning if the features are NOT the ones
// the outcome was derived from. Commits define the value cohort (a value session
// committed by construction); stop-events and interrupts define the waste cohort (a
// Stopped outcome is exactly "no commit AND a stop/interrupt"). "Discovering" that
// those separate the cohorts is tautology, not a finding. So contrastFeatures
// deliberately EXCLUDES commits, stop_events, and interrupts, and ranks only the
// behaviors an agent genuinely controls mid-session -- guard friction, tool-error
// pressure, reconnaissance depth, session length, goal framing -- whose correlation
// with the outcome is a real, non-circular signal.
//
// It stays pure and deterministic like the rest of the package: stdlib-only, no clock,
// no RNG, same corpus in -> same contrast out. That determinism is what lets a finding
// be trusted as more than a one-run fluke, the same discipline the scorer obeys.

import (
	"fmt"
	"io"
	"sort"
)

const contrastSchema = "fak.sessionobs.contrast.v1"

// minContrastCohort is the floor on each cohort for the contrast to be reported as
// SEPARABLE and a headline finding emitted. Below it, a mean over one or two sessions
// is noise, so the report honestly abstains rather than trumpet a fluke -- the same
// empty-journal-honesty law the guard-verdict loop obeys.
const minContrastCohort = 3

// minHeadlineSep is the separation a feature must clear to be promoted to the headline.
// A separable corpus whose strongest feature is still nearly flat has no actionable
// discriminator, and the report says exactly that instead of over-reading the noise.
const minHeadlineSep = 0.15

// featureDef binds a feature's stable wire name to the per-record extractor that
// projects it. The slice order is fixed so the unranked feature set is deterministic
// before sorting.
type featureDef struct {
	name string
	of   func(Record) float64
}

// contrastFeatures is the FIXED, ordered set of NON-DEFINITIONAL behaviors the contrast
// ranks. See the non-circularity law in the file header for why commits / stop_events /
// interrupts are absent: they define the cohorts, so they cannot be evidence about them.
var contrastFeatures = []featureDef{
	{"guard_refusals", func(r Record) float64 { return float64(r.Signals.GuardRefusals) }},
	{"tool_errors", func(r Record) float64 { return float64(r.Signals.ToolErrors) }},
	{"tool_error_rate", func(r Record) float64 { return safeRate(r.Signals.ToolErrors, r.ToolCalls) }},
	{"read_only_frac", func(r Record) float64 { return safeRate(r.ReadOnlyCalls, r.ToolCalls) }},
	{"assistant_turns", func(r Record) float64 { return float64(r.AssistantTurns) }},
	{"tool_calls", func(r Record) float64 { return float64(r.ToolCalls) }},
	{"output_ktokens", func(r Record) float64 { return float64(r.OutputTokens) / 1000 }},
	{"goal_events", func(r Record) float64 { return float64(r.Signals.GoalEvents) }},
}

// FeatureContrast is one behavior feature's value-vs-waste comparison.
type FeatureContrast struct {
	Name        string  `json:"feature"`
	ValueMean   float64 `json:"value_mean"`   // mean of the feature over value-side sessions
	WasteMean   float64 `json:"waste_mean"`   // mean over waste-side sessions
	Separation  float64 `json:"separation"`   // |Δ| / (value+waste): scale-free, in [0,1]
	WasteMarker bool    `json:"waste_marker"` // true when MORE of this feature predicts WASTE
	Lift        string  `json:"lift"`         // human ratio of the larger cohort mean to the smaller
	Detail      string  `json:"detail"`       // one-line human summary
}

// ContrastReport is the learn-rung output: every behavior feature ranked by how
// strongly it separates value from waste, plus the single headline finding a loop would
// act on (or an honest reason why no finding is warranted yet).
type ContrastReport struct {
	Schema     string            `json:"schema"`
	ValueN     int               `json:"value_n"`     // value-side (Shipped+Claimed) session count
	WasteN     int               `json:"waste_n"`     // waste-side (Stopped) session count
	Separable  bool              `json:"separable"`   // both cohorts meet minContrastCohort
	TopFeature string            `json:"top_feature"` // strongest discriminator name ("" if none warranted)
	Headline   string            `json:"headline"`    // the one actionable finding, or the why-not
	NextAction string            `json:"next_action"` // worst-first recovery / next step
	Features   []FeatureContrast `json:"features"`    // ranked by Separation desc, then name asc
}

// Contrast is the whole learn engine: a pure, deterministic function from a scrubbed
// corpus to the value-vs-waste behavior contrast. NoOp and Unknown records sit out --
// they are neither value nor waste, so they would only dilute the comparison.
func Contrast(corpus []Record) ContrastReport {
	var value, waste []Record
	for _, r := range corpus {
		switch {
		case r.Outcome.value():
			value = append(value, r)
		case r.Outcome == OutcomeStopped:
			waste = append(waste, r)
		}
	}

	rep := ContrastReport{Schema: contrastSchema, ValueN: len(value), WasteN: len(waste)}
	for _, f := range contrastFeatures {
		vm := meanOf(value, f.of)
		wm := meanOf(waste, f.of)
		fc := FeatureContrast{
			Name:        f.name,
			ValueMean:   round3(vm),
			WasteMean:   round3(wm),
			Separation:  round3(separation(vm, wm)),
			WasteMarker: wm > vm,
			Lift:        liftStr(vm, wm),
		}
		dir := "more predicts value"
		if fc.WasteMarker {
			dir = "more predicts waste"
		}
		fc.Detail = fmt.Sprintf("value avg %.2f vs waste avg %.2f (%s); %s",
			fc.ValueMean, fc.WasteMean, fc.Lift, dir)
		rep.Features = append(rep.Features, fc)
	}
	// Rank by separation (strongest discriminator first); ties broken by name so the
	// order is total and deterministic.
	sort.SliceStable(rep.Features, func(i, j int) bool {
		if rep.Features[i].Separation != rep.Features[j].Separation {
			return rep.Features[i].Separation > rep.Features[j].Separation
		}
		return rep.Features[i].Name < rep.Features[j].Name
	})

	rep.Separable = len(value) >= minContrastCohort && len(waste) >= minContrastCohort
	rep.Headline, rep.TopFeature, rep.NextAction = contrastVerdict(rep)
	return rep
}

// contrastVerdict picks the headline finding from a ranked report, honestly abstaining
// when the corpus is too thin to separate the cohorts or too flat to carry a signal.
func contrastVerdict(rep ContrastReport) (headline, top, next string) {
	if !rep.Separable {
		return fmt.Sprintf("insufficient contrast: need >=%d value and >=%d waste sessions, have value=%d waste=%d",
				minContrastCohort, minContrastCohort, rep.ValueN, rep.WasteN),
			"",
			"ingest more sessions (`fak sessions score --corpus ...`) until both the value and waste cohorts are populated"
	}
	if len(rep.Features) == 0 || rep.Features[0].Separation < minHeadlineSep {
		return "no behavior feature strongly separates value from waste in this corpus",
			"",
			"the corpus is separable but flat -- widen the sample or the feature set before acting on any single feature"
	}
	t := rep.Features[0]
	dir := "predicts value"
	if t.WasteMarker {
		dir = "predicts waste"
	}
	headline = fmt.Sprintf("%s %s: waste sessions average %.2f vs %.2f in value sessions (%s)",
		t.Name, dir, t.WasteMean, t.ValueMean, t.Lift)
	return headline, t.Name, featureAction(t)
}

// featureAction maps the headline feature to a concrete, worst-first next step. It is
// the bridge from "this feature separates the cohorts" to "here is what to change",
// keyed on the direction so a value marker reads as "do more of this" and a waste
// marker as "catch this early".
func featureAction(t FeatureContrast) string {
	switch t.Name {
	case "guard_refusals":
		if t.WasteMarker {
			return "guard friction precedes stalls -- surface the AGENTS.md recovery table earlier when refusals spike, before the session fights the guard into a STOP"
		}
		return "value sessions provoke more refusals and still ship -- refusals alone are not the waste signal here"
	case "tool_errors", "tool_error_rate":
		if t.WasteMarker {
			return "clustered tool errors precede stalls -- add an early replan/abort when errors repeat instead of grinding the same call"
		}
		return "value sessions tolerate more tool errors and still ship -- error count alone is not the waste signal"
	case "read_only_frac":
		if t.WasteMarker {
			return "waste sessions read more without acting -- watch for reconnaissance that never converts to an edit"
		}
		return "value sessions read more before acting -- bias toward reconnaissance before the first mutation"
	case "assistant_turns", "tool_calls", "output_ktokens":
		if t.WasteMarker {
			return "waste sessions run long with nothing to show -- cap the expensive tail; a session past the value-cohort mean with no commit is a stall candidate"
		}
		return "value sessions do more work per session -- length here tracks delivery, not waste"
	case "goal_events":
		if t.WasteMarker {
			return "sessions under a /goal directive stall more often -- check whether goal framing is over-scoping the work"
		}
		return "sessions under a /goal directive ship more often -- the goal framing is helping; keep it"
	default:
		return "investigate why this feature separates the cohorts, then encode it as an early-warning the agent acts on"
	}
}

// RenderContrast writes the human work-list: the cohort sizes and headline, then every
// feature ranked worst-first (strongest discriminator at the top).
func RenderContrast(w io.Writer, rep ContrastReport) {
	fmt.Fprintf(w, "session-obs learn (value-vs-waste contrast): value=%d waste=%d separable=%v\n",
		rep.ValueN, rep.WasteN, rep.Separable)
	fmt.Fprintf(w, "  headline: %s\n", rep.Headline)
	fmt.Fprintf(w, "  next: %s\n", rep.NextAction)
	for _, f := range rep.Features {
		dir := "->value"
		if f.WasteMarker {
			dir = "->waste"
		}
		fmt.Fprintf(w, "  %-16s sep %4.2f  value %8.2f  waste %8.2f  %-10s %s\n",
			f.Name, f.Separation, f.ValueMean, f.WasteMean, f.Lift, dir)
	}
}

// --- math helpers (pure) ---------------------------------------------------

// meanOf is the arithmetic mean of a feature over a cohort; an empty cohort is 0.
func meanOf(rs []Record, of func(Record) float64) float64 {
	if len(rs) == 0 {
		return 0
	}
	var sum float64
	for _, r := range rs {
		sum += of(r)
	}
	return sum / float64(len(rs))
}

// separation is the scale-free distance between two non-negative means: |a-b|/(a+b),
// in [0,1]. 0 means identical (or both zero); 1 means one cohort is entirely zero.
func separation(a, b float64) float64 {
	d := a - b
	if d < 0 {
		d = -d
	}
	s := a + b
	if s == 0 {
		return 0
	}
	return d / s
}

// liftStr renders the ratio of the larger cohort mean to the smaller as a human token.
// When the smaller mean is zero, the larger cohort is the sole carrier of the feature.
func liftStr(value, waste float64) string {
	hi, lo := value, waste
	if waste > value {
		hi, lo = waste, value
	}
	if lo <= 0 {
		if hi <= 0 {
			return "flat"
		}
		if waste > value {
			return "waste-only"
		}
		return "value-only"
	}
	return fmt.Sprintf("%.1fx", hi/lo)
}

// safeRate is a/b as a float, 0 when b is non-positive (a session with no tool calls
// has no error rate and no read-only fraction).
func safeRate(a, b int) float64 {
	if b <= 0 {
		return 0
	}
	return float64(a) / float64(b)
}
