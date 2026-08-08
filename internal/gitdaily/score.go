package gitdaily

// The Daily lock-aware Git hygiene HEALTH scorecard (#5587, follow-on to the #5577 spine).
//
// WHY A CARD AND NOT A STATUS DUMP. `fak git-daily --status` already prints the recent rows
// and the #5586 outcome counters, but reading "is this job still good?" off that dump is an
// AUDIT: a human has to notice that the last row is nine days old, or that four of the last
// six ticks deferred. This fold turns the same rows into one graded verdict with named
// evidence, so the question is a command instead.
//
// WHY IT SCORES THE LEDGER AND NOTHING ELSE. The job's only durable witness is the
// `fak-git-daily/1` rows it appends itself; there is no separate telemetry to consult and no
// self-report to trust. Every number below is derived from those rows plus the reference
// day, so the card cannot claim health the ledger does not show. The ledger PATH travels in
// the corpus so a captured card names the file it was folded from.
//
// THE THREE LENSES, AND WHY THEY ARE SEPARATE. A daily job fails in three independent ways
// and each has a different repair:
//
//   - usage  -- it is installed but does not fire every day (repair: the scheduler)
//   - health -- it fires but its tiers keep deferring (repair: the refusal reason)
//   - drift  -- it fired for a while and then stopped (repair: find out when and why)
//
// A single blended number would let a perfect success rate over three ancient ticks read as
// healthy, which is exactly the #4602 shape this card exists to catch.
//
// WHY THE USAGE WINDOW STARTS AT THE FIRST RECORDED RUN. A clone cannot have covered a day
// before the job existed in it. Grading a two-day-old ledger against a fixed 14-day
// denominator would score every fresh install an F and train an operator to ignore the card.
// The window is therefore [max(first recorded day, ref-window+1) .. ref], and a job that
// STOPPED still reads as missed days, because those days are after its first run.

import (
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// HealthSchema tags this card so a control-pane consumer reads it apart from every other one.
const HealthSchema = "fak-git-daily-health-scorecard/1"

// HealthDebtKey is the corpus debt integer: the count of lenses below the pass line.
const HealthDebtKey = "git_daily_debt"

// HealthMissedKey is the one UNBOUNDED headline -- graded days in the window with no
// recorded tick. Unlike the 0..1 lens fractions it does not saturate, so a job that has been
// dark twice as long reads twice as heavy.
const HealthMissedKey = "git_daily_missed_days"

// DefaultHealthWindow is the trailing day count graded when the caller names none. Two weeks
// is long enough that a single skipped day does not swing the grade and short enough that a
// job repaired last week reads as repaired.
const DefaultHealthWindow = 14

// HealthPassLine is the 0..1 fraction at/above which a lens is not in debt. It is ONE
// conservative floor an operator TIGHTENS over time (the ratchet knob), not a per-lens tuned
// target: every lens is retired by fixing the REAL defect (install the schedule, clear the
// refusal, restart the job), never by lowering this number.
const HealthPassLine = 0.8

// The canonical lens keys. Exported so a consumer addresses one lens without string-matching
// a detail line.
const (
	LensUsage  = "git_daily_usage"
	LensHealth = "git_daily_health"
	LensDrift  = "git_daily_drift"
	// LensEvidence stands in for all three when the ledger carries no dated row at all.
	LensEvidence = "git_daily_evidence"
)

// HealthFacts is the caller-supplied input: the rows to grade, the reference day, the
// trailing window, and the ledger path the rows came from. Now is injectable so the fold is
// deterministic under test; zero means time.Now() at call. Ledger is evidence only -- it is
// never read here, so this core stays pure over its arguments.
type HealthFacts struct {
	Rows   []Row
	Now    time.Time
	Window int
	Ledger string
}

// Health is the raw evidence behind the grade: the graded span, what it covered, and the
// #5586 outcome tally restricted to that span. Exported so a caller can assert on the
// evidence rather than parse a rendered detail line.
type Health struct {
	// WindowDays is the EFFECTIVE graded span in days (WindowFrom..WindowTo inclusive), not
	// the requested window -- see the file header for why the two differ on a young ledger.
	// Zero means the ledger carried no dated row, so nothing could be graded.
	WindowDays int    `json:"window_days"`
	WindowFrom string `json:"window_from"`
	WindowTo   string `json:"window_to"`
	// FirstDay / LastDay bound the WHOLE ledger, not the window, so a card records that
	// older history exists outside the span it graded.
	FirstDay string `json:"first_day,omitempty"`
	LastDay  string `json:"last_day,omitempty"`
	// DaysCovered / MissedDays partition the graded span by "did a tick record that day?".
	DaysCovered int `json:"days_covered"`
	MissedDays  int `json:"missed_days"`
	// StaleDays is the reference day minus LastDay: how long the job has been dark.
	StaleDays int `json:"stale_days"`
	// Outcomes is the #5586 tally over the rows INSIDE the graded span.
	Outcomes Outcomes `json:"outcomes"`
	// Ledger echoes the path the rows were read from, so the evidence names its source.
	Ledger string `json:"ledger,omitempty"`
}

// FoldHealth derives the graded span and its coverage/outcome evidence. Pure and total: an
// empty (or wholly undated) history folds to a zero-span Health, which ComposeHealth reports
// as missing evidence rather than as a perfect score. A row whose Day does not parse is
// skipped -- it cannot be placed in a span, and guessing a day for it would invent coverage.
func FoldHealth(f HealthFacts) Health {
	now := f.Now
	if now.IsZero() {
		now = time.Now()
	}
	window := f.Window
	if window <= 0 {
		window = DefaultHealthWindow
	}
	// Round-trip the reference instant through the ledger's own day layout so both sides of
	// every comparison below are midnight-UTC dates and the arithmetic carries no zone.
	ref, err := time.Parse(DayLayout, now.Format(DayLayout))
	if err != nil {
		return Health{Ledger: f.Ledger}
	}

	h := Health{WindowTo: ref.Format(DayLayout), Ledger: f.Ledger}

	var first, last time.Time
	for _, r := range f.Rows {
		d, err := time.Parse(DayLayout, r.Day)
		if err != nil {
			continue
		}
		if first.IsZero() || d.Before(first) {
			first = d
		}
		if last.IsZero() || d.After(last) {
			last = d
		}
	}
	if first.IsZero() {
		h.WindowFrom = h.WindowTo
		return h
	}
	h.FirstDay = first.Format(DayLayout)
	h.LastDay = last.Format(DayLayout)

	start := ref.AddDate(0, 0, -(window - 1))
	if start.Before(first) {
		start = first
	}
	// A ledger dated in the FUTURE (clock skew, or a row written on a machine hours ahead)
	// must not produce a negative span; clamp so the span is at least the reference day.
	if start.After(ref) {
		start = ref
	}
	h.WindowFrom = start.Format(DayLayout)
	h.WindowDays = int(ref.Sub(start).Hours()/24) + 1

	covered := map[string]bool{}
	inWindow := make([]Row, 0, len(f.Rows))
	for _, r := range f.Rows {
		d, parseErr := time.Parse(DayLayout, r.Day)
		if parseErr != nil || d.Before(start) || d.After(ref) {
			continue
		}
		covered[r.Day] = true
		inWindow = append(inWindow, r)
	}
	h.DaysCovered = len(covered)
	if missed := h.WindowDays - h.DaysCovered; missed > 0 {
		h.MissedDays = missed
	}
	h.Outcomes = FoldOutcomes(inWindow)
	if stale := int(ref.Sub(last).Hours() / 24); stale > 0 {
		h.StaleDays = stale
	}
	return h
}

// UsageFraction is the share of graded days that recorded a tick: "is the daily job actually
// running daily?". A zero span (no dated evidence) is 0, not 1 -- an unwired clone has no
// adoption, and reporting it as perfect is the failure this card exists to prevent.
func (h Health) UsageFraction() float64 {
	if h.WindowDays <= 0 {
		return 0
	}
	return clampHealth(float64(h.DaysCovered) / float64(h.WindowDays))
}

// HealthFraction is the share of recorded ticks that did their work with no tier held back.
// With no recorded tick in the span it is 0 by the same rule as UsageFraction; callers that
// must distinguish "no runs" from "all runs bad" read Outcomes.Runs.
func (h Health) HealthFraction() float64 {
	if h.Outcomes.Runs <= 0 {
		return 0
	}
	return clampHealth(float64(h.Outcomes.OK) / float64(h.Outcomes.Runs))
}

// DriftFraction grades how recently the job last ran, on the window's own scale. A tick
// today or yesterday is 1.0 -- a DAILY job whose trigger has not fired yet today is not
// drifting -- and each further dark day costs 1/window until it reaches 0.
func (h Health) DriftFraction() float64 {
	if h.LastDay == "" {
		return 0
	}
	span := h.WindowDays
	if span <= 0 {
		span = DefaultHealthWindow
	}
	dark := h.StaleDays - 1
	if dark <= 0 {
		return 1
	}
	return clampHealth(1 - float64(dark)/float64(span))
}

// healthKPIs builds one KPI per lens, each carrying the named evidence in its detail and the
// REAL repair in its defect. With no dated row at all it returns the single evidence KPI, so
// the card reports "never ran here" instead of grading three lenses off nothing.
func healthKPIs(h Health) []scorecard.KPI {
	if h.WindowDays <= 0 {
		return []scorecard.KPI{{
			Key:      LensEvidence,
			Group:    "git_daily",
			Score:    0,
			PassLine: 100 * HealthPassLine,
			Detail: fmt.Sprintf("no dated `%s` row in %s -- the daily tick has never recorded a run in this clone",
				Schema, ledgerLabel(h.Ledger)),
			Defects: []string{fmt.Sprintf(
				"%s: no recorded tick -- install the daily job (`fak cron install`) and run `fak git-daily` once to open the ledger at %s",
				LensEvidence, ledgerLabel(h.Ledger))},
		}}
	}

	usage := h.UsageFraction()
	kpis := []scorecard.KPI{{
		Key:      LensUsage,
		Group:    "git_daily",
		Score:    100 * usage,
		PassLine: 100 * HealthPassLine,
		Detail: fmt.Sprintf("recorded a tick on %d/%d graded days (%s..%s, %d missed) from %s",
			h.DaysCovered, h.WindowDays, h.WindowFrom, h.WindowTo, h.MissedDays, ledgerLabel(h.Ledger)),
	}}
	if below(usage) {
		kpis[0].Defects = []string{fmt.Sprintf(
			"%s: %d of %d graded days recorded no tick (%.0f%% < %.0f%% pass line) -- repair the SCHEDULE (`fak cron install`, then check the trigger actually fires), never the floor",
			LensUsage, h.MissedDays, h.WindowDays, usage*100, HealthPassLine*100)}
	}

	// The health lens is EXCLUDED (not scored 0) when the span holds no recorded tick: the
	// usage and drift lenses already carry that failure, and a third copy of it would treble
	// the debt for one defect.
	if h.Outcomes.Runs > 0 {
		health := h.HealthFraction()
		k := scorecard.KPI{
			Key:      LensHealth,
			Group:    "git_daily",
			Score:    100 * health,
			PassLine: 100 * HealthPassLine,
			Detail: fmt.Sprintf("%d/%d recorded ticks healthy (%d refused, %d error; folded %d loose objects)%s",
				h.Outcomes.OK, h.Outcomes.Runs, h.Outcomes.Refused, h.Outcomes.Errors,
				h.Outcomes.LooseFolded, reasonSuffix(h.Outcomes)),
		}
		if below(health) {
			k.Defects = []string{fmt.Sprintf(
				"%s: %d of %d recorded ticks held a tier back or hit an incident (%.0f%% < %.0f%% pass line)%s -- clear the REASON, never the floor",
				LensHealth, h.Outcomes.Runs-h.Outcomes.OK, h.Outcomes.Runs, health*100, HealthPassLine*100,
				reasonSuffix(h.Outcomes))}
		}
		kpis = append(kpis, k)
	}

	drift := h.DriftFraction()
	k := scorecard.KPI{
		Key:      LensDrift,
		Group:    "git_daily",
		Score:    100 * drift,
		PassLine: 100 * HealthPassLine,
		Detail: fmt.Sprintf("last recorded tick %s, %d day(s) before the %s reference day",
			dayLabel(h.LastDay), h.StaleDays, h.WindowTo),
	}
	if below(drift) {
		k.Defects = []string{fmt.Sprintf(
			"%s: the job has been dark for %d days (last tick %s; %.0f%% < %.0f%% pass line) -- find out WHY it stopped and restart it, never lower the floor",
			LensDrift, h.StaleDays, dayLabel(h.LastDay), drift*100, HealthPassLine*100)}
	}
	return append(kpis, k)
}

// ComposeHealth folds the ledger into the control-pane payload: corpus["value"] is the mean
// lens fraction, corpus[HealthMissedKey] the unbounded dark-day headline, and
// corpus[HealthDebtKey] the count of lenses below the pass line (ok == debt is 0). The
// standard grade curve is used because this is an OPERATIONAL health card, not a
// provenance-honesty card.
func ComposeHealth(f HealthFacts) scorecard.Payload {
	h := FoldHealth(f)
	return scorecard.Fold(HealthSchema, healthKPIs(h), HealthDebtKey, nil, scorecard.Messages{
		Finding: "the daily lock-aware Git hygiene job is NOT healthy: a lens fell below the pass line -- " +
			"it is not firing daily, its tiers are deferring, or it has gone dark",
		FindingClean: "the daily lock-aware Git hygiene job is healthy: it fires daily, its tiers do their work, " +
			"and its last tick is recent",
		NextAction: "repair the failing lens at its source: install/fix the schedule (usage), clear the structured " +
			"refusal reason (health), or find out why the job stopped and restart it (drift)",
		NextActionClean: "hold the line; keep the job on cadence and tighten the ratchet",
		Grade:           scorecard.GradeStd,
		ExtraCorpus: map[string]any{
			HealthMissedKey:          h.MissedDays,
			"git_daily_window_days":  h.WindowDays,
			"git_daily_window_from":  h.WindowFrom,
			"git_daily_window_to":    h.WindowTo,
			"git_daily_days_covered": h.DaysCovered,
			"git_daily_stale_days":   h.StaleDays,
			"git_daily_runs":         h.Outcomes.Runs,
			"git_daily_ok":           h.Outcomes.OK,
			"git_daily_refused":      h.Outcomes.Refused,
			"git_daily_errors":       h.Outcomes.Errors,
			"git_daily_loose_folded": h.Outcomes.LooseFolded,
			"git_daily_first_day":    h.FirstDay,
			"git_daily_last_day":     h.LastDay,
			"git_daily_ledger":       h.Ledger,
			"pass_line":              HealthPassLine,
		},
	})
}

// RenderHealth is the one-call operator readout: the graded card as a terminal work-list.
func RenderHealth(f HealthFacts) string {
	return scorecard.Render(ComposeHealth(f), HealthDebtKey)
}

// reasonSuffix appends the refusal breakdown (commonest first) when there is one, so a
// deferring job names WHICH reason without the reader opening the ledger.
func reasonSuffix(o Outcomes) string {
	names := o.ReasonsByCount()
	if len(names) == 0 {
		return ""
	}
	out := " ["
	for i, n := range names {
		if i > 0 {
			out += ", "
		}
		out += fmt.Sprintf("%s x%d", n, o.Reasons[n])
	}
	return out + "]"
}

// ledgerLabel names the evidence source, falling back to the ledger's file name when the
// caller supplied no path (the fold never needs one, but a rendered card should still say
// what it read).
func ledgerLabel(path string) string {
	if path == "" {
		return LedgerName
	}
	return path
}

// dayLabel keeps an absent day readable in a detail line.
func dayLabel(day string) string {
	if day == "" {
		return "never"
	}
	return day
}

// below reports whether a lens fraction is under the pass line, with the same epsilon the
// scorecard kernel uses so a lens exactly ON the line is not debt.
func below(fraction float64) bool { return fraction+healthEps < HealthPassLine }

const healthEps = 1e-9

func clampHealth(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
