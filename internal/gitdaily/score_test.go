package gitdaily

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// refDay is the fixed reference day every fixture below is dated against, so no test here
// depends on the wall clock.
const refDay = "2026-08-14"

func mustDay(t *testing.T, day string) time.Time {
	t.Helper()
	d, err := time.Parse(DayLayout, day)
	if err != nil {
		t.Fatalf("bad fixture day %q: %v", day, err)
	}
	return d
}

// dayBefore renders the day n days before refDay, so a fixture reads in "days ago" terms.
func dayBefore(t *testing.T, n int) string {
	t.Helper()
	return mustDay(t, refDay).AddDate(0, 0, -n).Format(DayLayout)
}

// healthy builds one clean recorded tick that folded work. PRUNE_OFF rides along because
// every DEFAULT run carries it (#5586) — a fixture without it would not be a real row.
func healthy(day string) Row {
	return Row{Schema: Schema, Day: day, LooseBefore: 400, LooseAfter: 0, GracePruneRefused: "PRUNE_OFF"}
}

func facts(t *testing.T, rows []Row) HealthFacts {
	t.Helper()
	return HealthFacts{Rows: rows, Now: mustDay(t, refDay), Ledger: ".git/" + LedgerName}
}

// TestHealthCardOnAnUnwiredCloneReportsNoEvidence is the adoption floor. A clone where the
// daily job has never run must NOT read as perfect: an empty ledger means zero adoption, and
// a card that graded "nothing recorded" as A would be exactly the green-looking lie the
// scorecard exists to prevent.
func TestHealthCardOnAnUnwiredCloneReportsNoEvidence(t *testing.T) {
	p := ComposeHealth(facts(t, nil))
	if p.OK {
		t.Fatalf("an unwired clone graded ok=true: %+v", p.Corpus)
	}
	if p.Verdict != "ACTION" {
		t.Errorf("verdict = %q, want ACTION", p.Verdict)
	}
	if got := p.Corpus[HealthDebtKey]; got != 1 {
		t.Errorf("%s = %v, want 1 (the missing-evidence lens)", HealthDebtKey, got)
	}
	if len(p.KPIs) != 1 || p.KPIs[0].Key != LensEvidence {
		t.Fatalf("KPIs = %+v, want the single %s lens", p.KPIs, LensEvidence)
	}
	// Named evidence: the card says WHICH file it found nothing in.
	if !strings.Contains(p.KPIs[0].Detail, LedgerName) {
		t.Errorf("evidence detail %q does not name the ledger", p.KPIs[0].Detail)
	}
}

// TestHealthCardGradesAFreshInstallOnItsOwnAge is the window-capping contract. A job
// installed three days ago that has run every day since is HEALTHY; grading it against a
// fixed 14-day denominator would score it 3/14 and train an operator to ignore the card.
func TestHealthCardGradesAFreshInstallOnItsOwnAge(t *testing.T) {
	rows := []Row{healthy(dayBefore(t, 2)), healthy(dayBefore(t, 1)), healthy(dayBefore(t, 0))}
	h := FoldHealth(facts(t, rows))
	if h.WindowDays != 3 {
		t.Fatalf("window_days = %d, want 3 (capped at the first recorded run, not %d)", h.WindowDays, DefaultHealthWindow)
	}
	if h.DaysCovered != 3 || h.MissedDays != 0 {
		t.Errorf("covered/missed = %d/%d, want 3/0", h.DaysCovered, h.MissedDays)
	}
	p := ComposeHealth(facts(t, rows))
	if !p.OK {
		t.Fatalf("a three-day-old install that ran every day graded not-ok: %v", p.Reason)
	}
	if got := p.Corpus["grade"]; got != "A" {
		t.Errorf("grade = %v, want A", got)
	}
	if got := p.Corpus[HealthMissedKey]; got != 0 {
		t.Errorf("%s = %v, want 0", HealthMissedKey, got)
	}
}

// TestHealthCardCatchesADarkJobDespiteAPerfectSuccessRate is the #4602 shape and the reason
// the lenses are separate. Five ticks, all healthy, then nine dark days: a single blended
// number would report 100% success. Usage and drift must BOTH red while the health lens
// stays clean, because the runs that happened really were fine — the job stopped.
func TestHealthCardCatchesADarkJobDespiteAPerfectSuccessRate(t *testing.T) {
	var rows []Row
	for n := 13; n >= 9; n-- {
		rows = append(rows, healthy(dayBefore(t, n)))
	}
	h := FoldHealth(facts(t, rows))
	if h.StaleDays != 9 {
		t.Fatalf("stale_days = %d, want 9", h.StaleDays)
	}
	if h.MissedDays != 9 {
		t.Fatalf("missed_days = %d, want 9 (span %d, covered %d)", h.MissedDays, h.WindowDays, h.DaysCovered)
	}
	if got := h.HealthFraction(); got != 1 {
		t.Errorf("health fraction = %v, want 1 (every recorded tick was fine)", got)
	}

	p := ComposeHealth(facts(t, rows))
	if p.OK {
		t.Fatal("a job dark for nine days graded ok=true")
	}
	debted := map[string]bool{}
	for _, k := range p.KPIs {
		if len(k.Defects) > 0 {
			debted[k.Key] = true
		}
	}
	if !debted[LensUsage] || !debted[LensDrift] {
		t.Errorf("debted lenses = %v, want both %s and %s", debted, LensUsage, LensDrift)
	}
	if debted[LensHealth] {
		t.Errorf("%s red on a history whose every recorded tick was healthy", LensHealth)
	}
	if got := p.Corpus[HealthMissedKey]; got != 9 {
		t.Errorf("%s = %v, want 9 (the unbounded dark-day headline)", HealthMissedKey, got)
	}
}

// TestHealthCardCatchesARefusalStreakOnACadencedJob is the mirror case: the job fires every
// single day (usage and drift perfect) but its fold tier keeps deferring. Only the health
// lens may red, and its detail must NAME the reason so the operator does not have to open
// the ledger to learn it.
func TestHealthCardCatchesARefusalStreakOnACadencedJob(t *testing.T) {
	var rows []Row
	for n := 5; n >= 0; n-- {
		r := healthy(dayBefore(t, n))
		if n > 0 {
			r.GraceRefused = "LOCKED"
		}
		rows = append(rows, r)
	}
	p := ComposeHealth(facts(t, rows))
	if p.OK {
		t.Fatal("a five-day LOCKED streak graded ok=true")
	}
	var health, usage, drift *struct {
		defects int
		detail  string
	}
	for _, k := range p.KPIs {
		v := &struct {
			defects int
			detail  string
		}{len(k.Defects), k.Detail}
		switch k.Key {
		case LensHealth:
			health = v
		case LensUsage:
			usage = v
		case LensDrift:
			drift = v
		}
	}
	if health == nil || health.defects == 0 {
		t.Fatalf("%s did not red on a 5/6 refusal rate: %+v", LensHealth, p.KPIs)
	}
	if usage == nil || usage.defects != 0 || drift == nil || drift.defects != 0 {
		t.Errorf("a perfectly cadenced job redded usage/drift: %+v %+v", usage, drift)
	}
	if !strings.Contains(health.detail, "LOCKED") {
		t.Errorf("health detail %q does not name the refusal reason", health.detail)
	}
}

// TestHealthCardExcludesTheHealthLensWithNoRunsInTheSpan keeps one defect from being counted
// three times. When the graded span holds no recorded tick, usage and drift already carry
// the failure; a third zero-scored lens would treble the debt for a single problem.
func TestHealthCardExcludesTheHealthLensWithNoRunsInTheSpan(t *testing.T) {
	// One tick 40 days before the reference day: older than the 14-day window, so the span
	// holds nothing, but the ledger is not empty either.
	rows := []Row{healthy(dayBefore(t, 40))}
	p := ComposeHealth(facts(t, rows))
	for _, k := range p.KPIs {
		if k.Key == LensHealth {
			t.Fatalf("%s scored with zero runs in the span: %+v", LensHealth, k)
		}
	}
	if got := p.Corpus[HealthDebtKey]; got != 2 {
		t.Errorf("%s = %v, want 2 (usage + drift only)", HealthDebtKey, got)
	}
	if got := p.Corpus["git_daily_window_days"]; got != DefaultHealthWindow {
		t.Errorf("window_days = %v, want %d (the ledger predates the window)", got, DefaultHealthWindow)
	}
}

// TestFoldHealthIgnoresUndatedAndFutureRows pins the two malformed-input paths: a row whose
// day cannot be parsed is skipped (guessing a day would invent coverage), and a row dated in
// the FUTURE must not produce a negative span.
func TestFoldHealthIgnoresUndatedAndFutureRows(t *testing.T) {
	undated := FoldHealth(facts(t, []Row{{Schema: Schema, Day: "not-a-day"}}))
	if undated.WindowDays != 0 || undated.FirstDay != "" {
		t.Errorf("an undated row produced a span: %+v", undated)
	}

	future := FoldHealth(facts(t, []Row{healthy(mustDay(t, refDay).AddDate(0, 0, 5).Format(DayLayout))}))
	if future.WindowDays < 1 {
		t.Errorf("a future-dated row produced a non-positive span: %+v", future)
	}
	if future.StaleDays != 0 {
		t.Errorf("stale_days = %d on a future row, want 0", future.StaleDays)
	}
}

// TestRenderHealthNamesItsEvidence is the operator-readout contract the make target prints:
// the render must carry the grade, the debt headline, and every lens's evidence line.
func TestRenderHealthNamesItsEvidence(t *testing.T) {
	rows := []Row{healthy(dayBefore(t, 1)), healthy(dayBefore(t, 0))}
	out := RenderHealth(facts(t, rows))
	for _, want := range []string{"git daily health scorecard", HealthDebtKey, LensUsage, LensDrift, LedgerName} {
		if !strings.Contains(out, want) {
			t.Errorf("render does not mention %q:\n%s", want, out)
		}
	}
}

// TestHealthCardOverTheLiveLedger is the real-data witness (#5587): it folds THIS clone's
// own ledger and prints the card, so `make gitdaily-score` produces captured output on real
// data rather than a fixture. It asserts only structure — never a floor — because a clone
// where the job legitimately has not run must not red the tree's test run.
func TestHealthCardOverTheLiveLedger(t *testing.T) {
	path := filepath.Join("..", "..", ".git", LedgerName)
	if st, err := os.Stat(path); err != nil || st.IsDir() {
		t.Skipf("no live ledger at %s (a linked worktree or a clone that never ran the job) -- fixtures cover the fold", path)
	}
	rows := Status(path, 0)
	p := ComposeHealth(HealthFacts{Rows: rows, Ledger: path})
	if p.Schema != HealthSchema {
		t.Fatalf("schema = %q, want %q", p.Schema, HealthSchema)
	}
	if grade, _ := p.Corpus["grade"].(string); grade == "" {
		t.Fatalf("live card emitted no grade: %+v", p.Corpus)
	}
	if _, ok := p.Corpus[HealthDebtKey]; !ok {
		t.Fatalf("live card emitted no %s", HealthDebtKey)
	}
	t.Logf("live git-daily health card over %d rows of %s:\n%s", len(rows), path, scorecard.Render(p, HealthDebtKey))
}
