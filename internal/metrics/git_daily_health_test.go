package metrics

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// realLedgerCapture is the CAPTURED WITNESS this card was built against: the live
// `fak-git-daily/1` ledger of this clone (.git/fak-git-daily.jsonl) as of 2026-08-05,
// five recorded ticks over two days. Pinning the real capture — rather than only
// hand-built fixtures — is what makes `make test-fast` / `make ci` grade the card against
// real ledger data on every run.
//
//	{"day":"2026-08-04","loose_before":4129,"loose_after":4129,"grace_prune_refused":"PRUNE_OFF"}
//	{"day":"2026-08-04","loose_before":4129,"loose_after":0,   "grace_prune_refused":"PRUNE_OFF"}
//	{"day":"2026-08-04","loose_before":48,  "loose_after":0,   "grace_prune_refused":"PRUNE_OFF"}
//	{"day":"2026-08-04","loose_before":34,  "loose_after":0,   "grace_prune_refused":"PRUNE_OFF"}
//	{"day":"2026-08-05","loose_before":5827,"loose_after":0,   "grace_prune_refused":"PRUNE_OFF"}
//
// PRUNE_OFF is absent from Reasons on purpose: the grace-prune tier is opt-in and
// default-OFF, so its refusal is the configured posture of a healthy run, not a held tier.
var realLedgerCapture = GitDailyHealthInput{
	Runs:          5,
	OK:            5,
	FirstDay:      "2026-08-04",
	LastDay:       "2026-08-05",
	LooseFolded:   0 + 4129 + 48 + 34 + 5827,
	RefusedStreak: 0,
	Today:         "2026-08-05",
	LedgerPath:    ".git/fak-git-daily.jsonl",
}

func gitDailyGrade(t *testing.T, p scorecard.Payload) string {
	t.Helper()
	grade, ok := p.Corpus["grade"].(string)
	if !ok {
		t.Fatalf("corpus[grade] = %v (%T), want a string", p.Corpus["grade"], p.Corpus["grade"])
	}
	return grade
}

func gitDailyDebt(t *testing.T, p scorecard.Payload) int {
	t.Helper()
	debt, ok := p.Corpus[GitDailyDebtKey].(int)
	if !ok {
		t.Fatalf("corpus[%s] = %v (%T), want an int", GitDailyDebtKey, p.Corpus[GitDailyDebtKey], p.Corpus[GitDailyDebtKey])
	}
	return debt
}

func gitDailyKPI(t *testing.T, p scorecard.Payload, key string) scorecard.KPI {
	t.Helper()
	for _, k := range p.KPIs {
		if k.Key == key {
			return k
		}
	}
	t.Fatalf("payload carries no %q KPI; got %v", key, p.KPIs)
	return scorecard.KPI{}
}

// TestGradeGitDailyHealthOnRealLedger is the real-data witness for #5587: the card graded
// against the captured live ledger, not a fixture. A healthy clone whose tick ran today
// and folded 10038 loose objects grades A with zero debt.
func TestGradeGitDailyHealthOnRealLedger(t *testing.T) {
	p := GradeGitDailyHealth(realLedgerCapture)

	if p.Schema != GitDailyHealthSchema {
		t.Errorf("schema = %q, want %q", p.Schema, GitDailyHealthSchema)
	}
	if got := gitDailyGrade(t, p); got != "A" {
		t.Errorf("grade = %q, want A (5/5 ok ticks, folded today)", got)
	}
	if got := gitDailyDebt(t, p); got != 0 {
		t.Errorf("%s = %d, want 0: %s", GitDailyDebtKey, got, p.Reason)
	}
	if !p.OK || p.Verdict != "OK" {
		t.Errorf("ok/verdict = %v/%q, want true/OK", p.OK, p.Verdict)
	}

	want := "git-daily=A runs=5 ok=5 refused=0 error=0 folded=10038 streak=0 debt=0"
	if got := GitDailyHealthFragment(realLedgerCapture, p); got != want {
		t.Errorf("fragment =\n  %q\nwant\n  %q", got, want)
	}

	// The evidence must NAME the witness it was scored from, not just report a number.
	adoption := gitDailyKPI(t, p, "adoption")
	for _, want := range []string{".git/fak-git-daily.jsonl", "2026-08-04..2026-08-05", "5 recorded tick"} {
		if !strings.Contains(adoption.Detail, want) {
			t.Errorf("adoption detail %q does not name %q", adoption.Detail, want)
		}
	}
	if drift := gitDailyKPI(t, p, "fold_drift"); !strings.Contains(drift.Detail, "10038 loose object(s) folded") {
		t.Errorf("fold_drift detail %q does not name the folded volume", drift.Detail)
	}
}

// TestGradeGitDailyHealthFirstRun pins the first-run contract: an empty ledger is not
// "healthy by default" — with no witness there is no health claim, and exactly ONE defect
// names the single root cause instead of three KPIs each reporting the same absence.
func TestGradeGitDailyHealthFirstRun(t *testing.T) {
	p := GradeGitDailyHealth(GitDailyHealthInput{LedgerPath: "/tmp/clone/.git/fak-git-daily.jsonl"})

	if got := gitDailyGrade(t, p); got != "F" {
		t.Errorf("grade = %q, want F for an unwitnessed job", got)
	}
	if got := gitDailyDebt(t, p); got != 1 {
		t.Errorf("%s = %d, want exactly 1 (one root cause, not one per axis): %s", GitDailyDebtKey, got, p.Reason)
	}
	if p.OK || p.Verdict != "ACTION" {
		t.Errorf("ok/verdict = %v/%q, want false/ACTION", p.OK, p.Verdict)
	}
	adoption := gitDailyKPI(t, p, "adoption")
	if len(adoption.Defects) != 1 || !strings.Contains(adoption.Defects[0], "/tmp/clone/.git/fak-git-daily.jsonl") {
		t.Errorf("adoption defects = %v, want one naming the ledger path", adoption.Defects)
	}
	// The other two axes must stay SOFT: soft never counts as debt, which is what keeps
	// the empty-ledger case from triple-charging one problem.
	for _, key := range []string{"outcome_health", "fold_drift"} {
		k := gitDailyKPI(t, p, key)
		if len(k.Defects) != 0 {
			t.Errorf("%s defects = %v, want none on an empty ledger", key, k.Defects)
		}
		if len(k.Soft) == 0 {
			t.Errorf("%s carries no soft note explaining why it is not gradable", key)
		}
	}
}

// TestGradeGitDailyHealthAdoptionLapse pins the usage axis: a job whose trigger stopped
// landing runs is the failure nothing else reports, because a stopped scheduler errors
// nowhere.
func TestGradeGitDailyHealthAdoptionLapse(t *testing.T) {
	in := GitDailyHealthInput{
		Runs: 3, OK: 3,
		FirstDay: "2026-07-26", LastDay: "2026-07-28",
		LooseFolded: 900,
		Today:       "2026-08-05",
		LedgerPath:  "ledger.jsonl",
	}
	p := GradeGitDailyHealth(in)

	adoption := gitDailyKPI(t, p, "adoption")
	if adoption.Score != 0 {
		t.Errorf("adoption score = %v, want 0 at an 8-day gap", adoption.Score)
	}
	if len(adoption.Defects) != 1 {
		t.Fatalf("adoption defects = %v, want exactly 1", adoption.Defects)
	}
	for _, want := range []string{"2026-07-28", "8 day(s) stale", "2026-08-05", "ledger.jsonl"} {
		if !strings.Contains(adoption.Defects[0], want) {
			t.Errorf("adoption defect %q does not name %q", adoption.Defects[0], want)
		}
	}
	// Ticks that DID run were healthy, so the other two axes must stay clean: the grade
	// has to say "it stopped", not "it broke".
	if got := gitDailyDebt(t, p); got != 1 {
		t.Errorf("%s = %d, want 1 (adoption only): %s", GitDailyDebtKey, got, p.Reason)
	}
	if got := gitDailyGrade(t, p); got != "D" {
		t.Errorf("grade = %q, want D", got)
	}
}

// TestGradeGitDailyHealthAdoptionWithinGrace pins the deliberately-loose recency gate: a
// daily job scheduled overnight legitimately has YESTERDAY as its newest row for most of
// the current day, so one day of gap must not red the card.
func TestGradeGitDailyHealthAdoptionWithinGrace(t *testing.T) {
	for _, tc := range []struct{ name, last, today string }{
		{"same day", "2026-08-05", "2026-08-05"},
		{"yesterday", "2026-08-04", "2026-08-05"},
		{"clock stepped backwards", "2026-08-06", "2026-08-05"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := GradeGitDailyHealth(GitDailyHealthInput{
				Runs: 2, OK: 2, FirstDay: tc.last, LastDay: tc.last, Today: tc.today, LooseFolded: 12,
			})
			if k := gitDailyKPI(t, p, "adoption"); len(k.Defects) != 0 {
				t.Errorf("adoption defects = %v, want none", k.Defects)
			}
		})
	}
}

// TestGradeGitDailyHealthRefusalStreak pins the #4602 regression the card exists for: nine
// consecutive days of "success" that folded nothing because a wedged lock pinned the tier.
func TestGradeGitDailyHealthRefusalStreak(t *testing.T) {
	in := GitDailyHealthInput{
		Runs: 9, Refused: 9,
		Reasons:  map[string]int{"LOCKED": 9},
		FirstDay: "2026-07-28", LastDay: "2026-08-05",
		RefusedStreak: 9,
		Today:         "2026-08-05",
		LedgerPath:    "ledger.jsonl",
	}
	p := GradeGitDailyHealth(in)

	drift := gitDailyKPI(t, p, "fold_drift")
	if drift.Score != 0 {
		t.Errorf("fold_drift score = %v, want 0 at a 9-tick streak", drift.Score)
	}
	if len(drift.Defects) != 1 {
		t.Fatalf("fold_drift defects = %v, want exactly 1", drift.Defects)
	}
	for _, want := range []string{"newest 9 recorded ticks", "LOCKED x9", "#4602", "ledger.jsonl"} {
		if !strings.Contains(drift.Defects[0], want) {
			t.Errorf("fold_drift defect %q does not name %q", drift.Defects[0], want)
		}
	}
	// A majority-refused window is a held tier, so the failure-rate axis charges it too.
	outcome := gitDailyKPI(t, p, "outcome_health")
	if len(outcome.Defects) != 1 || !strings.Contains(outcome.Defects[0], "9 of 9 recorded ticks") {
		t.Errorf("outcome_health defects = %v, want one naming the majority-refused share", outcome.Defects)
	}
	if got := gitDailyGrade(t, p); got != "F" {
		t.Errorf("grade = %q, want F", got)
	}
	if got := gitDailyDebt(t, p); got != 2 {
		t.Errorf("%s = %d, want 2 (held tier + drift streak): %s", GitDailyDebtKey, got, p.Reason)
	}
}

// TestGradeGitDailyHealthLoneRefusalIsNotDebt pins the anti-noise rule that makes the
// streak signal legible: one tick refusing because a peer held a transaction lock is the
// expected steady state, so it lands SOFT and never creates debt.
func TestGradeGitDailyHealthLoneRefusalIsNotDebt(t *testing.T) {
	in := GitDailyHealthInput{
		Runs: 10, OK: 9, Refused: 1,
		Reasons:  map[string]int{"LOCKED": 1},
		FirstDay: "2026-07-27", LastDay: "2026-08-05",
		LooseFolded: 500, RefusedStreak: 1,
		Today: "2026-08-05",
	}
	p := GradeGitDailyHealth(in)

	outcome := gitDailyKPI(t, p, "outcome_health")
	if len(outcome.Defects) != 0 {
		t.Errorf("outcome_health defects = %v, want none for a 1-in-10 refusal", outcome.Defects)
	}
	if len(outcome.Soft) != 1 || !strings.Contains(outcome.Soft[0], "LOCKED x1") {
		t.Errorf("outcome_health soft = %v, want one note naming the reason", outcome.Soft)
	}
	if got := gitDailyDebt(t, p); got != 0 {
		t.Errorf("%s = %d, want 0: %s", GitDailyDebtKey, got, p.Reason)
	}
	if got := gitDailyGrade(t, p); got != "A" {
		t.Errorf("grade = %q, want A", got)
	}
}

// TestGradeGitDailyHealthIncidentCapsTheGrade pins the incident rule: a lock cleanup that
// failed does not self-heal, so a single one caps the failure-rate axis at half rather
// than averaging away across a long healthy window.
func TestGradeGitDailyHealthIncidentCapsTheGrade(t *testing.T) {
	in := GitDailyHealthInput{
		Runs: 4, OK: 3, Errors: 1,
		FirstDay: "2026-08-02", LastDay: "2026-08-05",
		LooseFolded: 120, RefusedStreak: 1,
		Today:      "2026-08-05",
		LedgerPath: "ledger.jsonl",
	}
	p := GradeGitDailyHealth(in)

	outcome := gitDailyKPI(t, p, "outcome_health")
	if outcome.Score != 50 {
		t.Errorf("outcome_health score = %v, want the 50 incident cap (uncapped would be 75)", outcome.Score)
	}
	if len(outcome.Defects) != 1 || !strings.Contains(outcome.Defects[0], "1 of 4 recorded ticks in ledger.jsonl") {
		t.Fatalf("outcome_health defects = %v, want one naming the incident share and the ledger", outcome.Defects)
	}
	if got := gitDailyGrade(t, p); got != "B" {
		t.Errorf("grade = %q, want B", got)
	}
	// The letter still reads high on a mostly-healthy window, so debt — not the grade —
	// is the gate an operator acts on.
	if p.OK || p.Verdict != "ACTION" {
		t.Errorf("ok/verdict = %v/%q, want false/ACTION even at grade B", p.OK, p.Verdict)
	}
}

// TestGradeGitDailyHealthEveryDefectNamesTheLedger pins the "named evidence" half of the
// done condition: no defect may report a bare number without naming the witness file it
// was scored from.
func TestGradeGitDailyHealthEveryDefectNamesTheLedger(t *testing.T) {
	const ledger = "/srv/clone/.git/fak-git-daily.jsonl"
	for _, in := range []GitDailyHealthInput{
		{LedgerPath: ledger},
		{Runs: 3, OK: 3, FirstDay: "2026-07-01", LastDay: "2026-07-03", Today: "2026-08-05", LedgerPath: ledger},
		{Runs: 6, Refused: 5, Errors: 1, Reasons: map[string]int{"POSTURE_DRIFT": 5}, RefusedStreak: 6,
			FirstDay: "2026-07-31", LastDay: "2026-08-05", Today: "2026-08-05", LedgerPath: ledger},
	} {
		p := GradeGitDailyHealth(in)
		defects := 0
		for _, k := range p.KPIs {
			for _, d := range k.Defects {
				defects++
				if !strings.Contains(d, ledger) {
					t.Errorf("%s defect does not name the ledger: %q", k.Key, d)
				}
			}
		}
		if defects == 0 {
			t.Errorf("input %+v produced no defect; the case is not exercising the evidence rule", in)
		}
	}
}

// TestGradeGitDailyHealthIsDeterministic pins the "deterministic scorecard" claim: the
// refusal breakdown folds a MAP, and map iteration order is randomized per run, so a naive
// render would churn the evidence string between two runs on identical input.
func TestGradeGitDailyHealthIsDeterministic(t *testing.T) {
	in := GitDailyHealthInput{
		Runs: 20, Refused: 20,
		Reasons:  map[string]int{"LOCKED": 9, "POSTURE_DRIFT": 9, "SESSION_LIVE": 2},
		FirstDay: "2026-07-17", LastDay: "2026-08-05",
		RefusedStreak: 20,
		Today:         "2026-08-05",
	}
	first := GradeGitDailyHealth(in)
	wantReasons := "LOCKED x9, POSTURE_DRIFT x9, SESSION_LIVE x2"
	if got := gitDailyKPI(t, first, "outcome_health"); !strings.Contains(got.Detail, wantReasons) {
		t.Errorf("outcome_health detail %q does not render reasons commonest-first, ties by name (%q)", got.Detail, wantReasons)
	}
	for i := range 50 {
		again := GradeGitDailyHealth(in)
		if again.Reason != first.Reason {
			t.Fatalf("run %d reason drifted:\n  %q\nvs\n  %q", i, again.Reason, first.Reason)
		}
		for j := range again.KPIs {
			if again.KPIs[j].Detail != first.KPIs[j].Detail {
				t.Fatalf("run %d KPI %q detail drifted:\n  %q\nvs\n  %q", i, again.KPIs[j].Key, again.KPIs[j].Detail, first.KPIs[j].Detail)
			}
		}
	}
}

// TestGitDailyRefusedStreak pins the streak rule itself: it counts from the NEWEST tick
// backwards and stops at the first ok, and an incident extends it exactly like a refusal
// (both mean the newest ticks stopped doing the work).
func TestGitDailyRefusedStreak(t *testing.T) {
	for _, tc := range []struct {
		name     string
		outcomes []string
		want     int
	}{
		{"no history", nil, 0},
		{"newest ok", []string{"ok"}, 0},
		{"old refusals healed", []string{"refused", "refused", "ok"}, 0},
		{"one trailing refusal", []string{"ok", "refused"}, 1},
		{"refusal then error", []string{"ok", "refused", "error"}, 2},
		{"never ok", []string{"refused", "error", "refused"}, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := GitDailyRefusedStreak(tc.outcomes); got != tc.want {
				t.Errorf("GitDailyRefusedStreak(%v) = %d, want %d", tc.outcomes, got, tc.want)
			}
		})
	}
}

// TestGradeGitDailyHealthCorpusCarriesTheWitness pins the readout contract: the corpus
// carries the raw ledger counts, so a control-pane consumer can re-derive the grade
// instead of trusting it.
func TestGradeGitDailyHealthCorpusCarriesTheWitness(t *testing.T) {
	p := GradeGitDailyHealth(realLedgerCapture)
	for key, want := range map[string]any{
		"runs":             5,
		"ok":               5,
		"refused":          0,
		"errors":           0,
		"loose_folded":     10038,
		"refused_streak":   0,
		"first_day":        "2026-08-04",
		"last_day":         "2026-08-05",
		"today":            "2026-08-05",
		"ledger_path":      ".git/fak-git-daily.jsonl",
		"stale_after_days": DefaultGitDailyStaleDays,
	} {
		if got := p.Corpus[key]; got != want {
			t.Errorf("corpus[%s] = %v, want %v", key, got, want)
		}
	}
}

// realLedgerJSONL is the VERBATIM capture of this clone's live `fak-git-daily/1` ledger,
// .git/fak-git-daily.jsonl, on 2026-08-05 — the raw rows the daily tick appended, copied
// unedited rather than summarized.
//
// The rest of this file grades realLedgerCapture, a tally TYPED from these rows, against
// expectations ALSO typed from the same reading. Those two agreeing proves only that the
// author was self-consistent: misread a `loose_before`, drop a row when refreshing the
// capture, or classify PRUNE_OFF wrongly, and the typed tally and the typed fragment move
// together and every assertion still passes. That is the "score from witnesses, never
// from self-report" failure this card exists to catch, reproduced in its own test.
//
// Pinning the raw rows closes it two ways: the tally below is DERIVED from them rather
// than asserted against a second transcription, and a reviewer can diff this constant
// against .git/fak-git-daily.jsonl byte-for-byte — which no tally permits.
const realLedgerJSONL = `{"schema":"fak-git-daily/1","day":"2026-08-04","at":"2026-08-04T11:14:37-07:00","lease_locks_reaped":0,"index_locks_reaped":7,"lock_actions":1,"loose_before":4129,"loose_after":4129,"packs_before":4,"packs_after":4,"grace_prune_refused":"PRUNE_OFF"}
{"schema":"fak-git-daily/1","day":"2026-08-04","at":"2026-08-04T11:18:58-07:00","lease_locks_reaped":0,"index_locks_reaped":0,"lock_actions":0,"loose_before":4129,"loose_after":0,"packs_before":4,"packs_after":2,"grace_prune_refused":"PRUNE_OFF"}
{"schema":"fak-git-daily/1","day":"2026-08-04","at":"2026-08-04T11:59:46-07:00","lease_locks_reaped":0,"index_locks_reaped":0,"lock_actions":1,"loose_before":48,"loose_after":0,"packs_before":2,"packs_after":4,"grace_prune_refused":"PRUNE_OFF"}
{"schema":"fak-git-daily/1","day":"2026-08-04","at":"2026-08-04T12:15:16-07:00","lease_locks_reaped":0,"index_locks_reaped":0,"lock_actions":1,"loose_before":34,"loose_after":0,"packs_before":4,"packs_after":4,"grace_prune_refused":"PRUNE_OFF"}
{"schema":"fak-git-daily/1","day":"2026-08-05","at":"2026-08-05T04:39:17-07:00","lease_locks_reaped":0,"index_locks_reaped":1,"lock_actions":2,"loose_before":5827,"loose_after":0,"packs_before":4,"packs_after":4,"grace_prune_refused":"PRUNE_OFF"}`

// foldRealLedgerJSONL re-derives the graded input from raw `fak-git-daily/1` rows, using
// only the three classification rules this card's header states: an incident is an error,
// a grace refusal is a refusal EXCEPT the opt-in default-OFF PRUNE_OFF tier, and a
// NEGATIVE loose delta is a peer's concurrent commit volume rather than a fold that ran
// backwards. Kept in the test (not in the card) because reading the ledger is I/O and the
// card is a pure fold — see the file header on why gitdaily cannot be imported here.
func foldRealLedgerJSONL(t *testing.T, jsonl string) GitDailyHealthInput {
	t.Helper()
	var in GitDailyHealthInput
	var outcomes []string
	for _, line := range strings.Split(strings.TrimSpace(jsonl), "\n") {
		var row struct {
			Schema            string `json:"schema"`
			Day               string `json:"day"`
			LooseBefore       int    `json:"loose_before"`
			LooseAfter        int    `json:"loose_after"`
			GraceRefused      string `json:"grace_refused"`
			GracePruneRefused string `json:"grace_prune_refused"`
			Incident          bool   `json:"incident"`
		}
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			t.Fatalf("ledger row %q does not parse: %v", line, err)
		}
		if row.Schema != "fak-git-daily/1" {
			t.Fatalf("ledger row schema = %q, want fak-git-daily/1", row.Schema)
		}
		in.Runs++
		if in.FirstDay == "" {
			in.FirstDay = row.Day
		}
		in.LastDay = row.Day
		if d := row.LooseBefore - row.LooseAfter; d > 0 {
			in.LooseFolded += d
		}
		reason := row.GraceRefused
		if reason == "" && row.GracePruneRefused != "" && row.GracePruneRefused != "PRUNE_OFF" {
			reason = row.GracePruneRefused
		}
		switch {
		case row.Incident:
			in.Errors++
			outcomes = append(outcomes, "error")
		case reason != "":
			in.Refused++
			if in.Reasons == nil {
				in.Reasons = map[string]int{}
			}
			in.Reasons[reason]++
			outcomes = append(outcomes, "refused")
		default:
			in.OK++
			outcomes = append(outcomes, GitDailyOutcomeOK)
		}
	}
	in.RefusedStreak = GitDailyRefusedStreak(outcomes)
	in.Today = "2026-08-05"
	in.LedgerPath = ".git/fak-git-daily.jsonl"
	return in
}

// TestRealLedgerCaptureIsDerivedFromTheRawRows is the anti-self-report check: the tally
// every other test in this file grades must be exactly what the RAW ledger rows fold to.
// It fails if realLedgerCapture was mistyped, if a row was dropped when the capture was
// refreshed, or if PRUNE_OFF ever starts counting as a refusal — none of which the
// assertions on the tally alone can see.
func TestRealLedgerCaptureIsDerivedFromTheRawRows(t *testing.T) {
	derived := foldRealLedgerJSONL(t, realLedgerJSONL)

	// DeepEqual, not ==: GitDailyHealthInput carries a Reasons map, so the struct is not
	// comparable. Both sides are nil here (PRUNE_OFF is not a refusal), and DeepEqual is
	// what keeps that true if the capture is ever refreshed onto a ledger that refused.
	if !reflect.DeepEqual(derived, realLedgerCapture) {
		t.Fatalf("the pinned tally is not what the raw ledger folds to:\n  derived %+v\n  pinned  %+v", derived, realLedgerCapture)
	}
	// DeepEqual compares the fold against the pinned tally; this compares it against the
	// pinned ROWS, so a fold that silently skipped a line (and a tally refreshed to match
	// that skip) still fails. Without it both sides could agree on a short read.
	if got, want := derived.Runs, strings.Count(strings.TrimSpace(realLedgerJSONL), "\n")+1; got != want {
		t.Errorf("folded %d run(s) from %d raw row(s)", got, want)
	}

	// And the captured OUTPUT — the issue's witness — must be what the raw rows grade to,
	// not merely what the typed tally grades to.
	want := "git-daily=A runs=5 ok=5 refused=0 error=0 folded=10038 streak=0 debt=0"
	if got := GitDailyHealthFragment(derived, GradeGitDailyHealth(derived)); got != want {
		t.Errorf("captured output from the raw ledger =\n  %q\nwant\n  %q", got, want)
	}
}

func TestGitDailyCalendarCoverageMatrix(t *testing.T) {
	cases := []struct {
		name         string
		first, today string
		hour         int
		days         []string
		wantScore    float64
		wantDefect   bool
		wantText     string
	}{
		{"every day", "2026-07-01", "2026-07-11", 4, daysEvery(1, 11), 100, false, "11 of 11"},
		{"every other day", "2026-07-01", "2026-07-11", 4, daysEvery(2, 11), 100 * 6.0 / 11.0, true, "5 missed calendar day"},
		{"four of eleven", "2026-07-01", "2026-07-11", 4, []string{"2026-07-01", "2026-07-04", "2026-07-07", "2026-07-10"}, 100 * 4.0 / 11.0, true, "7 missed calendar day"},
		{"pre tick today grace", "2026-07-01", "2026-07-11", 2, daysEvery(1, 10), 100, false, "10 of 10"},
		{"young ledger", "2026-07-10", "2026-07-11", 4, []string{"2026-07-10"}, 100, false, "young ledger"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			k := gitDailyCoverageKPI(GitDailyHealthInput{FirstDay: tc.first, Today: tc.today, CurrentHour: tc.hour, RunDays: tc.days}, "ledger.jsonl")
			if diff := k.Score - tc.wantScore; diff < -0.001 || diff > 0.001 {
				t.Fatalf("score=%v want %v: %+v", k.Score, tc.wantScore, k)
			}
			if (len(k.Defects) > 0) != tc.wantDefect {
				t.Fatalf("defects=%v want defect=%v", k.Defects, tc.wantDefect)
			}
			joined := k.Detail + " " + strings.Join(k.Defects, " ") + " " + strings.Join(k.Soft, " ")
			if !strings.Contains(joined, tc.wantText) {
				t.Fatalf("%q missing %q", joined, tc.wantText)
			}
			if tc.wantDefect && !strings.Contains(k.Detail, "powered off") {
				t.Fatalf("confound not communicated: %q", k.Detail)
			}
		})
	}
}

func TestGitDailyCalendarCoverageEmptyIsUngradable(t *testing.T) {
	k := gitDailyCoverageKPI(GitDailyHealthInput{}, "ledger.jsonl")
	if k.Score != 0 || len(k.Defects) != 0 || !strings.Contains(k.Detail, "not gradable") {
		t.Fatalf("unexpected empty coverage: %+v", k)
	}
}

func daysEvery(step, n int) []string {
	start, _ := time.Parse(gitDailyDayLayout, "2026-07-01")
	var out []string
	for i := 0; i < n; i += step {
		out = append(out, start.AddDate(0, 0, i).Format(gitDailyDayLayout))
	}
	return out
}
