package metrics

// git_daily_health.go — the deterministic scorecard fold that grades Daily lock-aware
// Git hygiene (`fak git-daily`, the #5577 spine) from its own ledger, so "is it still
// good?" is a graded command rather than a hand audit of `--status` rows (#5587).
//
// WHAT IT GRADES, AND WHY THOSE THREE. The spine's failure mode is not a crash — it is a
// job that keeps *reporting* success. #4602 is the canonical shape: the scheduled tick ran
// daily, exited 0, and folded nothing for a week while a wedged lock pinned the fold tier
// at LOCKED and the loose-object backlog grew to 67,885. A grade that only asked "did it
// exit 0?" would have read A for that entire week. So the card folds the three axes that
// can each go bad silently:
//
//   - adoption (usage) — is the OS trigger still LANDING runs? A daily job whose newest
//     recorded tick is days old has stopped, and nothing errors when it does.
//   - outcome_health (failure rate) — of the ticks that DID run, how many refused a tier
//     or hit an incident? One refusal is a peer holding a lock; a majority is a held tier.
//   - fold_drift (drift) — the #4602 signature itself: a TRAILING streak of non-ok ticks,
//     i.e. the newest runs all deferring while the job keeps reporting daily success.
//
// SCORED FROM WITNESSES, NEVER FROM SELF-REPORT. Every number here descends from the
// `fak-git-daily/1` ledger rows the tick already appends (loose_before/loose_after,
// grace_refused, grace_prune_refused, incident, day) — the same rows `fak git-daily
// --status` reads back. The card invents no counter of its own, so it cannot disagree
// with the ledger, and it grades retroactively over history recorded before it existed.
//
// WHY IT IS A PURE FOLD IN metrics AND NOT IN gitdaily. internal/gitdaily sits a layer
// above internal/metrics, so importing it here would red architest with
// ARCH_LAYER_VIOLATION (the same constraint stale_lock_wedge.go documents). The card
// therefore takes the ledger-derived tallies as input — a one-to-one projection of the
// counters gitdaily.FoldOutcomes already computes, plus the trailing streak — and the I/O
// (open the ledger, read the rows, print the payload) belongs to whichever lane wires the
// operator readout. Keeping the decision pure is also what makes it deterministic: the
// same tallies always grade the same, with no clock, no filesystem, and no git.
//
// WORKED EXAMPLE — the real ledger on this clone, .git/fak-git-daily.jsonl, folded on
// 2026-08-05 (5 recorded ticks, 2026-08-04..2026-08-05, 10038 loose objects folded, every
// tick ok, PRUNE_OFF excluded from the refusal tally because the grace-prune tier is
// default-OFF):
//
//	git-daily=A runs=5 ok=5 refused=0 error=0 folded=10038 streak=0 debt=0
//
// That capture is pinned as a test case in git_daily_health_test.go, so the card is graded
// against real ledger data on every `make test-fast` / `make ci` run, not just fixtures.

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// GitDailyHealthSchema versions the control-pane payload this card emits.
const GitDailyHealthSchema = "fak-git-daily-health/1"

// GitDailyDebtKey is the card's `<name>_debt` headline in the control-pane corpus: the
// count of concrete, re-derivable repairs the grade found.
const GitDailyDebtKey = "git_daily_debt"

// DefaultGitDailyStaleDays is how many days may pass between the newest recorded tick and
// today before adoption counts as lapsed. Two, not one: a daily job scheduled for 03:00
// legitimately has YESTERDAY as its newest row for most of the current day, so a
// one-day gate would red every morning.
const DefaultGitDailyStaleDays = 2

// gitDailyDayLayout mirrors gitdaily.DayLayout — the ledger's LOCAL calendar day key.
// Duplicated as a literal rather than imported because gitdaily sits a layer up.
const gitDailyDayLayout = "2006-01-02"

// GitDailyOutcomeOK mirrors gitdaily's ok outcome by VALUE (not by import, see the file
// header). It is the one member of that closed vocabulary this card needs to name: every
// other value counts as a non-ok tick for the trailing-streak rule.
const GitDailyOutcomeOK = "ok"

// gitDailyCoverageKey names the calendar-coverage axis. It is a const because
// GradeGitDailyHealth also has to name it in the weight map that drops the axis from the
// composite when it is not gradable, and a typo there would silently restore the zero.
const gitDailyCoverageKey = "calendar_coverage"

// GitDailyHealthInput is the ledger-derived witness set the card grades. Every field is a
// one-to-one projection of what gitdaily.FoldOutcomes already computes over the
// `fak-git-daily/1` rows, so a caller passes tallies through rather than re-deriving them.
type GitDailyHealthInput struct {
	// Runs is the number of RECORDED ticks in the window. A deliberately-skipped tick
	// (ALREADY_RAN_TODAY, TICK_BUSY) writes no row by design, so this counts the ticks
	// that DID work — never the times the OS scheduler fired.
	Runs int
	// OK / Refused / Errors partition Runs by recorded outcome.
	OK      int
	Refused int
	Errors  int
	// Reasons counts refusals by their structured reason (LOCKED, POSTURE_DRIFT, …), so a
	// defect names WHICH tier was held rather than just how often. PRUNE_OFF is expected
	// to be absent: the grace-prune tier is opt-in and default-OFF, so its refusal is the
	// configured posture of a healthy run, not a tier being held back.
	Reasons map[string]int
	// FirstDay / LastDay bound the graded window, so a count is never read as "all time"
	// when the ledger has rotated or the caller asked for the last N rows.
	FirstDay string
	LastDay  string
	// LooseFolded is the loose objects consolidated across the window — the volume that
	// makes an ok count mean work rather than a no-op.
	LooseFolded int
	// RefusedStreak is the count of TRAILING consecutive non-ok ticks (0 when the newest
	// recorded tick was ok). This is the #4602 signal: one refusal is ordinary, a streak
	// is a wedged lock pinning the tier while the job keeps reporting daily success.
	// GitDailyRefusedStreak derives it so the rule lives with the grade.
	RefusedStreak int
	// Today is the LOCAL day key the recency check measures against ("" disables it).
	Today string
	// CurrentHour is the local wall-clock hour used only for today's 03:00 schedule grace.
	CurrentHour int
	// RunDays carries ledger day keys; duplicates are folded before coverage grading.
	RunDays []string
	// LedgerPath names the witness in every evidence string; "" falls back to the bare
	// ledger file name.
	LedgerPath string
	// StaleAfterDays overrides DefaultGitDailyStaleDays; <= 0 uses the default.
	StaleAfterDays int
}

// GitDailyRefusedStreak counts the trailing consecutive non-ok ticks in an OLDEST-FIRST
// run of recorded outcomes. It is the streak rule itself, kept here (rather than at the
// caller) so the definition of "a streak" cannot drift from the card that grades it.
// Anything that is not exactly the ok outcome — a refusal or an incident — extends the
// streak: both mean the newest ticks stopped doing the work.
func GitDailyRefusedStreak(outcomes []string) int {
	streak := 0
	for i := len(outcomes) - 1; i >= 0; i-- {
		if outcomes[i] == GitDailyOutcomeOK {
			break
		}
		streak++
	}
	return streak
}

// GradeGitDailyHealth folds the ledger witnesses into the control-pane payload: a 0-100
// score, an A-F grade, a `git_daily_debt` headline, and one KPI per axis carrying NAMED
// evidence (each defect states the count, the window, and the ledger it came from). Pure
// and total — an empty input grades the first-run contract (no witness => no health
// claim) rather than panicking.
func GradeGitDailyHealth(in GitDailyHealthInput) scorecard.Payload {
	ledger := in.LedgerPath
	if ledger == "" {
		ledger = "fak-git-daily.jsonl"
	}
	stale := in.StaleAfterDays
	if stale <= 0 {
		stale = DefaultGitDailyStaleDays
	}

	// Adoption owns a TRAILING gap. When it has already charged "the trigger stopped",
	// the days coverage would enumerate are that same single repair seen at a finer
	// resolution, so coverage reports them soft rather than charging a second debt unit —
	// the rule the empty-ledger case already applies to outcome_health and fold_drift.
	gap, gapKnown := gitDailyDayGap(in.LastDay, in.Today)
	adoptionStale := in.Runs > 0 && gapKnown && gap >= stale

	coverage, coverageGraded := gitDailyCoverageKPI(in, ledger, adoptionStale)
	kpis := []scorecard.KPI{
		gitDailyAdoptionKPI(in, ledger, stale),
		coverage,
		gitDailyOutcomeKPI(in, ledger),
		gitDailyDriftKPI(in, ledger),
	}

	// A not-gradable axis must not average into the composite: "no witness was supplied"
	// is a different claim from "this scored zero", and Fold's mean cannot tell them
	// apart. Weight it out instead, so an absent witness costs the grade nothing. Group
	// is looked up before Key, and no weight names the "usage" group, so this reaches
	// exactly the coverage axis and leaves adoption at its default weight of 1.
	var weights map[string]float64
	if !coverageGraded {
		weights = map[string]float64{gitDailyCoverageKey: 0}
	}

	msgs := scorecard.Messages{
		Finding:         fmt.Sprintf("daily lock-aware git hygiene is degraded on the witness of %s", ledger),
		FindingClean:    fmt.Sprintf("daily lock-aware git hygiene is healthy on the witness of %s", ledger),
		NextAction:      "repair the named defect, re-run `fak git-daily`, and re-grade against the new ledger row",
		NextActionClean: "no action: re-grade after the next recorded tick",
		Grade:           scorecard.GradeStd,
		ExtraCorpus: map[string]any{
			"runs":             in.Runs,
			"ok":               in.OK,
			"refused":          in.Refused,
			"errors":           in.Errors,
			"loose_folded":     in.LooseFolded,
			"refused_streak":   in.RefusedStreak,
			"first_day":        in.FirstDay,
			"last_day":         in.LastDay,
			"today":            in.Today,
			"ledger_path":      ledger,
			"stale_after_days": stale,
		},
	}
	return scorecard.Fold(GitDailyHealthSchema, kpis, GitDailyDebtKey, weights, msgs)
}

// gitDailyAdoptionKPI grades usage: is the trigger still landing runs at all?
func gitDailyAdoptionKPI(in GitDailyHealthInput, ledger string, stale int) scorecard.KPI {
	k := scorecard.KPI{Key: "adoption", Group: "usage"}
	if in.Runs <= 0 {
		k.Detail = "no fak-git-daily/1 row recorded"
		k.Defects = []string{fmt.Sprintf(
			"%s carries no fak-git-daily/1 row: the daily tick has never recorded an applied run, so no claim about its health is witnessed",
			ledger)}
		return k
	}

	k.Detail = fmt.Sprintf("%d recorded tick(s) in %s over %s", in.Runs, ledger, gitDailyWindow(in))
	gap, known := gitDailyDayGap(in.LastDay, in.Today)
	if !known {
		k.Score = 100
		k.Soft = []string{"recency not graded: no parsable today/last-day key, so only the recorded-run count is witnessed"}
		return k
	}

	k.Detail += fmt.Sprintf("; newest %s is %d day(s) before today %s", in.LastDay, gap, in.Today)
	if gap < stale {
		k.Score = 100
		return k
	}
	// Past the staleness gate the score decays a quarter per extra day, so a job stopped
	// for a week reads worse than one that missed a single cycle.
	k.Score = float64(max(0, 100-25*(gap-stale+1)))
	k.Defects = []string{fmt.Sprintf(
		"newest recorded tick in %s is %s, %d day(s) stale as of %s (>= %d): the schedule has stopped landing runs — re-check the OS trigger, and note a powered-down host reads the same way because a skipped tick writes no row",
		ledger, in.LastDay, gap, in.Today, stale)}
	return k
}

// gitDailyCoverageKPI grades each due calendar day. Today becomes due at the 03:00
// local schedule. Missing days are typed debt, while detail preserves the powered-off-host
// confound. The KPI is additive under fak-git-daily-health/1.
//
// The second return says whether the axis was GRADED at all. A day is only "missed" on the
// witness of the run-day keys, so a caller that supplies none has witnessed nothing about
// coverage — folding that silence as "every due day missed" would manufacture debt out of
// an absent witness, which is exactly what the file header forbids. The caller weights an
// ungraded axis out of the composite rather than averaging its zero in.
//
// adoptionStale reports that the adoption axis has already charged the trailing gap. The
// missing days are then the same repair at a finer resolution, so they stay soft: the
// score still falls, but `git_daily_debt` counts one repair, not two.
func gitDailyCoverageKPI(in GitDailyHealthInput, ledger string, adoptionStale bool) (scorecard.KPI, bool) {
	k := scorecard.KPI{Key: gitDailyCoverageKey, Group: "usage"}
	first, err1 := time.Parse(gitDailyDayLayout, in.FirstDay)
	today, err2 := time.Parse(gitDailyDayLayout, in.Today)
	if err1 != nil || err2 != nil || today.Before(first) {
		k.Detail = "not gradable: calendar window is missing or invalid"
		k.Soft = []string{"calendar coverage needs valid first-day and today witnesses"}
		return k, false
	}
	if len(in.RunDays) == 0 {
		k.Detail = "not gradable: no run-day witness for the graded window"
		k.Soft = []string{fmt.Sprintf(
			"calendar coverage needs the per-run day keys behind %s: %d recorded tick(s) arrived without them, and an absent witness is not a missed day",
			ledger, in.Runs)}
		return k, false
	}
	end := today
	if in.CurrentHour < 3 {
		end = end.AddDate(0, 0, -1)
	}
	if end.Before(first) {
		k.Score = 100
		k.Detail = "today is still inside the pre-03:00 schedule grace; no day is due yet"
		return k, true
	}
	due := int(end.Sub(first).Hours()/24) + 1
	seen := make(map[string]bool, len(in.RunDays))
	for _, day := range in.RunDays {
		seen[day] = true
	}
	missing := make([]string, 0)
	for day := first; !day.After(end); day = day.AddDate(0, 0, 1) {
		key := day.Format(gitDailyDayLayout)
		if !seen[key] {
			missing = append(missing, key)
		}
	}
	covered := due - len(missing)
	k.Detail = fmt.Sprintf("%d of %d due calendar day(s) recorded from %s through %s; a missing day may mean the host was powered off, which this ledger cannot distinguish from trigger failure", covered, due, in.FirstDay, end.Format(gitDailyDayLayout))
	if due < 3 {
		k.Score = 100
		k.Soft = append(k.Soft, "young ledger: fewer than 3 due days, so missing-day coverage is observed but not debt")
		return k, true
	}
	k.Score = 100 * float64(covered) / float64(due)
	if len(missing) > 0 {
		gaps := fmt.Sprintf("%d missed calendar day(s) in %s (%s): verify both host availability and the scheduled trigger", len(missing), ledger, strings.Join(missing, ", "))
		if adoptionStale {
			k.Soft = append(k.Soft, gaps+"; kept soft because the adoption axis already charges this stopped trigger as the one repair")
		} else {
			k.Defects = append(k.Defects, gaps)
		}
	}
	return k, true
}

// gitDailyOutcomeKPI grades the failure rate of the ticks that did run. An incident is
// fully disqualifying for its tick (only an operator can clear posture drift or a lock
// cleanup that failed); a refusal costs half, because one refusal is a peer holding a
// lock and only a majority means a tier is actually held.
func gitDailyOutcomeKPI(in GitDailyHealthInput, ledger string) scorecard.KPI {
	k := scorecard.KPI{Key: "outcome_health", Group: "failure_rate"}
	if in.Runs <= 0 {
		k.Detail = "not gradable: no recorded tick to classify"
		k.Soft = []string{"outcome health needs at least one recorded tick; the adoption defect already names the single root cause"}
		return k
	}

	runs := float64(in.Runs)
	k.Score = max(0, 100-100*float64(in.Errors)/runs-50*float64(in.Refused)/runs)
	if in.Errors > 0 {
		// A single incident in a LONG window would otherwise average away to a passing
		// score, and an incident is precisely the outcome that does not self-heal: the
		// maintenance wedge sits there until an operator clears it. Any incident at all
		// therefore caps this axis at half, so the letter grade moves too and not just the
		// debt count.
		k.Score = min(k.Score, 50)
	}
	reasons := gitDailyReasons(in.Reasons)
	k.Detail = fmt.Sprintf("%d ok, %d refused, %d error across %d recorded tick(s) over %s",
		in.OK, in.Refused, in.Errors, in.Runs, gitDailyWindow(in))
	if reasons != "" {
		k.Detail += "; refusals: " + reasons
	}

	if in.Errors > 0 {
		k.Defects = append(k.Defects, fmt.Sprintf(
			"%d of %d recorded ticks in %s recorded an incident (posture drift, or a lock cleanup that failed and left the maintenance wedge in place): only an operator can clear it",
			in.Errors, in.Runs, ledger))
	}
	switch {
	case in.Refused*2 > in.Runs:
		k.Defects = append(k.Defects, fmt.Sprintf(
			"%d of %d recorded ticks in %s refused a tier (%s): a majority-refused window is a held tier, not a peer's passing lock",
			in.Refused, in.Runs, ledger, gitDailyReasonsOrUnnamed(reasons)))
	case in.Refused > 0:
		k.Soft = append(k.Soft, fmt.Sprintf(
			"%d of %d recorded ticks refused a tier (%s): ordinary at this share — a peer holding a transaction lock is expected",
			in.Refused, in.Runs, gitDailyReasonsOrUnnamed(reasons)))
	}
	return k
}

// gitDailyDriftKPI grades the #4602 drift signature: the newest ticks all deferring while
// the job keeps reporting daily success.
func gitDailyDriftKPI(in GitDailyHealthInput, ledger string) scorecard.KPI {
	k := scorecard.KPI{Key: "fold_drift", Group: "drift"}
	if in.Runs <= 0 {
		k.Detail = "not gradable: no recorded tick to trend"
		k.Soft = []string{"drift needs at least one recorded tick; the adoption defect already names the single root cause"}
		return k
	}

	k.Detail = fmt.Sprintf("%d loose object(s) folded across %d recorded tick(s) over %s; trailing non-ok streak %d",
		in.LooseFolded, in.Runs, gitDailyWindow(in), in.RefusedStreak)

	if in.RefusedStreak <= 1 {
		k.Score = 100
	} else {
		k.Score = float64(max(0, 100-25*in.RefusedStreak))
		k.Defects = append(k.Defects, fmt.Sprintf(
			"the newest %d recorded ticks in %s all refused or errored (%s): one is a peer's lock, a streak is a wedged lock pinning the tier while the job keeps reporting daily success (#4602)",
			in.RefusedStreak, ledger, gitDailyReasonsOrUnnamed(gitDailyReasons(in.Reasons))))
	}

	// Zero folded volume is NOT scored as a defect: an already-packed object DB folds
	// nothing and is perfectly healthy. It is surfaced soft so an operator can compare it
	// against loose_before, which is the one reading that tells the two apart.
	if in.OK > 0 && in.LooseFolded <= 0 {
		k.Soft = append(k.Soft, fmt.Sprintf(
			"%d ok tick(s) folded 0 loose objects: healthy when the object DB was already packed, the #4602 shape when it was not — compare loose_before in %s",
			in.OK, ledger))
	}
	return k
}

// GitDailyHealthFragment renders a compact one-line readout of a graded payload for an
// operator surface (a status pane, a tick-stream row). Pure and deterministic — the same
// input and payload always render the same bytes. Example:
// "git-daily=A runs=5 ok=5 refused=0 error=0 folded=10038 streak=0 debt=0".
func GitDailyHealthFragment(in GitDailyHealthInput, p scorecard.Payload) string {
	grade, _ := p.Corpus["grade"].(string)
	if grade == "" {
		grade = "?"
	}
	debt, _ := p.Corpus[GitDailyDebtKey].(int)
	return fmt.Sprintf("git-daily=%s runs=%d ok=%d refused=%d error=%d folded=%d streak=%d debt=%d",
		grade, in.Runs, in.OK, in.Refused, in.Errors, in.LooseFolded, in.RefusedStreak, debt)
}

// gitDailyWindow names the graded window for an evidence string.
func gitDailyWindow(in GitDailyHealthInput) string {
	switch {
	case in.FirstDay == "" && in.LastDay == "":
		return "an unnamed window"
	case in.FirstDay == in.LastDay:
		return in.FirstDay
	case in.FirstDay == "":
		return ".." + in.LastDay
	case in.LastDay == "":
		return in.FirstDay + ".."
	default:
		return in.FirstDay + ".." + in.LastDay
	}
}

// gitDailyDayGap reports whole days between the newest recorded day and today. Both keys
// parse as UTC midnight, so the difference is an exact multiple of 24h regardless of the
// local zone the keys were written in. A negative gap (a clock stepped backwards) clamps
// to 0 rather than crediting the future.
func gitDailyDayGap(lastDay, today string) (int, bool) {
	if lastDay == "" || today == "" {
		return 0, false
	}
	last, err := time.Parse(gitDailyDayLayout, lastDay)
	if err != nil {
		return 0, false
	}
	now, err := time.Parse(gitDailyDayLayout, today)
	if err != nil {
		return 0, false
	}
	return max(0, int(now.Sub(last).Hours()/24)), true
}

// gitDailyReasons renders the refusal breakdown commonest-first, ties broken by name, so
// a readout diffed across two days does not churn on map ordering.
func gitDailyReasons(reasons map[string]int) string {
	if len(reasons) == 0 {
		return ""
	}
	names := make([]string, 0, len(reasons))
	for name := range reasons {
		names = append(names, name)
	}
	sort.Slice(names, func(i, j int) bool {
		if a, b := reasons[names[i]], reasons[names[j]]; a != b {
			return a > b
		}
		return names[i] < names[j]
	})
	parts := make([]string, 0, len(names))
	for _, name := range names {
		parts = append(parts, fmt.Sprintf("%s x%d", name, reasons[name]))
	}
	return strings.Join(parts, ", ")
}

// gitDailyReasonsOrUnnamed keeps an evidence string readable when a refusal was tallied
// without a structured reason, rather than emitting an empty parenthetical.
func gitDailyReasonsOrUnnamed(reasons string) string {
	if reasons == "" {
		return "no structured reason recorded"
	}
	return reasons
}
