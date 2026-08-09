package gitdaily

// Outcome counters for the daily tick (#5586).
//
// WHY A FOLD OVER THE LEDGER AND NOT A SEPARATE COUNTER STORE. Every applied tick
// already appends one `fak-git-daily/1` row, and `fak git-daily --status` already reads
// those rows back. A second, separately-incremented counter file would be a RIVAL
// witness: it can disagree with the ledger, it starts at zero (blind to every run the
// job has already made), and it needs its own write path to keep correct. Folding the
// rows that are already on disk keeps one source of truth and makes the counts
// retroactive over the entire retained history — including the runs recorded before
// this counter existed.
//
// WHAT "RUNS" DOES NOT COUNT. A skipped tick (ALREADY_RAN_TODAY, TICK_BUSY) writes no
// ledger row by design — that is what lets a coarse catch-up-on-wake trigger fire hourly
// without inflating the history. So these counters tally the ticks that DID WORK, not
// the times the OS scheduler fired. "0 refused, 0 error" therefore means "every run that
// reached the tiers was healthy", not "nothing was ever skipped".
//
// WHY PRUNE_OFF IS NOT COUNTED AS A REFUSAL. gitgate's grace-prune tier is opt-in and
// default-OFF, and it records exactly PRUNE_OFF when the caller did not opt in. That is
// the configured posture of a healthy default run, not a tier being held back — counting
// it would mark 100% of default runs "refused" and bury the one signal these counters
// exist for. Every OTHER GracePruneRefused reason (LOCKED, SESSION_LIVE, POSTURE_DRIFT,
// PRUNE_EXPIRE_UNSAFE) can only appear when the operator DID opt in, so those do count.

import (
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gitgate"
)

// Outcome is the closed vocabulary for one recorded tick's result. Closed, not free
// text, so a readback (and an alert built on one) can match on it.
type Outcome string

const (
	// OutcomeOK: the tick ran and no tier was held back — the healthy steady state.
	OutcomeOK Outcome = "ok"
	// OutcomeRefused: a tier declined for a structured reason. One is ordinary (a peer
	// held a lock); a STREAK is the #4602 regression, where the job reports success
	// daily while the backlog grows.
	OutcomeRefused Outcome = "refused"
	// OutcomeError: the run surfaced something only an operator can repair — posture
	// drift, or a lock cleanup that failed and left the maintenance wedge in place.
	OutcomeError Outcome = "error"
)

// RefusalReason names the structured reason that makes this row a refusal, or "" when
// the row is not one. See the file header for why PRUNE_OFF is excluded.
func (r Row) RefusalReason() string {
	if r.GraceRefused != "" {
		return r.GraceRefused
	}
	if r.GracePruneRefused != "" && r.GracePruneRefused != string(gitgate.MaintReasonPruneOff) {
		return r.GracePruneRefused
	}
	return ""
}

// Outcome classifies one recorded tick. An error outranks a refusal: a run that drifted
// posture AND deferred its fold tier is one thing an operator must fix, and reporting it
// as a mere refusal would let it sit.
func (r Row) Outcome() Outcome {
	switch {
	case r.Incident:
		return OutcomeError
	case r.RefusalReason() != "":
		return OutcomeRefused
	default:
		return OutcomeOK
	}
}

// Outcomes is the invocation tally over a run history: the counts that answer "is this
// job still working?" without anyone diffing the rows by hand.
type Outcomes struct {
	// Runs is the number of RECORDED ticks folded (see the file header: skips write no
	// row, so this is not the number of trigger fires).
	Runs    int `json:"runs"`
	OK      int `json:"ok"`
	Refused int `json:"refused"`
	Errors  int `json:"errors"`
	// Reasons counts refusals by their structured reason, so "LOCKED x9" and
	// "POSTURE_DRIFT x1" stay distinguishable — they are different operator stories.
	Reasons map[string]int `json:"reasons,omitempty"`
	// FirstDay / LastDay bound the window these counts cover, so a count is never read
	// as "all time" when the ledger has rotated or the caller asked for the last N.
	FirstDay string `json:"first_day,omitempty"`
	LastDay  string `json:"last_day,omitempty"`
	// LooseFolded is the loose objects consolidated across the window — the volume that
	// makes the ok-count mean work rather than a no-op.
	LooseFolded int `json:"loose_folded"`
}

// FoldOutcomes tallies a run history into its counters. Pure and total: an empty history
// folds to a zero Outcomes, which is the first-run contract every readback here shares.
func FoldOutcomes(rows []Row) Outcomes {
	var out Outcomes
	for _, r := range rows {
		out.Runs++
		if out.FirstDay == "" {
			out.FirstDay = r.Day
		}
		out.LastDay = r.Day
		// A NEGATIVE delta means peers wrote more loose objects during the run than it
		// folded — that is their commit volume, not a fold that ran backwards, so it
		// must not be subtracted from the work this job actually did.
		if d := r.LooseFolded(); d > 0 {
			out.LooseFolded += d
		}
		switch r.Outcome() {
		case OutcomeError:
			out.Errors++
		case OutcomeRefused:
			out.Refused++
			if out.Reasons == nil {
				out.Reasons = map[string]int{}
			}
			out.Reasons[r.RefusalReason()]++
		default:
			out.OK++
		}
	}
	return out
}

// WeekOutcome is the per-ISO-week invocation/outcome fold used by the adoption
// readback. Week is the Monday that begins the UTC week (YYYY-MM-DD), making the
// bucket stable across locale settings and year boundaries.
type WeekOutcome struct {
	Week    string `json:"week"`
	Total   int    `json:"total"`
	OK      int    `json:"ok"`
	Refused int    `json:"refused"`
	Errors  int    `json:"errors"`
}

// FoldOutcomesByWeek groups recorded invocations by their RFC3339 timestamp. Legacy
// rows without a parseable At value are intentionally omitted: assigning them to the
// current week would fabricate adoption. Results are chronological and deterministic.
func FoldOutcomesByWeek(rows []Row) []WeekOutcome {
	byWeek := make(map[string]*WeekOutcome)
	for _, r := range rows {
		at, err := time.Parse(time.RFC3339, r.At)
		if err != nil {
			continue
		}
		at = at.UTC()
		weekday := (int(at.Weekday()) + 6) % 7 // Monday = 0.
		monday := at.AddDate(0, 0, -weekday).Format("2006-01-02")
		w := byWeek[monday]
		if w == nil {
			w = &WeekOutcome{Week: monday}
			byWeek[monday] = w
		}
		w.Total++
		switch r.Outcome() {
		case OutcomeError:
			w.Errors++
		case OutcomeRefused:
			w.Refused++
		default:
			w.OK++
		}
	}
	weeks := make([]string, 0, len(byWeek))
	for week := range byWeek {
		weeks = append(weeks, week)
	}
	sort.Strings(weeks)
	out := make([]WeekOutcome, 0, len(weeks))
	for _, week := range weeks {
		out = append(out, *byWeek[week])
	}
	return out
}

// ReasonsByCount renders the refusal breakdown in a stable order — commonest first, ties
// broken by name — so a readout diffed across two days does not churn on map ordering.
func (o Outcomes) ReasonsByCount() []string {
	names := make([]string, 0, len(o.Reasons))
	for name := range o.Reasons {
		names = append(names, name)
	}
	sort.Slice(names, func(i, j int) bool {
		if a, b := o.Reasons[names[i]], o.Reasons[names[j]]; a != b {
			return a > b
		}
		return names[i] < names[j]
	})
	return names
}
