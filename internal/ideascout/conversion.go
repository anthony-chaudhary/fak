package ideascout

// The CONVERSION GATE (#6506): what the scout's own filing history says about the
// stock it created and the flow out of it, the filing pause that follows from it,
// and the per-lane source health that decides whether a run may call itself a
// plain success.
//
// The defect this answers: the daily acting scout had filed 160 issues, 117 of
// which were still open AND still carrying `needs-triage` (oldest 41.5 days), and
// it kept planning three more every morning. Every run exited 0 — including the
// ones where all six Reddit queries 403'd. Discovery volume was measured
// (`candidates_gathered`, `planned`); conversion never was, so nothing in the loop
// could see that filing had outrun triage, and no run could report itself as
// anything other than a success.
//
// Three numbers therefore enter the run record, and one of them is allowed to stop
// the run from filing:
//
//	backlog/conversion — the filed → triaged → closed chain, read off the scout's
//	                     own label-targeted filing history (rung 2's corpus, already
//	                     fetched; this adds no network round trip).
//	filing gate        — pause new filing while the untriaged stock exceeds the
//	                     declared `untriaged_cap`. It re-arms itself: filing resumes
//	                     the moment the stock is drained back under the cap, so
//	                     "re-enable only after draining" is mechanical rather than a
//	                     note in an issue.
//	source health      — per-lane attempted/failed, so a lane that is down for EVERY
//	                     topic degrades the run's status instead of vanishing into a
//	                     list of strings nobody reads.

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

// The headline verdict a run reports about itself, as a closed vocabulary. A
// caller that only reads `status` still learns the two things that change what the
// run means: whether the source pool was complete, and whether filing was allowed.
//
// Precedence is degraded > paused > ok, and it is deliberate: a degraded run's
// candidate pool is incomplete, which invalidates every conclusion drawn from that
// pool — including "nothing new worth filing today". The gate's own verdict stays
// readable in FilingGate regardless of which status won.
const (
	StatusOK       = "ok"
	StatusDegraded = "degraded"
	StatusPaused   = "paused"
)

// Per-lane health. `down` means the lane failed on every topic that armed it —
// i.e. the source itself is unavailable (the Reddit 403s), not a flaky query.
const (
	LaneOK      = "ok"
	LanePartial = "partial"
	LaneDown    = "down"
)

// Why the filing gate held, as a closed vocabulary an operator can grep for.
const (
	GateUntriagedCap      = "untriaged-cap"
	GateIndexUnclassified = "index-unclassified"
)

// DefaultUntriagedCap is the declared ceiling on untriaged OPEN scout issues, in
// units of "days of filing at the default max_issues=3": ~2 weeks of unattended
// filing may accumulate before the scout stops adding to it. It is a config
// threshold (`untriaged_cap`), so a maintainer who is actively draining can raise
// it for a run; a NEGATIVE value disables the gate outright, and 0 means "pause
// while any untriaged issue is open at all".
const DefaultUntriagedCap = 40

// BacklogStats is the conversion ledger for the scout's own filings, computed from
// the SAME label-targeted corpus the durable dedup rung already fetches.
//
// What it can and cannot witness, stated plainly so the numbers are not read as
// more than they are:
//
//   - filed → triaged is exact: `needs-triage` is put on by the scout and can only
//     come off by a human. A CLOSED issue that still carries the label is NOT
//     counted as triaged — closing without ever clearing it is the observed habit,
//     and folding the two together would erase the finding.
//   - triaged → shipped is bounded, not exact. GitHub reports a closure as
//     COMPLETED both when a `Fixes #N` commit closed it and when a human closed it
//     by hand as "completed"; NOT_PLANNED covers the no-action/redundant closures.
//     So Converted is an UPPER BOUND on shipped effect and NoAction is a lower
//     bound on waste. The corpus cannot see the closing commit without a per-issue
//     timeline call, and a 160-call fan-out is not worth putting on a daily tick.
//   - Unclassified counts corpus rows that carry no state at all (a fixture replay,
//     or a `gh` that stopped returning the field). It exists so a blind index reads
//     as blind instead of as a backlog of zero.
type BacklogStats struct {
	Filed          int     `json:"filed"`
	Open           int     `json:"open"`
	Closed         int     `json:"closed"`
	Untriaged      int     `json:"untriaged_open"`
	Triaged        int     `json:"triaged"`
	Converted      int     `json:"converted"`
	NoAction       int     `json:"no_action"`
	Unclassified   int     `json:"unclassified"`
	OldestOpenDays int     `json:"oldest_open_days"`
	MedianOpenDays int     `json:"median_open_days"`
	ConversionRate float64 `json:"conversion_rate"`
	NoActionRate   float64 `json:"no_action_rate"`
	TriageRate     float64 `json:"triage_rate"`
}

// FilingGate is the decision, carried in the run record whether it held or not: a
// reader can always see the cap that was in force and the stock it was measured
// against, so "the scout filed nothing today" is never ambiguous between "nothing
// was new" and "the gate stopped it".
type FilingGate struct {
	Cap       int    `json:"untriaged_cap"`
	Untriaged int    `json:"untriaged_open"`
	Paused    bool   `json:"paused"`
	Reason    string `json:"reason,omitempty"`
	Detail    string `json:"detail,omitempty"`
}

// LaneHealth is one source lane's outcome across the topics that armed it.
type LaneHealth struct {
	Lane      string `json:"lane"`
	Attempted int    `json:"attempted"`
	Failed    int    `json:"failed"`
	Status    string `json:"status"`
}

// Backlog folds the scout's filing history into the conversion ledger. `now` dates
// the open-age numbers; an issue whose created_at cannot be parsed contributes to
// the counts but not to the ages, so a malformed timestamp cannot silently reset
// "oldest open" to zero.
func Backlog(issues []ExistingIssue, now time.Time) BacklogStats {
	stats := BacklogStats{Filed: len(issues)}
	var openAges []int
	for _, iss := range issues {
		switch {
		case iss.IsOpen():
			stats.Open++
			if iss.HasLabel(TriageLabel) {
				stats.Untriaged++
			} else {
				stats.Triaged++
			}
			if age, ok := ageDays(iss.CreatedAt, now); ok {
				openAges = append(openAges, age)
			}
		case iss.IsClosed():
			stats.Closed++
			if !iss.HasLabel(TriageLabel) {
				stats.Triaged++
			}
			switch closeReason(iss.StateReason) {
			case "completed":
				stats.Converted++
			case "not_planned":
				stats.NoAction++
			}
		default:
			stats.Unclassified++
		}
	}
	sort.Ints(openAges)
	if len(openAges) > 0 {
		stats.OldestOpenDays = openAges[len(openAges)-1]
		stats.MedianOpenDays = median(openAges)
	}
	stats.ConversionRate = rate(stats.Converted, stats.Filed)
	stats.NoActionRate = rate(stats.NoAction, stats.Filed)
	stats.TriageRate = rate(stats.Triaged, stats.Filed)
	return stats
}

// GateFiling decides whether the scout may add to the stock it already created.
//
// Two ways it holds. The first is the point of the gate: untriaged OPEN filings
// above the declared cap. The second is a fail-closed backstop — a corpus big
// enough to matter that reports NO state at all cannot be shown to be under the
// cap, and "we could not tell" must not read as "we are fine", which is exactly
// how a census that fails open loses its meaning.
func GateFiling(stats BacklogStats, cap int) FilingGate {
	gate := FilingGate{Cap: cap, Untriaged: stats.Untriaged}
	if cap < 0 {
		return gate // explicitly disabled by config
	}
	if stats.Untriaged > cap {
		gate.Paused = true
		gate.Reason = GateUntriagedCap
		gate.Detail = fmt.Sprintf("%d untriaged open %s issue(s) exceed untriaged_cap=%d; draining/triaging the stock re-opens filing automatically", stats.Untriaged, ScoutLabel, cap)
		return gate
	}
	if stats.Filed > cap && stats.Unclassified == stats.Filed {
		gate.Paused = true
		gate.Reason = GateIndexUnclassified
		gate.Detail = fmt.Sprintf("the filed-issue index returned no state for any of its %d issue(s), so the untriaged stock cannot be measured against untriaged_cap=%d", stats.Filed, cap)
	}
	return gate
}

// SourceHealth attributes the run's per-lane fetch errors to the lanes that were
// actually attempted. Attempts are derived from the config (which topic arms which
// lane) rather than counted by the gatherer, so the table covers lanes that failed
// before they could report anything.
func SourceHealth(topics []Topic, cfg Config, errs []string) []LaneHealth {
	attempts := laneAttempts(topics, cfg)
	failures := laneFailures(errs)
	var out []LaneHealth
	for _, lane := range sourceLanes {
		attempted := attempts[lane.label]
		if attempted == 0 {
			continue
		}
		health := LaneHealth{Lane: lane.label, Attempted: attempted, Failed: failures[lane.label], Status: LaneOK}
		switch {
		case health.Failed >= health.Attempted:
			health.Status = LaneDown
		case health.Failed > 0:
			health.Status = LanePartial
		}
		out = append(out, health)
	}
	return out
}

// Degraded is true when at least one lane failed on EVERY topic that armed it.
// That is the "persistent source error" case — the six Reddit 403s — as opposed to
// one flaky query, which is reported as `partial` and left alone.
func Degraded(health []LaneHealth) bool {
	for _, lane := range health {
		if lane.Status == LaneDown {
			return true
		}
	}
	return false
}

// RunStatus is the headline verdict; see the status vocabulary above for why
// degraded outranks paused.
func RunStatus(gate FilingGate, health []LaneHealth) string {
	switch {
	case Degraded(health):
		return StatusDegraded
	case gate.Paused:
		return StatusPaused
	default:
		return StatusOK
	}
}

// laneAttempts counts, per lane, the topics that armed it — the same arming
// conditions GatherCandidates applies, including the fresh lane's fresh_per_topic
// switch, so an unarmed lane never shows up as healthy-but-idle.
func laneAttempts(topics []Topic, cfg Config) map[string]int {
	out := map[string]int{}
	for _, topic := range topics {
		for _, lane := range sourceLanes {
			if topicQuery(topic, lane.topicKey) == "" {
				continue
			}
			if lane.label == "github-fresh" && cfg.FreshPerTopic <= 0 {
				continue
			}
			out[lane.label]++
		}
	}
	return out
}

// laneFailures counts the `lane[topic]: …` errors GatherCandidates records, keyed
// by lane. It matches against the declared lane vocabulary rather than any prefix,
// so the same-shaped `create[source-id]: …` filing errors are not mistaken for a
// dead source lane.
func laneFailures(errs []string) map[string]int {
	known := map[string]bool{}
	for _, lane := range sourceLanes {
		known[lane.label] = true
	}
	out := map[string]int{}
	for _, e := range errs {
		open := strings.Index(e, "[")
		if open <= 0 || !known[e[:open]] {
			continue
		}
		if !strings.Contains(e[open:], "]: ") {
			continue
		}
		out[e[:open]]++
	}
	return out
}

// closeReason normalises GitHub's state_reason ("COMPLETED", "NOT_PLANNED", and
// the "not planned" spelling some payloads use) to the two tokens this package
// counts.
func closeReason(raw string) string {
	s := strings.ToLower(strings.TrimSpace(raw))
	s = strings.ReplaceAll(s, " ", "_")
	s = strings.ReplaceAll(s, "-", "_")
	return s
}

func ageDays(created string, now time.Time) (int, bool) {
	if strings.TrimSpace(created) == "" {
		return 0, false
	}
	t, err := time.Parse(time.RFC3339, created)
	if err != nil {
		return 0, false
	}
	days := int(now.Sub(t).Hours() / 24)
	if days < 0 {
		days = 0
	}
	return days, true
}

// median of a sorted slice; the even case takes the floor of the two middles so
// the number stays an integer day count.
func median(sorted []int) int {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n%2 == 1 {
		return sorted[n/2]
	}
	return (sorted[n/2-1] + sorted[n/2]) / 2
}

func rate(part, whole int) float64 {
	if whole <= 0 {
		return 0
	}
	return math.Round(float64(part)/float64(whole)*10000) / 10000
}
