// This file holds the span fold: one Span per issue, carrying the four
// time-in-state facts. The package doc lives in doc.go.
//
// It derives queue-theory flow facts about issues from two sources that already
// exist: the GitHub issue record (opened/closed) and the local git history (the
// only place a "work actually started" timestamp lives).
//
// WHY THIS PACKAGE EXISTS. The repo measures throughput (superloop.IssueTarget
// counts witnessed closes) and it measures ticket SHAPE (issuesmallness counts
// deliverables, issuecontract grades scale S0-S4). Neither measures TIME-IN-
// STATE, so the one question an operator actually asks — "how long did this sit
// before anyone touched it, and how much of its life was real work?" — has no
// answer. Lead time alone cannot answer it: a 9-day lead time is healthy if 8 of
// those days were active work and pathological if 8 were queue.
//
// THE START-TIME PROBLEM. GitHub hands us createdAt and closedAt and nothing in
// between: there is no "in progress" transition in this workflow (99.9% of open
// issues are unassigned, so assignment is not a start signal either). The
// insight this package is built on is that the FIRST COMMIT referencing #N is a
// hard, auditable start timestamp — it is the moment work provably touched the
// tree. That makes the classic Kanban decomposition computable:
//
//	lead  = closed - opened          (what the requester experiences)
//	queue = started - opened         (pure waiting; the WIP tax)
//	active = closed - started        (touch time)
//	flow efficiency = active / lead  (the fraction of life that was work)
//
// Flow efficiency is the load-bearing number. Throughput can look excellent
// while efficiency is 5%, and that gap IS the WIP problem: work sits.
//
// The core here is pure and clock-free — every function takes its facts as
// arguments so the whole fold is testable from fixtures. The impure gather
// (git log, gh) lives in gather.go.

package flowmetrics

import (
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/maputil"
)

// Schema is the payload contract version, following the repo's scorecard
// convention of a versioned schema string on every emitted envelope.
const Schema = "fak-flow-metrics/1"

// Issue is the minimal GitHub fact the fold needs. A nil ClosedAt means open.
// Kept deliberately small so a `gh issue list --json` dump maps onto it
// directly and a fixture can be written by hand.
type Issue struct {
	Number    int       `json:"number"`
	Title     string    `json:"title"`
	CreatedAt time.Time `json:"createdAt"`
	// ClosedAt is a pointer because "not closed" and "closed at the zero
	// time" are different facts and conflating them silently turns every
	// open issue into one with a 2000-year lead time.
	ClosedAt *time.Time `json:"closedAt,omitempty"`
	Labels   []string   `json:"labels,omitempty"`
	Body     string     `json:"body,omitempty"`
}

// Closed reports whether the issue has a real close timestamp.
func (i Issue) Closed() bool { return i.ClosedAt != nil && !i.ClosedAt.IsZero() }

// Commit is the minimal git fact the fold needs: when it landed, which issues
// its message referenced, and which lane (`(fak <leaf>)`) it claimed.
type Commit struct {
	SHA     string    `json:"sha"`
	When    time.Time `json:"when"`
	Subject string    `json:"subject"`
	Leaf    string    `json:"leaf,omitempty"`
	Issues  []int     `json:"issues,omitempty"`
	Files   int       `json:"files,omitempty"`
	Adds    int       `json:"adds,omitempty"`
	Dels    int       `json:"dels,omitempty"`
}

// issueRefRE matches a `#N` issue reference. It deliberately does NOT match
// inside a word, so a sha-like `abc#123` is skipped.
//
// The bound on WHICH numbers count is a value test, not a digit-count test, and
// that distinction is load-bearing. A digit-count fence (say 2-5 digits) is
// wrong in both directions on this repo: it drops the real single-digit issues
// (#3, #26, #40, #80 all exist) and it admits five-digit foreign-repo citations
// (llama.cpp #14762, vLLM #35021, claude-code #53063) that would otherwise be
// minted as phantom fak issues. Filter on ForeignRefFloor instead.
var issueRefRE = regexp.MustCompile(`(^|[^0-9A-Za-z_])#([0-9]+)\b`)

// ForeignRefFloor is the first issue number treated as belonging to another
// repository. fak's own numbering is in the 8000s, so 10000 leaves headroom
// while fencing out the upstream-project citations that appear in commit prose.
// Raise it when fak's numbering approaches it.
const ForeignRefFloor = 10000

// fixesTrailerRE matches the anchored close trailers, which are the only body
// references precise enough to attribute work.
//
// Measured on this history, mining raw body text for `#N` raises recall by
// ~11.5pp but inflates the multi-commit span p90 from 4.5d to 13.1d, because
// prose ("the #5822 shape") reads as an attribution it is not. Anchoring to a
// whole line beginning with fixes/closes/resolves keeps the recall that is real.
var fixesTrailerRE = regexp.MustCompile(`(?im)^\s*(?:fixes|closes|resolves)\s+#([0-9]+)\s*$`)

// leafRE matches the mandated `(fak <leaf>)` commit trailer that binds a commit
// to a lane. AGENTS.md requires it on every ship commit, so its ABSENCE is
// itself a measurable hygiene fact, not a parse failure.
var leafRE = regexp.MustCompile(`\(fak ([a-z0-9][a-z0-9._/-]*)\)\s*$`)

// ParseIssueRefs pulls the distinct issue numbers out of a commit SUBJECT, in
// ascending order so the result is stable for fixtures and diffs.
func ParseIssueRefs(subject string) []int {
	seen := map[int]bool{}
	var out []int
	collectRefs(issueRefRE, subject, 2, seen, &out)
	sort.Ints(out)
	return out
}

// ParseCommitRefs is the attribution used for a whole commit: every `#N` in the
// subject, plus only the anchored fixes/closes/resolves trailers from the body.
// Pass the body separately so a caller holding just a subject cannot
// accidentally widen attribution by concatenating the two.
func ParseCommitRefs(subject, body string) []int {
	seen := map[int]bool{}
	var out []int
	collectRefs(issueRefRE, subject, 2, seen, &out)
	collectRefs(fixesTrailerRE, body, 1, seen, &out)
	sort.Ints(out)
	return out
}

// collectRefs appends the distinct in-range issue numbers found by re in s,
// reading capture group grp, skipping any already in seen.
func collectRefs(re *regexp.Regexp, s string, grp int, seen map[int]bool, out *[]int) {
	for _, m := range re.FindAllStringSubmatch(s, -1) {
		n, err := strconv.Atoi(m[grp])
		if err != nil || n <= 0 || n >= ForeignRefFloor || seen[n] {
			continue
		}
		seen[n] = true
		*out = append(*out, n)
	}
}

// ParseLeaf pulls the `(fak <leaf>)` lane trailer from a commit subject,
// returning "" when the commit carries none.
func ParseLeaf(subject string) string {
	if m := leafRE.FindStringSubmatch(strings.TrimSpace(subject)); m != nil {
		return m[1]
	}
	return ""
}

// Span is one issue's derived flow record: the four time-in-state facts plus
// the work-shape facts (how many commits and lanes it took) that say whether it
// behaved like one atom or like a project.
type Span struct {
	Issue    int    `json:"issue"`
	Title    string `json:"title,omitempty"`
	OpenedAt time.Time
	// StartedAt is the first commit referencing this issue. nil means NO
	// commit ever referenced it — the issue is unstarted, which is the
	// single most important WIP state and the one GitHub cannot show.
	StartedAt *time.Time
	ClosedAt  *time.Time

	Commits int      `json:"commits"`
	Leaves  []string `json:"leaves,omitempty"`
	Files   int      `json:"files,omitempty"`
	Churn   int      `json:"churn,omitempty"`

	LeadHours   float64 `json:"lead_hours"`
	QueueHours  float64 `json:"queue_hours"`
	ActiveHours float64 `json:"active_hours"`
	// FlowEfficiency is active/lead in [0,1]. It is -1 when undefined
	// (unstarted, or a non-positive lead) so a caller can never mistake
	// "unknown" for "zero efficiency".
	FlowEfficiency float64 `json:"flow_efficiency"`
}

// Started reports whether any commit ever referenced the issue.
func (s Span) Started() bool { return s.StartedAt != nil }

// Closed reports whether the issue reached a close timestamp.
func (s Span) Closed() bool { return s.ClosedAt != nil }

// Atomic reports whether the issue landed as exactly one commit in exactly one
// lane — the shape AGENTS.md mandates ("one issue, one commit; one commit, one
// leaf"). Two commits is not automatically churn, but one is provably not.
func (s Span) Atomic() bool { return s.Commits == 1 && len(s.Leaves) <= 1 }

// AgeHours is how long an OPEN span has been in flight as of now: measured from
// the start when work began, else from the open. Closed spans report 0.
func (s Span) AgeHours(now time.Time) float64 {
	if s.Closed() {
		return 0
	}
	from := s.OpenedAt
	if s.Started() {
		from = *s.StartedAt
	}
	return hours(from, now)
}

// hours returns the non-negative hour delta between two instants. Clock skew or
// a rebased commit older than the issue it references would otherwise produce a
// negative duration that silently corrupts every percentile downstream, so the
// floor at zero is deliberate.
func hours(from, to time.Time) float64 {
	d := to.Sub(from).Hours()
	if d < 0 {
		return 0
	}
	return d
}

// BuildSpans folds issues plus commits into one Span per issue. Commits are
// matched to issues purely by `#N` reference, so a commit referencing three
// issues counts toward all three: that over-attribution is intentional and is
// itself the signal that the commit was not atomic.
//
// Commits referencing an issue not present in the issue set are ignored rather
// than synthesising a span, because without a createdAt there is no queue term
// and a partial span would pollute the percentiles.
func BuildSpans(issues []Issue, commits []Commit) []Span {
	byIssue := make(map[int][]Commit, len(issues))
	for _, c := range commits {
		for _, n := range c.Issues {
			byIssue[n] = append(byIssue[n], c)
		}
	}
	out := make([]Span, 0, len(issues))
	for _, iss := range issues {
		s := Span{
			Issue:          iss.Number,
			Title:          iss.Title,
			OpenedAt:       iss.CreatedAt,
			FlowEfficiency: -1,
		}
		if iss.Closed() {
			c := *iss.ClosedAt
			s.ClosedAt = &c
		}
		cs := byIssue[iss.Number]
		sort.Slice(cs, func(a, b int) bool { return cs[a].When.Before(cs[b].When) })
		if len(cs) > 0 {
			first := cs[0].When
			s.StartedAt = &first
			s.Commits = len(cs)
			leaves := map[string]bool{}
			for _, c := range cs {
				if c.Leaf != "" {
					leaves[c.Leaf] = true
				}
				s.Files += c.Files
				s.Churn += c.Adds + c.Dels
			}
			s.Leaves = sortedKeys(leaves)
		}
		// The three durations are only defined for the transitions that
		// actually happened; every one left at 0 means "not reached yet".
		if s.Started() {
			s.QueueHours = hours(s.OpenedAt, *s.StartedAt)
		}
		if s.Closed() {
			s.LeadHours = hours(s.OpenedAt, *s.ClosedAt)
			if s.Started() {
				s.ActiveHours = hours(*s.StartedAt, *s.ClosedAt)
				if s.LeadHours > 0 {
					s.FlowEfficiency = s.ActiveHours / s.LeadHours
				} else {
					// Opened and closed inside the same
					// instant: fully efficient by
					// definition, not undefined.
					s.FlowEfficiency = 1
				}
			}
		}
		out = append(out, s)
	}
	sort.Slice(out, func(a, b int) bool { return out[a].Issue < out[b].Issue })
	return out
}

func sortedKeys(m map[string]bool) []string {
	if len(m) == 0 {
		return nil
	}
	out := maputil.SortedKeys(m)
	return out
}

// Percentile returns the INCLUSIVE nearest-rank percentile of xs for p in
// [0,100]: index ceil(p/100*n)-1 over the ascending values. That definition is
// pinned deliberately — the exclusive and interpolated variants disagree by a
// whole rank on small cohorts, and every published flow number for this repo was
// computed with this one, so changing it would silently invalidate comparisons.
//
// It sorts a copy so the caller's slice is never reordered underneath it, and
// reports 0 for an empty input rather than panicking, because an empty cohort is
// a normal state for a freshly filtered slice.
func Percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	cp := append([]float64(nil), xs...)
	sort.Float64s(cp)
	if p <= 0 {
		return cp[0]
	}
	if p >= 100 {
		return cp[len(cp)-1]
	}
	rank := int(math.Ceil(p/100*float64(len(cp)))) - 1
	if rank < 0 {
		rank = 0
	}
	if rank >= len(cp) {
		rank = len(cp) - 1
	}
	return cp[rank]
}

// DayWIP is one day on the WIP curve.
type DayWIP struct {
	Date     string `json:"date"`
	Opened   int    `json:"opened"`
	Started  int    `json:"started"`
	Closed   int    `json:"closed"`
	InFlight int    `json:"in_flight"`
	Backlog  int    `json:"backlog"`
}

// WIPCurve reports, for each UTC day in [from,to], how many issues opened,
// started, and closed that day, and how many were IN FLIGHT across it.
//
// In-flight uses OVERLAP, not an end-of-day snapshot: an issue counts for a day
// if it had started before the day ended and had not yet closed when the day
// began. The snapshot definition is the more conventional one, and it is wrong
// for this repo — 53.9% of issues here close within 24h of opening, so a
// close-of-business reading would omit the majority of all work from the WIP
// curve entirely and show a fleet running at a fraction of its real concurrency.
// Overlap answers "how many distinct items were in hand that day", which is the
// question a WIP cap is actually set against.
//
// In-flight counts STARTED work only. An unstarted issue is backlog, not WIP:
// conflating the two is what makes a 1300-issue backlog look like 1300 units of
// work in progress when the real concurrent count is far smaller.
func WIPCurve(spans []Span, from, to time.Time) []DayWIP {
	if to.Before(from) {
		return nil
	}
	from = from.UTC().Truncate(24 * time.Hour)
	to = to.UTC().Truncate(24 * time.Hour)
	var out []DayWIP
	for d := from; !d.After(to); d = d.Add(24 * time.Hour) {
		end := d.Add(24 * time.Hour)
		row := DayWIP{Date: d.Format("2006-01-02")}
		for _, s := range spans {
			if !s.OpenedAt.Before(d) && s.OpenedAt.Before(end) {
				row.Opened++
			}
			if s.Started() && !s.StartedAt.Before(d) && s.StartedAt.Before(end) {
				row.Started++
			}
			if s.Closed() && !s.ClosedAt.Before(d) && s.ClosedAt.Before(end) {
				row.Closed++
			}
			if s.Started() && s.StartedAt.Before(end) && (!s.Closed() || !s.ClosedAt.Before(d)) {
				row.InFlight++
			}
			if s.OpenedAt.Before(end) && (!s.Closed() || !s.ClosedAt.Before(d)) && (!s.Started() || !s.StartedAt.Before(end)) {
				row.Backlog++
			}
		}
		out = append(out, row)
	}
	return out
}

// AgingWIP returns the open spans that have been STARTED but not closed,
// oldest-first. This is the actionable WIP list: every row is work someone
// already put commits into and then left, so it is the cheapest possible source
// of throughput — finishing one costs less than starting a new issue.
//
// A limit <= 0 returns every row.
func AgingWIP(spans []Span, now time.Time, limit int) []Span {
	var out []Span
	for _, s := range spans {
		if s.Started() && !s.Closed() {
			out = append(out, s)
		}
	}
	sort.Slice(out, func(a, b int) bool {
		if ai, bi := out[a].AgeHours(now), out[b].AgeHours(now); ai != bi {
			return ai > bi
		}
		return out[a].Issue < out[b].Issue
	})
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}
