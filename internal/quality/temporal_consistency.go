package quality

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
)

// tmpTemporalConsistency is the temporal-consistency oracle for executive
// reports (#4556): the report must not assert a fact that a NEWER datapoint in
// the case's ground truth contradicts, and must not present a stale date as
// current. It is the time axis of the report-quality layer — claim-grounding
// (#4551) checks a claim is backed by SOME evidence; this oracle checks a
// claim is not contradicted by the NEWEST evidence, which is the direction
// stale rollups enter from (last week's status pasted into this week's
// report).
//
// Ground truth travels in the case as a dated fact block in Reference.Text,
// one fact per line:
//
//	YYYY-MM-DD | topic | status
//
// Lines that do not parse (wrong field count, empty topic/status, invalid
// date) are skipped, never panicked on. Facts sharing a topic (matched
// case-insensitively) form that topic's timeline: the newest-dated status is
// CURRENT and every older status is SUPERSEDED; on a date tie the
// later-declared line wins. Validated dates compare chronologically as plain
// YYYY-MM-DD strings.
//
// Two claim classes are checked against the block, both case-insensitively:
//
//  1. Status claims: each sentence of eng.Text that mentions a declared topic
//     is a temporal claim. A sentence containing the topic's CURRENT status is
//     consistent — even when it also narrates a superseded one ("previously in
//     progress, now completed"). A sentence asserting only a SUPERSEDED status
//     is a stale-fact contradiction. A sentence mentioning the topic without
//     any declared status asserts nothing checkable and is consistent.
//  2. As-of claims: every "as of YYYY-MM-DD" phrase in eng.Text claims the
//     report's currency. A date older than the newest fact date in the block
//     presents a stale date as current and is a violation; a malformed date is
//     skipped (it is not an adjudicable temporal claim).
//
// Score = consistent claims / checked claims; Pass iff Score >=
// Rubric.MinScore (default 1: no stale claim tolerated). On failure Detail
// names the FIRST violation — the stale claim and the newer dated fact that
// contradicts it — localizing the staleness per the spine contract. A case
// with no parseable facts, or a report with no checkable temporal claims,
// passes at score 1 with a Detail note.
type tmpTemporalConsistency struct{}

func (tmpTemporalConsistency) Name() string { return "temporal-consistency" }
func (tmpTemporalConsistency) Kind() string { return "rubric" }

func init() { Register(tmpTemporalConsistency{}) }

func (tmpTemporalConsistency) Judge(ref, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "temporal-consistency", Kind: "rubric", Pass: true, Score: 1}
	topics := tmpTopicTimelines(tmpParseFacts(ref.Text))
	if len(topics) == 0 {
		v.Detail = "no dated facts declared in the reference; nothing to check temporally"
		return v
	}
	newestOverall := ""
	for _, tl := range topics {
		if tl.current.date > newestOverall {
			newestOverall = tl.current.date
		}
	}

	checked, consistent := 0, 0
	firstViolation := ""
	note := func(msg string) {
		if firstViolation == "" {
			firstViolation = msg
		}
	}

	for _, claim := range reportSentences(eng.Text) {
		low := strings.ToLower(claim)
		for _, tl := range topics {
			if !strings.Contains(low, tl.key) {
				continue
			}
			checked++
			if strings.Contains(low, strings.ToLower(tl.current.status)) {
				consistent++ // current status asserted; a superseded mention is narrative
				continue
			}
			stale := tmpFact{}
			for _, s := range tl.superseded {
				if strings.Contains(low, strings.ToLower(s.status)) {
					stale = s
					break
				}
			}
			if stale.status == "" {
				consistent++ // mentions the topic but asserts no declared status
				continue
			}
			note(fmt.Sprintf("stale claim %q asserts superseded status %q (%s) for %q; newer fact (%s) says %q",
				claim, stale.status, stale.date, tl.topic, tl.current.date, tl.current.status))
		}
	}

	for _, d := range tmpAsOfDates(eng.Text) {
		checked++
		if d >= newestOverall {
			consistent++
			continue
		}
		note(fmt.Sprintf("stale date presented as current: report claims currency %q but ground truth carries a newer datapoint dated %s",
			"as of "+d, newestOverall))
	}

	if checked == 0 {
		v.Detail = "report makes no checkable temporal claims about the declared facts"
		return v
	}
	min, short := rubricScore(&v, c, consistent, checked)
	if short {
		v.Detail = fmt.Sprintf("temporal consistency %.2f < %.2f (%d/%d claims consistent); first violation: %s",
			v.Score, min, consistent, checked, firstViolation)
		return v
	}
	if firstViolation != "" {
		v.Detail = fmt.Sprintf("temporal consistency %.2f >= %.2f (%d/%d claims consistent; tolerated: %s)",
			v.Score, min, consistent, checked, firstViolation)
		return v
	}
	v.Detail = fmt.Sprintf("all %d temporal claim(s) consistent with the dated fact block", checked)
	return v
}

// tmpFact is one dated ground-truth datapoint parsed from the reference fact
// block: the ISO date it was recorded, the topic it concerns, and the status
// asserted for that topic on that date.
type tmpFact struct {
	date   string
	topic  string
	status string
}

// tmpDateLayout is the fact-block date layout. Dates validated against it
// compare chronologically as plain strings.
const tmpDateLayout = "2006-01-02"

// tmpParseFacts parses a "YYYY-MM-DD | topic | status" fact block, keeping
// declaration order and skipping malformed lines (wrong field count, empty
// topic/status, invalid date).
func tmpParseFacts(text string) []tmpFact {
	var out []tmpFact
	for _, line := range strings.Split(text, "\n") {
		parts := strings.SplitN(line, "|", 3)
		if len(parts) != 3 {
			continue
		}
		date := strings.TrimSpace(parts[0])
		topic := strings.TrimSpace(parts[1])
		status := strings.TrimSpace(parts[2])
		if topic == "" || status == "" {
			continue
		}
		if _, err := time.Parse(tmpDateLayout, date); err != nil {
			continue
		}
		out = append(out, tmpFact{date: date, topic: topic, status: status})
	}
	return out
}

// tmpTimeline is one topic's ordered ground truth: the newest fact is the
// CURRENT status, everything older is SUPERSEDED. key is the lowercased topic
// used for matching; topic keeps the author's casing for Details.
type tmpTimeline struct {
	key        string
	topic      string
	current    tmpFact
	superseded []tmpFact
}

// tmpTopicTimelines folds facts into per-topic timelines, preserving the fact
// block's topic declaration order so the first violation reported is
// deterministic. Within a topic, facts are stable-sorted by date: the last is
// current (a date tie resolves to the later-declared line).
func tmpTopicTimelines(facts []tmpFact) []tmpTimeline {
	byKey := map[string][]tmpFact{}
	var order []string
	for _, f := range facts {
		k := strings.ToLower(f.topic)
		if _, seen := byKey[k]; !seen {
			order = append(order, k)
		}
		byKey[k] = append(byKey[k], f)
	}
	out := make([]tmpTimeline, 0, len(order))
	for _, k := range order {
		fs := byKey[k]
		sort.SliceStable(fs, func(i, j int) bool { return fs[i].date < fs[j].date })
		out = append(out, tmpTimeline{
			key:        k,
			topic:      fs[len(fs)-1].topic,
			current:    fs[len(fs)-1],
			superseded: fs[:len(fs)-1],
		})
	}
	return out
}

// tmpAsOfRE is the documented currency-claim grammar: the phrase "as of"
// followed by an ISO date, matched case-insensitively.
var tmpAsOfRE = regexp.MustCompile(`(?i)\bas of\s+(\d{4}-\d{2}-\d{2})`)

// tmpAsOfDates extracts the validly-dated "as of" claims from report text in
// order of appearance; a matched but invalid date (e.g. month 13) is skipped.
func tmpAsOfDates(text string) []string {
	var out []string
	for _, m := range tmpAsOfRE.FindAllStringSubmatch(text, -1) {
		if _, err := time.Parse(tmpDateLayout, m[1]); err != nil {
			continue
		}
		out = append(out, m[1])
	}
	return out
}
