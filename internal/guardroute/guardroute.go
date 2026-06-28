// Package guardroute is the bridge that closes the guard RSI loop: it turns a
// guarded session's worst journal bucket (found by internal/guardrsi) into a
// routed, idempotent, escalating finding -- a pickable findings-queue row, and,
// for a real honesty-hole, a deduped GitHub issue.
//
// THE GAP THIS CLOSES
// -------------------
// internal/guardrsi folds a session's hash-chained decision journal and finds
// the dominant honesty-hole (guardrsi.WorstBucket). Today that result is
// rendered to a human and dropped -- nothing picks it up, so the RSI loop never
// CLOSES on our own usage. Two idempotent sinks already exist but the review
// half never called them: tools/findings_route.py (the queue) and
// internal/dogfoodissues (the deduped gh issue). This package is the missing
// DECISION layer between them -- the only genuinely new logic. It produces a
// RouteDecision; the caller materializes it through those existing sinks.
//
// The decision is a PURE function of the fold (no wall-clock, no RNG): the same
// journal yields the same decision, the same discipline guardrsi and
// findings_route already hold. That keeps the closure rung non-forgeable and
// replayable.
package guardroute

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
	"github.com/anthony-chaudhary/fak/internal/guardrsi"
)

// Schema is the stable schema tag stamped on a machine-readable route verdict.
const Schema = "fak.guard-route.v1"

// DefaultReasonThreshold is the minimum count of a single denial reason before
// that reason bucket is route-worthy. Below it, one or two denials of the same
// reason are advisory noise, not a finding worth a queue row -- only a denial
// reason that RECURS clears the bar. Honesty-holes (blank reason on a DENY, an
// out-of-vocabulary verdict) route regardless of count: even one is a defect the
// loop exists to close.
const DefaultReasonThreshold = 3

// Severity tokens, aligned with the findings_route.py ladder (P3<P2<P1<P0).
const (
	SevP2 = "P2" // an advisory recurring denial reason
	SevP1 = "P1" // a real honesty-hole (unexplained block / unknown verdict)
)

// RouteDecision is the pure verdict over a session's fold + worst bucket: should
// it route, at what severity, with what stable identity, and does it warrant a
// tracked issue (not just a queue row).
type RouteDecision struct {
	Schema string `json:"schema"`
	// Route is the one-bit gate: true when the worst bucket is a real finding.
	Route bool `json:"route"`
	// Severity is the findings_route ladder rung (P2/P1); empty when !Route.
	Severity string `json:"severity,omitempty"`
	// CauseKey is the CONTENT-STABLE cause identity (no run-id / timestamp), so
	// concurrent sessions damping-fold onto one queue row and a recurrence after
	// a "fixed" close escalates -- exactly what findings_route.py keys on.
	CauseKey string `json:"cause_key,omitempty"`
	// Pattern is the short recurring-pattern label for the queue row.
	Pattern string `json:"pattern,omitempty"`
	// Item is the one-line pickable work item.
	Item string `json:"item,omitempty"`
	// Reason is the human-readable why (also the self-diagnosing why-NOT when
	// Route is false).
	Reason string `json:"reason"`
	// FileIssue is true when this finding warrants a deduped GitHub issue (an
	// honesty-hole), not merely a queue row.
	FileIssue bool `json:"file_issue"`
}

// Decide is the pure routing decision over a guardrsi fold + its worst bucket.
// threshold defaults to DefaultReasonThreshold when <= 0. It NEVER fabricates a
// finding: an empty journal or a clean fold returns Route=false with a
// self-diagnosing reason, mirroring guardrsi's empty-journal honesty.
func Decide(fold guardrsi.Fold, bucket guardrsi.Bucket, threshold int) RouteDecision {
	if threshold <= 0 {
		threshold = DefaultReasonThreshold
	}
	d := RouteDecision{Schema: Schema}

	if fold.TotalRows == 0 {
		d.Reason = "empty journal -- no adjudicated row to review; nothing to route"
		return d
	}

	switch {
	case bucket.Bucket == "blank_reason_on_deny":
		d.Route = true
		d.Severity = SevP1
		d.FileIssue = true
		d.CauseKey = "guard-journal:blank_reason_on_deny"
		d.Pattern = "guard-unexplained-block"
		d.Item = fmt.Sprintf("%d DENY/QUARANTINE row(s) carry no reason -- require a closed-vocabulary reason on every block so no unexplained block reaches the journal", bucket.Count)
		d.Reason = "honesty-hole: an unexplained block reached the journal (" + bucket.Lever + ")"

	case bucket.Bucket == "unknown_verdict":
		d.Route = true
		d.Severity = SevP1
		d.FileIssue = true
		d.CauseKey = "guard-journal:unknown_verdict"
		d.Pattern = "guard-unknown-verdict"
		d.Item = fmt.Sprintf("%d row(s) carry a verdict outside the closed set -- constrain verdicts to the closed vocabulary; an UNCLASSIFIED verdict is a bug to declare, not journal", bucket.Count)
		d.Reason = "honesty-hole: an out-of-vocabulary verdict reached the journal (" + bucket.Lever + ")"

	case strings.HasPrefix(bucket.Bucket, "reason:"):
		reason := strings.TrimPrefix(bucket.Bucket, "reason:")
		if bucket.Count >= threshold {
			d.Route = true
			d.Severity = SevP2
			d.FileIssue = false // a recurring denial reason is advisory: queue row, no issue
			d.CauseKey = "guard-journal:reason:" + reason
			d.Pattern = "guard-recurring-denial"
			d.Item = fmt.Sprintf("denial reason %q recurred %dx (>= threshold %d) -- a floor refinement could pre-empt it", reason, bucket.Count, threshold)
			d.Reason = fmt.Sprintf("recurring denial: %q hit %dx (>= %d); advisory queue row, no honesty hole", reason, bucket.Count, threshold)
		} else {
			d.Reason = fmt.Sprintf("largest denial reason %q hit only %dx (< threshold %d) -- advisory noise, not route-worthy", reason, bucket.Count, threshold)
		}

	default:
		d.Reason = "no honesty hole and no recurring denial -- nothing to retire this session (" + bucket.Lever + ")"
	}
	return d
}

// ToActionItem maps a route decision onto the EXISTING dogfoodissues.ActionItem,
// so the GitHub-issue half is pure reuse (IssueBody/BuildPlan/Sync) -- no new gh
// code. The stable Key is derived from the content-stable CauseKey, so a re-run
// updates the same issue in place instead of opening a duplicate.
func ToActionItem(d RouteDecision, evidencePath string) dogfoodissues.ActionItem {
	return dogfoodissues.ActionItem{
		Key:          "guard-rsi-route/" + d.CauseKey,
		Title:        "guard RSI ACTION: " + d.Pattern,
		SourceProbe:  "guard-verdict-rsi",
		ScoreName:    "severity",
		Score:        d.Severity,
		Grade:        d.Severity,
		DebtName:     "guard_honesty_hole",
		DebtCount:    1,
		EvidencePath: evidencePath,
		NextAction:   d.Item,
		Finding:      d.CauseKey,
	}
}

// RouteArgv builds the tools/findings_route.py route_finding argv for a routed
// decision. source is the originating session/run id, recorded on the queue row.
// The --key is the per-stop idempotency key (source-scoped so two distinct
// sessions each get one fold onto the shared cause), while --cause-key is the
// content-stable cause identity the queue damps + escalates on.
func RouteArgv(d RouteDecision, source string) []string {
	if source == "" {
		source = "guard-verdict-rsi"
	}
	return []string{
		"route_finding",
		"--key", source + ":" + d.CauseKey,
		"--sev", d.Severity,
		"--pattern", d.Pattern,
		"--item", d.Item,
		"--source", source,
		"--owning-plan", "GUARD-RSI",
		"--cause-key", d.CauseKey,
	}
}

// Envelope is the control-pane envelope the route subcommand emits, so the
// garden bundle folds it with zero new fold code (gardenbundle.Interpret already
// handles the standard ok/verdict/finding/reason/next_action shape).
type Envelope struct {
	Schema     string         `json:"schema"`
	OK         bool           `json:"ok"`
	Verdict    string         `json:"verdict"`
	Finding    string         `json:"finding"`
	Reason     string         `json:"reason"`
	NextAction string         `json:"next_action"`
	Decision   RouteDecision  `json:"decision"`
	Routed     map[string]any `json:"routed,omitempty"`
	Issue      map[string]any `json:"issue,omitempty"`
}

// Fold builds the control-pane envelope for a decision plus the (optional) sink
// outcomes. It is OK=true whenever the review ran -- a routed finding is the
// pass WORKING, not a broken pass (the same advisory rationale the garden's
// loop_audit member uses); only an internal failure makes it not-ok.
func Fold(d RouteDecision, routed, issue map[string]any) Envelope {
	e := Envelope{
		Schema:   Schema,
		OK:       true,
		Decision: d,
		Routed:   routed,
		Issue:    issue,
	}
	switch {
	case !d.Route:
		e.Verdict, e.Finding = "OK", "guard_route_clear"
		e.Reason = "guard session reviewed; " + d.Reason
		e.NextAction = "hold the line; the next guarded session re-reviews the journal"
	default:
		e.Verdict, e.Finding = "ACTION", "guard_route_routed"
		e.Reason = fmt.Sprintf("guard session reviewed; routed a %s finding -- %s", d.Severity, d.Reason)
		if d.FileIssue {
			e.NextAction = "pick the routed queue row / tracked issue and close the honesty hole worst-first"
		} else {
			e.NextAction = "pick the routed queue row (advisory) and consider a floor refinement"
		}
	}
	return e
}
