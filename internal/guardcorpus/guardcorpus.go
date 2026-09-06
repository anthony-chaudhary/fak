// Package guardcorpus folds a guarded session's decision-journal rows into the
// durable, policy-attributable GUARD-SESSION dataset: one SessionRecord per
// session plus the replayable, redacted Example rows for training and regression.
package guardcorpus

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// Schema identifiers for dataset records.
const (
	// SessionSchema is the schema identifier for version 1 session records.
	SessionSchema = "fak-guard-session/1"
	// ExampleSchema is the schema identifier for version 1 example records.
	ExampleSchema = "fak-guard-example/1"

	// maxAllowExamples bounds the sample count of ALLOW decisions emitted per session.
	maxAllowExamples = 8
)

// knownVerdicts defines the closed decision verdict vocabulary recognized as legitimate,
// including non-blocking ADVISORY verdicts emitted by monitor passes.
var knownVerdicts = map[string]bool{
	"ALLOW": true, "DENY": true, "TRANSFORM": true, "QUARANTINE": true,
	"WITNESS": true, "DEFER": true, "INDETERMINATE": true, "ADVISORY": true,
}

// Outcome classes for a session.
const (
	// OutcomeClean indicates that the guarded session completed without abnormal termination.
	OutcomeClean = "CLEAN"
	// OutcomeCrashed indicates that the child process wrapped by the guard terminated abnormally.
	OutcomeCrashed = "CRASHED"
	// OutcomeRateLimited indicates that the session terminated due to provider capacity or rate limits.
	OutcomeRateLimited = "RATE_LIMITED"
)

// SessionMeta captures out-of-band session identity and effective policy digest.
type SessionMeta struct {
	TraceID       string `json:"trace_id,omitempty"`
	Agent         string `json:"agent,omitempty"`
	HostClass     string `json:"host_class,omitempty"`
	PolicyDigest  string `json:"policy_digest,omitempty"`
	ChainVerified bool   `json:"chain_verified"`
}

// HonestyHoles records per-session integrity gaps that indicate accounting defects.
type HonestyHoles struct {
	BlankReasonOnDeny int `json:"blank_reason_on_deny"`
	UnknownVerdict    int `json:"unknown_verdict"`
	WitnesslessBlock  int `json:"witnessless_block"`
	ChildCrash        int `json:"child_crash"`
}

// SessionRecord captures folded aggregate metrics and outcome classification for a session.
type SessionRecord struct {
	Schema          string         `json:"schema"`
	TraceID         string         `json:"trace_id,omitempty"`
	Agent           string         `json:"agent,omitempty"`
	HostClass       string         `json:"host_class,omitempty"`
	PolicyDigest    string         `json:"policy_digest,omitempty"`
	StartedUnixNano int64          `json:"started_unix_nano,omitempty"`
	EndedUnixNano   int64          `json:"ended_unix_nano,omitempty"`
	ToolCalls       int            `json:"tool_calls"`
	ByVerdict       map[string]int `json:"by_verdict,omitempty"`
	ByReason        map[string]int `json:"by_reason,omitempty"`
	ByGate          map[string]int `json:"by_gate,omitempty"`
	HonestyHoles    HonestyHoles   `json:"honesty_holes"`
	Outcome         string         `json:"outcome"`
	ChainVerified   bool           `json:"chain_verified"`
}

// Example represents one redacted, labeled adjudication suitable for replay or training.
type Example struct {
	Schema       string `json:"schema"`
	TraceID      string `json:"trace_id,omitempty"`
	PolicyDigest string `json:"policy_digest,omitempty"`
	Tool         string `json:"tool,omitempty"`
	ArgsLabel    string `json:"args_label,omitempty"`
	Kind         string `json:"kind,omitempty"`
	Verdict      string `json:"verdict,omitempty"`
	Reason       string `json:"reason,omitempty"`
	By           string `json:"by,omitempty"`
	Taint        string `json:"taint,omitempty"`
	Witness      string `json:"witness,omitempty"`
}

// Fold projects journal rows into a SessionRecord summary and bounded Example records.
func Fold(meta SessionMeta, rows []journal.Row) (SessionRecord, []Example) {
	rec := SessionRecord{
		Schema:        SessionSchema,
		TraceID:       meta.TraceID,
		Agent:         meta.Agent,
		HostClass:     meta.HostClass,
		PolicyDigest:  meta.PolicyDigest,
		Outcome:       OutcomeClean,
		ChainVerified: meta.ChainVerified,
		ByVerdict:     map[string]int{},
		ByReason:      map[string]int{},
		ByGate:        map[string]int{},
	}
	var examples []Example
	allowExamples := 0

	for _, r := range rows {
		if r.TSUnixNano > 0 {
			if rec.StartedUnixNano == 0 || r.TSUnixNano < rec.StartedUnixNano {
				rec.StartedUnixNano = r.TSUnixNano
			}
			if r.TSUnixNano > rec.EndedUnixNano {
				rec.EndedUnixNano = r.TSUnixNano
			}
		}

		// Child crashes record process termination rather than a tool decision.
		if strings.EqualFold(strings.TrimSpace(r.Kind), "CHILD_CRASH") {
			if rateLimitClass(r) != "" {
				if rec.Outcome == OutcomeClean {
					rec.Outcome = OutcomeRateLimited
				}
				continue
			}
			rec.HonestyHoles.ChildCrash++
			rec.Outcome = OutcomeCrashed
			examples = append(examples, exampleFrom(meta, r, "CHILD_CRASH", ""))
			continue
		}

		verdict := normalizeVerdict(r.Verdict, r.Kind)
		if verdict == "" {
			continue
		}
		rec.ToolCalls++
		rec.ByVerdict[verdict]++
		if r.By != "" {
			rec.ByGate[r.By]++
		}
		if !knownVerdicts[verdict] {
			rec.HonestyHoles.UnknownVerdict++
		}

		reason := strings.TrimSpace(r.Reason)
		isBlock := verdict == "DENY" || verdict == "QUARANTINE"
		if isBlock {
			if reason == "" {
				rec.HonestyHoles.BlankReasonOnDeny++
			} else {
				rec.ByReason[reason]++
				if strings.TrimSpace(r.Witness) == "" {
					rec.HonestyHoles.WitnesslessBlock++
				}
			}
		}

		switch {
		case isBlock || verdict == "WITNESS":
			examples = append(examples, exampleFrom(meta, r, r.Kind, verdict))
		case verdict == "ALLOW" && allowExamples < maxAllowExamples:
			examples = append(examples, exampleFrom(meta, r, r.Kind, verdict))
			allowExamples++
		}
	}

	if len(rec.ByVerdict) == 0 {
		rec.ByVerdict = nil
	}
	if len(rec.ByReason) == 0 {
		rec.ByReason = nil
	}
	if len(rec.ByGate) == 0 {
		rec.ByGate = nil
	}
	return rec, examples
}

func exampleFrom(meta SessionMeta, r journal.Row, kind, verdict string) Example {
	return Example{
		Schema:       ExampleSchema,
		TraceID:      meta.TraceID,
		PolicyDigest: meta.PolicyDigest,
		Tool:         r.Tool,
		ArgsLabel:    r.ArgsLabel,
		Kind:         strings.TrimSpace(kind),
		Verdict:      verdict,
		Reason:       strings.TrimSpace(r.Reason),
		By:           r.By,
		Taint:        r.Taint,
		Witness:      strings.TrimSpace(r.Witness),
	}
}

// normalizeVerdict canonicalizes the verdict, falling back to Kind if verdict is blank.
func normalizeVerdict(verdict, kind string) string {
	if v := strings.TrimSpace(verdict); v != "" {
		return strings.ToUpper(v)
	}
	switch strings.ToUpper(strings.TrimSpace(kind)) {
	case "DENY":
		return "DENY"
	case "RESULT_DENY":
		return "DENY"
	case "QUARANTINE":
		return "QUARANTINE"
	}
	return ""
}

// rateLimitClass identifies provider rate-limit indicators in a CHILD_CRASH event.
func rateLimitClass(r journal.Row) string {
	for _, raw := range []string{r.Reason, r.Witness, r.ArgsLabel} {
		low := strings.ToLower(strings.TrimSpace(raw))
		switch {
		case low == "":
			continue
		case strings.Contains(low, "session_limit"):
			return "session_limit"
		case strings.Contains(low, "weekly_limit"):
			return "weekly_limit"
		case strings.Contains(low, "usage_limit"):
			return "usage_limit"
		case strings.Contains(low, "rate_limited"), strings.Contains(low, "rate limit"),
			strings.Contains(low, "ratelimit"), strings.Contains(low, "rate_limit_exit"):
			return "rate_limited"
		}
	}
	return ""
}
