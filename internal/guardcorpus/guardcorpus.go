// Package guardcorpus folds a guarded session's decision-journal rows into the
// durable, policy-attributable GUARD-SESSION dataset: one SessionRecord per
// session plus the replayable, redacted Example rows a training/regression
// consumer needs (docs/GUARD-SESSION-DATASET-PLAN.md).
//
// WHY IT EXISTS. The raw journal (internal/journal) is host-local, gitignored,
// and reaped (internal/guardaudit); guardrsi scores it and auditusage rolls it
// up, but nothing turns it into a committed, schema-versioned, per-session
// corpus that survives the reaper and stays attributable to the policy that
// produced each verdict. This package is that fold.
//
// PURE BY CONTRACT. Fold is a pure function: []journal.Row (+ the session
// identity the CLI shell already resolved from the spawn sidecar) in, records
// out. No disk read, no wall-clock, no RNG — same rows in, same records out, so
// the fold is hermetically testable and the dataset is reproducible.
//
// REDACTION IS INHERITED, NEVER RE-DERIVED. Every field an Example carries
// (ArgsLabel, Witness) is one the journal producer already bounded and scrubbed,
// by a rule chosen per field:
//
//   - ArgsLabel — internal/journal ArgsLabelForBytes: reduced to a shape label
//     (command stem / path stem / key names), bounded to 96 bytes, and dropped
//     WHOLE when secretish() fires.
//   - Witness — internal/journal boundWitness: bounded to 512 bytes with the
//     value of any secretish `key=value` assignment redacted. Deliberately NOT
//     the ArgsLabel rule. The 96-byte bound would truncate ~9% of witness rows
//     and a quarter of all witness prose bytes, and the long values are the
//     gate's own remedy text — the thing that makes a refusal recoverable. The
//     whole-string secretish() drop would blank the witness on exactly the
//     SECRET_EXFIL refusals an operator most needs to read.
//
// This package copies those already-safe fields verbatim and never reaches for a
// rawer source, so the dataset cannot be a looser disclosure surface than the
// journal it folds.
//
// WHAT THE WITNESS BOUND DOES NOT PROMISE. The claim is not rung-authored by
// construction — several live rungs concatenate call-derived bytes into it (an
// egress host parsed from the call args; under the opt-in LintWrites rung, the
// target path plus a parser message that embeds the offending source token). The
// journal cannot tell rung prose from call bytes, so it bounds and scrubs rather
// than admitting a closed vocabulary the way DenyRule does. A rung that wants a
// hard guarantee should stamp a DenyRule, not lengthen its claim.
package guardcorpus

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// Invariant: guard corpus entries maintain deterministic immutable hash signatures and valid category taxonomy.

// Schema identifiers for the two dataset record kinds. Bump on any shape change.
const (
	// SessionSchema is the schema identifier for the version 1 session record dataset.
	SessionSchema = "fak-guard-session/1"
	// ExampleSchema is the schema identifier for the version 1 guard example record dataset.
	ExampleSchema = "fak-guard-example/1"

	// maxAllowExamples bounds how many ALLOW rows a single session contributes as
	// examples. Blocks/quarantines/witness/crashes are always kept (they are the
	// scarce, interesting class); allows are sampled so the committed corpus stays
	// small and class-balanced. Deterministic: the FIRST N allows in row order.
	maxAllowExamples = 8
)

// knownVerdicts is the kernel's verdict vocabulary this fold recognizes as
// legitimate (i.e. NOT an unknown-verdict honesty hole). It is the seven closed
// decision verdicts PLUS "ADVISORY".
//
// ADVISORY is a real, non-blocking verdict emitted by guard monitors — e.g. the
// tool-definition-pruner (kind TOOL_DEFINITION_PRUNED, by "tool-definition-pruner")
// and the SHELL_DIALECT monitor. Folding this host's real journals showed it is
// the DOMINANT verdict (~39 rows/session across 510 sessions).
//
// guardrsi.KnownVerdicts used to OMIT it, so the guard-RSI verdict-quality metric
// miscounted every advisory as an unknown-verdict honesty hole and pinned its worst
// bucket on "unknown_verdict" — a lever aimed at a defect that did not exist. That
// fold now carries ADVISORY too (TestAdvisoryIsNotAnUnknownVerdict pins it), so the
// two vocabularies agree; the note is kept because the divergence is what the
// A-advisory fan-out item in docs/GUARD-SESSION-DATASET-PLAN.md was filed about.
var knownVerdicts = map[string]bool{
	"ALLOW": true, "DENY": true, "TRANSFORM": true, "QUARANTINE": true,
	"WITNESS": true, "DEFER": true, "INDETERMINATE": true, "ADVISORY": true,
}

// Outcome classes for a session.
const (
	// OutcomeClean indicates that the guarded session completed without crashes or rate limits.
	OutcomeClean = "CLEAN"
	// OutcomeCrashed indicates that the child process wrapped by the guard terminated abnormally.
	OutcomeCrashed = "CRASHED"
	// OutcomeRateLimited indicates that the session terminated due to provider capacity or rate limits.
	OutcomeRateLimited = "RATE_LIMITED"
)

// SessionMeta is the out-of-band session identity the CLI shell resolves from
// the spawn sidecar (.dispatch-runs/*.startup.json) and hands to the fold. The
// PolicyDigest is the Gap-2 data element: the effective policy in force for the
// session (base floor + allow/deny overlays), computed by
// guardEffectivePolicyDigest at guard startup. Carrying it here makes every
// folded verdict attributable to the policy that produced it even after the raw
// journals are aggregated or reaped away from their sidecars.
type SessionMeta struct {
	TraceID       string `json:"trace_id,omitempty"`
	Agent         string `json:"agent,omitempty"`
	HostClass     string `json:"host_class,omitempty"`
	PolicyDigest  string `json:"policy_digest,omitempty"`
	ChainVerified bool   `json:"chain_verified"`
}

// HonestyHoles mirrors guardrsi's honesty accounting: the per-session gaps a
// good guard journal should have zero of.
type HonestyHoles struct {
	BlankReasonOnDeny int `json:"blank_reason_on_deny"`
	UnknownVerdict    int `json:"unknown_verdict"`
	WitnesslessBlock  int `json:"witnessless_block"`
	ChildCrash        int `json:"child_crash"`
}

// SessionRecord represents one row of the fak-guard-session/1 dataset capturing aggregate session metrics.
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

// Example represents one row of the fak-guard-example/1 dataset: a redacted, labeled,
// replayable adjudication. Only journal-redacted fields are carried.
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

// Fold projects one session's rows into its SessionRecord and Example rows. It
// is the pure core; the CLI shell owns discovery, chain verification, and the
// SessionMeta join. Rows are taken in journal order (the caller preserves it).
//
// Contract: Fold is pure, deterministic, and preserves input journal row ordering.
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

		// A CHILD_CRASH is not a kernel decision — the wrapped child died. It never
		// enters the verdict accounting; it decides the session outcome and (unless
		// it is a provider rate-limit exit) counts as the worst honesty hole. This
		// mirrors guardrsi.FoldRows so the two folds agree on what a crash is.
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
			continue // an operational row with no verdict (not a decision)
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

		// Emit an example for every interesting adjudication (block / witness) and
		// a bounded, deterministic sample of allows.
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

// Invariant: Example redaction bounds are inherited directly from journal-scrubbed fields without secondary expansion.
// Postcondition: Returns an immutable Example record populated with sanitized arguments, taints, and witness prose.
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

// normalizeVerdict resolves the row's verdict, falling back to the decision Kind
// when the explicit verdict field is blank (a DENY row's Kind is "DENY"). It
// mirrors guardrsi's resolution so the two folds classify identically.
//
// Precondition: Candidate verdict and kind tokens represent raw classification strings extracted from journal rows.
// Postcondition: Returns an uppercase canonical verdict or empty string if the row represents an operational event.
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

// rateLimitClass names the provider-capacity class of a CHILD_CRASH row whose
// bounded evidence points at rate limiting, else "". A rate-limit exit is a
// terminal outcome but not process instability, so it must not count as a crash.
// Mirrors guardrsi.childRateLimitExitClass over the same bounded fields.
//
// Postcondition: Returns a normalized rate limit category string when matching evidence is present, or empty string otherwise.
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
