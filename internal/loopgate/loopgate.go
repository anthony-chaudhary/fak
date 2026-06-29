// Package loopgate adjudicates a loop turn's self-reported "done" claim
// against an external witness. The package owns no process spawning: callers
// provide the witness function, so unit tests use fixtures and production hosts
// can bind the request to dos commit-audit, dos verify, or another witness verb.
package loopgate

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// Verdict is the exit-gate decision a loop driver consumes.
type Verdict string

const (
	VerdictWitnessed Verdict = "WITNESSED"
	VerdictNotYet    Verdict = "NOT_YET"
	VerdictRefused   Verdict = "REFUSED"
)

const (
	// ReasonDoneUnwitnessed is the re-arm reason for a done claim that no
	// external witness corroborated. It is declared in dos.toml so loop drivers
	// can surface a closed reason token rather than prose.
	ReasonDoneUnwitnessed  = "LOOP_DONE_UNWITNESSED"
	ReasonSchemaUnreadable = "SCHEMA_UNREADABLE"
)

// CriterionKind names the witness surface that should adjudicate the turn.
type CriterionKind string

const (
	CriterionCommitAudit     CriterionKind = "commit-audit"
	CriterionVerify          CriterionKind = "verify"
	CriterionTestWitness     CriterionKind = "test-witness"
	CriterionCitationResolve CriterionKind = "citation-resolve"
	CriterionWitness         CriterionKind = "witness"
	CriterionMetric          CriterionKind = "metric"
)

// Criterion is the goal's witness requirement. Empty Kind defaults to
// commit-audit over the turn ref.
type Criterion struct {
	Kind      CriterionKind
	Ref       string
	Plan      string
	Phase     string
	Source    string
	Subject   string
	Baseline  string
	Candidate string
}

// Turn is the loop turn state visible to the gate.
type Turn struct {
	ClaimedDone bool
	Claim       string
	HeadRef     string
	Criterion   Criterion
}

// Request is the normalized witness call the loop host must satisfy.
type Request struct {
	Kind      CriterionKind `json:"kind"`
	Ref       string        `json:"ref,omitempty"`
	Plan      string        `json:"plan,omitempty"`
	Phase     string        `json:"phase,omitempty"`
	Source    string        `json:"source,omitempty"`
	Subject   string        `json:"subject,omitempty"`
	Baseline  string        `json:"baseline,omitempty"`
	Candidate string        `json:"candidate,omitempty"`
	Claim     string        `json:"claim,omitempty"`
}

// Argv returns the dos CLI argv corresponding to this request, excluding the
// leading "dos" binary. It is a convenience for hosts that bind the gate to the
// CLI; Adjudicate itself never shells out.
func (r Request) Argv() []string {
	switch r.Kind {
	case CriterionVerify:
		return []string{"verify", "--json", r.Plan, r.Phase}
	case CriterionTestWitness:
		return []string{"test-witness", "--json", "--baseline", r.Baseline, "--candidate", r.Candidate}
	case CriterionCitationResolve:
		return []string{"witness", "--json", "citation_resolve", r.Subject}
	case CriterionWitness:
		return []string{"witness", "--json", r.Source, r.Subject}
	default:
		ref := strings.TrimSpace(r.Ref)
		if ref == "" {
			ref = "HEAD"
		}
		return []string{"commit-audit", "--json", ref}
	}
}

// WitnessOutcome is the normalized outcome returned by the witness adapter.
type WitnessOutcome string

const (
	OutcomeWitnessed WitnessOutcome = "witnessed"
	OutcomeNotYet    WitnessOutcome = "not_yet"
	OutcomeRefused   WitnessOutcome = "refused"
)

// WitnessResult is the adapter's evidence summary.
type WitnessResult struct {
	Outcome    WitnessOutcome `json:"outcome"`
	Reason     string         `json:"reason,omitempty"`
	Detail     string         `json:"detail,omitempty"`
	RawVerdict string         `json:"raw_verdict,omitempty"`
	Rung       string         `json:"rung,omitempty"`
}

// WitnessFunc satisfies one normalized witness request.
type WitnessFunc func(context.Context, Request) (WitnessResult, error)

// Decision is the gate's full typed result.
type Decision struct {
	Verdict Verdict `json:"verdict"`
	Reason  string  `json:"reason,omitempty"`
	Summary string  `json:"summary,omitempty"`
	Request Request `json:"request,omitempty"`
	Witness string  `json:"witness,omitempty"`
}

// Adjudicate maps a turn's done claim and witness criterion to the loop exit
// verdict. A done claim is accepted only on OutcomeWitnessed. Unwitnessed claims
// re-arm with ReasonDoneUnwitnessed. Malformed criteria or witness adapter
// failures terminate as structured refusals.
func Adjudicate(ctx context.Context, turn Turn, witness WitnessFunc) Decision {
	if !turn.ClaimedDone {
		return Decision{
			Verdict: VerdictNotYet,
			Reason:  ReasonDoneUnwitnessed,
			Summary: "turn did not claim done; continue",
		}
	}
	req, err := turn.request()
	if err != nil {
		return Decision{Verdict: VerdictRefused, Reason: ReasonSchemaUnreadable, Summary: err.Error()}
	}
	if req.Kind == CriterionMetric {
		return Decision{
			Verdict: VerdictNotYet,
			Reason:  ReasonDoneUnwitnessed,
			Summary: "raw metric is not an external witness criterion",
			Request: req,
		}
	}
	if witness == nil {
		return Decision{
			Verdict: VerdictRefused,
			Reason:  ReasonSchemaUnreadable,
			Summary: "loopgate witness adapter is nil",
			Request: req,
		}
	}
	res, err := witness(ctx, req)
	if err != nil {
		return Decision{
			Verdict: VerdictRefused,
			Reason:  ReasonSchemaUnreadable,
			Summary: "witness adapter failed: " + err.Error(),
			Request: req,
		}
	}
	switch res.Outcome {
	case OutcomeWitnessed:
		return Decision{
			Verdict: VerdictWitnessed,
			Reason:  firstNonEmpty(res.Reason, "WITNESSED"),
			Summary: firstNonEmpty(res.Detail, "done claim witnessed"),
			Request: req,
			Witness: res.Rung,
		}
	case OutcomeNotYet:
		return Decision{
			Verdict: VerdictNotYet,
			Reason:  ReasonDoneUnwitnessed,
			Summary: firstNonEmpty(res.Detail, res.Reason, "done claim was not witnessed"),
			Request: req,
			Witness: res.Rung,
		}
	case OutcomeRefused:
		return Decision{
			Verdict: VerdictRefused,
			Reason:  firstNonEmpty(res.Reason, ReasonSchemaUnreadable),
			Summary: firstNonEmpty(res.Detail, "witness refused the claim"),
			Request: req,
			Witness: res.Rung,
		}
	default:
		return Decision{
			Verdict: VerdictRefused,
			Reason:  ReasonSchemaUnreadable,
			Summary: fmt.Sprintf("unknown witness outcome %q", res.Outcome),
			Request: req,
			Witness: res.Rung,
		}
	}
}

func (t Turn) request() (Request, error) {
	c := t.Criterion
	kind := c.Kind
	if kind == "" {
		kind = CriterionCommitAudit
	}
	req := Request{
		Kind:      kind,
		Ref:       firstNonEmpty(c.Ref, t.HeadRef, "HEAD"),
		Plan:      strings.TrimSpace(c.Plan),
		Phase:     strings.TrimSpace(c.Phase),
		Source:    strings.TrimSpace(c.Source),
		Subject:   strings.TrimSpace(c.Subject),
		Baseline:  strings.TrimSpace(c.Baseline),
		Candidate: strings.TrimSpace(c.Candidate),
		Claim:     strings.TrimSpace(t.Claim),
	}
	switch kind {
	case CriterionCommitAudit:
		return req, nil
	case CriterionVerify:
		if req.Plan == "" || req.Phase == "" {
			return Request{}, errors.New("verify criterion requires plan and phase")
		}
		return req, nil
	case CriterionTestWitness:
		if req.Baseline == "" || req.Candidate == "" {
			return Request{}, errors.New("test-witness criterion requires baseline and candidate outcomes")
		}
		return req, nil
	case CriterionCitationResolve:
		if req.Subject == "" {
			return Request{}, errors.New("citation-resolve criterion requires a subject citation")
		}
		return req, nil
	case CriterionWitness:
		if req.Source == "" || req.Subject == "" {
			return Request{}, errors.New("witness criterion requires source and subject")
		}
		return req, nil
	case CriterionMetric:
		return req, nil
	default:
		return Request{}, fmt.Errorf("unsupported witness criterion %q", kind)
	}
}

func firstNonEmpty(xs ...string) string {
	for _, x := range xs {
		if s := strings.TrimSpace(x); s != "" {
			return s
		}
	}
	return ""
}
