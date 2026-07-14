package operatorresolve

import (
	"context"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
	"github.com/anthony-chaudhary/fak/internal/operatorquestion"
)

// Reason is the closed resolver vocabulary consumed by the pre-adjudication rung.
type Reason string

const (
	ReasonWitnessedOption    Reason = "WITNESSED_OPTION"
	ReasonNeedsInvestigation Reason = "NEEDS_INVESTIGATION"
	ReasonAuthorityFork      Reason = "AUTHORITY_FORK"
	ReasonEvidenceTie        Reason = "EVIDENCE_TIE"
)

// Evidence is one oracle-authored fact attached to an option.
type Evidence struct {
	Oracle  string `json:"oracle"`
	Claim   string `json:"claim"`
	Witness string `json:"witness,omitempty"`
	Score   int    `json:"score"`
}

// OptionEvidence binds all read-only evidence to one normalized option.
type OptionEvidence struct {
	Option   operatorquestion.Option `json:"option"`
	Evidence []Evidence              `json:"evidence"`
	Score    int                     `json:"score"`
}

// Verdict is the evidence-first answer or the irreducible structured escalation.
type Verdict struct {
	Disposition choicetriage.Disposition `json:"disposition"`
	Reason      Reason                   `json:"reason"`
	Action      string                   `json:"action,omitempty"`
	Options     []OptionEvidence         `json:"options,omitempty"`
}

// Oracle is read-only by contract. Implementations inspect evidence but never mutate it.
type Oracle interface {
	Name() string
	Inspect(context.Context, operatorquestion.OperatorQuestion, operatorquestion.Option) (Evidence, bool, error)
}

// Resolver folds a normalized clarify/approach question through independent evidence.
type Resolver struct {
	Oracles []Oracle
}

// ReadOnlyRunner is the process seam for repository evidence. Implementations receive
// argv directly (never a shell string), which keeps command auditing exact.
type ReadOnlyRunner interface {
	Run(context.Context, string, ...string) ([]byte, error)
}

// GitIsolationOracle resolves the recurring shared-tree commit-isolation clarify from
// git state/history. Its command vocabulary is fixed and mutation-free.
type GitIsolationOracle struct {
	Runner ReadOnlyRunner
}

func (GitIsolationOracle) Name() string { return "git-isolation-readonly" }

func (o GitIsolationOracle) Inspect(ctx context.Context, q operatorquestion.OperatorQuestion, option operatorquestion.Option) (Evidence, bool, error) {
	if o.Runner == nil || !containsFold(q.Question+" "+q.Detail, "commit") {
		return Evidence{}, false, nil
	}
	status, err := o.Runner.Run(ctx, "git", "status", "--short")
	if err != nil {
		return Evidence{}, false, err
	}
	history, err := o.Runner.Run(ctx, "git", "log", "-1", "--oneline", "--", ".")
	if err != nil {
		return Evidence{}, false, err
	}
	label := option.Label + " " + option.Rationale
	witness := "git status --short: " + strings.TrimSpace(string(status)) + "; git log -1 --oneline -- .: " + strings.TrimSpace(string(history))
	if len(strings.TrimSpace(string(status))) > 0 && containsAnyFold(label, "explicit path", "owned path", "path-scoped") {
		return Evidence{Claim: "peer-dirty state makes explicit owned paths the isolation-preserving choice", Witness: witness, Score: 10}, true, nil
	}
	return Evidence{Claim: "option has no positive path-isolation witness", Witness: witness, Score: 0}, true, nil
}

func (r Resolver) Resolve(ctx context.Context, q operatorquestion.OperatorQuestion) (Verdict, error) {
	if q.Kind != operatorquestion.Clarify && q.Kind != operatorquestion.ChooseApproach {
		return Verdict{}, fmt.Errorf("operatorresolve: unsupported question kind %q", q.Kind)
	}
	if authorityFork(q) {
		return Verdict{
			Disposition: choicetriage.HumanResidual,
			Reason:      ReasonAuthorityFork,
			Action:      "ask the operator to make the named policy, priority, release, credential, or approval decision",
			Options:     blankOptionEvidence(q.Options),
		}, nil
	}
	if len(q.Options) == 0 {
		return Verdict{Disposition: choicetriage.FreshContext, Reason: ReasonNeedsInvestigation, Action: "investigate the missing evidence in a fresh context"}, nil
	}

	out := make([]OptionEvidence, 0, len(q.Options))
	best, bestIndex, ties := 0, -1, 0
	for i, option := range q.Options {
		item := OptionEvidence{Option: option}
		for _, oracle := range r.Oracles {
			if oracle == nil {
				continue
			}
			evidence, ok, err := oracle.Inspect(ctx, q, option)
			if err != nil {
				return Verdict{}, fmt.Errorf("operatorresolve: oracle %s: %w", oracle.Name(), err)
			}
			if !ok {
				continue
			}
			if strings.TrimSpace(evidence.Oracle) == "" {
				evidence.Oracle = oracle.Name()
			}
			item.Evidence = append(item.Evidence, evidence)
			item.Score += evidence.Score
		}
		out = append(out, item)
		if item.Score > best {
			best, bestIndex, ties = item.Score, i, 1
		} else if item.Score == best && item.Score > 0 {
			ties++
		}
	}
	if bestIndex >= 0 && best > 0 && ties == 1 {
		return Verdict{Disposition: choicetriage.TakeObvious, Reason: ReasonWitnessedOption, Action: q.Options[bestIndex].Label, Options: out}, nil
	}
	reason := ReasonNeedsInvestigation
	if best > 0 && ties > 1 {
		reason = ReasonEvidenceTie
	}
	return Verdict{Disposition: choicetriage.FreshContext, Reason: reason, Action: "investigate and break the evidence tie in a fresh context", Options: out}, nil
}

// OracleFunc makes deterministic read-only oracle injection concise in tests and callers.
type OracleFunc struct {
	OracleName string
	InspectFn  func(context.Context, operatorquestion.OperatorQuestion, operatorquestion.Option) (Evidence, bool, error)
}

func (f OracleFunc) Name() string { return f.OracleName }
func (f OracleFunc) Inspect(ctx context.Context, q operatorquestion.OperatorQuestion, o operatorquestion.Option) (Evidence, bool, error) {
	return f.InspectFn(ctx, q, o)
}

func blankOptionEvidence(options []operatorquestion.Option) []OptionEvidence {
	out := make([]OptionEvidence, 0, len(options))
	for _, option := range options {
		out = append(out, OptionEvidence{Option: option})
	}
	return out
}

func authorityFork(q operatorquestion.OperatorQuestion) bool {
	hay := strings.ToUpper(q.Question + " " + q.Detail)
	for _, option := range q.Options {
		hay += " " + strings.ToUpper(option.Label+" "+option.Rationale)
	}
	for _, token := range []string{"POLICY", "PRIORITY", "RELEASE", "PUBLISH", "CREDENTIAL", "APPROVAL", "AUTHORITY"} {
		if strings.Contains(hay, token) {
			return true
		}
	}
	return false
}

func containsFold(haystack, needle string) bool {
	return strings.Contains(strings.ToUpper(haystack), strings.ToUpper(needle))
}

func containsAnyFold(haystack string, needles ...string) bool {
	for _, needle := range needles {
		if containsFold(haystack, needle) {
			return true
		}
	}
	return false
}
