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
	// ReasonWitnessedOption indicates an option is backed by positive evidence.
	ReasonWitnessedOption Reason = "WITNESSED_OPTION"
	// ReasonNeedsInvestigation indicates evidence is absent or insufficient to decide.
	ReasonNeedsInvestigation Reason = "NEEDS_INVESTIGATION"
	// ReasonAuthorityFork indicates a policy or authority decision requiring human escalation.
	ReasonAuthorityFork Reason = "AUTHORITY_FORK"
	// ReasonEvidenceTie indicates competing options have identical positive evidence scores.
	ReasonEvidenceTie Reason = "EVIDENCE_TIE"
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

// Name returns the identifier for GitIsolationOracle.
func (GitIsolationOracle) Name() string { return "git-isolation-readonly" }

// Inspect checks git status and history to evaluate commit isolation.
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

// TrackedArtifactOracle resolves the recurring create-vs-edit CHOOSE_APPROACH — "add a
// new file or edit the existing one?" — from a fact about the tree rather than a
// preference: an option naming a path git already tracks is the duplicate-avoiding
// choice. It is the repo-fact rung of the CLARIFY ladder (docs/notes
// CONCEPT-AGENTIC-OPERATOR-QUESTIONS §3). Its command vocabulary is fixed to the single
// read-only `git ls-files -- <path>`; it never mutates the tree, and it abstains (ok
// false, no git call) unless the question is framed as create-vs-edit AND the option
// names an unambiguous repo-relative path, so it cannot misfire on the isolation fold.
type TrackedArtifactOracle struct {
	Runner ReadOnlyRunner
}

// ScorecardAxis is one independently witnessed comparable axis. Scores are
// keyed by option label. Higher is better unless LowerIsBetter is set.
type ScorecardAxis struct {
	Name          string
	Scores        map[string]float64
	LowerIsBetter bool
	Witness       string
}

// ScorecardAxisOracle chooses only a Pareto-dominant reversible option. It does
// not infer scores from question prose: the caller supplies scorecard values and
// the witness that produced each axis.
type ScorecardAxisOracle struct {
	Axes       []ScorecardAxis
	Reversible map[string]bool
}

// Name returns the identifier for ScorecardAxisOracle.
func (ScorecardAxisOracle) Name() string { return "scorecard-axis-readonly" }

// Inspect checks Pareto dominance across declared scorecard axes.
func (o ScorecardAxisOracle) Inspect(_ context.Context, q operatorquestion.OperatorQuestion, option operatorquestion.Option) (Evidence, bool, error) {
	if len(q.Options) < 2 || len(o.Axes) == 0 || !o.Reversible[option.Label] {
		return Evidence{}, false, nil
	}
	deciding := make([]string, 0, len(o.Axes))
	witnesses := make([]string, 0, len(o.Axes))
	for _, other := range q.Options {
		if !o.Reversible[other.Label] {
			return Evidence{}, false, nil
		}
		if other.Label == option.Label {
			continue
		}
		strict := false
		for _, axis := range o.Axes {
			av, aok := axis.Scores[option.Label]
			bv, bok := axis.Scores[other.Label]
			if !aok || !bok || strings.TrimSpace(axis.Name) == "" || strings.TrimSpace(axis.Witness) == "" {
				return Evidence{}, false, nil
			}
			if axis.LowerIsBetter {
				av, bv = -av, -bv
			}
			if av < bv {
				return Evidence{}, false, nil
			}
			if av > bv {
				strict = true
				deciding = appendUnique(deciding, axis.Name)
			}
			witnesses = appendUnique(witnesses, axis.Name+"="+axis.Witness)
		}
		if !strict {
			return Evidence{}, false, nil
		}
	}
	return Evidence{Oracle: o.Name(), Claim: "dominates on axis " + strings.Join(deciding, ", "), Witness: strings.Join(witnesses, ", "), Score: 100}, true, nil
}

func appendUnique(values []string, value string) []string {
	for _, existing := range values {
		if existing == value {
			return values
		}
	}
	return append(values, value)
}

// Name returns the identifier for TrackedArtifactOracle.
func (TrackedArtifactOracle) Name() string { return "tracked-artifact-readonly" }

// Inspect verifies whether candidate file paths are already tracked in git.
func (o TrackedArtifactOracle) Inspect(ctx context.Context, q operatorquestion.OperatorQuestion, option operatorquestion.Option) (Evidence, bool, error) {
	if o.Runner == nil || !containsAnyFold(q.Question+" "+q.Detail, "new file", "create", "existing", "edit", "rewrite", "which file", "separate file", "add to") {
		return Evidence{}, false, nil
	}
	path := firstRepoPath(option.Label + " " + option.Rationale)
	if path == "" {
		return Evidence{}, false, nil
	}
	out, err := o.Runner.Run(ctx, "git", "ls-files", "--", path)
	if err != nil {
		return Evidence{}, false, err
	}
	tracked := strings.TrimSpace(string(out))
	witness := "git ls-files -- " + path + ": " + tracked
	if tracked != "" {
		return Evidence{Claim: "option targets an already-tracked file (" + path + "); editing the existing artifact avoids a parallel duplicate", Witness: witness, Score: 8}, true, nil
	}
	return Evidence{Claim: "option references a path git does not track (" + path + ")", Witness: witness, Score: 0}, true, nil
}

// Resolve evaluates question options against configured oracles and produces a verdict.
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

// Name returns the configured oracle name.
func (f OracleFunc) Name() string { return f.OracleName }

// Inspect delegates inspection to InspectFn.
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

// firstRepoPath extracts the first unambiguous repo-relative path from free option
// text: a backtick-wrapped literal that looks like a path, else the first bare token
// that does. It is deliberately strict — a token that is not clearly a path yields ""
// so TrackedArtifactOracle abstains rather than guessing (a false path would run a
// pointless git query and, worse, could witness the wrong option).
func firstRepoPath(text string) string {
	if i := strings.IndexByte(text, '`'); i >= 0 {
		if j := strings.IndexByte(text[i+1:], '`'); j >= 0 {
			if cand := strings.TrimSpace(text[i+1 : i+1+j]); looksLikePath(cand) {
				return cand
			}
		}
	}
	for _, tok := range strings.FieldsFunc(text, func(r rune) bool {
		return r == ' ' || r == '\t' || r == '\n' || r == ',' || r == ';' || r == '(' || r == ')' || r == '"' || r == '\''
	}) {
		if cand := strings.Trim(tok, ".,:;`"); looksLikePath(cand) {
			return cand
		}
	}
	return ""
}

// looksLikePath reports whether tok is unambiguously a repo-relative path: it carries a
// forward-slash directory separator (git paths are always forward-slash) and a final
// component with a real extension. Prose words and bare filenames are rejected on
// purpose — the oracle only speaks when the option clearly names a tree location.
func looksLikePath(tok string) bool {
	if len(tok) < 3 || strings.ContainsAny(tok, " \t") || !strings.Contains(tok, "/") {
		return false
	}
	base := tok[strings.LastIndex(tok, "/")+1:]
	dot := strings.LastIndex(base, ".")
	return dot > 0 && dot < len(base)-1
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
