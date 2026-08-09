package trajctl

// judgescorer.go — issue #2543, the W1 rung of the trajectory-control epic
// (#2533): a judge scorer that scores current state against the objective
// STATEMENT via a structured, pinned-schema verdict call. It fills the non-code
// gap — research, docs, and triage objectives whose progress is neither commit-
// nor test-shaped have no deterministic scorer at all — WITHOUT ever outranking
// stronger evidence: every row it emits is W1, honestly below the W2 activity
// scorer and the W3 witnessed-commit scorer, so a consumer that has real
// evidence (curve.go's commit-progress fold) always prefers it.
//
// The scorer stays PURE the same way every other scorer in this package does:
// the impure model call is an INJECTED JudgeClient (the gateway-backed
// GatewayJudgeClient lives in judgeclient.go), so the fold itself is
// deterministic given a client and tier-1. Two things are enforced, not merely
// documented:
//   - a per-call token budget cap (MaxCallTokens) travels INTO the request so
//     the model cannot generate past it, AND a returned usage over the cap is
//     rejected fail-closed — a runaway judge call earns no credited progress and
//     cannot blow the per-session cost.
//   - a nil client or a non-positive cap yields NO row: no budget, no spend.

import (
	"encoding/json"
	"fmt"
)

const (
	// JudgeScorerMethod is the stable method id of the judge scorer. It keys the
	// registry and travels in every W1 row it emits.
	JudgeScorerMethod = "judge-objective-progress"
	// JudgeScorerVersion is this implementation's version.
	JudgeScorerVersion = "1"

	// DefaultJudgeMaxCallTokens is the conservative per-call output-token cap used
	// when a JudgeScorer is constructed with MaxCallTokens<=0 through NewJudgeScorer.
	// A verdict is a few tokens of JSON; 512 is generous headroom, not a licence to
	// run a full generation.
	DefaultJudgeMaxCallTokens = 512
)

// JudgeVerdict is the PINNED structured schema a judge call returns — the
// forced-tool-choice tool's arguments. Progress is the unit-interval progress
// estimate the W1 row carries as its value; the whole blob is recorded as
// evidence so a later audit can read the model's own justification.
type JudgeVerdict struct {
	// Progress is the model's estimate of objective completion in [0,1]. It is
	// clamped into range before it becomes a row value.
	Progress float64 `json:"progress"`
	// Met is the model's boolean read of whether the objective is fully satisfied.
	Met bool `json:"met"`
	// Rationale is a short justification, carried verbatim as evidence.
	Rationale string `json:"rationale"`
	// Criteria is the per-criterion attribution a rubric-based verdict carries
	// (#2544): one finding per rubric criterion the judge scored. Empty on a
	// bare-statement verdict.
	Criteria []RubricFinding `json:"criteria,omitempty"`
}

// JudgeUsage is what one judge call cost, so the scorer can enforce the budget
// cap on the returned spend, not only on the request ceiling.
type JudgeUsage struct {
	// Tokens is the total tokens the call consumed (prompt + completion).
	Tokens int
}

// JudgeRequest is the cache-friendly prompt shape handed to a JudgeClient: a
// stable objective/state pair plus the per-call token cap the client MUST pass
// to the model as its output ceiling. The client builds a stable instruction
// prefix around these so a re-scored objective reuses the provider prompt cache.
type JudgeRequest struct {
	// Objective is the objective statement being scored against.
	Objective string
	// State describes the current state the model judges progress from.
	State string
	// MaxTokens is the per-call output-token cap. A client MUST forward it as the
	// request's max_tokens so a runaway generation is bounded at the source.
	MaxTokens int
	// Rubric, when non-empty, is the formatted rubric block (#2544,
	// FormatRubricForPrompt) the judge scores against instead of the bare
	// objective statement alone. Empty keeps the #2543 bare-statement prompt.
	Rubric string
}

// JudgeClient serves one structured verdict call. It is the injected impurity:
// the pure scorer folds its result, the client owns the network I/O and the
// pinned-schema / forced-tool-choice request shape.
type JudgeClient interface {
	Judge(req JudgeRequest) (JudgeVerdict, JudgeUsage, error)
}

// JudgeScorer emits a W1 objective-progress row from a structured judge verdict.
// It is constructed per run by the caller (the `fak trajctl score --method
// judge` path), which injects the client and the budget cap; the Score fold
// itself is deterministic given that client.
type JudgeScorer struct {
	// Client serves the verdict call. A nil client makes Score emit no row.
	Client JudgeClient
	// MaxCallTokens is the per-call token budget cap. It travels into the request
	// as the output ceiling AND bounds the accepted return: a call whose reported
	// usage exceeds it is rejected. A non-positive cap makes Score emit no row.
	MaxCallTokens int
	// State, when set, is the current-state description handed to the judge. Empty
	// derives a compact summary from the evidence window (prior curve + sessions).
	State string
}

// NewJudgeScorer builds a judge scorer with the default cap applied when maxCall
// is non-positive, so a caller that forgets to set a budget still gets the
// conservative ceiling rather than the no-spend fail-closed path.
func NewJudgeScorer(client JudgeClient, maxCall int) JudgeScorer {
	if maxCall <= 0 {
		maxCall = DefaultJudgeMaxCallTokens
	}
	return JudgeScorer{Client: client, MaxCallTokens: maxCall}
}

// Method implements Scorer.
func (JudgeScorer) Method() string { return JudgeScorerMethod }

// Version implements Scorer.
func (JudgeScorer) Version() string { return JudgeScorerVersion }

// Score folds one structured verdict into a single W1 progress row. It fails
// closed at every boundary: a closed objective, a nil client, a non-positive
// budget cap, a client error, or a returned spend over the cap all yield NO row
// rather than crediting unverified or over-budget progress.
func (s JudgeScorer) Score(obj Objective, win EvidenceWindow) []ScoreRow {
	if obj.Status != StatusActive && obj.Status != StatusPaused {
		return nil
	}
	if s.Client == nil || s.MaxCallTokens <= 0 {
		return nil // no client or no budget: no spend, no row
	}
	state := s.State
	if state == "" {
		state = judgeStateSummary(obj, win)
	}
	req := JudgeRequest{Objective: obj.Statement, State: state, MaxTokens: s.MaxCallTokens, Rubric: FormatRubricForPrompt(obj.Rubric)}
	verdict, usage, err := s.Client.Judge(req)
	if err != nil {
		return nil // fail-closed: a failed judge call credits no progress
	}
	if usage.Tokens > s.MaxCallTokens {
		return nil // budget cap enforced on the RETURN, not just the request
	}
	blob, err := json.Marshal(verdict)
	if err != nil {
		return nil
	}
	evidence := []EvidenceRef{{
		Kind:   "judge-verdict",
		Ref:    fmt.Sprintf("tokens=%d", usage.Tokens),
		Detail: string(blob),
	}}
	// Per-criterion attribution (#2544): one ref per rubric finding, keyed by
	// criterion id, so a row cites WHICH criteria moved, not just the aggregate.
	// Findings without an id are dropped — an unattributable finding would fail
	// ledger validation and cannot be cited.
	for _, f := range verdict.Criteria {
		if f.ID == "" {
			continue
		}
		fb, err := json.Marshal(f)
		if err != nil {
			continue
		}
		evidence = append(evidence, EvidenceRef{
			Kind:   "rubric-criterion",
			Ref:    f.ID,
			Detail: string(fb),
		})
	}
	// Conjunctive fold (#3926): when the cached rubric is marked conjunctive, the
	// W1 value is a hard AND of the per-criterion findings — 1.0 only if every
	// criterion passes, else 0.0 — instead of the soft clamp01(progress). The
	// per-criterion refs above are preserved; a summary ref names which criterion
	// broke the AND so the failure is attributable. Default (non-conjunctive)
	// objectives keep the soft-progress fold untouched.
	value := clamp01(verdict.Progress)
	// A soft rubric verdict carries two representations of progress: the scalar
	// and its per-criterion itemization. Never credit a scalar that is more
	// optimistic than the arithmetic support in that same call. This is a
	// consistency check, not an alternative scoring policy: conservative scalars
	// remain valid, and bare-statement verdicts have no itemization to bind.
	if obj.Rubric != nil && !obj.Rubric.Conjunctive && len(verdict.Criteria) > 0 {
		// Fold only declared criterion IDs, once each. A missing finding supports
		// zero, while an unknown or duplicate finding cannot inflate support.
		byID := make(map[string]float64, len(verdict.Criteria))
		for _, finding := range verdict.Criteria {
			if _, seen := byID[finding.ID]; !seen {
				byID[finding.ID] = clamp01(finding.Progress)
			}
		}
		var supported float64
		for _, criterion := range obj.Rubric.Criteria {
			supported += byID[criterion.ID]
		}
		if len(obj.Rubric.Criteria) > 0 {
			supported /= float64(len(obj.Rubric.Criteria))
		}
		if value > supported+1e-9 {
			return nil
		}
	}
	if obj.Rubric != nil && obj.Rubric.Conjunctive {
		v, failed := conjunctiveValue(obj.Rubric, verdict.Criteria)
		value = v
		ref := EvidenceRef{Kind: "rubric-conjunctive", Ref: "all-pass", Detail: "all rubric criteria met the pass threshold"}
		if v == 0 {
			ref.Ref = "fail"
			if failed != "" {
				ref.Detail = fmt.Sprintf("unmet criterion %q broke the conjunctive AND", failed)
			} else {
				ref.Detail = "no rubric criteria to satisfy the conjunctive AND"
			}
		}
		evidence = append(evidence, ref)
	}
	return []ScoreRow{{
		ObjectiveID: obj.ID,
		Value:       value,
		Method:      JudgeScorerMethod,
		Version:     JudgeScorerVersion,
		Witness:     W1,
		Evidence:    evidence,
		UnixMillis:  win.UnixMillis,
	}}
}

// judgeStateSummary derives a compact, deterministic current-state description
// from the evidence window when the caller supplied none: the prior curve's
// depth and best witnessed progress plus the observed session count. It is the
// cache-friendly default — a short, stable string rather than a dump.
func judgeStateSummary(obj Objective, win EvidenceWindow) string {
	best, n := 0.0, 0
	for _, r := range win.PriorScores {
		if r.ObjectiveID != obj.ID {
			continue
		}
		n++
		if r.Value > best {
			best = r.Value
		}
	}
	return fmt.Sprintf("prior_scores=%d best_progress=%.2f sessions=%d", n, best, len(win.Sessions))
}
