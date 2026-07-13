// Package quality is the runnable spine of the "missing middle" validation
// ladder (epic #4509): the layer between primitive correctness tests (too local
// to catch a fluent-but-wrong decode) and end benchmarks (too coarse and late to
// localize an engine-caused regression).
//
// The spine runs one versioned prompt case through a REFERENCE path and one fak
// ENGINE path, captures request/config/token/logit/output provenance, applies at
// least one deterministic comparator and one rubric scorer, and emits a
// machine-readable result with a pass/fail verdict and — on failure — a portable,
// replay-complete failure bundle. A quality claim is never green from a single
// stochastic sample: every oracle names the baseline it compares against and
// reports the FIRST divergence so a defect localizes to a token/step rather than
// being observed only in prose.
//
// The package is deliberately stdlib-only and hermetic: the reference and engine
// paths are Runner adapters (case.go / runner.go), so the spine is CI-runnable
// without a live model, and real engine adapters wire in behind the same seam.
// The child cohort under #4509 extends this by ADDING its own oracle/runner files
// and registering them — it does not edit these cores.
package quality

// CaseSchema is the versioned envelope tag for a quality case. Bumping the
// trailing version is a breaking change to the case shape; readers reject an
// unknown major so a stale corpus can never masquerade as current.
const CaseSchema = "fak-quality-case/1"

// QualityCase is one versioned, replayable unit of quality evidence: a prompt,
// the sampling configuration it must be decoded under, the reference expectation
// it is judged against, and the named oracles to apply. It carries everything a
// runner needs to reproduce the run — nothing about a case depends on wall-clock
// or ambient state.
type QualityCase struct {
	Schema  string         `json:"schema"`
	ID      string         `json:"id"`
	Version int            `json:"version"`
	Prompt  string         `json:"prompt"`
	Params  SamplingParams `json:"params"`
	// Reference is the golden trace the engine path is compared against. For a
	// deterministic (temperature-zero / greedy) case this is an exact oracle; a
	// stochastic case carries a distribution instead (added by the sampling
	// children #4529–#4530 without touching this struct).
	Reference Trace `json:"reference"`
	// Oracles names the comparators/scorers to apply, by Oracle.Name(). An empty
	// list is a case authored without any judge — RunCase reports it rather than
	// silently passing (a case that checks nothing is not a green case).
	Oracles []string `json:"oracles"`
	// Rubric configures the rubric scorer(s) for this case (grounding phrases,
	// forbidden claims, threshold). Optional; empty means no rubric dimension.
	Rubric RubricSpec `json:"rubric,omitempty"`
}

// SamplingParams is the decode configuration a case is pinned to. It is part of
// the replay contract: a result is only reproducible if the params that produced
// it travel with it. Seed is honored by stochastic runners; a temperature-zero
// case must decode identically regardless of seed (frozen by #4525).
type SamplingParams struct {
	Temperature float64 `json:"temperature"`
	TopK        int     `json:"top_k,omitempty"`
	TopP        float64 `json:"top_p,omitempty"`
	MaxTokens   int     `json:"max_tokens"`
	Seed        int64   `json:"seed,omitempty"`
}

// Trace is the captured output of one decode path: the emitted token sequence,
// optional per-step logits (top candidates), and the assembled text. Tokens are
// the primary differential surface — text scoring happens only after the token
// stream is proven equal or explicitly allowed to differ, because two paths can
// assemble identical-looking text from divergent tokens.
type Trace struct {
	Runner string      `json:"runner"`
	Tokens []string    `json:"tokens"`
	Logits [][]float64 `json:"logits,omitempty"`
	Text   string      `json:"text"`
}

// RubricSpec is the deterministic rubric configuration for a case: phrases the
// grounded output must contain, claims it must not contain, and the minimum
// fraction of required phrases to pass. It is intentionally a deterministic
// stand-in the report-quality children (#4550–#4565) replace with calibrated
// judges — but even the stand-in makes a "reads fine" report fail when it drops a
// material, required claim.
type RubricSpec struct {
	Required  []string `json:"required,omitempty"`
	Forbidden []string `json:"forbidden,omitempty"`
	MinScore  float64  `json:"min_score,omitempty"`
}

// Valid reports whether the case envelope is well-formed enough to run. It is the
// admission gate: a case with the wrong schema, no ID, or no oracles is refused
// with a reason rather than run to a misleading green.
func (c QualityCase) Valid() (bool, string) {
	if c.Schema != CaseSchema {
		return false, "case schema " + c.Schema + " != " + CaseSchema
	}
	if c.ID == "" {
		return false, "case has empty id"
	}
	if len(c.Oracles) == 0 {
		return false, "case declares no oracles (a case that checks nothing is not green)"
	}
	return true, ""
}
