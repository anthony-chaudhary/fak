package quality

import (
	"fmt"
	"sort"
	"strings"
)

// Oracle judges an engine trace against a reference trace for one case and
// returns a Verdict. Two kinds ship in the spine: a DETERMINISTIC comparator
// (token-by-token differential) and a RUBRIC scorer (deterministic grounding
// stand-in). The child cohort registers more oracles against this same interface
// — logit-parity tolerances (#4523), distribution comparison (#4530), calibrated
// report judges (#4563) — without editing this file.
type Oracle interface {
	// Name is the stable identifier a case's Oracles list references.
	Name() string
	// Kind is "differential" (exact/tolerance comparison to a reference) or
	// "rubric" (a scored quality dimension). It drives how explain localizes.
	Kind() string
	// Judge compares eng against ref for case c. It must be a pure function of its
	// inputs so a result replays identically.
	Judge(ref, eng Trace, c QualityCase) Verdict
}

// Verdict is one oracle's decision. FirstDivergence is set by differential
// oracles to the exact step where the engine first departed from the reference —
// the localizing evidence the epic exists to produce. Score is set by rubric
// oracles.
type Verdict struct {
	Oracle          string      `json:"oracle"`
	Kind            string      `json:"kind"`
	Pass            bool        `json:"pass"`
	Score           float64     `json:"score,omitempty"`
	FirstDivergence *Divergence `json:"first_divergence,omitempty"`
	Detail          string      `json:"detail"`
}

// Divergence is the first step at which two token streams disagree: its index and
// the reference vs engine token there. It is what turns "the report looked wrong"
// into "token 7 was 'increased' where the reference emitted 'decreased'".
type Divergence struct {
	Index     int    `json:"index"`
	Reference string `json:"reference"`
	Engine    string `json:"engine"`
}

// rubricFail is the verdict a rubric oracle returns when the run fails OUTRIGHT
// rather than scoring a fraction — the payload would not parse, the case declared
// no usable evidence, a forbidden claim appeared, the wrong tool was called. Such a
// run is not a low score, it is a zero: the oracle had nothing to grade. Callers
// pass their own already-formatted reason so the Detail still names the specific
// fault, and return the result directly.
func rubricFail(v Verdict, reason string) Verdict {
	v.Pass = false
	v.Score = 0
	v.Detail = reason
	return v
}

// GreedyTokenDiff is the deterministic comparator: for a temperature-zero / greedy
// case, the engine token stream must equal the reference token stream exactly, and
// the first mismatch (or a length difference) is reported as the first divergence.
// This is the spine realization of #4522 (compare greedy decode token by token
// BEFORE text scoring) — a fluent report built from one wrong token is caught here,
// not left for a downstream text metric to average away.
type GreedyTokenDiff struct{}

func (GreedyTokenDiff) Name() string { return "greedy-token-diff" }
func (GreedyTokenDiff) Kind() string { return "differential" }

func (GreedyTokenDiff) Judge(ref, eng Trace, _ QualityCase) Verdict {
	v := Verdict{Oracle: "greedy-token-diff", Kind: "differential", Pass: true}
	n := len(ref.Tokens)
	if len(eng.Tokens) < n {
		n = len(eng.Tokens)
	}
	for i := 0; i < n; i++ {
		if ref.Tokens[i] != eng.Tokens[i] {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: i, Reference: ref.Tokens[i], Engine: eng.Tokens[i]}
			v.Detail = fmt.Sprintf("token %d diverged: reference %q, engine %q", i, ref.Tokens[i], eng.Tokens[i])
			return v
		}
	}
	if len(ref.Tokens) != len(eng.Tokens) {
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: n, Reference: tokenAt(ref.Tokens, n), Engine: tokenAt(eng.Tokens, n)}
		v.Detail = fmt.Sprintf("token length diverged at %d: reference has %d, engine has %d",
			n, len(ref.Tokens), len(eng.Tokens))
		return v
	}
	v.Detail = fmt.Sprintf("%d tokens matched the reference", len(ref.Tokens))
	return v
}

func tokenAt(toks []string, i int) string {
	if i < len(toks) {
		return toks[i]
	}
	return "<end>"
}

// GroundingRubric is the deterministic rubric scorer: it scores the engine text on
// the fraction of the case's required phrases present, fails if any forbidden claim
// appears, and gates on the case's MinScore. It is the spine stand-in for the
// report-quality rubric layer (#4550–#4565): even without a calibrated judge, an
// executive report that silently omits a material required claim, or invents a
// forbidden one, fails a gate instead of being merely observed in prose.
type GroundingRubric struct{}

func (GroundingRubric) Name() string { return "grounding-rubric" }
func (GroundingRubric) Kind() string { return "rubric" }

func (GroundingRubric) Judge(_ Trace, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "grounding-rubric", Kind: "rubric", Pass: true, Score: 1}
	text := strings.ToLower(eng.Text)
	for _, f := range c.Rubric.Forbidden {
		if f != "" && strings.Contains(text, strings.ToLower(f)) {
			return rubricFail(v, fmt.Sprintf("forbidden claim present: %q", f))
		}
	}
	req := c.Rubric.Required
	if len(req) == 0 {
		v.Detail = "no required phrases declared; forbidden checks passed"
		return v
	}
	present := 0
	var missing []string
	for _, r := range req {
		if r != "" && strings.Contains(text, strings.ToLower(r)) {
			present++
		} else {
			missing = append(missing, r)
		}
	}
	v.Score = float64(present) / float64(len(req))
	min := c.Rubric.MinScore
	if min == 0 {
		min = 1 // default: all required phrases must be present
	}
	if v.Score < min {
		v.Pass = false
		v.Detail = fmt.Sprintf("grounding score %.2f < %.2f; missing required: %s",
			v.Score, min, strings.Join(missing, ", "))
		return v
	}
	v.Detail = fmt.Sprintf("grounding score %.2f >= %.2f", v.Score, min)
	return v
}

// registry is the oracle name → constructor table. Registration is the additive
// seam the child cohort uses: a new oracle file calls Register in an init() and
// becomes available to any case that names it, with no edit to run.go/case.go.
var registry = map[string]Oracle{}

// Register adds an oracle to the shared registry. It panics on a duplicate name so
// two children silently shadowing each other's oracle is a build-time failure, not
// a runtime surprise.
func Register(o Oracle) {
	if _, dup := registry[o.Name()]; dup {
		panic("quality: duplicate oracle registration: " + o.Name())
	}
	registry[o.Name()] = o
}

func init() {
	Register(GreedyTokenDiff{})
	Register(GroundingRubric{})
}

// Lookup resolves the oracles a case names, preserving the case's order. It
// returns an error naming the first unknown oracle so a typo'd or unshipped oracle
// is refused rather than silently skipped (a skipped oracle is not a pass).
func Lookup(names []string) ([]Oracle, error) {
	out := make([]Oracle, 0, len(names))
	for _, n := range names {
		o, ok := registry[n]
		if !ok {
			return nil, fmt.Errorf("unknown oracle %q (registered: %s)", n, strings.Join(registeredNames(), ", "))
		}
		out = append(out, o)
	}
	return out, nil
}

func registeredNames() []string {
	names := make([]string, 0, len(registry))
	for n := range registry {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}
