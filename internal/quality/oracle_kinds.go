package quality

import (
	"fmt"
	"math"
)

// oracle_kinds.go completes the oracle-kind taxonomy of the quality spine
// (#4517). The spine ships a DIFFERENTIAL comparator and a RUBRIC scorer; this
// file adds the remaining two kinds:
//
//   - "exact": byte/sequence-identical equality of the engine trace against the
//     reference — the strictest comparator, for cases where ANY departure (a
//     token, a text byte, a length) is a defect. Its verdicts localize the
//     first mismatch as a FirstDivergence.
//   - "statistical": an agreement-RATE judgment over N paired samples with a
//     confidence interval, for surfaces where per-sample equality is too strict
//     but the population agreement must stay provably high. It passes iff the
//     CI lower bound clears the case's threshold, so a green is a bounded
//     statistical claim, not a lucky point estimate.
//
// Both oracles register in init() and are named by cases like any other; no
// core file changes.

// kindsExactMatch is the EXACT-kind oracle: the engine trace must be
// sequence-identical to the reference in tokens AND byte-identical in assembled
// text. It is stricter than greedy-token-diff, which stops at the token stream:
// two paths can emit equal tokens yet assemble different text (a detokenizer
// bug), and exact-match catches that byte too. Logit closeness is out of scope
// — tolerance oracles own numeric drift; exact means exact.
type kindsExactMatch struct{}

func (kindsExactMatch) Name() string { return "exact-match" }
func (kindsExactMatch) Kind() string { return "exact" }

func (kindsExactMatch) Judge(ref, eng Trace, _ QualityCase) Verdict {
	v := Verdict{Oracle: "exact-match", Kind: "exact", Pass: true}
	n := len(ref.Tokens)
	if len(eng.Tokens) < n {
		n = len(eng.Tokens)
	}
	for i := 0; i < n; i++ {
		if ref.Tokens[i] != eng.Tokens[i] {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: i, Reference: ref.Tokens[i], Engine: eng.Tokens[i]}
			v.Detail = fmt.Sprintf("exact match broken at token %d: reference %q, engine %q",
				i, ref.Tokens[i], eng.Tokens[i])
			return v
		}
	}
	if len(ref.Tokens) != len(eng.Tokens) {
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: n, Reference: tokenAt(ref.Tokens, n), Engine: tokenAt(eng.Tokens, n)}
		v.Detail = fmt.Sprintf("exact match broken at token %d: reference has %d tokens, engine has %d",
			n, len(ref.Tokens), len(eng.Tokens))
		return v
	}
	if i := kindsFirstByteDiff(ref.Text, eng.Text); i >= 0 {
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: i, Reference: kindsByteAt(ref.Text, i), Engine: kindsByteAt(eng.Text, i)}
		v.Detail = fmt.Sprintf("tokens identical but assembled text diverged at byte %d: reference %q, engine %q",
			i, kindsByteAt(ref.Text, i), kindsByteAt(eng.Text, i))
		return v
	}
	v.Detail = fmt.Sprintf("exact: %d tokens and %d text bytes identical to the reference",
		len(ref.Tokens), len(ref.Text))
	return v
}

// kindsFirstByteDiff returns the index of the first byte at which a and b
// differ (the shorter string's length when one is a strict prefix of the
// other), or -1 when they are byte-identical.
func kindsFirstByteDiff(a, b string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return n
	}
	return -1
}

// kindsByteAt renders the byte of s at i for divergence evidence, or "<end>"
// when i is past the end of s (the length-mismatch side of a text divergence).
func kindsByteAt(s string, i int) string {
	if i < len(s) {
		return string(s[i])
	}
	return "<end>"
}

// kindsAgreementZ is the z value of the two-sided 95% normal interval, and
// kindsAgreementDefaultMin the pass threshold used when a case declares none
// via Rubric.MinScore.
const (
	kindsAgreementZ          = 1.96
	kindsAgreementDefaultMin = 0.95
)

// kindsStatisticalAgreement is the STATISTICAL-kind oracle: it treats each
// token position as one paired (reference, engine) sample, computes the
// agreement rate over the N pairs, bounds it with a 95% normal-approximation
// confidence interval, and passes iff the CI LOWER bound clears the case's
// threshold (Rubric.MinScore; default 0.95). Gating on the lower bound — not
// the point estimate — is what makes a pass a statistical claim: with few
// samples the interval is wide and a marginal rate cannot sneak through, while
// a genuinely high rate over many samples passes with room to spare. Positions
// present in only one trace count as disagreements, so over-generation and
// truncation depress the rate instead of hiding from it.
type kindsStatisticalAgreement struct{}

func (kindsStatisticalAgreement) Name() string { return "statistical-agreement" }
func (kindsStatisticalAgreement) Kind() string { return "statistical" }

func (kindsStatisticalAgreement) Judge(ref, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "statistical-agreement", Kind: "statistical"}
	agree, n := kindsAgreementSamples(ref, eng)
	if n == 0 {
		v.Pass = false
		v.Detail = "no paired samples: both traces are empty, so agreement cannot be bounded (an unmeasured case is not green)"
		return v
	}
	p, lo, hi := kindsAgreementCI(agree, n)
	v.Score = p
	threshold := c.Rubric.MinScore
	if threshold == 0 {
		threshold = kindsAgreementDefaultMin
	}
	if lo < threshold {
		v.Pass = false
		v.Detail = fmt.Sprintf("agreement %d/%d = %.4f, 95%% CI [%.4f, %.4f]: lower bound %.4f < threshold %.4f",
			agree, n, p, lo, hi, lo, threshold)
		return v
	}
	v.Pass = true
	v.Detail = fmt.Sprintf("agreement %d/%d = %.4f, 95%% CI [%.4f, %.4f]: lower bound %.4f >= threshold %.4f",
		agree, n, p, lo, hi, lo, threshold)
	return v
}

// kindsAgreementSamples pairs the two token streams position by position:
// N is the longer stream's length, and a position agrees only when both
// streams carry the same token there. A position past either stream's end is
// a disagreement by construction.
func kindsAgreementSamples(ref, eng Trace) (agree, n int) {
	n = len(ref.Tokens)
	if len(eng.Tokens) > n {
		n = len(eng.Tokens)
	}
	for i := 0; i < n; i++ {
		if i < len(ref.Tokens) && i < len(eng.Tokens) && ref.Tokens[i] == eng.Tokens[i] {
			agree++
		}
	}
	return agree, n
}

// kindsAgreementCI returns the observed agreement rate over n paired samples
// and its two-sided 95% normal-approximation (Wald) confidence interval,
// clamped to [0, 1]. It is a pure function of (agree, n), so a verdict built
// from it replays identically.
func kindsAgreementCI(agree, n int) (p, lo, hi float64) {
	p = float64(agree) / float64(n)
	se := math.Sqrt(p * (1 - p) / float64(n))
	lo = p - kindsAgreementZ*se
	hi = p + kindsAgreementZ*se
	if lo < 0 {
		lo = 0
	}
	if hi > 1 {
		hi = 1
	}
	return p, lo, hi
}

func init() {
	Register(kindsExactMatch{})
	Register(kindsStatisticalAgreement{})
}
