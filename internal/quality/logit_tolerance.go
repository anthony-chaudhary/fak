package quality

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// logit_tolerance.go is the backend logit-parity child of the quality spine
// (#4523): two backends decoding the same case — a reference kernel and an
// engine build — may legitimately disagree in the last bits of a logit
// (different FMA contraction, different reduction order, a fused kernel), so
// exact float equality is the wrong gate. Instead each case DECLARES the
// tolerance its backends are allowed to differ by, and the oracle passes iff
// the maximum absolute difference between the step-0 logit vectors
// (Reference.Logits[0] vs the engine trace's Logits[0]) is within that band.
// In-band numeric noise passes; a kernel defect that drifts one logit beyond
// the declared band fails, localized to that logit index with the measured
// difference — "backend B is off" becomes "logit 5 drifted by 0.008 against
// tol=0.001".

// ltolOracleName is the stable registry identifier of the logit-parity oracle.
const ltolOracleName = "logit-parity-tolerance"

// ltolDefaultTolerance is the band applied when a case declares no tolerance at
// all — deliberately near float64 noise, so an undeclared case behaves almost
// like exact parity rather than silently admitting real drift.
const ltolDefaultTolerance = 1e-9

// ltolTolerance resolves the declared tolerance for a case. A "tol=<float>"
// field in the prompt wins (the tolerance travels with the case text, e.g.
// "compare logits within tol=1e-3"); otherwise a positive Rubric.MinScore is
// reused as the band; otherwise the tight default applies. The returned source
// string is echoed into the verdict detail so a reader sees WHICH declaration
// the judgment was made under.
func ltolTolerance(c QualityCase) (float64, string) {
	for _, f := range strings.Fields(c.Prompt) {
		if !strings.HasPrefix(f, "tol=") {
			continue
		}
		raw := strings.TrimRight(strings.TrimPrefix(f, "tol="), ".,;:!?)")
		if t, err := strconv.ParseFloat(raw, 64); err == nil && t > 0 {
			return t, "declared in prompt"
		}
	}
	if c.Rubric.MinScore > 0 {
		return c.Rubric.MinScore, "declared via rubric min_score"
	}
	return ltolDefaultTolerance, "default (case declared no tolerance)"
}

// ltolFirstStep returns the step-0 logit vector of a trace's logits, or nil
// when the trace carries none. Step 0 is the declared parity surface of this
// child: both paths score the same prompt prefix there, so their logit vectors
// are directly comparable before any sampled token can fork the context.
func ltolFirstStep(logits [][]float64) []float64 {
	if len(logits) == 0 {
		return nil
	}
	return logits[0]
}

// ltolFmt renders a logit exactly (shortest round-trip form) for divergence
// reporting, so the verdict carries the numbers themselves, not a lossy view.
func ltolFmt(x float64) string {
	return strconv.FormatFloat(x, 'g', -1, 64)
}

// ltolLogitAt is tokenAt's numeric sibling: the formatted logit at i, or a
// sentinel when the vector already ended.
func ltolLogitAt(vec []float64, i int) string {
	if i < len(vec) {
		return ltolFmt(vec[i])
	}
	return "<end>"
}

// ltolOracle is the differential oracle for backend logit parity within a
// declared tolerance (#4523). Fail-closed properties: a trace without step-0
// logits cannot pass (a comparison that never ran is not a green), a length
// mismatch fails at the first missing index, and a NaN/Inf difference is
// treated as beyond any tolerance.
type ltolOracle struct{}

func (ltolOracle) Name() string { return ltolOracleName }
func (ltolOracle) Kind() string { return "differential" }

func init() { Register(ltolOracle{}) }

func (ltolOracle) Judge(ref, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: ltolOracleName, Kind: "differential", Pass: true}
	tol, src := ltolTolerance(c)

	refL, engL := ltolFirstStep(ref.Logits), ltolFirstStep(eng.Logits)
	if len(refL) == 0 || len(engL) == 0 {
		v.Pass = false
		v.Detail = fmt.Sprintf(
			"logit parity cannot be judged: reference carries %d step-0 logits, engine %d (a trace without logits is not a pass)",
			len(refL), len(engL))
		return v
	}
	if len(refL) != len(engL) {
		n := len(refL)
		if len(engL) < n {
			n = len(engL)
		}
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: n, Reference: ltolLogitAt(refL, n), Engine: ltolLogitAt(engL, n)}
		v.Detail = fmt.Sprintf("logit vector length diverged at %d: reference has %d, engine has %d",
			n, len(refL), len(engL))
		return v
	}

	worst, worstIdx := 0.0, 0
	for i := range refL {
		d := math.Abs(refL[i] - engL[i])
		if d > worst {
			worst, worstIdx = d, i
		}
		if d > tol || math.IsNaN(d) || math.IsInf(d, 0) {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: i, Reference: ltolFmt(refL[i]), Engine: ltolFmt(engL[i])}
			v.Detail = fmt.Sprintf("logit %d drifted beyond tolerance: |%s - %s| = %.6g > tol=%.6g (%s)",
				i, ltolFmt(refL[i]), ltolFmt(engL[i]), d, tol, src)
			return v
		}
	}
	v.Detail = fmt.Sprintf("%d logits within tol=%.6g (%s); max abs diff %.6g at logit %d",
		len(refL), tol, src, worst, worstIdx)
	return v
}

// ltolReferenceLogits is the fixed golden step-0 logit vector of the demo case:
// eight exactly-representable values, so the reference side contributes no
// noise of its own and every measured difference is authored by the engine.
func ltolReferenceLogits() []float64 {
	return []float64{2.5, 1.25, 0.5, -0.75, -1.5, -2.25, -3, -4.5}
}

const (
	// ltolDriftIndex is the single logit the "drift" mutant corrupts —
	// mid-vector, so the in-band prefix proves the localization does work.
	ltolDriftIndex = 5
	// ltolDriftFactor scales the injected drift to 8x the declared tolerance:
	// decisively out of band for any tolerance, never a rounding accident.
	ltolDriftFactor = 8
	// ltolNoiseFactor scales the legitimate cross-backend jitter to a quarter
	// of the declared tolerance: real, nonzero, and safely in band.
	ltolNoiseFactor = 0.25
)

// ltolCase builds the backend-parity case pinned to a declared tolerance: a
// one-step greedy decode whose prompt carries "tol=<v>" and whose reference
// trace carries the golden step-0 logit vector.
func ltolCase(tol float64) QualityCase {
	return QualityCase{
		Schema:  CaseSchema,
		ID:      "logit-parity-tolerance-demo",
		Version: 1,
		Prompt:  "Compare backend step-0 logits within tol=" + strconv.FormatFloat(tol, 'g', -1, 64),
		Params:  SamplingParams{Temperature: 0, MaxTokens: 1},
		Reference: Trace{
			Tokens: []string{"Throughput"},
			Logits: [][]float64{ltolReferenceLogits()},
			Text:   "Throughput",
		},
		Oracles: []string{ltolOracleName},
	}
}

// ltolEngine returns a scripted runner modeling a second backend under the
// declared tolerance: "" reproduces the reference logits with deterministic
// in-band noise (tol/4, alternating sign) — the legitimate numeric jitter two
// backends are ALLOWED, which must PASS; "drift" additionally moves the single
// logit at ltolDriftIndex by 8x the tolerance — the kernel-defect class that
// must FAIL, localized to that index with the measured difference.
func ltolEngine(defect string, tol float64) ScriptedRunner {
	ref := ltolReferenceLogits()
	logits := make([]float64, len(ref))
	for i, x := range ref {
		noise := tol * ltolNoiseFactor
		if i%2 == 1 {
			noise = -noise
		}
		logits[i] = x + noise
	}
	label := "engine-backend-clean"
	if defect == "drift" {
		logits[ltolDriftIndex] = ref[ltolDriftIndex] + tol*ltolDriftFactor
		label = "engine-backend-drift"
	}
	return ScriptedRunner{
		Label: label,
		Trace: Trace{Tokens: []string{"Throughput"}, Logits: [][]float64{logits}, Text: "Throughput"},
	}
}
