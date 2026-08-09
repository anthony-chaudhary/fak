// Command negframe-steerability-ab is the offline A/B harness for issue #3546: does
// affordance-first ("do this") guard-directive framing lift steer compliance over the same
// directive framed as a prohibition ("don't do that")?
//
// PROVENANCE, up front and unambiguous: this harness has NO live model and NO GPU. Every number
// it prints is a MODELED / OFFLINE PROXY, derived from a documented, admittedly-simple linear
// cost model over internal/negframe's deterministic negation classification -- never a measured
// agent-behavior outcome. Treat every "compliance" figure below as "what a stated cost model
// predicts," not "what an agent actually did." See README.md in this directory for the full
// design, the real-witness upgrade path, and why that distinction matters (the repo's net-true-
// value standard: an unwitnessed gain is `not yet`, never a shipped claim).
//
// What it does:
//   - Runs the MEASURED A/B (#5851): the same fak-authored SessionStart stream emitted twice
//     through the real binary, once with #3568's lever set (`FAK_ABLATE=negframe_reframe`, the
//     unreframed CONTROL arm) and once with it unset (the reframed TREATMENT arm), reading each
//     run's arm label and negation counts back off the per-turn journal
//     (`.fak/negframe/journal.jsonl`). See measured.go. The arm labels are measured; the
//     compliance rate they map to is still the modeled proxy below.
//   - Holds a fixture corpus of paired guard directives: Arm A (negative/prohibition-framed,
//     drawn from the repo's own steer-prose idiom) and Arm B (the same instruction reframed
//     affordance-first -- mechanically via internal/negframe's reframe rules where a rule
//     applies, hand-authored otherwise, since negframe's judgement tier intentionally does not
//     auto-rewrite).
//   - Scores each arm's negation load via negframe.Classify (a real, deterministic, ungameable
//     signal: the same lexicon `fak score negframe` gates on).
//   - Maps negation load to a MODELED compliance proxy via a stated linear cost function, and
//     reports the paired A/B delta plus an exact, dependency-free paired sign test.
//   - Runs a self-check (-selfcheck) that (a) proves the fixture corpus is well-formed (Arm A
//     truly negative-framed, Arm B truly clean) and (b) proves the modeled delta point the
//     direction the thesis predicts, printing one clear PASS/FAIL line.
//
// Run:
//
//	go run ./experiments/negframe-steerability-ab              # human-readable report
//	go run ./experiments/negframe-steerability-ab -json         # machine-readable result
//	go run ./experiments/negframe-steerability-ab -selfcheck     # corpus + direction PASS/FAIL
package main

import (
	"encoding/json"
	"flag"
	"os"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

// --- the modeled cost model (see README.md "Compliance metric") --------------------------------
//
// These three constants are the entire "prediction" this harness makes. They are NOT fit to any
// data -- there is none to fit to -- they are a stated, monotone, bounded proxy standing in for
// the hypothesis "each negation a reader/agent must invert before finding the actionable
// instruction costs some compliance." Swap them, or replace complianceProxy entirely, once a
// real witness (see README.md "Upgrading to a live witness") supplies calibration data.
const (
	// complianceCeiling is the modeled compliance rate for a directive with ZERO negation
	// findings (Arm B, by fixture construction): not 100%, because even a clean affordance-first
	// instruction is not read/executed perfectly every time.
	complianceCeiling = 0.97
	// mechanicalCost is the modeled compliance points lost per MECHANICAL negframe finding (a
	// negative with a confident positive rewrite, e.g. "don't forget to X") -- the class the
	// reframe makes disappear entirely in Arm B.
	mechanicalCost = 0.07
	// judgementCost is the modeled points lost per JUDGEMENT-tier finding (negatively framed,
	// e.g. "never X", but no auto-rewrite) -- assumed cheaper per instance than a mechanical
	// finding because these are shorter/blunter idioms ("never", "avoid") with a well-worn
	// positive counterpart a reader infers quickly, vs. a two-clause "don't forget to" that
	// requires holding the negation while parsing the embedded clause.
	judgementCost = 0.04
	// complianceFloor bounds the proxy below zero-compliance directives at a non-zero rate: even
	// a heavily negative-framed directive is presumed followed some of the time.
	complianceFloor = 0.10
)

// complianceProxy maps a directive's negation load to the modeled compliance rate in [floor,
// ceiling]. Linear and monotone-decreasing in both finding counts by construction.
func complianceProxy(mechanical, judgement int) float64 {
	v := complianceCeiling - mechanicalCost*float64(mechanical) - judgementCost*float64(judgement)
	if v < complianceFloor {
		return complianceFloor
	}
	if v > complianceCeiling {
		return complianceCeiling
	}
	return v
}

func main() {
	asJSON := flag.Bool("json", false, "emit machine-readable JSON result")
	selfCheck := flag.Bool("selfcheck", false, "run corpus + direction self-check, print PASS/FAIL, exit 1 on FAIL")
	measure := flag.Bool("measure", true, "run the measured A/B arms through the real fak binary (#5851)")
	flag.Parse()

	report := runExperiment()
	if *measure {
		report.Measured, report.MeasuredUnavailable = measureArms()
	} else {
		report.MeasuredUnavailable = "skipped (-measure=false)"
	}

	switch {
	case *selfCheck:
		ok := selfCheckReport(os.Stdout, report)
		if !ok {
			os.Exit(1)
		}
		return
	case *asJSON:
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(report)
		return
	default:
		printHuman(os.Stdout, report)
	}
}

// PairResult is one fixture pair's negframe scoring and modeled compliance for both arms.
type PairResult struct {
	ID                  string  `json:"id"`
	ArmAText            string  `json:"arm_a_negative_framed"`
	ArmBText            string  `json:"arm_b_affordance_first"`
	ArmAMechanical      int     `json:"arm_a_mechanical_findings"`
	ArmAJudgement       int     `json:"arm_a_judgement_findings"`
	ArmBMechanical      int     `json:"arm_b_mechanical_findings"`
	ArmBJudgement       int     `json:"arm_b_judgement_findings"`
	ArmACompliance      float64 `json:"arm_a_modeled_compliance"`
	ArmBCompliance      float64 `json:"arm_b_modeled_compliance"`
	Delta               float64 `json:"delta_b_minus_a"`
	ReframeIsMechanical bool    `json:"reframe_is_mechanical"` // true if negframe's own rule produced Arm B, false if hand-authored (judgement tier)
}

// SignTest is the exact one-sided binomial sign test over the non-tied paired deltas: H0 is
// P(Arm B compliance > Arm A compliance) == 0.5; the reported p is P(X >= k) under that null.
type SignTest struct {
	Pairs      int     `json:"pairs_total"`
	Ties       int     `json:"pairs_tied"`
	NUsed      int     `json:"pairs_used_n"`
	Favoring_B int     `json:"pairs_favoring_b_k"`
	PValue     float64 `json:"one_sided_p_value"`
}

// Report is the full experiment output: the MEASURED arms (#5851) + the modeled fixture half
// (fixtures + aggregate stats + the sign test) + an explicit provenance label. This is what -json
// emits and what printHuman/selfCheckReport render.
type Report struct {
	Schema     string       `json:"schema"`
	Issue      string       `json:"issue"`
	Provenance string       `json:"provenance"`
	Hypothesis string       `json:"hypothesis"`
	Pairs      []PairResult `json:"pairs"`
	MeanA      float64      `json:"mean_modeled_compliance_arm_a"`
	MeanB      float64      `json:"mean_modeled_compliance_arm_b"`
	MeanDelta  float64      `json:"mean_delta_b_minus_a"`
	SignTest   SignTest     `json:"sign_test"`
	// Measured is the #5851 half: the two arms produced by #3568's env lever over the real
	// injected stream, with their arm labels read back off the per-turn journal. Nil when no fak
	// binary could be resolved, in which case MeasuredUnavailable states why — the fixture half
	// below then stands alone and is labelled MODELED throughout.
	Measured            *MeasuredAB `json:"measured_arms,omitempty"`
	MeasuredUnavailable string      `json:"measured_arms_unavailable,omitempty"`
}

const provenanceLabel = "MODELED / OFFLINE PROXY -- not a live-model measurement. See README.md."

// measureArms resolves a fak binary and runs the measured A/B through it, returning either the
// measured half or a one-line reason it is unavailable. It never fails the run: a host with no
// fak binary still gets the modeled fixture report, explicitly labelled as standing alone.
func measureArms() (*MeasuredAB, string) {
	work, err := os.MkdirTemp("", "negframe-ab-measured-")
	if err != nil {
		return nil, "scratch workspace: " + err.Error()
	}
	bin, source, err := resolveFakBinary(work)
	if err != nil {
		return nil, "resolve fak binary: " + err.Error()
	}
	measured, err := runMeasuredAB(bin, source, work)
	if err != nil {
		return nil, "run measured arms: " + err.Error()
	}
	return measured, ""
}

func runExperiment() Report {
	pairs := make([]PairResult, 0, len(fixtures))
	var sumA, sumB float64
	favoringB, ties, n := 0, 0, 0
	for _, fx := range fixtures {
		fa := negframe.Classify("arm-a/"+fx.ID, fx.ArmA)
		fb := negframe.Classify("arm-b/"+fx.ID, fx.ArmB)
		mechA, judgeA := tally(fa)
		mechB, judgeB := tally(fb)
		compA := complianceProxy(mechA, judgeA)
		compB := complianceProxy(mechB, judgeB)
		delta := compB - compA
		pairs = append(pairs, PairResult{
			ID:                  fx.ID,
			ArmAText:            fx.ArmA,
			ArmBText:            fx.ArmB,
			ArmAMechanical:      mechA,
			ArmAJudgement:       judgeA,
			ArmBMechanical:      mechB,
			ArmBJudgement:       judgeB,
			ArmACompliance:      compA,
			ArmBCompliance:      compB,
			Delta:               delta,
			ReframeIsMechanical: fx.ReframeIsMechanical,
		})
		sumA += compA
		sumB += compB
		switch {
		case delta > 0:
			favoringB++
			n++
		case delta < 0:
			n++
		default:
			ties++
		}
	}
	count := float64(len(fixtures))
	st := SignTest{
		Pairs:      len(fixtures),
		Ties:       ties,
		NUsed:      n,
		Favoring_B: favoringB,
		PValue:     signTestPValue(n, favoringB),
	}
	return Report{
		Schema:     "fak-negframe-steerability-ab/1",
		Issue:      "#3546 (design) / #3568 (lever) / #5851 (measured-arm consumer wiring)",
		Provenance: provenanceLabel,
		Hypothesis: "Affordance-first (positive) guard-directive framing raises modeled compliance vs. the same directive framed as a prohibition.",
		Pairs:      pairs,
		MeanA:      sumA / count,
		MeanB:      sumB / count,
		MeanDelta:  (sumB - sumA) / count,
		SignTest:   st,
	}
}

func tally(findings []negframe.Finding) (mechanical, judgement int) {
	for _, f := range findings {
		if f.Mechanical() {
			mechanical++
		} else {
			judgement++
		}
	}
	return
}

// signTestPValue returns the exact one-sided binomial p-value P(X >= k) for X ~ Binomial(n, 0.5),
// the classic paired sign test statistic. n is the count of non-tied pairs; k is how many of
// those favor Arm B. Computed via the multiplicative binomial-coefficient recurrence (stable for
// the small n this harness ever sees, no external stats dependency).
func signTestPValue(n, k int) float64 {
	if n <= 0 {
		return 1
	}
	// P(X = i) = C(n,i) * 0.5^n; sum i=k..n.
	total := 0.0
	c := 1.0 // C(n,0)
	for i := 0; i <= n; i++ {
		if i > 0 {
			c = c * float64(n-i+1) / float64(i)
		}
		if i >= k {
			total += c
		}
	}
	return total / pow2(n)
}

func pow2(n int) float64 {
	v := 1.0
	for i := 0; i < n; i++ {
		v *= 2
	}
	return v
}
