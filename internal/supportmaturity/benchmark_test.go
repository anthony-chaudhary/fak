package supportmaturity

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/covmatrix"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

var (
	benchSinkRung           Rung
	benchSinkAction         NextAction
	benchSinkRecord         PromotionRecord
	benchSinkDecision       shipgate.Decision
	benchSinkBudgetCheck    BudgetCheck
	benchSinkPlaybook       RegimePlaybook
	benchSinkScoredFeatures []ScoredFeature
	benchSinkClaimTag       ClaimTag
	benchSinkBool           bool
)

const benchClaimsCorpus = `- [SHIPPED] Hardware-aware cache placement & lifecycle (CXL/NUMA-far).
- [SHIPPED] DDR cache tiers (engine cache-event visibility).
- [SHIPPED] Planned-elision → KV-eviction residency bridge.
- [SIMULATED] Multi-model weight-residency layer.
- [SIMULATED] metrics-service scrape adapter / KV-residency / token-per-watt.
- [SHIPPED] A pure-Go SmolLM2-135M forward pass (softmax attention).
- [SHIPPED] Native paged KV opt-in path.
- [SHIPPED] glm_moe_dsa (DeepSeek sparse attention).
- [SHIPPED] Poly-model serving core.
- [SHIPPED] Serving-latency observability (TTFT/TPOT histograms).
- [SHIPPED] Classed runtime device OOM recovery.
- [SHIPPED] LIVE gateway route dispatch executes routing decisions.
- [STUB] No LIVE transport attaching a real external KV cache yet.`

func BenchmarkFromSupport(b *testing.B) {
	supports := []covmatrix.Support{
		covmatrix.Supported,
		covmatrix.ProofPathOnly,
		covmatrix.Fenced,
		covmatrix.Undefined,
		covmatrix.Support("UNRECOGNIZED"),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkRung = FromSupport(supports[i%len(supports)])
	}
}

func BenchmarkFromPreflightVerdict(b *testing.B) {
	verdicts := []string{
		ggufload.PreflightReady,
		ggufload.PreflightRefuseTooBig,
		ggufload.PreflightRefuseArch,
		ggufload.PreflightRefuseHeader,
		"UNKNOWN_VERDICT",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkRung = FromPreflightVerdict(verdicts[i%len(verdicts)])
	}
}

func BenchmarkFromCorrectnessClass(b *testing.B) {
	classes := []compute.CorrectnessClass{
		compute.Reference,
		compute.Approx,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkRung = FromCorrectnessClass(classes[i%len(classes)])
	}
}

func BenchmarkNextActionFor(b *testing.B) {
	rungs := Rungs
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkAction = NextActionFor(rungs[i%len(rungs)])
	}
}

func BenchmarkRegimePlaybookAndStepBudget(b *testing.B) {
	steps := []int{5, 50, 5000, 50000}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := Rungs[i%len(Rungs)]
		regime, pb := PlaybookFor(r)
		benchSinkPlaybook = pb
		benchSinkBudgetCheck = regime.CheckStepBudget(steps[i%len(steps)])
	}
}

func BenchmarkPromote(b *testing.B) {
	witness := shipgate.Witness{
		Class:       shipgate.ClassProofCarrying,
		Metric:      "tokens_per_sec",
		Before:      100.0,
		After:       150.0,
		LowerBetter: false,
		TruthClean:  true,
		SuiteGreen:  true,
	}
	b.Run("ValidPromotion", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			next, dec := Promote(M4Correct, WitnessBenchCommitted, witness, nil)
			benchSinkRung = next
			benchSinkDecision = dec
		}
	})
	b.Run("MismatchedWitness", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			next, dec := Promote(M4Correct, WitnessPreflight, witness, nil)
			benchSinkRung = next
			benchSinkDecision = dec
		}
	})
	b.Run("WithGateBreaker", func(b *testing.B) {
		gate := shipgate.NewGate(2)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			next, dec := Promote(M4Correct, WitnessBenchCommitted, witness, gate)
			benchSinkRung = next
			benchSinkDecision = dec
		}
	})
}

func BenchmarkPromoteWithRecord(b *testing.B) {
	witness := shipgate.Witness{
		Class:       shipgate.ClassProofCarrying,
		Metric:      "tokens_per_sec",
		Before:      100.0,
		After:       150.0,
		LowerBetter: false,
		TruthClean:  true,
		SuiteGreen:  true,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkRecord = PromoteWithRecord(M4Correct, WitnessBenchCommitted, witness, nil)
	}
}

func BenchmarkDrop(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := Rungs[i%len(Rungs)]
		benchSinkRung = Drop(r, (i&1) == 1)
	}
}

func BenchmarkOptimizeCell(b *testing.B) {
	cell := OptimizeCell{
		Name:    "qwen3 x cuda",
		Current: M4Correct,
		Target:  M6Parity,
	}
	decisions := []shipgate.Decision{shipgate.KEEP, shipgate.REVERT}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		target, ok := cell.PromotionTarget()
		benchSinkRung = target
		benchSinkBool = ok
		promoted := cell.PromoteOnRun(decisions[i%len(decisions)])
		benchSinkAction = promoted.NextAction()
	}
}

func BenchmarkFeatureRung(b *testing.B) {
	features := FeatureRoster
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f := features[i%len(features)]
		r, tag, ok := FeatureRung(benchClaimsCorpus, f)
		benchSinkRung = r
		benchSinkClaimTag = tag
		benchSinkBool = ok
	}
}

func BenchmarkScoreFeatures(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkScoredFeatures = ScoreFeatures(benchClaimsCorpus)
	}
}
