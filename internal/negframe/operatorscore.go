package negframe

import "github.com/anthony-chaudhary/fak/pkg/scorecard"

const NegationOperatorSchema = "fak-negation-operator-scorecard/1"
const NegationOperatorDebtKey = "negation_operator_debt"

// NegationOperatorSignals is the stable, machine-readable health input. BenchmarkDelta
// is operator-minus-baseline in percentage points; UnknownFallbackRate is [0,1].
type NegationOperatorSignals struct {
	BenchmarkDelta      float64 `json:"benchmark_delta"`
	DomainCoverage      float64 `json:"enumerable_domain_coverage"`
	UnknownFallbackRate float64 `json:"unknown_fallback_rate"`
	EnumerableDomains   int     `json:"enumerable_domains"`
	HandledDomains      int     `json:"handled_domains"`
}

// MeasureNegationOperator derives deterministic structural signals. Benchmark observations
// are record-then-view data and therefore remain zero until a witnessed baseline is supplied.
func MeasureNegationOperator(benchmarkDelta, unknownFallbackRate float64) NegationOperatorSignals {
	domains := Domains()
	handled := 0
	for _, domain := range domains {
		valid := len(domain.Members) > 1 && len(domain.Members) <= MaxPositiveDomainMembers
		if valid {
			for _, member := range domain.Members {
				if !ResolvePositive("not "+member, domain).Exact {
					valid = false
					break
				}
			}
		}
		if valid {
			handled++
		}
	}
	coverage := 0.0
	if len(domains) > 0 {
		coverage = float64(handled) / float64(len(domains))
	}
	if unknownFallbackRate < 0 {
		unknownFallbackRate = 0
	}
	if unknownFallbackRate > 1 {
		unknownFallbackRate = 1
	}
	return NegationOperatorSignals{BenchmarkDelta: benchmarkDelta, DomainCoverage: coverage, UnknownFallbackRate: unknownFallbackRate, EnumerableDomains: len(domains), HandledDomains: handled}
}

func BuildNegationOperatorScore(signals NegationOperatorSignals) scorecard.Payload {
	debt := signals.EnumerableDomains - signals.HandledDomains
	if signals.BenchmarkDelta < 0 {
		debt++
	}
	kpi := scorecard.KPI{Key: "negation_operator", Group: "cards", Score: 100, Detail: "operator structural coverage and witnessed benchmark delta"}
	if debt > 0 {
		kpi.Score = 0
		kpi.Defects = []string{"negation operator health has a regression"}
	}
	return scorecard.Fold(NegationOperatorSchema, []scorecard.KPI{kpi}, NegationOperatorDebtKey, nil, scorecard.Messages{
		Grade: scorecard.GradeStrict, Finding: "negation operator regression observed", FindingClean: "negation operator structural health holds", NextAction: "record a clean operator witness", NextActionClean: "continue record-then-view soak",
		ExtraCorpus: map[string]any{NegationOperatorDebtKey: debt, "benchmark_delta": signals.BenchmarkDelta, "enumerable_domain_coverage": signals.DomainCoverage, "unknown_fallback_rate": signals.UnknownFallbackRate, "enumerable_domains": signals.EnumerableDomains, "handled_domains": signals.HandledDomains, "family": "Cards", "sentinel": "deferred"},
	})
}
