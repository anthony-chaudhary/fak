package answershape

import (
	"strings"
	"time"
)

var compareTexts = []string{"The cache preserves shared setup while requests remain semantically aligned. Distinct turns still receive independent tool results.", strings.Repeat("refund payment now ", 30), strings.Repeat("ABCD", 80), strings.Repeat("same line\n", 35)}

type ComparisonArm struct {
	Name, Kind                                                          string
	Available, Correct                                                  bool
	Latency                                                             time.Duration
	Cases, TruePositives, TrueNegatives, FalsePositives, FalseNegatives int
	CPUSeconds                                                          float64
	PeakRSSBytes, InputBytes, NetworkBytes                              int64
	OperatorSeconds, CostUSD                                            float64
	Note                                                                string
}
type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

func nativeArm() ComparisonArm {
	a := ComparisonArm{Name: "fak native multi-signal answer degeneration detector", Kind: "native", Available: true, Cases: 4, Note: "n-gram, repeated-line, short-period, and compression signals with explicit reasons"}
	st := time.Now()
	for i, x := range compareTexts {
		a.InputBytes += int64(len(x))
		got := Measure([]byte(x), Limits{MaxRepeat: DefaultMaxRepeat, NGram: DefaultNGram}).Degenerate
		want := i > 0
		if got && want {
			a.TruePositives++
		} else if !got && !want {
			a.TrueNegatives++
		} else if got {
			a.FalsePositives++
		} else {
			a.FalseNegatives++
		}
	}
	a.Latency = time.Since(st)
	a.Correct = a.TruePositives == 3 && a.TrueNegatives == 1
	return a
}
func exactLineArm() ComparisonArm {
	a := ComparisonArm{Name: "exact repeated-line ratio", Kind: "baseline", Available: true, Cases: 4, Note: "tuned baseline catches line loops but misses token and short-period loops"}
	st := time.Now()
	for i, x := range compareTexts {
		lines := strings.Split(strings.TrimSpace(x), "\n")
		seen := map[string]bool{}
		dup := false
		for _, l := range lines {
			if seen[l] {
				dup = true
			}
			seen[l] = true
		}
		want := i > 0
		if dup && want {
			a.TruePositives++
		} else if !dup && !want {
			a.TrueNegatives++
		} else if dup {
			a.FalsePositives++
		} else {
			a.FalseNegatives++
		}
	}
	a.Latency = time.Since(st)
	return a
}
func ua(n, k, s string) ComparisonArm { return ComparisonArm{Name: n, Kind: k, Note: s} }
func CompareLocal() ComparisonResult {
	return ComparisonResult{Workload: "classify the same coherent response, phrase loop, short-period byte loop, and repeated-line loop without false positives", Arms: []ComparisonArm{nativeArm(), exactLineArm(), ua("fak + OpenAI response guard", "integration", "requires real OpenAI generation stream and independent labels"), ua("fak + Anthropic response guard", "integration", "requires real Anthropic generation stream and independent labels"), ua("llama.cpp repetition controls", "external", "requires pinned generation and equivalent degeneration corpus"), ua("vLLM repetition penalties", "external", "requires pinned vLLM generation"), ua("Hugging Face transformers repetition penalty", "external", "requires pinned transformers generation"), ua("NeMo Guardrails output rail", "external", "requires equivalent output rail")}}
}
