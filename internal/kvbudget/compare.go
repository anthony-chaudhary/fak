package kvbudget

import "time"

type ComparisonArm struct {
	Name      string        `json:"name"`
	Kind      string        `json:"kind"`
	Available bool          `json:"available"`
	Correct   bool          `json:"correct"`
	Latency   time.Duration `json:"latency"`
	Bytes     float64       `json:"bytes"`
	CostUSD   float64       `json:"cost_usd"`
	Note      string        `json:"note,omitempty"`
}

type ComparisonResult struct {
	Workload string          `json:"workload"`
	Arms     []ComparisonArm `json:"arms"`
}

func CompareLocal() ComparisonResult {
	start := time.Now()
	native := GLM52DSA.KVBytesPerToken(F16)
	elapsed := time.Since(start)
	return ComparisonResult{
		Workload: "predict FP16 KV bytes per token for the declared GLM-5.2 DSA shape",
		Arms: []ComparisonArm{
			{Name: "fak native KV budget model", Kind: "native", Available: true, Correct: native == 129536, Latency: elapsed, Bytes: native},
			{Name: "full-MHA closed form", Kind: "baseline", Available: true, Correct: false, Bytes: float64(GLM52DSA.MHAElemsPerToken()) * F16.BytesPerElem, Note: "tuned conventional baseline ignores MLA/DSA compression"},
			{Name: "vLLM memory profiler", Kind: "external", Note: "requires the matching model, engine build, and GPU allocation trace"},
			{Name: "SGLang memory pool", Kind: "external", Note: "requires the matching model, engine build, and GPU allocation trace"},
			{Name: "NVIDIA GenAI-Perf", Kind: "external", Note: "requires a real serving deployment and GPU telemetry"},
		},
	}
}
