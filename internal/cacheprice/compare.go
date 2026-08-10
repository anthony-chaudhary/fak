package cacheprice

import "time"

// ComparisonArm is one independently reportable cache-pricing arm.
type ComparisonArm struct {
	Name      string        `json:"name"`
	Kind      string        `json:"kind"`
	Available bool          `json:"available"`
	Correct   bool          `json:"correct"`
	Latency   time.Duration `json:"latency"`
	Tokens    int           `json:"tokens"`
	Bytes     int64         `json:"bytes"`
	CostUSD   float64       `json:"cost_usd"`
	Note      string        `json:"note,omitempty"`
}

// ComparisonResult prices one provider-observed prompt and resident prefix.
type ComparisonResult struct {
	Workload string          `json:"workload"`
	Arms     []ComparisonArm `json:"arms"`
}

// CompareLocal executes the native arithmetic and full-prompt baseline. Cloud
// pricing calculators remain unavailable until real billed traces are supplied.
func CompareLocal() ComparisonResult {
	const promptTokens = 4096
	const residentPrefixTokens = 3072
	start := time.Now()
	native := AdmissionTokens(promptTokens, residentPrefixTokens)
	elapsed := time.Since(start)
	return ComparisonResult{
		Workload: "price a 4,096-token prompt with 3,072 provider-observed resident prefix tokens",
		Arms: []ComparisonArm{
			{Name: "fak native cache pricing", Kind: "native", Available: true, Correct: native == 1024, Latency: elapsed, Tokens: native},
			{Name: "charge full prompt", Kind: "baseline", Available: true, Correct: false, Tokens: promptTokens, Note: "zero-metadata baseline ignores witnessed resident prefix"},
			{Name: "AWS Pricing Calculator", Kind: "external", Note: "requires matching live service prices and billed cache telemetry"},
			{Name: "Google Cloud Pricing Calculator", Kind: "external", Note: "requires matching live service prices and billed cache telemetry"},
			{Name: "Azure Pricing Calculator", Kind: "external", Note: "requires matching live service prices and billed cache telemetry"},
		},
	}
}
