package canon

import (
	"bytes"
	"encoding/json"
	"time"
)

type UsageComparisonArm struct {
	Name              string
	Kind              string
	Available         bool
	Correct           bool
	Latency           time.Duration
	Cases             int
	CorrectCases      int
	RejectionErrors   int
	ClassErrors       int
	RawLosses         int
	InputBytes        int64
	RepresentedTokens int64
	CPUSeconds        float64
	PeakRSSBytes      int64
	NetworkBytes      int64
	ModelTokens       int64
	OperatorSeconds   float64
	CostUSD           float64
	Note              string
}
type UsageComparisonResult struct {
	Workload string
	Arms     []UsageComparisonArm
}
type usageCase struct {
	provider    string
	raw         json.RawMessage
	wantTotal   int64
	wantClasses int
	wantError   bool
	rawNeedle   string
}

var usageCases = []usageCase{
	{"openai", json.RawMessage(` {"input_tokens":100,"output_tokens":30,"input_tokens_details":{"cached_tokens":40},"output_tokens_details":{"reasoning_tokens":10},"future":"keep"} `), 130, 4, false, "future"},
	{"openai", json.RawMessage(`{"prompt_tokens":80,"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":30},"completion_tokens_details":{"reasoning_tokens":5}}`), 100, 4, false, "prompt_tokens"},
	{"anthropic", json.RawMessage(`{"input_tokens":20,"cache_read_input_tokens":50,"cache_creation_input_tokens":5,"output_tokens":11,"future":"keep"}`), 86, 4, false, "future"},
	{"local", json.RawMessage(`{"prompt_tokens":17,"completion_tokens":9,"native_extra":1}`), 26, 2, false, "native_extra"},
	{"openai", json.RawMessage(`{"input_tokens":`), 0, 0, true, ""},
	{"local", json.RawMessage(`{"prompt_tokens":-1,"completion_tokens":2}`), 0, 0, true, ""},
	{"openai", json.RawMessage(`{"input_tokens":2,"output_tokens":1,"input_tokens_details":{"cached_tokens":3}}`), 0, 0, true, ""},
}

func usageInputBytes() int64 {
	var n int64
	for _, tc := range usageCases {
		n += int64(len(tc.raw))
	}
	return n
}
func runNativeUsage() UsageComparisonArm {
	a := UsageComparisonArm{Name: "fak native canonical token-usage adapter", Kind: "native", Available: true, Cases: len(usageCases), InputBytes: usageInputBytes(), Note: "provider-specific class normalization, reconciliation refusals, and lossless raw provenance"}
	start := time.Now()
	for _, tc := range usageCases {
		got, err := AdaptTokenUsage(tc.provider, tc.raw)
		if (err != nil) != tc.wantError {
			a.RejectionErrors++
			continue
		}
		if tc.wantError {
			a.CorrectCases++
			continue
		}
		if got.Total != tc.wantTotal || len(got.Classes) != tc.wantClasses {
			a.ClassErrors++
			continue
		}
		if !bytes.Contains(got.Raw, []byte(tc.rawNeedle)) {
			a.RawLosses++
			continue
		}
		a.CorrectCases++
		a.RepresentedTokens += got.Total
	}
	a.Latency = time.Since(start)
	a.Correct = a.CorrectCases == len(usageCases) && a.RejectionErrors == 0 && a.ClassErrors == 0 && a.RawLosses == 0
	return a
}
func totalsOnly(tc usageCase) (int64, error) {
	var x map[string]any
	if err := json.Unmarshal(tc.raw, &x); err != nil {
		return 0, err
	}
	num := func(k string) int64 { v, _ := x[k].(float64); return int64(v) }
	switch tc.provider {
	case "openai":
		in, out := num("input_tokens"), num("output_tokens")
		if in == 0 {
			in = num("prompt_tokens")
		}
		if out == 0 {
			out = num("completion_tokens")
		}
		return in + out, nil
	case "anthropic":
		return num("input_tokens") + num("output_tokens"), nil
	default:
		return num("prompt_tokens") + num("completion_tokens"), nil
	}
}
func runTotalsUsage() UsageComparisonArm {
	a := UsageComparisonArm{Name: "provider total fields only", Kind: "baseline", Available: true, Cases: len(usageCases), InputBytes: usageInputBytes(), Note: "tuned no-normalization baseline parses top-level totals and discards detail classes/raw provenance"}
	start := time.Now()
	for _, tc := range usageCases {
		total, err := totalsOnly(tc)
		if tc.wantError {
			if err != nil {
				a.CorrectCases++
			} else {
				a.RejectionErrors++
			}
			continue
		}
		if err == nil && total == tc.wantTotal {
			a.CorrectCases++
		} else {
			a.ClassErrors++
		}
		a.RawLosses++
		a.RepresentedTokens += total
	}
	a.Latency = time.Since(start)
	a.Correct = a.CorrectCases == len(usageCases) && a.ClassErrors == 0 && a.RawLosses == 0 && a.RejectionErrors == 0
	return a
}
func CompareTokenUsageLocal() UsageComparisonResult {
	return UsageComparisonResult{Workload: "normalize seven ordered multi-provider usage cases with exact classes/totals, refusal behavior, and raw-provenance oracle", Arms: []UsageComparisonArm{runNativeUsage(), runTotalsUsage(), {Name: "fak + OpenAI", Kind: "integration", Note: "requires real OpenAI response usage"}, {Name: "fak + Anthropic", Kind: "integration", Note: "requires real Anthropic response usage"}, {Name: "fak + local provider", Kind: "integration", Note: "requires real local provider usage"}, {Name: "fak + OpenTelemetry", Kind: "integration", Note: "requires real GenAI telemetry export/read-back"}, {Name: "OpenAI SDK usage models", Kind: "external", Note: "requires pinned SDK parsing"}, {Name: "Anthropic SDK usage models", Kind: "external", Note: "requires pinned SDK parsing"}, {Name: "LiteLLM usage normalization", Kind: "external", Note: "requires real LiteLLM normalization"}, {Name: "OpenTelemetry GenAI semantic conventions", Kind: "external", Note: "requires real semantic-convention pipeline"}, {Name: "LangSmith token and cost tracking", Kind: "external", Note: "requires real trace ingestion and read-back"}}}
}
