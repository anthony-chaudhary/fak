package modelroute

import (
	"fmt"
	"runtime"
	"time"
)

// ComparisonSchema identifies the stable, machine-readable local routing report.
const ComparisonSchema = "fak-model-routing-comparison/1"

// ComparisonCase is one routing decision in a frozen same-workload corpus.
type ComparisonCase struct {
	Name          string  `json:"name"`
	Subject       Subject `json:"subject"`
	ExpectedModel string  `json:"expected_model"`
}

// ComparisonArm reports one required arm without turning absence into success.
type ComparisonArm struct {
	Name               string  `json:"name"`
	Class              string  `json:"class"`
	Available          bool    `json:"available"`
	Decisions          int     `json:"decisions,omitempty"`
	ExpectedSelections int     `json:"expected_selections,omitempty"`
	SelectionAccuracy  float64 `json:"selection_accuracy,omitempty"`
	ElapsedNS          int64   `json:"elapsed_ns,omitempty"`
	NanosecondsEach    float64 `json:"elapsed_per_item_ns,omitempty"`
	TaskSuccess        float64 `json:"task_success,omitempty"`
	InputTokens        int64   `json:"input_tokens,omitempty"`
	OutputTokens       int64   `json:"output_tokens,omitempty"`
	PeakRSSBytes       int64   `json:"peak_rss_bytes,omitempty"`
	TotalCostUSD       float64 `json:"total_cost_usd,omitempty"`
	UnavailableReason  string  `json:"unavailable_reason,omitempty"`
}

// ComparisonReport is intentionally incomplete until every process-level arm has
// run the identical prompts/models and supplied quality, token, RSS, and cost data.
type ComparisonReport struct {
	Schema   string          `json:"schema"`
	Workload string          `json:"workload"`
	GOOS     string          `json:"goos"`
	GOARCH   string          `json:"goarch"`
	Arms     []ComparisonArm `json:"arms"`
	Complete bool            `json:"complete"`
	Verdict  string          `json:"verdict"`
}

// ComparisonCorpus returns a small deterministic decision corpus. It witnesses
// local policy selection only; it is not a substitute for inference quality.
func ComparisonCorpus() []ComparisonCase {
	return []ComparisonCase{
		{Name: "refund", Subject: Subject{Aspect: AspectToolCall, Tool: "refund_payment"}, ExpectedModel: "strong"},
		{Name: "search", Subject: Subject{Aspect: AspectToolCall, Tool: "search_kb"}, ExpectedModel: "small"},
		{Name: "long-query", Subject: Subject{Aspect: AspectQuery, PromptTokens: 9000}, ExpectedModel: "long-context"},
		{Name: "short-query", Subject: Subject{Aspect: AspectQuery, PromptTokens: 300}, ExpectedModel: "small"},
		{Name: "hard-step", Subject: Subject{Aspect: AspectStep, Complexity: ComplexityHigh}, ExpectedModel: "strong"},
		{Name: "fast-step", Subject: Subject{Aspect: AspectStep, Latency: LatencyInteractive}, ExpectedModel: "small"},
	}
}

func comparisonManifest() Manifest {
	return Manifest{
		Version: Version,
		Default: Plan{Members: []Member{{Model: "small"}}},
		Rules: []Rule{
			{Name: "refund", Match: Match{Aspect: AspectToolCall, Tool: "refund_payment"}, Plan: Plan{Members: []Member{{Model: "strong"}}}},
			{Name: "search", Match: Match{Aspect: AspectToolCall, Tool: "search_*"}, Plan: Plan{Members: []Member{{Model: "small"}}}},
			{Name: "long", Match: Match{Aspect: AspectQuery, MinPromptTokens: 8000}, Plan: Plan{Members: []Member{{Model: "long-context"}}}},
			{Name: "hard", Match: Match{Aspect: AspectStep, MinComplexity: ComplexityHigh}, Plan: Plan{Members: []Member{{Model: "strong"}}}},
			{Name: "fast", Match: Match{Aspect: AspectStep, Latency: LatencyInteractive}, Plan: Plan{Members: []Member{{Model: "small"}}}},
		},
	}
}

// CompareLocal runs fak-native routing and a tuned fixed-strong-model baseline
// over the exact same decision corpus. External process arms remain explicit and
// unavailable rather than being represented by mocks or adapter registration.
func CompareLocal(iterations int) (ComparisonReport, error) {
	if iterations <= 0 {
		iterations = 10000
	}
	corpus := ComparisonCorpus()
	manifest := comparisonManifest()
	if err := manifest.Validate(); err != nil {
		return ComparisonReport{}, fmt.Errorf("comparison manifest: %w", err)
	}

	nativeCorrect := 0
	start := time.Now()
	for i := 0; i < iterations; i++ {
		for _, c := range corpus {
			if manifest.Route(c.Subject).Plan.Primary() == c.ExpectedModel {
				nativeCorrect++
			}
		}
	}
	nativeElapsed := time.Since(start)

	baselineCorrect := 0
	start = time.Now()
	for i := 0; i < iterations; i++ {
		for _, c := range corpus {
			if "strong" == c.ExpectedModel {
				baselineCorrect++
			}
		}
	}
	baselineElapsed := time.Since(start)
	total := iterations * len(corpus)

	arms := []ComparisonArm{
		localArm("fak native aspect-aware manifest", "native", total, nativeCorrect, nativeElapsed),
		localArm("fixed strongest model", "tuned_baseline", total, baselineCorrect, baselineElapsed),
		unavailableArm("RouteLLM", "next_best", "requires a frozen RouteLLM checkpoint, identical candidate models, prompts, and independently graded responses"),
		unavailableArm("fak + LiteLLM Router", "first_class_integration", "requires a live LiteLLM router using the identical candidate models and workload"),
		unavailableArm("fak + OpenRouter routing", "first_class_integration", "requires live OpenRouter routing using the identical candidate models and workload"),
		unavailableArm("fak + Portkey router", "first_class_integration", "requires a live Portkey router using the identical candidate models and workload"),
	}
	return ComparisonReport{
		Schema: ComparisonSchema, GOOS: runtime.GOOS, GOARCH: runtime.GOARCH,
		Workload: "frozen local decision corpus; process-level completion uses identical prompts, candidate models, quality grader, concurrency, and warmup",
		Arms:     arms, Complete: false,
		Verdict: "local selection overhead and fixture accuracy only; no net-true routing winner until all live arms report task quality, latency, tokens/resources, and cost",
	}, nil
}

func localArm(name, class string, total, correct int, elapsed time.Duration) ComparisonArm {
	return ComparisonArm{Name: name, Class: class, Available: true, Decisions: total,
		ExpectedSelections: correct, SelectionAccuracy: float64(correct) / float64(total),
		ElapsedNS: elapsed.Nanoseconds(), NanosecondsEach: float64(elapsed.Nanoseconds()) / float64(total)}
}

func unavailableArm(name, class, reason string) ComparisonArm {
	return ComparisonArm{Name: name, Class: class, Available: false, UnavailableReason: reason}
}
