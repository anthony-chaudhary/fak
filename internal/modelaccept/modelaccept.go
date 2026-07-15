package modelaccept

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

const Schema = "fak.modelaccept.report/1"

type Verdict string

const (
	Pass Verdict = "PASS"
	Hold Verdict = "HOLD"
)

type Task struct {
	ID               string `json:"id"`
	Class            string `json:"class,omitempty"`
	Tier             int    `json:"tier"`
	Repetitions      int    `json:"repetitions"`
	Prompt           string `json:"prompt,omitempty"`
	Expected         string `json:"expected"`
	ResultMatch      string `json:"result_match,omitempty"`
	ToolRequired     bool   `json:"tool_required,omitempty"`
	MinToolCalls     int    `json:"min_tool_calls,omitempty"`
	ExpectedRefusal  string `json:"expected_refusal,omitempty"`
	RetryRequired    bool   `json:"retry_required,omitempty"`
	RecoveryRequired bool   `json:"recovery_required,omitempty"`
}

type Thresholds struct {
	MinSuccessRate        float64 `json:"min_success_rate"`
	MaxP95LatencyMS       int64   `json:"max_p95_latency_ms"`
	MaxProviderErrorRate  float64 `json:"max_provider_error_rate"`
	MaxInvalidToolRate    float64 `json:"max_invalid_tool_rate"`
	MaxAverageInputTokens float64 `json:"max_average_input_tokens"`
	MaxAverageCostUSD     float64 `json:"max_average_cost_usd"`
}

type Corpus struct {
	ID               string     `json:"id"`
	DeclaredAt       string     `json:"declared_at"`
	Tasks            []Task     `json:"tasks"`
	Thresholds       Thresholds `json:"thresholds"`
	ReplacementRules string     `json:"replacement_rules,omitempty"`
	StoppingRule     string     `json:"stopping_rule,omitempty"`
}

type Run struct {
	Model         string  `json:"model"`
	ActualModel   string  `json:"actual_model"`
	Task          string  `json:"task"`
	Repetition    int     `json:"repetition"`
	Result        string  `json:"result"`
	ProviderError bool    `json:"provider_error"`
	ToolValid     bool    `json:"tool_valid"`
	ToolCalls     int     `json:"tool_calls,omitempty"`
	Refusal       string  `json:"refusal,omitempty"`
	RetryCount    int     `json:"retry_count,omitempty"`
	Recovered     bool    `json:"recovered,omitempty"`
	LatencyMS     int64   `json:"latency_ms"`
	InputTokens   int64   `json:"input_tokens"`
	CostUSD       float64 `json:"cost_usd"`
	ObservedAt    string  `json:"observed_at"`
	FailureClass  string  `json:"failure_class,omitempty"`
	FailureDetail string  `json:"failure_detail,omitempty"`
}

type ModelRequest struct {
	Model         string `json:"model"`
	RequestedTier int    `json:"requested_tier"`
}

type Input struct {
	Schema string         `json:"schema"`
	Corpus Corpus         `json:"corpus"`
	Models []ModelRequest `json:"models"`
	Runs   []Run          `json:"runs"`
}

type ModelDecision struct {
	Model              string   `json:"model"`
	RequestedTier      int      `json:"requested_tier"`
	Verdict            Verdict  `json:"verdict"`
	Samples            int      `json:"samples"`
	SuccessRate        float64  `json:"success_rate"`
	ProviderErrorRate  float64  `json:"provider_error_rate"`
	InvalidToolRate    float64  `json:"invalid_tool_rate"`
	P50LatencyMS       int64    `json:"p50_latency_ms"`
	P95LatencyMS       int64    `json:"p95_latency_ms"`
	P95InputTokens     int64    `json:"p95_input_tokens"`
	P95CostUSD         float64  `json:"p95_cost_usd"`
	RefusalRate        float64  `json:"refusal_rate"`
	RetryRate          float64  `json:"retry_rate"`
	RecoveryRate       float64  `json:"recovery_rate"`
	AverageInputTokens float64  `json:"average_input_tokens"`
	AverageCostUSD     float64  `json:"average_cost_usd"`
	Reasons            []string `json:"reasons"`
}

type Decision struct {
	Schema   string          `json:"schema"`
	Verdict  Verdict         `json:"verdict"`
	CorpusID string          `json:"corpus_id"`
	Models   []ModelDecision `json:"models"`
	Reasons  []string        `json:"reasons,omitempty"`
}

func Evaluate(in Input) Decision {
	out := Decision{Schema: Schema, Verdict: Pass, CorpusID: in.Corpus.ID}
	if reasons := validate(in); len(reasons) != 0 {
		out.Verdict, out.Reasons = Hold, reasons
		return out
	}
	tasks := map[string]Task{}
	for _, t := range in.Corpus.Tasks {
		tasks[t.ID] = t
	}
	for _, req := range in.Models {
		d := ModelDecision{Model: req.Model, RequestedTier: req.RequestedTier, Verdict: Pass}
		var rr []Run
		for _, r := range in.Runs {
			task, known := tasks[r.Task]
			if r.Model == req.Model && known && task.Tier >= req.RequestedTier {
				rr = append(rr, r)
			}
		}
		d.Samples = len(rr)
		expectedSamples := 0
		for _, t := range in.Corpus.Tasks {
			if t.Tier >= req.RequestedTier {
				expectedSamples += t.Repetitions
			}
		}
		seen := map[string]bool{}
		successes, providerErrors, invalidTools := 0, 0, 0
		var lat, tokenSamples []int64
		var costSamples []float64
		var tokens int64
		var cost float64
		refusals, retries, recoveries := 0, 0, 0
		for _, r := range rr {
			key := fmt.Sprintf("%s/%d", r.Task, r.Repetition)
			if seen[key] {
				d.Reasons = append(d.Reasons, "duplicate run: "+key)
			}
			seen[key] = true
			t, ok := tasks[r.Task]
			if !ok {
				d.Reasons = append(d.Reasons, "unknown task: "+r.Task)
				continue
			}
			if r.ActualModel != req.Model {
				d.Reasons = append(d.Reasons, fmt.Sprintf("wrong actual model for %s: %s", key, r.ActualModel))
			}
			if r.ProviderError {
				providerErrors++
			}
			toolOK := !t.ToolRequired || (r.ToolValid && (t.MinToolCalls == 0 || r.ToolCalls >= t.MinToolCalls))
			if !toolOK {
				invalidTools++
				d.Reasons = append(d.Reasons, "tool behavior mismatch: "+key)
			}
			refusalOK := r.Refusal == t.ExpectedRefusal
			if !refusalOK {
				d.Reasons = append(d.Reasons, "refusal mismatch: "+key)
			}
			retryOK := !t.RetryRequired || r.RetryCount > 0
			if !retryOK {
				d.Reasons = append(d.Reasons, "required retry missing: "+key)
			}
			recoveryOK := !t.RecoveryRequired || (r.RetryCount > 0 && r.Recovered)
			if !recoveryOK {
				d.Reasons = append(d.Reasons, "required recovery missing: "+key)
			}
			if r.Refusal != "" {
				refusals++
			}
			if r.RetryCount > 0 {
				retries++
			}
			if r.Recovered {
				recoveries++
			}
			if !r.ProviderError && r.ActualModel == req.Model && ResultMatches(t, r.Result) && toolOK && refusalOK && retryOK && recoveryOK {
				successes++
			}
			lat = append(lat, r.LatencyMS)
			tokenSamples = append(tokenSamples, r.InputTokens)
			costSamples = append(costSamples, r.CostUSD)
			tokens += r.InputTokens
			cost += r.CostUSD
		}
		for _, t := range in.Corpus.Tasks {
			if t.Tier >= req.RequestedTier {
				for i := 1; i <= t.Repetitions; i++ {
					key := fmt.Sprintf("%s/%d", t.ID, i)
					if !seen[key] {
						d.Reasons = append(d.Reasons, "missing run: "+key)
					}
				}
			}
		}
		if len(rr) > 0 {
			n := float64(len(rr))
			d.SuccessRate = float64(successes) / n
			d.ProviderErrorRate = float64(providerErrors) / n
			d.InvalidToolRate = float64(invalidTools) / n
			d.RefusalRate = float64(refusals) / n
			d.RetryRate = float64(retries) / n
			d.RecoveryRate = float64(recoveries) / n
			d.AverageInputTokens = float64(tokens) / n
			d.AverageCostUSD = cost / n
			sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
			sort.Slice(tokenSamples, func(i, j int) bool { return tokenSamples[i] < tokenSamples[j] })
			sort.Float64s(costSamples)
			d.P50LatencyMS = lat[(50*len(lat)-1)/100]
			d.P95LatencyMS = lat[(95*len(lat)-1)/100]
			d.P95InputTokens = tokenSamples[(95*len(tokenSamples)-1)/100]
			d.P95CostUSD = costSamples[(95*len(costSamples)-1)/100]
		}
		th := in.Corpus.Thresholds
		if len(rr) != expectedSamples {
			d.Reasons = append(d.Reasons, fmt.Sprintf("samples %d != required %d", len(rr), expectedSamples))
		}
		if d.SuccessRate < th.MinSuccessRate {
			d.Reasons = append(d.Reasons, fmt.Sprintf("success_rate %.4f < %.4f", d.SuccessRate, th.MinSuccessRate))
		}
		if d.ProviderErrorRate > th.MaxProviderErrorRate {
			d.Reasons = append(d.Reasons, "provider_error_rate exceeds threshold")
		}
		if d.InvalidToolRate > th.MaxInvalidToolRate {
			d.Reasons = append(d.Reasons, "invalid_tool_rate exceeds threshold")
		}
		if d.P95LatencyMS > th.MaxP95LatencyMS {
			d.Reasons = append(d.Reasons, "p95_latency_ms exceeds threshold")
		}
		if d.AverageInputTokens > th.MaxAverageInputTokens {
			d.Reasons = append(d.Reasons, "average_input_tokens exceeds threshold")
		}
		if d.AverageCostUSD > th.MaxAverageCostUSD {
			d.Reasons = append(d.Reasons, "average_cost_usd exceeds threshold")
		}
		if len(d.Reasons) > 0 {
			d.Verdict = Hold
			out.Verdict = Hold
		}
		sort.Strings(d.Reasons)
		out.Models = append(out.Models, d)
	}
	return out
}

func validate(in Input) []string {
	var r []string
	if in.Schema != Schema {
		r = append(r, "schema must be "+Schema)
	}
	if strings.TrimSpace(in.Corpus.ID) == "" {
		r = append(r, "corpus id is required")
	}
	if strings.TrimSpace(in.Corpus.DeclaredAt) == "" {
		r = append(r, "corpus declared_at is required")
	}
	if len(in.Models) == 0 {
		r = append(r, "at least one model is required")
	}
	declaredAt, declaredErr := time.Parse(time.RFC3339, in.Corpus.DeclaredAt)
	if declaredErr != nil {
		r = append(r, "corpus declared_at must be RFC3339")
	}
	declaredModels := map[string]bool{}
	for _, model := range in.Models {
		if strings.TrimSpace(model.Model) == "" {
			r = append(r, "every model needs an exact non-empty id")
		}
		if model.RequestedTier < 0 {
			r = append(r, "every model needs a non-negative requested tier")
		}
		if declaredModels[model.Model] {
			r = append(r, "duplicate model: "+model.Model)
		}
		declaredModels[model.Model] = true
	}
	seen := map[string]bool{}
	structured := false
	classes := map[string]bool{}
	for _, t := range in.Corpus.Tasks {
		if t.ID == "" || t.Repetitions <= 0 || t.Tier < 0 {
			r = append(r, "every task needs id, positive repetitions, and non-negative tier")
		}
		if t.ResultMatch != "" && t.ResultMatch != "exact" && t.ResultMatch != "sentinel_line" {
			r = append(r, "task "+t.ID+" has invalid result_match")
		}
		if t.ResultMatch == "sentinel_line" {
			structured = true
			classes[t.Class] = true
			if strings.ContainsAny(t.Expected, "\r\n") || strings.TrimSpace(t.Expected) != t.Expected || t.Expected == "" {
				r = append(r, "task "+t.ID+" sentinel must be one non-empty, unpadded line")
			}
			if strings.TrimSpace(t.Class) == "" {
				r = append(r, "task "+t.ID+" structured campaign class is required")
			}
		}
		if t.MinToolCalls < 0 || (t.MinToolCalls > 0 && !t.ToolRequired) {
			r = append(r, "task "+t.ID+" has invalid minimum tool calls")
		}
		if t.ExpectedRefusal != "" && t.ExpectedRefusal != "policy" && t.ExpectedRefusal != "safety" {
			r = append(r, "task "+t.ID+" has invalid expected refusal")
		}
		if t.RecoveryRequired && !t.RetryRequired {
			r = append(r, "task "+t.ID+" requires recovery without retry")
		}
		if seen[t.ID] {
			r = append(r, "duplicate task: "+t.ID)
		}
		seen[t.ID] = true
	}
	if structured {
		if strings.TrimSpace(in.Corpus.ReplacementRules) == "" {
			r = append(r, "structured campaign replacement_rules are required")
		}
		if strings.TrimSpace(in.Corpus.StoppingRule) == "" {
			r = append(r, "structured campaign stopping_rule is required")
		}
		delete(classes, "")
		if len(classes) < 2 {
			r = append(r, "structured campaign needs at least two task classes")
		}
	}
	for _, run := range in.Runs {
		if run.FailureClass != "" && run.FailureClass != "capability" && run.FailureClass != "policy_refusal" && run.FailureClass != "provider_infrastructure" && run.FailureClass != "harness" {
			r = append(r, fmt.Sprintf("run %s/%s/%d has invalid failure class", run.Model, run.Task, run.Repetition))
		}
		if run.ToolCalls < 0 || run.RetryCount < 0 {
			r = append(r, fmt.Sprintf("run %s/%s/%d has negative behavior count", run.Model, run.Task, run.Repetition))
		}
		if run.Refusal != "" && run.Refusal != "policy" && run.Refusal != "safety" {
			r = append(r, fmt.Sprintf("run %s/%s/%d has invalid refusal class", run.Model, run.Task, run.Repetition))
		}
		if run.Recovered && run.RetryCount == 0 {
			r = append(r, fmt.Sprintf("run %s/%s/%d recovered without retry", run.Model, run.Task, run.Repetition))
		}
		if !declaredModels[run.Model] {
			r = append(r, "run uses undeclared model: "+run.Model)
		}
		if !seen[run.Task] {
			r = append(r, "unknown run task: "+run.Task)
		}
		observedAt, err := time.Parse(time.RFC3339, run.ObservedAt)
		if err != nil {
			r = append(r, fmt.Sprintf("run %s/%s/%d observed_at must be RFC3339", run.Model, run.Task, run.Repetition))
		} else if declaredErr == nil && observedAt.Before(declaredAt) {
			r = append(r, fmt.Sprintf("run %s/%s/%d predates corpus declaration", run.Model, run.Task, run.Repetition))
		}
	}
	th := in.Corpus.Thresholds
	rates := []float64{th.MinSuccessRate, th.MaxProviderErrorRate, th.MaxInvalidToolRate}
	for _, v := range rates {
		if v < 0 || v > 1 {
			r = append(r, "rate thresholds must be within [0,1]")
			break
		}
	}
	if th.MaxP95LatencyMS <= 0 || th.MaxAverageInputTokens <= 0 || th.MaxAverageCostUSD <= 0 {
		r = append(r, "latency, token, and cost thresholds must be positive")
	}
	sort.Strings(r)
	return r
}

// ResultMatches applies the task's predeclared output contract. sentinel_line
// matches only a complete line after CRLF normalization; it deliberately does
// not accept an embedded substring or padded line.
func ResultMatches(task Task, result string) bool {
	if task.ResultMatch == "sentinel_line" {
		result = strings.ReplaceAll(result, "\r\n", "\n")
		result = strings.ReplaceAll(result, "\r", "\n")
		for _, line := range strings.Split(result, "\n") {
			if line == task.Expected {
				return true
			}
		}
		return false
	}
	return result == task.Expected
}
