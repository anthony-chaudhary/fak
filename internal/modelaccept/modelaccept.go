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
	ID           string `json:"id"`
	Tier         int    `json:"tier"`
	Repetitions  int    `json:"repetitions"`
	Expected     string `json:"expected"`
	ToolRequired bool   `json:"tool_required,omitempty"`
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
	ID         string     `json:"id"`
	DeclaredAt string     `json:"declared_at"`
	Tasks      []Task     `json:"tasks"`
	Thresholds Thresholds `json:"thresholds"`
}

type Run struct {
	Model         string  `json:"model"`
	ActualModel   string  `json:"actual_model"`
	Task          string  `json:"task"`
	Repetition    int     `json:"repetition"`
	Result        string  `json:"result"`
	ProviderError bool    `json:"provider_error"`
	ToolValid     bool    `json:"tool_valid"`
	LatencyMS     int64   `json:"latency_ms"`
	InputTokens   int64   `json:"input_tokens"`
	CostUSD       float64 `json:"cost_usd"`
	ObservedAt    string  `json:"observed_at"`
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
	P95LatencyMS       int64    `json:"p95_latency_ms"`
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
		var lat []int64
		var tokens int64
		var cost float64
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
			if t.ToolRequired && !r.ToolValid {
				invalidTools++
			}
			if !r.ProviderError && r.ActualModel == req.Model && r.Result == t.Expected && (!t.ToolRequired || r.ToolValid) {
				successes++
			}
			lat = append(lat, r.LatencyMS)
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
			d.AverageInputTokens = float64(tokens) / n
			d.AverageCostUSD = cost / n
			sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
			d.P95LatencyMS = lat[(95*len(lat)-1)/100]
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
	seen := map[string]bool{}
	for _, t := range in.Corpus.Tasks {
		if t.ID == "" || t.Repetitions <= 0 || t.Tier < 0 {
			r = append(r, "every task needs id, positive repetitions, and non-negative tier")
		}
		if seen[t.ID] {
			r = append(r, "duplicate task: "+t.ID)
		}
		seen[t.ID] = true
	}
	for _, run := range in.Runs {
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
