package livecodebench

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
)

func rawArmProblems(n int) []Problem {
	ps := make([]Problem, n)
	for i := range ps {
		ps[i] = Problem{QuestionID: fmt.Sprintf("q%d", i), Scenario: ScenarioCodeGeneration, Prompt: "solve it"}
	}
	return ps
}

func TestRunRawArmCollectsNSamplesPerProblemInOrder(t *testing.T) {
	probs := rawArmProblems(3)
	cfg := RawArmConfig{Model: "glm-4.6", Endpoint: "http://127.0.0.1:8080/v1", N: 2, Temperature: 0.8, Concurrency: 4}

	sample := func(_ context.Context, p Problem, i int) (string, RawSampleUsage, error) {
		return fmt.Sprintf("%s#%d", p.QuestionID, i), RawSampleUsage{}, nil
	}
	rep, err := RunRawArm(context.Background(), cfg, probs, sample)
	if err != nil {
		t.Fatalf("RunRawArm: %v", err)
	}
	if rep.Arm != "raw" || rep.Model != "glm-4.6" || rep.Endpoint != "http://127.0.0.1:8080/v1" {
		t.Fatalf("run identity not recorded: %+v", rep)
	}
	if rep.N != 2 || rep.Temperature != 0.8 || rep.Concurrency != 4 {
		t.Fatalf("n/temperature/concurrency not recorded: %+v", rep)
	}
	if len(rep.Problems) != 3 {
		t.Fatalf("want 3 problems, got %d", len(rep.Problems))
	}
	for pi, p := range rep.Problems {
		if p.QuestionID != fmt.Sprintf("q%d", pi) {
			t.Fatalf("problem %d out of order: %q", pi, p.QuestionID)
		}
		want := []string{fmt.Sprintf("q%d#0", pi), fmt.Sprintf("q%d#1", pi)}
		if len(p.Completions) != 2 || p.Completions[0] != want[0] || p.Completions[1] != want[1] {
			t.Fatalf("problem %d completions = %v, want %v", pi, p.Completions, want)
		}
	}
	if rep.Usage.Samples != 6 {
		t.Fatalf("want 6 samples, got %d", rep.Usage.Samples)
	}
}

func TestRunRawArmFoldsCachedPromptTokensNormalized(t *testing.T) {
	probs := rawArmProblems(1)
	cfg := RawArmConfig{Model: "m", Endpoint: "e", N: 3, Concurrency: 2}

	// Per-sample usage arrives already normalized by the sampler (the gateway
	// sampler folds each provider shape through agent.Usage.CachedPromptTokens()
	// before returning — witnessed end-to-end in cmd/livecodebench raw_test.go).
	usages := []RawSampleUsage{
		{PromptTokens: 100, CompletionTokens: 10, CachedPromptTokens: 40},
		{PromptTokens: 100, CompletionTokens: 10, CachedPromptTokens: 30},
		{PromptTokens: 100, CompletionTokens: 10, CachedPromptTokens: 25},
	}
	var idx int32 = -1
	sample := func(_ context.Context, _ Problem, _ int) (string, RawSampleUsage, error) {
		u := usages[atomic.AddInt32(&idx, 1)]
		return "code", u, nil
	}
	rep, err := RunRawArm(context.Background(), cfg, probs, sample)
	if err != nil {
		t.Fatalf("RunRawArm: %v", err)
	}
	if rep.Usage.PromptTokens != 300 || rep.Usage.CompletionTokens != 30 {
		t.Fatalf("token totals wrong: %+v", rep.Usage)
	}
	if rep.Usage.CachedPromptTokens != 40+30+25 {
		t.Fatalf("cached prompt tokens = %d, want %d (normalized across provider shapes)", rep.Usage.CachedPromptTokens, 40+30+25)
	}
}

func TestRunRawArmRespectsConcurrencyCap(t *testing.T) {
	probs := rawArmProblems(8)
	cfg := RawArmConfig{Model: "m", Endpoint: "e", N: 4, Concurrency: 3}

	var mu sync.Mutex
	var inFlight, maxSeen int
	sample := func(_ context.Context, p Problem, i int) (string, RawSampleUsage, error) {
		mu.Lock()
		inFlight++
		if inFlight > maxSeen {
			maxSeen = inFlight
		}
		mu.Unlock()
		for k := 0; k < 5000; k++ {
			_ = k
		}
		mu.Lock()
		inFlight--
		mu.Unlock()
		return "x", RawSampleUsage{}, nil
	}
	if _, err := RunRawArm(context.Background(), cfg, probs, sample); err != nil {
		t.Fatalf("RunRawArm: %v", err)
	}
	if maxSeen > 3 {
		t.Fatalf("concurrency cap breached: saw %d in flight, cap 3", maxSeen)
	}
}

func TestRunRawArmSamplerErrorAborts(t *testing.T) {
	probs := rawArmProblems(4)
	cfg := RawArmConfig{Model: "m", Endpoint: "e", N: 2, Concurrency: 2}
	boom := errors.New("gateway 500")
	sample := func(_ context.Context, p Problem, i int) (string, RawSampleUsage, error) {
		if p.QuestionID == "q1" {
			return "", RawSampleUsage{}, boom
		}
		return "ok", RawSampleUsage{}, nil
	}
	_, err := RunRawArm(context.Background(), cfg, probs, sample)
	if err == nil || !errors.Is(err, boom) {
		t.Fatalf("want wrapped sampler error, got %v", err)
	}
}

func TestRunRawArmRejectsBadConfig(t *testing.T) {
	if _, err := RunRawArm(context.Background(), RawArmConfig{N: 0}, nil, func(context.Context, Problem, int) (string, RawSampleUsage, error) { return "", RawSampleUsage{}, nil }); err == nil {
		t.Fatalf("want error for n<1")
	}
	if _, err := RunRawArm(context.Background(), RawArmConfig{N: 1}, nil, nil); err == nil {
		t.Fatalf("want error for nil sampler")
	}
}

func TestRunRawArmCountsTruncatedAndReasoningOnly(t *testing.T) {
	report, err := RunRawArm(context.Background(), RawArmConfig{Model: "m", Endpoint: "e", N: 1}, []Problem{{QuestionID: "q", Prompt: "p"}}, func(context.Context, Problem, int) (string, RawSampleUsage, error) {
		return "reasoning", RawSampleUsage{FinishReason: "length", ReasoningOnly: true}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if report.Usage.Truncated != 1 || report.Usage.ReasoningOnly != 1 {
		t.Fatalf("usage=%+v", report.Usage)
	}
	if report.ResultClaimAllowed {
		t.Fatal("truncated report must not allow a result claim")
	}
	problem := report.Problems[0]
	if len(problem.FinishReasons) != 1 || problem.FinishReasons[0] != "length" || len(problem.ReasoningOnly) != 1 || !problem.ReasoningOnly[0] {
		t.Fatalf("per-sample termination metadata=%+v", problem)
	}
}

func TestRunRawArmAllowsCompleteResultClaim(t *testing.T) {
	report, err := RunRawArm(context.Background(), RawArmConfig{Model: "m", Endpoint: "e", N: 1}, []Problem{{QuestionID: "q", Prompt: "p"}}, func(context.Context, Problem, int) (string, RawSampleUsage, error) {
		return "answer", RawSampleUsage{FinishReason: "stop"}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !report.ResultClaimAllowed {
		t.Fatal("complete report should allow a result claim")
	}
}
