package livecodebench

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
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

	sample := func(_ context.Context, p Problem, i int) (string, agent.Usage, error) {
		return fmt.Sprintf("%s#%d", p.QuestionID, i), agent.Usage{}, nil
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

	usages := []agent.Usage{
		{PromptTokens: 100, CompletionTokens: 10, PromptTokensDetails: &agent.UsageTokenDetails{CachedTokens: 40}},
		{PromptTokens: 100, CompletionTokens: 10, CacheReadInputTokens: 30},
		{PromptTokens: 100, CompletionTokens: 10, PromptCacheHitTokens: 25},
	}
	var idx int32 = -1
	sample := func(_ context.Context, _ Problem, _ int) (string, agent.Usage, error) {
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
	sample := func(_ context.Context, p Problem, i int) (string, agent.Usage, error) {
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
		return "x", agent.Usage{}, nil
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
	sample := func(_ context.Context, p Problem, i int) (string, agent.Usage, error) {
		if p.QuestionID == "q1" {
			return "", agent.Usage{}, boom
		}
		return "ok", agent.Usage{}, nil
	}
	_, err := RunRawArm(context.Background(), cfg, probs, sample)
	if err == nil || !errors.Is(err, boom) {
		t.Fatalf("want wrapped sampler error, got %v", err)
	}
}

func TestRunRawArmRejectsBadConfig(t *testing.T) {
	if _, err := RunRawArm(context.Background(), RawArmConfig{N: 0}, nil, func(context.Context, Problem, int) (string, agent.Usage, error) { return "", agent.Usage{}, nil }); err == nil {
		t.Fatalf("want error for n<1")
	}
	if _, err := RunRawArm(context.Background(), RawArmConfig{N: 1}, nil, nil); err == nil {
		t.Fatalf("want error for nil sampler")
	}
}
