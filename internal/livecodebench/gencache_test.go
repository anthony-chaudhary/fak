package livecodebench

import (
	"context"
	"fmt"
	"path/filepath"
	"sync"
	"testing"
)

func genProblems(ids ...string) []Problem {
	ps := make([]Problem, len(ids))
	for i, id := range ids {
		ps[i] = Problem{QuestionID: id, Prompt: "prompt for " + id}
	}
	return ps
}

// countingSampler returns a deterministic sampler that records how many times
// it was invoked — the witness that a re-run did or did not re-spend tokens.
func countingSampler(calls *int, mu *sync.Mutex) RawArmSampler {
	return func(_ context.Context, p Problem, i int) (string, RawSampleUsage, error) {
		mu.Lock()
		*calls++
		mu.Unlock()
		return fmt.Sprintf("gen:%s:%d", p.QuestionID, i), RawSampleUsage{PromptTokens: 10, CompletionTokens: 5}, nil
	}
}

func failingSampler(t *testing.T) RawArmSampler {
	return func(_ context.Context, p Problem, i int) (string, RawSampleUsage, error) {
		t.Errorf("sampler called for %q sample %d: a fully cached re-run must not regenerate", p.QuestionID, i)
		return "", RawSampleUsage{}, fmt.Errorf("must not be called")
	}
}

func TestDirGenCacheRoundTripAndMisses(t *testing.T) {
	c := DirGenCache{Dir: filepath.Join(t.TempDir(), "gencache")}
	key := CacheKeyInput{Model: "m", Prompt: "p", N: 2, Temperature: 0.2, Release: "release_v6"}.CacheKey()

	if _, ok := c.Get(key); ok {
		t.Fatalf("Get on empty cache: ok = true, want miss")
	}
	c.Put(key, []string{"a", "b"})
	got, ok := c.Get(key)
	if !ok || len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("Get after Put = %v, %v; want [a b], true", got, ok)
	}
	if _, ok := c.Get(key + "x"); ok {
		t.Fatalf("Get on unknown key: ok = true, want miss")
	}
}

func TestRunRawArmCachedReusesOnRerun(t *testing.T) {
	cache := DirGenCache{Dir: t.TempDir()}
	cfg := RawArmConfig{Model: "m", Endpoint: "e", N: 2, Temperature: 0.2}
	problems := genProblems("q1", "q2")

	var calls int
	var mu sync.Mutex
	first, err := RunRawArmCached(context.Background(), cfg, "release_v6", problems, countingSampler(&calls, &mu), cache, nil)
	if err != nil {
		t.Fatalf("first run: %v", err)
	}
	if calls != 4 {
		t.Fatalf("first run sampler calls = %d, want 4 (2 problems x n=2)", calls)
	}
	if first.Stats.Hits != 0 || first.Stats.Misses != 2 {
		t.Fatalf("first run stats = %+v, want 0 hits / 2 misses", first.Stats)
	}

	// Re-run with identical knobs: every completion must come from the cache.
	second, err := RunRawArmCached(context.Background(), cfg, "release_v6", problems, failingSampler(t), cache, nil)
	if err != nil {
		t.Fatalf("re-run: %v", err)
	}
	if second.Stats.Hits != 2 || second.Stats.Misses != 0 {
		t.Fatalf("re-run stats = %+v, want 2 hits / 0 misses", second.Stats)
	}
	if second.Report.Usage.Samples != 0 {
		t.Fatalf("re-run usage samples = %d, want 0 (no tokens re-spent)", second.Report.Usage.Samples)
	}
	if len(second.Report.Problems) != 2 || second.Report.Problems[0].Completions[0] != "gen:q1:0" {
		t.Fatalf("re-run report problems = %+v, want first run's completions in suite order", second.Report.Problems)
	}
}

func TestRunRawArmCachedKeyIncludesEveryKnob(t *testing.T) {
	base := RawArmConfig{Model: "m", N: 1, Temperature: 0.2}
	variants := map[string]struct {
		cfg     RawArmConfig
		release string
	}{
		"model":       {RawArmConfig{Model: "m2", N: 1, Temperature: 0.2}, "release_v6"},
		"n":           {RawArmConfig{Model: "m", N: 2, Temperature: 0.2}, "release_v6"},
		"temperature": {RawArmConfig{Model: "m", N: 1, Temperature: 0.7}, "release_v6"},
		"release":     {base, "release_v5"},
	}
	for name, v := range variants {
		t.Run(name, func(t *testing.T) {
			cache := DirGenCache{Dir: t.TempDir()}
			problems := genProblems("q1")
			var calls int
			var mu sync.Mutex
			if _, err := RunRawArmCached(context.Background(), base, "release_v6", problems, countingSampler(&calls, &mu), cache, nil); err != nil {
				t.Fatalf("seed run: %v", err)
			}
			res, err := RunRawArmCached(context.Background(), v.cfg, v.release, problems, countingSampler(&calls, &mu), cache, nil)
			if err != nil {
				t.Fatalf("variant run: %v", err)
			}
			if res.Stats.Misses != 1 || res.Stats.Hits != 0 {
				t.Fatalf("changed %s: stats = %+v, want a miss (different identity must not reuse)", name, res.Stats)
			}
		})
	}
	// Changing the prompt must also miss.
	cache := DirGenCache{Dir: t.TempDir()}
	var calls int
	var mu sync.Mutex
	if _, err := RunRawArmCached(context.Background(), base, "release_v6", genProblems("q1"), countingSampler(&calls, &mu), cache, nil); err != nil {
		t.Fatalf("seed run: %v", err)
	}
	changed := []Problem{{QuestionID: "q1", Prompt: "a different prompt"}}
	res, err := RunRawArmCached(context.Background(), base, "release_v6", changed, countingSampler(&calls, &mu), cache, nil)
	if err != nil {
		t.Fatalf("prompt variant: %v", err)
	}
	if res.Stats.Misses != 1 {
		t.Fatalf("changed prompt: stats = %+v, want a miss", res.Stats)
	}
}

func TestRunRawArmCachedContinueExisting(t *testing.T) {
	cfg := RawArmConfig{Model: "m", N: 2, Temperature: 0.2}
	problems := genProblems("q1", "q2", "q3")
	prior := &RawArmReport{
		Model: "m", N: 2, Temperature: 0.2,
		Problems: []RawArmProblem{
			{QuestionID: "q1", Completions: []string{"prior:q1:0", "prior:q1:1"}},
			{QuestionID: "q3", Completions: []string{"prior:q3:0"}}, // incomplete: only 1 of n=2
		},
		Usage: RawArmUsage{Samples: 2, PromptTokens: 20, CompletionTokens: 10},
	}

	var calls int
	var mu sync.Mutex
	res, err := RunRawArmCached(context.Background(), cfg, "release_v6", problems, countingSampler(&calls, &mu), nil, prior)
	if err != nil {
		t.Fatalf("resume run: %v", err)
	}
	// q1 resumed; q2 and q3 (incomplete in prior) regenerated: 2 problems x n=2.
	if calls != 4 {
		t.Fatalf("sampler calls = %d, want 4 (q2+q3 only; q1 must not be duplicated)", calls)
	}
	if res.Resumed != 1 {
		t.Fatalf("Resumed = %d, want 1", res.Resumed)
	}
	if res.Stats.Lookups() != 0 {
		t.Fatalf("stats lookups = %d, want 0 (resume skips are not cache lookups)", res.Stats.Lookups())
	}
	if len(res.Report.Problems) != 3 {
		t.Fatalf("merged report has %d problems, want 3", len(res.Report.Problems))
	}
	if res.Report.Problems[0].QuestionID != "q1" || res.Report.Problems[0].Completions[0] != "prior:q1:0" {
		t.Fatalf("q1 row = %+v, want the prior run's completions carried over", res.Report.Problems[0])
	}
	if res.Report.Problems[2].Completions[0] != "gen:q3:0" {
		t.Fatalf("q3 row = %+v, want regenerated (prior row was incomplete)", res.Report.Problems[2])
	}
	// Usage sums the prior segment and this segment.
	if res.Report.Usage.Samples != 2+4 || res.Report.Usage.PromptTokens != 20+40 {
		t.Fatalf("merged usage = %+v, want prior + regenerated", res.Report.Usage)
	}
}

func TestRunRawArmCachedContinueExistingIdentityMismatch(t *testing.T) {
	cfg := RawArmConfig{Model: "m", N: 2, Temperature: 0.2}
	prior := &RawArmReport{Model: "other-model", N: 2, Temperature: 0.2}
	_, err := RunRawArmCached(context.Background(), cfg, "release_v6", genProblems("q1"), failingSampler(t), nil, prior)
	if err == nil {
		t.Fatalf("identity mismatch accepted; want refusal")
	}
}

func TestRunRawArmCachedNoCacheNoPriorMatchesRunRawArm(t *testing.T) {
	cfg := RawArmConfig{Model: "m", Endpoint: "e", N: 1, Temperature: 0.2}
	problems := genProblems("q1", "q2")
	var calls int
	var mu sync.Mutex
	res, err := RunRawArmCached(context.Background(), cfg, "release_v6", problems, countingSampler(&calls, &mu), nil, nil)
	if err != nil {
		t.Fatalf("plain run: %v", err)
	}
	want, err := RunRawArm(context.Background(), cfg, problems, countingSampler(&calls, &mu))
	if err != nil {
		t.Fatalf("RunRawArm: %v", err)
	}
	if len(res.Report.Problems) != len(want.Problems) || res.Report.Usage != want.Usage {
		t.Fatalf("cached-path report %+v differs from RunRawArm %+v", res.Report, want)
	}
	if res.Stats.Lookups() != 0 {
		t.Fatalf("stats lookups = %d, want 0 with no cache", res.Stats.Lookups())
	}
}
