package livecodebench

import (
	"context"
	"errors"
	"flag"
	"strings"
	"sync/atomic"
	"testing"
)

// TestDefaultSamplingMatchesUpstream pins the #2106 acceptance: the shared
// defaults are the upstream lcb_runner ones (n=10, temperature=0.2), unseeded,
// with a bounded fan-out and a nonzero per-sample retry budget.
func TestDefaultSamplingMatchesUpstream(t *testing.T) {
	c := DefaultSampling()
	if c.N != 10 || c.Temperature != 0.2 {
		t.Fatalf("upstream defaults are n=10 temperature=0.2, got n=%d temperature=%v", c.N, c.Temperature)
	}
	if c.Seed != 0 {
		t.Fatalf("upstream default is unseeded, got seed=%d", c.Seed)
	}
	if c.Concurrency < 1 {
		t.Fatalf("concurrency must bound the fan-out, got %d", c.Concurrency)
	}
	if c.MaxRetries < 1 {
		t.Fatalf("a failed sample must be retried at least once by default, got max retries %d", c.MaxRetries)
	}
	if err := c.Validate(); err != nil {
		t.Fatalf("defaults must validate: %v", err)
	}
}

// TestRegisterFlagsSharedSurface: the arms register the identical sampling flag
// surface from one place; no-flag parses keep the upstream defaults and
// explicit flags land in the config.
func TestRegisterFlagsSharedSurface(t *testing.T) {
	c := DefaultSampling()
	fs := flag.NewFlagSet("arm", flag.ContinueOnError)
	c.RegisterFlags(fs)
	for _, name := range []string{"n", "temperature", "seed", "concurrency", "max-retries"} {
		if fs.Lookup(name) == nil {
			t.Fatalf("shared sampling flag -%s not registered", name)
		}
	}
	if err := fs.Parse(nil); err != nil {
		t.Fatal(err)
	}
	if c != DefaultSampling() {
		t.Fatalf("parsing no flags must keep upstream defaults, got %+v", c)
	}

	c2 := DefaultSampling()
	fs2 := flag.NewFlagSet("arm", flag.ContinueOnError)
	c2.RegisterFlags(fs2)
	if err := fs2.Parse([]string{"-n", "3", "-temperature", "0.7", "-seed", "42", "-concurrency", "2", "-max-retries", "0"}); err != nil {
		t.Fatal(err)
	}
	want := SamplingConfig{N: 3, Temperature: 0.7, Seed: 42, Concurrency: 2, MaxRetries: 0}
	if c2 != want {
		t.Fatalf("parsed sampling = %+v, want %+v", c2, want)
	}
}

func TestSamplingValidateRefusesBadConfigs(t *testing.T) {
	bad := []SamplingConfig{
		{N: 0, Temperature: 0.2, Concurrency: 1},
		{N: 1, Temperature: -0.1, Concurrency: 1},
		{N: 1, Temperature: 0.2, Concurrency: 0},
		{N: 1, Temperature: 0.2, Concurrency: 1, MaxRetries: -1},
	}
	for _, c := range bad {
		if err := c.Validate(); err == nil {
			t.Errorf("Validate(%+v) = nil, want error", c)
		}
	}
}

// TestArmConfigCarriesEveryValue: the centralized config reaches the arm run
// identity whole, so every sampling value lands in the report header.
func TestArmConfigCarriesEveryValue(t *testing.T) {
	c := SamplingConfig{N: 4, Temperature: 0.3, Seed: 7, Concurrency: 5, MaxRetries: 1}
	got := c.ArmConfig("glm-4.6", "http://e/v1")
	want := RawArmConfig{Model: "glm-4.6", Endpoint: "http://e/v1", N: 4, Temperature: 0.3, Seed: 7, Concurrency: 5, MaxRetries: 1}
	if got != want {
		t.Fatalf("ArmConfig = %+v, want %+v", got, want)
	}
}

// TestRunRawArmRetriesAndCounts witnesses the #2106 acceptance: a transiently
// failed sample is retried, the retry is counted in the report, and no
// completion is dropped. The sampling identity (seed, retry budget) is
// recorded in the report header.
func TestRunRawArmRetriesAndCounts(t *testing.T) {
	var failedOnce atomic.Bool
	sampler := func(ctx context.Context, p Problem, i int) (string, RawSampleUsage, error) {
		if p.QuestionID == "q1" && i == 0 && failedOnce.CompareAndSwap(false, true) {
			return "", RawSampleUsage{}, errors.New("transient gateway 502")
		}
		return "ok", RawSampleUsage{PromptTokens: 1, CompletionTokens: 1}, nil
	}
	cfg := RawArmConfig{Model: "m", Endpoint: "e", N: 2, Seed: 42, Concurrency: 2, MaxRetries: 2}
	problems := []Problem{{QuestionID: "q0", Prompt: "a"}, {QuestionID: "q1", Prompt: "b"}}
	rep, err := RunRawArm(context.Background(), cfg, problems, sampler)
	if err != nil {
		t.Fatalf("RunRawArm: %v", err)
	}
	if rep.Usage.Retries != 1 {
		t.Fatalf("want 1 counted retry, got %d", rep.Usage.Retries)
	}
	if rep.Usage.Samples != 4 {
		t.Fatalf("want all 4 samples collected, got %d", rep.Usage.Samples)
	}
	for _, p := range rep.Problems {
		for i, comp := range p.Completions {
			if comp != "ok" {
				t.Fatalf("problem %s sample %d dropped: %q", p.QuestionID, i, comp)
			}
		}
	}
	if rep.Seed != 42 || rep.MaxRetries != 2 {
		t.Fatalf("sampling identity not recorded in the report header: seed=%d max_retries=%d", rep.Seed, rep.MaxRetries)
	}
}

// TestRunRawArmRetryExhaustionAbortsLoudly: a sample that fails past its retry
// budget aborts the run naming the problem, sample, and attempt count — a
// failure is counted and surfaced, never silently dropped.
func TestRunRawArmRetryExhaustionAbortsLoudly(t *testing.T) {
	sampler := func(ctx context.Context, p Problem, i int) (string, RawSampleUsage, error) {
		return "", RawSampleUsage{}, errors.New("gateway down")
	}
	cfg := RawArmConfig{Model: "m", Endpoint: "e", N: 1, Concurrency: 1, MaxRetries: 1}
	_, err := RunRawArm(context.Background(), cfg, []Problem{{QuestionID: "q0", Prompt: "a"}}, sampler)
	if err == nil {
		t.Fatal("want error after retry exhaustion, got nil")
	}
	for _, needle := range []string{"q0", "2 attempt(s)", "gateway down"} {
		if !strings.Contains(err.Error(), needle) {
			t.Fatalf("exhaustion error must carry %q, got: %v", needle, err)
		}
	}
}

// TestRunFakArmCarriesSamplingIdentity: the fak arm's report header records the
// same centralized sampling identity the raw arm records.
func TestRunFakArmCarriesSamplingIdentity(t *testing.T) {
	sampler := func(ctx context.Context, p Problem, i int) (string, RawSampleUsage, FakSampleEvidence, error) {
		return "ok", RawSampleUsage{}, FakSampleEvidence{}, nil
	}
	cfg := RawArmConfig{Model: "m", Endpoint: "e", N: 1, Seed: 9, Concurrency: 1, MaxRetries: 3}
	rep, err := RunFakArm(context.Background(), cfg, "release_v6", []Problem{{QuestionID: "q0", Prompt: "a"}}, sampler)
	if err != nil {
		t.Fatalf("RunFakArm: %v", err)
	}
	if rep.Seed != 9 || rep.MaxRetries != 3 {
		t.Fatalf("fak report header must record seed and retry budget: seed=%d max_retries=%d", rep.Seed, rep.MaxRetries)
	}
}

// TestRunRawArmCachedRefusesSeedMismatch: resuming a prior report under a
// different seed is refused rather than silently mixing generations.
func TestRunRawArmCachedRefusesSeedMismatch(t *testing.T) {
	cfg := RawArmConfig{Model: "m", N: 1, Temperature: 0.2, Seed: 1}
	prior := &RawArmReport{Model: "m", N: 1, Temperature: 0.2, Seed: 2}
	sampler := func(context.Context, Problem, int) (string, RawSampleUsage, error) { return "", RawSampleUsage{}, nil }
	_, err := RunRawArmCached(context.Background(), cfg, "release_v6", nil, sampler, nil, prior)
	if err == nil || !strings.Contains(err.Error(), "seed") {
		t.Fatalf("want seed identity-mismatch refusal, got %v", err)
	}
}
