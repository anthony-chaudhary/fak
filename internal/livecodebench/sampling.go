package livecodebench

import (
	"flag"
	"fmt"
)

// sampling.go centralizes the sampling identity BOTH A/B arms share (#2106,
// epic #2085): n, temperature, seed, and the rate-limited fan-out width. The
// upstream lcb_runner defaults are pinned here once so the CLI flags, the
// codegen scorer, and both arm reports cannot drift from each other or from
// upstream. Every SamplingConfig value is recorded verbatim in the report
// header of whichever arm ran with it.

const (
	// UpstreamDefaultN mirrors lcb_runner.runner.main's default -n:
	// 10 samples per problem.
	UpstreamDefaultN = 10
	// UpstreamDefaultTemperature mirrors lcb_runner's default --temperature.
	UpstreamDefaultTemperature = 0.2
	// DefaultSamplingConcurrency bounds in-flight gateway requests (the
	// closed-API --multiprocess analog) so a run respects gateway rate limits.
	DefaultSamplingConcurrency = 8
	// DefaultMaxSampleRetries is the per-sample retry budget: a failed sample
	// is retried up to this many times, each retry counted in
	// RawArmUsage.Retries; exhausting the budget aborts the run loudly. A
	// failed sample is never silently dropped.
	DefaultMaxSampleRetries = 2
)

// SamplingConfig is the sampling identity shared by both arms. Both arms bind
// it into their run config through RawArmConfig so the shared knobs cannot
// drift between them.
type SamplingConfig struct {
	N           int     // samples per problem (upstream -n)
	Temperature float64 // sampling temperature (upstream --temperature)
	Seed        int64   // sampling seed sent when nonzero; 0 = provider default
	Concurrency int     // max in-flight requests (rate limit toward the gateway)
	MaxRetries  int     // per-sample retry budget before the run aborts
}

// DefaultSampling returns the upstream lcb_runner sampling defaults.
func DefaultSampling() SamplingConfig {
	return SamplingConfig{
		N:           UpstreamDefaultN,
		Temperature: UpstreamDefaultTemperature,
		Concurrency: DefaultSamplingConcurrency,
		MaxRetries:  DefaultMaxSampleRetries,
	}
}

// Validate refuses a sampling config no arm should run with, so a bad flag
// value is caught before any tokens are spent.
func (s SamplingConfig) Validate() error {
	if s.N < 1 {
		return fmt.Errorf("livecodebench sampling: n must be >= 1, got %d", s.N)
	}
	if s.Temperature < 0 {
		return fmt.Errorf("livecodebench sampling: temperature must be >= 0, got %v", s.Temperature)
	}
	if s.Concurrency < 1 {
		return fmt.Errorf("livecodebench sampling: concurrency must be >= 1, got %d", s.Concurrency)
	}
	if s.MaxRetries < 0 {
		return fmt.Errorf("livecodebench sampling: max retries must be >= 0, got %d", s.MaxRetries)
	}
	return nil
}

// RegisterFlags binds the shared sampling flag surface onto fs, defaulting to
// the receiver's current values. Both arm CLIs register their sampling flags
// through this one method, so the surface cannot drift between arms.
func (s *SamplingConfig) RegisterFlags(fs *flag.FlagSet) {
	fs.IntVar(&s.N, "n", s.N, "samples to generate per problem (mirrors lcb_runner -n)")
	fs.Float64Var(&s.Temperature, "temperature", s.Temperature, "sampling temperature sent and recorded (mirrors lcb_runner --temperature)")
	fs.Int64Var(&s.Seed, "seed", s.Seed, "sampling seed sent and recorded when nonzero (0 = provider default)")
	fs.IntVar(&s.Concurrency, "concurrency", s.Concurrency, "max in-flight gateway requests, respecting gateway rate limits (mirrors closed-API --multiprocess)")
	fs.IntVar(&s.MaxRetries, "max-retries", s.MaxRetries, "per-sample retry budget: a failed sample is retried and counted, never silently dropped; exhausting it aborts the run")
}

// ArmConfig binds this sampling identity to one arm's run identity (model +
// endpoint). RunFakArm takes the same RawArmConfig, so a config built here is
// the single source of truth for both arms of an A/B run.
func (s SamplingConfig) ArmConfig(model, endpoint string) RawArmConfig {
	return RawArmConfig{
		Model:       model,
		Endpoint:    endpoint,
		N:           s.N,
		Temperature: s.Temperature,
		Seed:        s.Seed,
		Concurrency: s.Concurrency,
		MaxRetries:  s.MaxRetries,
	}
}
