// Copyright 2026 The fak authors.
// SPDX-License-Identifier: Apache-2.0

// NOTICE / attribution: the scenario invariants exercised by this test are
// ported from ai-dynamo/aiperf@ab610ad (Apache-2.0), file
// src/aiperf/common/scenario/base.py (ScenarioSpec, ScenarioViolation,
// ScenarioOutcome, ScenarioLockError). Source pin:
// ai-dynamo/aiperf@ab610ad (full SHA ab610add4e8876fca223197596c728e17365f381).
// They are documented in docs/BENCHMARK-CONFORMANCE.md, section
// "Scenario invariants (aiperf scenario/base)".

package main

import (
	"errors"
	"fmt"
	"testing"
)

// scenarioSpec mirrors the frozen ScenarioSpec declaration from aiperf
// scenario/base.py (ai-dynamo/aiperf@ab610ad, Apache-2.0). The spec is
// immutable and extra-forbidding: applyScenario takes it by value and never
// mutates the caller's copy.
type scenarioSpec struct {
	name                              string
	timingMode                        string
	requireIgnoreEOS                  bool
	requireUseThinkTimeOnly           bool
	requireStreaming                  bool
	forbidIgnoreTraceDelays           bool
	forbidInputTruncation             bool
	requireLoader                     []string
	minBenchmarkDurationSeconds       int
	defaultBenchmarkDurationSeconds   *int
	defaultTrajectoryStartMinRatio    *float64
	defaultTrajectoryStartMaxRatio    *float64
	interTurnDelayCapSeconds          *float64
	traceIdleGapCapSeconds            *float64
	systemIdleGapCapSeconds           *float64
	forbidInterTurnDelayCap           bool
	requireCacheBust                  *string
	minimumProfileMetricCoverageRatio *float64
}

// validateSpec enforces the frozen-spec field constraints: duration floor
// > 0, trajectory ratios in [0,1], caps >= 0, coverage ratio in (0,1].
func validateSpec(s scenarioSpec) error {
	if s.name == "" {
		return fmt.Errorf("scenario spec: name is required")
	}
	if s.minBenchmarkDurationSeconds <= 0 {
		return fmt.Errorf("scenario spec: min_benchmark_duration_seconds must be > 0, got %d", s.minBenchmarkDurationSeconds)
	}
	if s.defaultBenchmarkDurationSeconds != nil && *s.defaultBenchmarkDurationSeconds <= 0 {
		return fmt.Errorf("scenario spec: default_benchmark_duration_seconds must be > 0")
	}
	for _, r := range []*float64{s.defaultTrajectoryStartMinRatio, s.defaultTrajectoryStartMaxRatio} {
		if r != nil && (*r < 0.0 || *r > 1.0) {
			return fmt.Errorf("scenario spec: trajectory start ratio must be in [0,1], got %v", *r)
		}
	}
	for _, c := range []*float64{s.interTurnDelayCapSeconds, s.traceIdleGapCapSeconds, s.systemIdleGapCapSeconds} {
		if c != nil && *c < 0 {
			return fmt.Errorf("scenario spec: delay cap must be >= 0, got %v", *c)
		}
	}
	if s.minimumProfileMetricCoverageRatio != nil &&
		(*s.minimumProfileMetricCoverageRatio <= 0.0 || *s.minimumProfileMetricCoverageRatio > 1.0) {
		return fmt.Errorf("scenario spec: minimum_profile_metric_coverage_ratio must be in (0,1], got %v", *s.minimumProfileMetricCoverageRatio)
	}
	if s.forbidInterTurnDelayCap && s.interTurnDelayCapSeconds != nil {
		return fmt.Errorf("scenario spec: forbid_inter_turn_delay_cap is set but inter_turn_delay_cap_seconds is present")
	}
	return nil
}

// userConfig is the resolved user-facing configuration the scenario lock is
// applied against. Nil pointers mean "user left the flag unset".
type userConfig struct {
	timingMode            string
	streaming             *bool
	ignoreEOS             *bool
	ignoreTraceDelays     *bool
	synthesisMaxISL       *int
	loader                string
	benchmarkDuration     *int
	trajectoryStartMin    *float64
	trajectoryStartMax    *float64
	interTurnDelayCap     *float64
	cacheBustTarget       *string
	profileMetricCoverage *float64 // observed fraction of profiling phases with TTFT/ITL
}

// scenarioViolation mirrors ScenarioViolation.
type scenarioViolation struct {
	flag          string
	currentValue  any
	requiredValue any
	message       string
}

func (v scenarioViolation) String() string {
	return fmt.Sprintf("%s: got %v, required %v (%s)", v.flag, v.currentValue, v.requiredValue, v.message)
}

// scenarioOutcome mirrors ScenarioOutcome.
type scenarioOutcome struct {
	scenarioName             *string
	appliedLocks             []string
	violations               []scenarioViolation
	submissionValid          *bool
	submissionInvalidReasons []string
}

// scenarioLockError mirrors ScenarioLockError: raised when a scenario lock is
// violated and cannot proceed. Bypassable violations can be downgraded to
// warnings under unsafeOverride (submission_valid=false); hard locks cannot.
type scenarioLockError struct {
	violations []scenarioViolation
	bypassable bool
}

func (e *scenarioLockError) Error() string {
	hint := "pass --unsafe-override to convert to warnings (run will be marked submission_valid=false)"
	if !e.bypassable {
		hint = "these violations cannot be bypassed with --unsafe-override"
	}
	return fmt.Sprintf("scenario invariants violated (%d conflict%s): %s; %s",
		len(e.violations), plural(len(e.violations)), joinViolations(e.violations), hint)
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}

func joinViolations(vs []scenarioViolation) string {
	out := ""
	for i, v := range vs {
		if i > 0 {
			out += "; "
		}
		out += v.String()
	}
	return out
}

func scBoolPtr(b bool) *bool    { return &b }
func scIntPtr(i int) *int       { return &i }
func scFPtr(f float64) *float64 { return &f }
func scStrPtr(s string) *string { return &s }

// applyScenario mirrors aiperf.common.scenario.apply_scenario: it validates
// the frozen spec, auto-fills unset user values, collects every invariant
// conflict in one pass, and returns the outcome. With unsafeOverride=false a
// non-empty violation list is a hard scenarioLockError; with
// unsafeOverride=true violations are downgraded to warnings and
// submission_valid is stamped false.
func applyScenario(spec scenarioSpec, user userConfig, unsafeOverride bool) (scenarioOutcome, error) {
	if err := validateSpec(spec); err != nil {
		return scenarioOutcome{}, err
	}

	var (
		outcome    scenarioOutcome
		violations []scenarioViolation
		locks      []string
	)
	add := func(flag string, current, required any, msg string) {
		violations = append(violations, scenarioViolation{flag, current, required, msg})
	}

	// Required timing mode.
	if user.timingMode != spec.timingMode {
		add("timing_mode", user.timingMode, spec.timingMode, "scenario requires the declared timing mode")
	} else {
		locks = append(locks, "timing_mode")
	}

	// require_ignore_eos: inject ignore_eos=true; explicit false is an error.
	if spec.requireIgnoreEOS {
		if user.ignoreEOS != nil && !*user.ignoreEOS {
			add("ignore_eos", *user.ignoreEOS, true, "scenario requires ignore_eos=true; explicit false is an error")
		} else {
			locks = append(locks, "ignore_eos")
		}
	}

	// require_streaming: explicit --no-streaming is an error.
	if spec.requireStreaming {
		if user.streaming != nil && !*user.streaming {
			add("streaming", *user.streaming, true, "streaming is required for per-token latency metrics (TTFT, ITL); explicit --no-streaming is an error")
		} else {
			locks = append(locks, "streaming")
		}
	}

	// forbid_ignore_trace_delays: replaying recorded trace timing.
	if spec.forbidIgnoreTraceDelays {
		if user.ignoreTraceDelays != nil && *user.ignoreTraceDelays {
			add("ignore_trace_delays", *user.ignoreTraceDelays, false,
				"scenario replays recorded trace timing; ignoring trace delays dispatches all turns back-to-back and falsifies the workload")
		} else {
			locks = append(locks, "trace_delays")
		}
	}

	// forbid_input_truncation: client-side input-length truncation rejected.
	if spec.forbidInputTruncation {
		if user.synthesisMaxISL != nil {
			add("synthesis_max_isl", *user.synthesisMaxISL, "unset", "client-side input-length truncation is rejected")
		} else {
			locks = append(locks, "input_truncation")
		}
	}

	// require_loader: detected loader must match one of the required names.
	if user.loader != "" {
		matched := false
		for _, name := range spec.requireLoader {
			if user.loader == name {
				matched = true
				break
			}
		}
		if !matched {
			add("loader", user.loader, spec.requireLoader, "detected loader must match the required loader name or one of the equivalent names")
		} else {
			locks = append(locks, "loader")
		}
	}

	// Benchmark duration: auto-fill when unset, then enforce the floor.
	duration := user.benchmarkDuration
	if duration == nil && spec.defaultBenchmarkDurationSeconds != nil {
		duration = spec.defaultBenchmarkDurationSeconds
		locks = append(locks, "duration_autofill")
	}
	if duration != nil && *duration < spec.minBenchmarkDurationSeconds {
		add("benchmark_duration", *duration, spec.minBenchmarkDurationSeconds,
			"benchmark duration is below the scenario floor")
	} else if duration != nil {
		locks = append(locks, "duration_floor")
	}

	// Trajectory start ratios: auto-fill when unset; explicit values honored.
	for _, r := range []*float64{user.trajectoryStartMin, user.trajectoryStartMax} {
		if r != nil && (*r < 0.0 || *r > 1.0) {
			add("trajectory_start_ratio", *r, "[0,1]", "trajectory start ratios must be in [0,1]")
		}
	}
	if user.trajectoryStartMin == nil && spec.defaultTrajectoryStartMinRatio != nil {
		locks = append(locks, "trajectory_start_min_autofill")
	}
	if user.trajectoryStartMax == nil && spec.defaultTrajectoryStartMaxRatio != nil {
		locks = append(locks, "trajectory_start_max_autofill")
	}

	// forbid_inter_turn_delay_cap: preserve original inter-turn timing.
	if spec.forbidInterTurnDelayCap {
		if user.interTurnDelayCap != nil {
			add("inter_turn_delay_cap", *user.interTurnDelayCap, "unset",
				"scenario preserves each trace's original inter-turn timing; a cap is forbidden")
		} else {
			locks = append(locks, "inter_turn_timing")
		}
	}

	// require_cache_bust: target must equal the required value.
	if spec.requireCacheBust != nil {
		if user.cacheBustTarget == nil || *user.cacheBustTarget != *spec.requireCacheBust {
			add("cache_bust_target", user.cacheBustTarget, *spec.requireCacheBust,
				"prompt cache-bust target must equal the required value")
		} else {
			locks = append(locks, "cache_bust")
		}
	}

	// minimum_profile_metric_coverage_ratio: below threshold the run is not
	// a valid submission.
	if spec.minimumProfileMetricCoverageRatio != nil {
		if user.profileMetricCoverage == nil || *user.profileMetricCoverage < *spec.minimumProfileMetricCoverageRatio {
			add("profile_metric_coverage", user.profileMetricCoverage, *spec.minimumProfileMetricCoverageRatio,
				"below the minimum fraction of profiling phases containing TTFT or ITL observations; the run exits non-zero")
		} else {
			locks = append(locks, "metric_coverage")
		}
	}

	outcome = scenarioOutcome{
		scenarioName: &spec.name,
		appliedLocks: locks,
		violations:   violations,
	}

	if len(violations) > 0 {
		if !unsafeOverride {
			outcome.submissionValid = scBoolPtr(false)
			return outcome, &scenarioLockError{violations: violations, bypassable: true}
		}
		// Downgraded to warnings: the run proceeds but is not a valid
		// submission.
		outcome.submissionValid = scBoolPtr(false)
		outcome.submissionInvalidReasons = append(outcome.submissionInvalidReasons, "unsafe_override")
		return outcome, nil
	}

	outcome.submissionValid = scBoolPtr(true)
	return outcome, nil
}

// clampInterTurnDelays applies the hard inter-turn delay cap to a trace's
// per-turn delays. The cap clamps observed gaps only: the input slice (the
// dataset) is never mutated, and request order is preserved.
func clampInterTurnDelays(delays []float64, capSeconds *float64) []float64 {
	out := make([]float64, len(delays))
	copy(out, delays)
	if capSeconds == nil {
		return out
	}
	for i, d := range out {
		if d > *capSeconds {
			out[i] = *capSeconds
		}
	}
	return out
}

// sampleScenario is a conforming scenario spec mirroring the documented
// invariants: frozen fields, required timing mode, ignore_eos and streaming
// required, trace delays and input truncation forbidden, loader pinned,
// duration floor with auto-fill, trajectory ratio auto-fill, inter-turn
// timing preserved, cache-bust pinned, and a metric coverage floor.
func sampleScenario() scenarioSpec {
	return scenarioSpec{
		name:                              "conformance-sample",
		timingMode:                        "trace_replay",
		requireIgnoreEOS:                  true,
		requireUseThinkTimeOnly:           false,
		requireStreaming:                  true,
		forbidIgnoreTraceDelays:           true,
		forbidInputTruncation:             true,
		requireLoader:                     []string{"weka_trace", "weka_trace_hf"},
		minBenchmarkDurationSeconds:       300,
		defaultBenchmarkDurationSeconds:   scIntPtr(600),
		defaultTrajectoryStartMinRatio:    scFPtr(0.1),
		defaultTrajectoryStartMaxRatio:    scFPtr(0.9),
		interTurnDelayCapSeconds:          nil,
		traceIdleGapCapSeconds:            scFPtr(120),
		systemIdleGapCapSeconds:           scFPtr(300),
		forbidInterTurnDelayCap:           true,
		requireCacheBust:                  scStrPtr("full"),
		minimumProfileMetricCoverageRatio: scFPtr(0.5),
	}
}

// conformingUser is a user configuration that satisfies every invariant:
// unset flags are auto-filled, set flags match the required values.
func conformingUser() userConfig {
	return userConfig{
		timingMode:            "trace_replay",
		streaming:             nil, // auto-enabled
		ignoreEOS:             nil, // injected by the scenario
		ignoreTraceDelays:     nil,
		synthesisMaxISL:       nil,
		loader:                "weka_trace",
		benchmarkDuration:     nil, // auto-filled to 600s
		trajectoryStartMin:    nil, // auto-filled to 0.1
		trajectoryStartMax:    nil, // auto-filled to 0.9
		interTurnDelayCap:     nil,
		cacheBustTarget:       scStrPtr("full"),
		profileMetricCoverage: scFPtr(0.8),
	}
}

func TestScenarioInvariants(t *testing.T) {
	spec := sampleScenario()
	if err := validateSpec(spec); err != nil {
		t.Fatalf("sample scenario spec failed frozen-spec validation: %v", err)
	}

	// 1. Conforming sample: submission_valid=true, no violations, and the
	// expected locks were applied in order.
	outcome, err := applyScenario(spec, conformingUser(), false)
	if err != nil {
		t.Fatalf("conforming sample must not raise a scenario lock error: %v", err)
	}
	if outcome.submissionValid == nil || !*outcome.submissionValid {
		t.Fatalf("conforming sample: submission_valid = %v, want true", outcome.submissionValid)
	}
	if len(outcome.violations) != 0 {
		t.Fatalf("conforming sample: violations = %v, want none", outcome.violations)
	}
	wantLocks := []string{"timing_mode", "ignore_eos", "streaming", "trace_delays",
		"input_truncation", "loader", "duration_autofill", "duration_floor",
		"trajectory_start_min_autofill", "trajectory_start_max_autofill",
		"inter_turn_timing", "cache_bust", "metric_coverage"}
	if len(outcome.appliedLocks) != len(wantLocks) {
		t.Fatalf("conforming sample: applied_locks = %v, want %v", outcome.appliedLocks, wantLocks)
	}
	for i, want := range wantLocks {
		if outcome.appliedLocks[i] != want {
			t.Fatalf("conforming sample: applied_locks[%d] = %q, want %q (full: %v)", i, outcome.appliedLocks[i], want, outcome.appliedLocks)
		}
	}

	// 2. Frozen spec: applyScenario takes the spec by value; the caller's
	// copy is byte-identical after application (no mutation, no extra
	// fields accepted).
	after, _ := applyScenario(spec, conformingUser(), false)
	_ = after
	if spec.name != "conformance-sample" || spec.minBenchmarkDurationSeconds != 300 ||
		!spec.requireStreaming || !spec.forbidIgnoreTraceDelays ||
		!spec.forbidInputTruncation || !spec.forbidInterTurnDelayCap ||
		spec.requireCacheBust == nil || *spec.requireCacheBust != "full" ||
		spec.minimumProfileMetricCoverageRatio == nil || *spec.minimumProfileMetricCoverageRatio != 0.5 {
		t.Fatalf("frozen spec was mutated by applyScenario: %+v", spec)
	}

	// 3. Explicit --no-streaming is an error (streaming required for
	// TTFT/ITL).
	u := conformingUser()
	u.streaming = scBoolPtr(false)
	_, err = applyScenario(spec, u, false)
	if err == nil {
		t.Fatal("explicit --no-streaming must be rejected")
	}
	var lockErr *scenarioLockError
	if !errors.As(err, &lockErr) {
		t.Fatalf("expected scenarioLockError, got %T: %v", err, err)
	}

	// 4. Explicit ignore_eos=false is an error.
	u = conformingUser()
	u.ignoreEOS = scBoolPtr(false)
	if _, err := applyScenario(spec, u, false); err == nil {
		t.Fatal("explicit ignore_eos=false must be rejected")
	}

	// 5. --ignore-trace-delays is forbidden (falsifies the workload).
	u = conformingUser()
	u.ignoreTraceDelays = scBoolPtr(true)
	if _, err := applyScenario(spec, u, false); err == nil {
		t.Fatal("--ignore-trace-delays must be rejected")
	}

	// 6. Client-side input truncation (--synthesis-max-isl) is rejected.
	u = conformingUser()
	u.synthesisMaxISL = scIntPtr(4096)
	if _, err := applyScenario(spec, u, false); err == nil {
		t.Fatal("--synthesis-max-isl must be rejected")
	}

	// 7. Loader mismatch is rejected; an equivalent loader name is accepted.
	u = conformingUser()
	u.loader = "synthetic_loader"
	if _, err := applyScenario(spec, u, false); err == nil {
		t.Fatal("non-matching loader must be rejected")
	}
	u.loader = "weka_trace_hf"
	if _, err := applyScenario(spec, u, false); err != nil {
		t.Fatalf("equivalent loader name must be accepted: %v", err)
	}

	// 8. Duration floor: below the floor is rejected; unset auto-fills to
	// the default; an explicit value above the floor is honored.
	u = conformingUser()
	u.benchmarkDuration = scIntPtr(60)
	if _, err := applyScenario(spec, u, false); err == nil {
		t.Fatal("benchmark duration below the floor must be rejected")
	}
	u = conformingUser()
	if _, err := applyScenario(spec, u, false); err != nil {
		t.Fatalf("unset duration must auto-fill to the default: %v", err)
	}
	u = conformingUser()
	u.benchmarkDuration = scIntPtr(900)
	if _, err := applyScenario(spec, u, false); err != nil {
		t.Fatalf("explicit duration above the floor must be honored: %v", err)
	}

	// 9. forbid_inter_turn_delay_cap: a user cap is rejected.
	u = conformingUser()
	u.interTurnDelayCap = scFPtr(30)
	if _, err := applyScenario(spec, u, false); err == nil {
		t.Fatal("inter-turn delay cap must be rejected when the scenario preserves original timing")
	}

	// 10. Cache-bust mismatch: rejected by default; under --unsafe-override
	// the violation is downgraded and submission_valid=false.
	u = conformingUser()
	u.cacheBustTarget = scStrPtr("none")
	if _, err := applyScenario(spec, u, false); err == nil {
		t.Fatal("cache-bust target mismatch must be rejected")
	}
	outcome, err = applyScenario(spec, u, true)
	if err != nil {
		t.Fatalf("unsafe-override must downgrade the cache-bust violation to a warning: %v", err)
	}
	if outcome.submissionValid == nil || *outcome.submissionValid {
		t.Fatalf("unsafe-override with violations: submission_valid = %v, want false", outcome.submissionValid)
	}
	if len(outcome.violations) != 1 || outcome.violations[0].flag != "cache_bust_target" {
		t.Fatalf("unsafe-override: violations = %v, want exactly [cache_bust_target]", outcome.violations)
	}
	if len(outcome.submissionInvalidReasons) != 1 || outcome.submissionInvalidReasons[0] != "unsafe_override" {
		t.Fatalf("unsafe-override: submission_invalid_reasons = %v, want [unsafe_override]", outcome.submissionInvalidReasons)
	}

	// 11. Metric coverage below the threshold is not a valid submission.
	u = conformingUser()
	u.profileMetricCoverage = scFPtr(0.2)
	if _, err := applyScenario(spec, u, false); err == nil {
		t.Fatal("profile metric coverage below the minimum ratio must be rejected")
	}

	// 12. Delay caps are hard ceilings that clamp observed gaps without
	// mutating the dataset or reordering requests.
	trace := []float64{0.5, 3.0, 10.0}
	original := []float64{0.5, 3.0, 10.0}
	capped := clampInterTurnDelays(trace, scFPtr(5.0))
	if capped[0] != 0.5 || capped[1] != 3.0 || capped[2] != 5.0 {
		t.Fatalf("delay cap: got %v, want [0.5 3 5]", capped)
	}
	if trace[2] != 10.0 {
		t.Fatalf("delay cap mutated the dataset: %v", trace)
	}
	// The cap clamps values only: same length, same order, and every
	// unclamped delay passes through unchanged.
	if len(capped) != len(original) {
		t.Fatalf("delay cap changed request length: %v vs %v", capped, original)
	}
	for i := range original {
		if original[i] <= 5.0 && capped[i] != original[i] {
			t.Fatalf("delay cap altered an unclamped delay at %d: %v vs %v", i, capped, original)
		}
		if capped[i] > original[i] {
			t.Fatalf("delay cap increased a delay at %d: %v vs %v", i, capped, original)
		}
	}
	// No cap: delays pass through unchanged.
	uncapped := clampInterTurnDelays(trace, nil)
	if uncapped[2] != 10.0 {
		t.Fatalf("no cap: delays must pass through unchanged, got %v", uncapped)
	}

	// 13. Frozen-spec field constraints: coverage ratio must be in (0,1],
	// trajectory ratios in [0,1], duration floor > 0.
	bad := sampleScenario()
	bad.minimumProfileMetricCoverageRatio = scFPtr(0.0)
	if err := validateSpec(bad); err == nil {
		t.Fatal("coverage ratio 0.0 must fail frozen-spec validation")
	}
	bad = sampleScenario()
	bad.minimumProfileMetricCoverageRatio = scFPtr(1.5)
	if err := validateSpec(bad); err == nil {
		t.Fatal("coverage ratio > 1 must fail frozen-spec validation")
	}
	bad = sampleScenario()
	bad.defaultTrajectoryStartMinRatio = scFPtr(1.2)
	if err := validateSpec(bad); err == nil {
		t.Fatal("trajectory ratio > 1 must fail frozen-spec validation")
	}
	bad = sampleScenario()
	bad.minBenchmarkDurationSeconds = 0
	if err := validateSpec(bad); err == nil {
		t.Fatal("duration floor 0 must fail frozen-spec validation")
	}
}
