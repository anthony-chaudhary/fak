// Package profileplan defines the immutable workload geometry for agentbench.
// It performs no I/O and records budgets and proposed SLOs, never measurements.
package profileplan

import (
	"errors"
	"fmt"
	"time"
)

const Schema = "fak.agentbench.profile-plan.v1"

type Options struct {
	Profile              string
	PreferredConcurrency int
	Duration             time.Duration
}

type Plan struct {
	Schema                      string         `json:"schema"`
	Profile                     string         `json:"profile"`
	Qualifying                  bool           `json:"qualifying"`
	Seed                        uint64         `json:"seed"`
	SeedHex                     string         `json:"seed_hex"`
	SeedDerivation              string         `json:"seed_derivation,omitempty"`
	Deadline                    time.Duration  `json:"deadline"`
	DeadlineText                string         `json:"deadline_text"`
	DrainGrace                  time.Duration  `json:"drain_grace,omitempty"`
	DrainGraceText              string         `json:"drain_grace_text,omitempty"`
	MinimumModelContextTokens   int            `json:"minimum_model_context_tokens"`
	MaximumPromptTokens         int            `json:"maximum_prompt_tokens,omitempty"`
	QuickFixtureContextTokens   int            `json:"quick_fixture_context_tokens,omitempty"`
	ProbeTurns                  int            `json:"probe_turns"`
	SteadyTurns                 int            `json:"steady_turns"`
	ControlTurns                int            `json:"control_turns"`
	ScheduledReplayTurns        int            `json:"scheduled_replay_turns"`
	MaxPrimingRequests          int            `json:"max_priming_requests"`
	MaxReplayRetries            int            `json:"max_replay_retries"`
	TaskFixtures                int            `json:"task_fixtures"`
	TaskConcurrency             int            `json:"task_concurrency"`
	MaxTaskTurns                int            `json:"max_task_turns"`
	PreferredConcurrency        int            `json:"preferred_concurrency"`
	NormalMaxServiceConcurrency int            `json:"normal_max_service_concurrency,omitempty"`
	RequestCeiling              int            `json:"request_ceiling"`
	MinimumEpochs               int            `json:"minimum_epochs,omitempty"`
	Epoch                       *Epoch         `json:"epoch,omitempty"`
	Phases                      []Phase        `json:"phases"`
	OutputCapMix                []OutputCap    `json:"output_cap_mix"`
	CapacitySelection           CapacityPolicy `json:"capacity_selection"`
	SLOs                        ProposedSLOs   `json:"proposed_slos"`
	Prerequisites               []Prerequisite `json:"prerequisites,omitempty"`
	Unknowns                    []string       `json:"unknowns"`
}

type Phase struct {
	Name                   string   `json:"name"`
	ConcurrencySchedule    []int    `json:"concurrency_schedule,omitempty"`
	SessionsPerConcurrency int      `json:"sessions_per_concurrency,omitempty"`
	TurnsPerSession        int      `json:"turns_per_session,omitempty"`
	Sessions               int      `json:"sessions,omitempty"`
	Turns                  int      `json:"turns"`
	TaskFixtures           int      `json:"task_fixtures,omitempty"`
	TaskConcurrency        int      `json:"task_concurrency,omitempty"`
	MaxTurnsPerTask        int      `json:"max_turns_per_task,omitempty"`
	Areas                  int      `json:"areas,omitempty"`
	Notes                  []string `json:"notes,omitempty"`
}

type Epoch struct {
	Sessions           int `json:"sessions"`
	TurnsPerSession    int `json:"turns_per_session"`
	SteadyTurns        int `json:"steady_turns"`
	ControlTurns       int `json:"control_turns"`
	TaskFixtures       int `json:"task_fixtures"`
	MaxTaskTurns       int `json:"max_task_turns"`
	MaxPrimingRequests int `json:"max_priming_requests"`
	MaxReplayRetries   int `json:"max_replay_retries"`
	RequestCeiling     int `json:"request_ceiling"`
}

type OutputCap struct {
	Turns  int `json:"turns"`
	Tokens int `json:"tokens"`
}

type CapacityPolicy struct {
	ProbeConcurrency                 []int   `json:"probe_concurrency"`
	CompleteCellsOnly                bool    `json:"complete_cells_only"`
	RequireObservedConcurrency       bool    `json:"require_observed_concurrency"`
	ThroughputWithinBest             float64 `json:"throughput_within_best"`
	Selection                        string  `json:"selection"`
	HighestPassingReportedSeparately bool    `json:"highest_passing_reported_separately"`
}

type ProposedSLOs struct {
	Status                                 string        `json:"status"`
	BootstrapReadyWithin                   time.Duration `json:"bootstrap_ready_within"`
	BootstrapReadyWithinText               string        `json:"bootstrap_ready_within_text"`
	ColdFirstUsefulOutputWithin            time.Duration `json:"cold_first_useful_output_within"`
	ColdFirstUsefulOutputWithinText        string        `json:"cold_first_useful_output_within_text"`
	ColdRequestCompleteWithin              time.Duration `json:"cold_request_complete_within"`
	ColdRequestCompleteWithinText          string        `json:"cold_request_complete_within_text"`
	NewAreaFirstUsefulOutputEachWithin     time.Duration `json:"new_area_first_useful_output_each_within"`
	NewAreaFirstUsefulOutputEachWithinText string        `json:"new_area_first_useful_output_each_within_text"`
	WarmFirstOutputP50Within               time.Duration `json:"warm_first_output_p50_within"`
	WarmFirstOutputP50WithinText           string        `json:"warm_first_output_p50_within_text"`
	WarmFirstOutputP90Within               time.Duration `json:"warm_first_output_p90_within"`
	WarmFirstOutputP90WithinText           string        `json:"warm_first_output_p90_within_text"`
	WarmShortToolCallP90Within             time.Duration `json:"warm_short_tool_call_p90_within"`
	WarmShortToolCallP90WithinText         string        `json:"warm_short_tool_call_p90_within_text"`
	MediumTurnP90Within                    time.Duration `json:"medium_turn_p90_within"`
	MediumTurnP90WithinText                string        `json:"medium_turn_p90_within_text"`
	LongTurnP90Within                      time.Duration `json:"long_turn_p90_within"`
	LongTurnP90WithinText                  string        `json:"long_turn_p90_within_text"`
	WarmUnexplainedProgressGapMaximum      time.Duration `json:"warm_unexplained_progress_gap_maximum"`
	WarmUnexplainedProgressGapMaximumText  string        `json:"warm_unexplained_progress_gap_maximum_text"`
	WarmRequestCompleteWithin              time.Duration `json:"warm_request_complete_within"`
	WarmRequestCompleteWithinText          string        `json:"warm_request_complete_within_text"`
	TaskReleaseThroughVerifierWithin       time.Duration `json:"task_release_through_verifier_within"`
	TaskReleaseThroughVerifierWithinText   string        `json:"task_release_through_verifier_within_text"`
	UnplannedQualifyingFailuresMaximum     int           `json:"unplanned_qualifying_failures_maximum"`
	RequireAllIndependentTaskChecks        bool          `json:"require_all_independent_task_checks"`
	OperatorRescuesMaximum                 int           `json:"operator_rescues_maximum"`
}

type Prerequisite struct {
	Name     string `json:"name"`
	Required bool   `json:"required"`
	Note     string `json:"note"`
}

var errProfile = errors.New("agentbench profile plan")

func Build(opts Options) (Plan, error) {
	profile := opts.Profile
	if profile == "" {
		profile = "normal"
	}
	switch profile {
	case "quick":
		return buildQuick(opts)
	case "normal":
		return buildNormal(opts)
	case "burn-in":
		return buildBurnIn(opts)
	default:
		return Plan{}, fmt.Errorf("%w: unsupported profile %q", errProfile, profile)
	}
}

func base(profile string, seed uint64, deadline time.Duration) Plan {
	return Plan{
		Schema: Schema, Profile: profile, Seed: seed, SeedHex: fmt.Sprintf("0x%08X", seed),
		Deadline: deadline, DeadlineText: deadline.String(),
		OutputCapMix: []OutputCap{{Turns: 8, Tokens: 128}, {Turns: 3, Tokens: 256}, {Turns: 1, Tokens: 1024}},
		CapacitySelection: CapacityPolicy{
			ProbeConcurrency: []int{1, 2, 4, 8}, CompleteCellsOnly: true,
			RequireObservedConcurrency: true, ThroughputWithinBest: .95,
			Selection:                        "smallest passing concurrency whose observed latency-qualified useful throughput is within 95 percent of the best passing rung",
			HighestPassingReportedSeparately: true,
		},
		SLOs:     proposedSLOs(),
		Unknowns: []string{"cost ceiling when model pricing is unavailable", "reference tokenizer counts until a tokenizer is selected", "actual token counts and cache reuse", "hardware, memory, thermal, batching, prefill, and decode measurements"},
	}
}

func buildQuick(opts Options) (Plan, error) {
	if opts.Duration != 0 && opts.Duration != 5*time.Minute {
		return Plan{}, fmt.Errorf("%w: quick duration is fixed at 5m", errProfile)
	}
	if opts.PreferredConcurrency != 0 && opts.PreferredConcurrency != 2 {
		return Plan{}, fmt.Errorf("%w: quick has fixed C1/C2 schedule", errProfile)
	}
	p := base("quick", 0xA63E0001, 5*time.Minute)
	p.Qualifying = false
	p.MinimumModelContextTokens = 8192
	p.QuickFixtureContextTokens = 8192
	p.ScheduledReplayTurns = 16
	p.TaskFixtures = 1
	p.TaskConcurrency = 1
	p.MaxTaskTurns = 12
	p.PreferredConcurrency = 2
	p.RequestCeiling = 28
	p.Phases = []Phase{
		{Name: "replay", ConcurrencySchedule: []int{1, 2}, SessionsPerConcurrency: 2, TurnsPerSession: 4, Turns: 16, Notes: []string{"execute C1 before C2; preserve request order within each session"}},
		{Name: "tasks", TaskFixtures: 1, TaskConcurrency: 1, MaxTurnsPerTask: 12},
	}
	return p, nil
}

func buildNormal(opts Options) (Plan, error) {
	if opts.Duration != 0 && opts.Duration != 30*time.Minute {
		return Plan{}, fmt.Errorf("%w: normal duration is fixed at 30m", errProfile)
	}
	c := opts.PreferredConcurrency
	if c == 0 {
		c = 2
	}
	if !normalConcurrency(c) {
		return Plan{}, fmt.Errorf("%w: normal preferred concurrency must be one of 1, 2, 4, or 8", errProfile)
	}
	p := base("normal", 0xA63E1001, 30*time.Minute)
	p.Qualifying = true
	p.MinimumModelContextTokens = 32768
	p.MaximumPromptTokens = 31744
	p.ProbeTurns, p.SteadyTurns, p.ControlTurns = 64, 96, 24
	p.ScheduledReplayTurns = 184
	p.MaxPrimingRequests, p.MaxReplayRetries = 8, 8
	p.TaskFixtures, p.TaskConcurrency, p.MaxTaskTurns = 6, 2, 12
	p.PreferredConcurrency = c
	p.NormalMaxServiceConcurrency = 8
	p.RequestCeiling = normalRequestCeiling(c)
	p.Phases = []Phase{
		{Name: "probe", ConcurrencySchedule: []int{1, 2, 4, 8}, SessionsPerConcurrency: 8, TurnsPerSession: 2, Turns: 64},
		{Name: "steady", ConcurrencySchedule: []int{c}, Sessions: 8, TurnsPerSession: 12, Turns: 96},
		{Name: "controls", Turns: 24, Notes: []string{"six four-turn chains: cold-prefix, no-share, partial-prefix, and invalidation controls"}},
		{Name: "tasks-c2", TaskFixtures: 6, TaskConcurrency: 2, MaxTurnsPerTask: 12},
	}
	if c != 2 {
		p.Phases = append(p.Phases, Phase{Name: "tasks-preferred", TaskFixtures: max(6, c), TaskConcurrency: c, MaxTurnsPerTask: 12})
	}
	return p, nil
}

func buildBurnIn(opts Options) (Plan, error) {
	duration := opts.Duration
	if duration == 0 {
		duration = 2 * time.Hour
	}
	if duration != 2*time.Hour && duration != 8*time.Hour {
		return Plan{}, fmt.Errorf("%w: burn-in duration must be 2h or 8h", errProfile)
	}
	c := opts.PreferredConcurrency
	if !normalConcurrency(c) {
		return Plan{}, fmt.Errorf("%w: burn-in requires preferred concurrency 1, 2, 4, or 8 from a qualifying normal receipt", errProfile)
	}
	tasks := max(6, c)
	epoch := Epoch{Sessions: 8, TurnsPerSession: 32, SteadyTurns: 256, ControlTurns: 24, TaskFixtures: tasks, MaxTaskTurns: tasks * 12, MaxPrimingRequests: 8, MaxReplayRetries: 8}
	epoch.RequestCeiling = epoch.SteadyTurns + epoch.ControlTurns + epoch.MaxTaskTurns + epoch.MaxPrimingRequests + epoch.MaxReplayRetries
	p := base("burn-in", 0xA63EB001, duration)
	p.Qualifying = true
	p.SeedDerivation = "epoch_seed = 0xA63EB001 + zero_based_epoch_ordinal"
	p.DrainGrace, p.DrainGraceText = 90*time.Second, "1m30s"
	p.MinimumModelContextTokens = 32768
	p.MaximumPromptTokens = 31744
	p.MinimumEpochs = 2
	p.Epoch = &epoch
	p.SteadyTurns, p.ControlTurns = 2*epoch.SteadyTurns, 2*epoch.ControlTurns
	p.ScheduledReplayTurns = p.SteadyTurns + p.ControlTurns
	p.MaxPrimingRequests, p.MaxReplayRetries = 16, 16
	p.TaskFixtures, p.TaskConcurrency, p.MaxTaskTurns = 2*tasks, c, 2*epoch.MaxTaskTurns
	p.PreferredConcurrency = c
	p.NormalMaxServiceConcurrency = 8
	p.RequestCeiling = 2 * epoch.RequestCeiling
	p.Phases = []Phase{{Name: "epoch", ConcurrencySchedule: []int{c}, Sessions: 8, TurnsPerSession: 32, Turns: 280, TaskFixtures: tasks, TaskConcurrency: c, MaxTurnsPerTask: 12, Areas: 4, Notes: []string{"minimum two complete epochs", "rotate areas; apply pressure and revisit; inject cancellation and verify recovery", "after minimum coverage, admit only whole versioned epochs permitted by duration policy"}}}
	p.Prerequisites = []Prerequisite{{Name: "matching_qualifying_normal_receipt", Required: true, Note: "model, configuration, workload identity, and preferred concurrency must match; an absent receipt requires a separately accounted normal preflight"}}
	return p, nil
}

func normalConcurrency(c int) bool { return c == 1 || c == 2 || c == 4 || c == 8 }

func normalRequestCeiling(c int) int {
	switch c {
	case 1, 4:
		return 344
	case 8:
		return 368
	default:
		return 272
	}
}

func proposedSLOs() ProposedSLOs {
	return ProposedSLOs{
		Status:               "proposed_v1_product_tolerances_not_measurements",
		BootstrapReadyWithin: 90 * time.Second, BootstrapReadyWithinText: "1m30s",
		ColdFirstUsefulOutputWithin: 60 * time.Second, ColdFirstUsefulOutputWithinText: "1m",
		ColdRequestCompleteWithin: 120 * time.Second, ColdRequestCompleteWithinText: "2m",
		NewAreaFirstUsefulOutputEachWithin: 30 * time.Second, NewAreaFirstUsefulOutputEachWithinText: "30s",
		WarmFirstOutputP50Within: 2 * time.Second, WarmFirstOutputP50WithinText: "2s",
		WarmFirstOutputP90Within: 5 * time.Second, WarmFirstOutputP90WithinText: "5s",
		WarmShortToolCallP90Within: 10 * time.Second, WarmShortToolCallP90WithinText: "10s",
		MediumTurnP90Within: 20 * time.Second, MediumTurnP90WithinText: "20s",
		LongTurnP90Within: 60 * time.Second, LongTurnP90WithinText: "1m",
		WarmUnexplainedProgressGapMaximum: 10 * time.Second, WarmUnexplainedProgressGapMaximumText: "10s",
		WarmRequestCompleteWithin: 90 * time.Second, WarmRequestCompleteWithinText: "1m30s",
		TaskReleaseThroughVerifierWithin: 5 * time.Minute, TaskReleaseThroughVerifierWithinText: "5m",
		UnplannedQualifyingFailuresMaximum: 0, RequireAllIndependentTaskChecks: true, OperatorRescuesMaximum: 0,
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
