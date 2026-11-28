package profileplan

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"
)

// fak-test:runtime fast est=10ms lane=default
func TestBuildDefaultAndQuickProfiles(t *testing.T) {
	def, err := Build(Options{})
	if err != nil {
		t.Fatalf("Build(Options{}) unexpected error: %v", err)
	}
	if def.Schema != Schema {
		t.Errorf("Schema = %q, want %q", def.Schema, Schema)
	}
	if def.Profile != "normal" || !def.Qualifying || def.PreferredConcurrency != 2 {
		t.Errorf("default plan identity = (profile=%q qualifying=%v c=%d)", def.Profile, def.Qualifying, def.PreferredConcurrency)
	}
	if def.RequestCeiling != 272 || len(def.Phases) != 4 {
		t.Errorf("default ceiling/phases = (%d, %d), want (272, 4)", def.RequestCeiling, len(def.Phases))
	}
	if !reflect.DeepEqual(def.CapacitySelection.ProbeConcurrency, []int{1, 2, 4, 8}) || def.CapacitySelection.ThroughputWithinBest != 0.95 {
		t.Errorf("CapacitySelection = %+v", def.CapacitySelection)
	}
	if def.SLOs.Status == "" || len(def.OutputCapMix) != 3 || len(def.Unknowns) == 0 {
		t.Errorf("SLOs/OutputCapMix/Unknowns incomplete: status=%q mix=%d unknowns=%d", def.SLOs.Status, len(def.OutputCapMix), len(def.Unknowns))
	}

	quick, err := Build(Options{Profile: "quick"})
	if err != nil {
		t.Fatalf("Build(quick) unexpected error: %v", err)
	}
	if quick.Qualifying || quick.Deadline != 5*time.Minute || quick.DeadlineText != "5m0s" {
		t.Errorf("quick timing/qualifying = (%v, %v, %q)", quick.Qualifying, quick.Deadline, quick.DeadlineText)
	}
	if quick.Seed != 0xA63E0001 || quick.SeedHex != "0xA63E0001" {
		t.Errorf("quick seed = (%#x, %q)", quick.Seed, quick.SeedHex)
	}
	if quick.MinimumModelContextTokens != 8192 || quick.QuickFixtureContextTokens != 8192 || quick.RequestCeiling != 28 {
		t.Errorf("quick budgets = (%d, %d, %d)", quick.MinimumModelContextTokens, quick.QuickFixtureContextTokens, quick.RequestCeiling)
	}
	if len(quick.Phases) != 2 || quick.Phases[0].Name != "replay" || quick.Phases[1].Name != "tasks" {
		t.Errorf("quick phases = %+v", quick.Phases)
	}

	quickExplicit, err := Build(Options{Profile: "quick", PreferredConcurrency: 2, Duration: 5 * time.Minute})
	if err != nil {
		t.Fatalf("Build(quick explicit) unexpected error: %v", err)
	}
	if !reflect.DeepEqual(quick, quickExplicit) {
		t.Errorf("explicit quick options altered plan")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestBuildNormalConcurrencySchedules(t *testing.T) {
	cases := []struct {
		concurrency int
		wantCeiling int
		wantPhases  int
		wantTaskFix int
	}{
		{concurrency: 1, wantCeiling: 344, wantPhases: 5, wantTaskFix: 6},
		{concurrency: 2, wantCeiling: 272, wantPhases: 4, wantTaskFix: 0},
		{concurrency: 4, wantCeiling: 344, wantPhases: 5, wantTaskFix: 6},
		{concurrency: 8, wantCeiling: 368, wantPhases: 5, wantTaskFix: 8},
	}
	for _, tc := range cases {
		p, err := Build(Options{Profile: "normal", PreferredConcurrency: tc.concurrency, Duration: 30 * time.Minute})
		if err != nil {
			t.Fatalf("Build(normal, c=%d) error: %v", tc.concurrency, err)
		}
		if p.PreferredConcurrency != tc.concurrency || p.RequestCeiling != tc.wantCeiling || len(p.Phases) != tc.wantPhases {
			t.Errorf("c=%d: got (c=%d ceiling=%d phases=%d), want (%d, %d, %d)",
				tc.concurrency, p.PreferredConcurrency, p.RequestCeiling, len(p.Phases),
				tc.concurrency, tc.wantCeiling, tc.wantPhases)
		}
		if p.MinimumModelContextTokens != 32768 || p.MaximumPromptTokens != 31744 || p.ScheduledReplayTurns != 184 {
			t.Errorf("c=%d: unexpected token/turn budget (%d, %d, %d)",
				tc.concurrency, p.MinimumModelContextTokens, p.MaximumPromptTokens, p.ScheduledReplayTurns)
		}
		if tc.wantPhases == 5 {
			last := p.Phases[len(p.Phases)-1]
			if last.Name != "tasks-preferred" || last.TaskFixtures != tc.wantTaskFix || last.TaskConcurrency != tc.concurrency {
				t.Errorf("c=%d: tasks-preferred phase = %+v", tc.concurrency, last)
			}
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestBuildBurnInProfileAndEpochMath(t *testing.T) {
	for _, tc := range []struct {
		name        string
		opts        Options
		wantDur     time.Duration
		wantTaskFix int
	}{
		{name: "default 2h c=4", opts: Options{Profile: "burn-in", PreferredConcurrency: 4}, wantDur: 2 * time.Hour, wantTaskFix: 6},
		{name: "explicit 8h c=8", opts: Options{Profile: "burn-in", PreferredConcurrency: 8, Duration: 8 * time.Hour}, wantDur: 8 * time.Hour, wantTaskFix: 8},
		{name: "explicit 2h c=1", opts: Options{Profile: "burn-in", PreferredConcurrency: 1, Duration: 2 * time.Hour}, wantDur: 2 * time.Hour, wantTaskFix: 6},
	} {
		p, err := Build(tc.opts)
		if err != nil {
			t.Fatalf("%s: Build error: %v", tc.name, err)
		}
		if !p.Qualifying || p.Deadline != tc.wantDur || p.MinimumEpochs != 2 || p.Epoch == nil {
			t.Fatalf("%s: unexpected header/epoch: %+v", tc.name, p)
		}
		if p.DrainGrace != 90*time.Second || p.DrainGraceText != "1m30s" || p.SeedDerivation == "" {
			t.Errorf("%s: drain/seed derivation = (%v, %q, %q)", tc.name, p.DrainGrace, p.DrainGraceText, p.SeedDerivation)
		}
		ep := p.Epoch
		wantEpCeiling := ep.SteadyTurns + ep.ControlTurns + ep.MaxTaskTurns + ep.MaxPrimingRequests + ep.MaxReplayRetries
		if ep.RequestCeiling != wantEpCeiling || p.RequestCeiling != 2*wantEpCeiling {
			t.Errorf("%s: ceiling mismatch ep=%d (want %d), plan=%d", tc.name, ep.RequestCeiling, wantEpCeiling, p.RequestCeiling)
		}
		if ep.TaskFixtures != tc.wantTaskFix || p.TaskFixtures != 2*tc.wantTaskFix {
			t.Errorf("%s: task fixtures mismatch ep=%d plan=%d want=%d", tc.name, ep.TaskFixtures, p.TaskFixtures, tc.wantTaskFix)
		}
		if len(p.Prerequisites) != 1 || !p.Prerequisites[0].Required || p.Prerequisites[0].Name != "matching_qualifying_normal_receipt" {
			t.Errorf("%s: prerequisites = %+v", tc.name, p.Prerequisites)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestBuildRejectsInvalidOptions(t *testing.T) {
	invalid := []struct {
		name string
		opts Options
	}{
		{name: "unsupported profile", opts: Options{Profile: "turbo"}},
		{name: "quick wrong duration", opts: Options{Profile: "quick", Duration: 10 * time.Minute}},
		{name: "quick wrong concurrency", opts: Options{Profile: "quick", PreferredConcurrency: 4}},
		{name: "normal wrong duration", opts: Options{Profile: "normal", Duration: 15 * time.Minute}},
		{name: "normal unsupported concurrency 3", opts: Options{Profile: "normal", PreferredConcurrency: 3}},
		{name: "normal negative concurrency", opts: Options{Profile: "normal", PreferredConcurrency: -1}},
		{name: "burn-in missing concurrency", opts: Options{Profile: "burn-in"}},
		{name: "burn-in unsupported concurrency", opts: Options{Profile: "burn-in", PreferredConcurrency: 3}},
		{name: "burn-in wrong duration", opts: Options{Profile: "burn-in", PreferredConcurrency: 2, Duration: time.Hour}},
	}
	for _, tc := range invalid {
		_, err := Build(tc.opts)
		if err == nil {
			t.Errorf("%s: expected error, got nil", tc.name)
			continue
		}
		if !errors.Is(err, errProfile) {
			t.Errorf("%s: error %v does not wrap errProfile", tc.name, err)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestBuildIsDeterministicAndJSONRoundTrips(t *testing.T) {
	for _, opts := range []Options{
		{Profile: "quick"},
		{Profile: "normal", PreferredConcurrency: 4},
		{Profile: "burn-in", PreferredConcurrency: 2, Duration: 2 * time.Hour},
	} {
		a, err := Build(opts)
		if err != nil {
			t.Fatal(err)
		}
		b, err := Build(opts)
		if err != nil {
			t.Fatal(err)
		}
		ja, err := json.Marshal(a)
		if err != nil {
			t.Fatal(err)
		}
		jb, err := json.Marshal(b)
		if err != nil {
			t.Fatal(err)
		}
		if string(ja) != string(jb) {
			t.Fatalf("non-deterministic Build output for %+v", opts)
		}
		var back Plan
		if err := json.Unmarshal(ja, &back); err != nil {
			t.Fatalf("Unmarshal(Plan): %v", err)
		}
		if !reflect.DeepEqual(a, back) {
			t.Errorf("JSON round-trip mismatch for %s:\ngot:  %+v\nwant: %+v", opts.Profile, back, a)
		}
	}
}
