package main

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/speedab"
)

func TestSpeedABNotYetExit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.json")
	if err := os.WriteFile(path, []byte(`{"schema":"fak.speed-ab.v1","runs":[]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var out strings.Builder
	if code := runSpeedAB(&out, &out, []string{"--manifest", path}); code != 3 {
		t.Fatalf("code=%d out=%s", code, out.String())
	}
	if !strings.Contains(out.String(), `"verdict": "NOT_YET"`) {
		t.Fatalf("out=%s", out.String())
	}
}

type stubSpeedABRecipe struct {
	out speedABRecipeOutput
	err error
}

func (stubSpeedABRecipe) Name() string        { return "stub" }
func (stubSpeedABRecipe) Description() string { return "test stub" }
func (s stubSpeedABRecipe) Expand(speedABRecipeContext) (speedABRecipeOutput, error) {
	return s.out, s.err
}

func TestRecipeExpand(t *testing.T) {
	pinned := speedABRecipeContext{
		Model: "model-x", Account: "acct-1", Revision: "rev-9",
		SLATargets: map[string]float64{"p90_sla_ms": 500},
	}

	t.Run("each preset compiles to exactly one branch", func(t *testing.T) {
		for name, want := range map[string]string{
			"paired-arms":               "scenarios",
			"arm-grid":                  "sweep_parameters",
			"min-samples-under-p90-sla": "adaptive_search",
		} {
			out, err := compileSpeedABRecipe(name, pinned)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			set := map[string]bool{
				"adaptive_search":  out.Search != nil,
				"sweep_parameters": len(out.Grid) > 0,
				"scenarios":        len(out.Scenarios) > 0,
			}
			for branch, isSet := range set {
				if isSet != (branch == want) {
					t.Errorf("%s: branch %s set=%v, want only %s", name, branch, isSet, want)
				}
			}
		}
	})

	t.Run("recipe name never reaches the plan", func(t *testing.T) {
		for _, r := range speedABRecipes {
			out, err := compileSpeedABRecipe(r.Name(), pinned)
			if err != nil {
				t.Fatalf("%s: %v", r.Name(), err)
			}
			b, err := json.Marshal(out)
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(string(b), r.Name()) {
				t.Errorf("plan for %q carries the recipe name: %s", r.Name(), b)
			}
			var keys map[string]json.RawMessage
			if err := json.Unmarshal(b, &keys); err != nil {
				t.Fatal(err)
			}
			for key := range keys {
				switch key {
				case "adaptive_search", "sweep_parameters", "scenarios", "sla_filters", "post_process":
				default:
					t.Errorf("plan for %q has non-canonical key %q", r.Name(), key)
				}
			}
		}
	})

	t.Run("paired arms pin every scenario", func(t *testing.T) {
		out, err := compileSpeedABRecipe("paired-arms", pinned)
		if err != nil {
			t.Fatal(err)
		}
		var names []string
		for _, s := range out.Scenarios {
			names = append(names, s.Name)
			if s.Model != "model-x" || s.Account != "acct-1" || s.Revision != "rev-9" {
				t.Errorf("scenario %q is not pinned: %+v", s.Name, s)
			}
		}
		if got, want := strings.Join(names, ","), "grind-fast,grind-standard,interactive-fast,interactive-standard"; got != want {
			t.Fatalf("scenarios = %s, want %s", got, want)
		}
		if _, err := compileSpeedABRecipe("paired-arms", speedABRecipeContext{Model: "model-x"}); err == nil {
			t.Fatal("paired-arms compiled without account and revision pins")
		}
	})

	t.Run("sla target and bounds", func(t *testing.T) {
		out, err := compileSpeedABRecipe("min-samples-under-p90-sla", pinned)
		if err != nil {
			t.Fatal(err)
		}
		if out.Search.Lo != 2 || out.Search.Hi != 32 || out.Search.Parameter != "samples" || out.Search.Maximize {
			t.Fatalf("default search = %+v", *out.Search)
		}
		if len(out.SLAFilters) != 1 || out.SLAFilters[0] != (speedABSLAFilter{Metric: "p90_ms", Op: "<=", Threshold: 500}) {
			t.Fatalf("sla filters = %+v", out.SLAFilters)
		}
		if out.PostProcess != "sla-gate" {
			t.Fatalf("post process = %q", out.PostProcess)
		}

		overridden := pinned
		overridden.Overrides = map[string]int{"samples_min": 4, "samples_max": 9}
		out, err = compileSpeedABRecipe("min-samples-under-p90-sla", overridden)
		if err != nil || out.Search.Lo != 4 || out.Search.Hi != 9 {
			t.Fatalf("overridden search = %+v err = %v", out.Search, err)
		}

		inverted := pinned
		inverted.Overrides = map[string]int{"samples_min": 9, "samples_max": 9}
		if _, err := compileSpeedABRecipe("min-samples-under-p90-sla", inverted); err == nil || !strings.Contains(err.Error(), "samples_min (9) must be < samples_max (9)") {
			t.Fatalf("inverted bounds err = %v", err)
		}

		nan := speedABRecipeContext{SLATargets: map[string]float64{"p90_sla_ms": math.NaN()}}
		if _, err := compileSpeedABRecipe("min-samples-under-p90-sla", nan); err == nil {
			t.Fatal("a NaN sla target compiled")
		}

		if _, err := compileSpeedABRecipe("min-samples-under-p90-sla", speedABRecipeContext{}); err == nil || !strings.Contains(err.Error(), "p90_sla_ms") {
			t.Fatalf("missing sla target err = %v", err)
		}
	})

	t.Run("unknown recipe lists the known ones", func(t *testing.T) {
		_, err := compileSpeedABRecipe("no-such-recipe", pinned)
		if err == nil || !strings.Contains(err.Error(), "known: arm-grid, min-samples-under-p90-sla, paired-arms") {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("output union is exactly one of", func(t *testing.T) {
		search := &speedABSearch{Parameter: "samples", Lo: 2, Hi: 4}
		grid := map[string][]string{"speed": {"fast"}}
		scenarios := []speedABScenario{{Name: "a"}}
		for name, tc := range map[string]struct {
			out speedABRecipeOutput
			ok  bool
		}{
			"none":            {speedABRecipeOutput{}, false},
			"empty grid only": {speedABRecipeOutput{Grid: map[string][]string{}}, false},
			"search":          {speedABRecipeOutput{Search: search}, true},
			"grid":            {speedABRecipeOutput{Grid: grid}, true},
			"scenarios":       {speedABRecipeOutput{Scenarios: scenarios}, true},
			"search and grid": {speedABRecipeOutput{Search: search, Grid: grid}, false},
			"all three":       {speedABRecipeOutput{Search: search, Grid: grid, Scenarios: scenarios}, false},
			"unknown hook":    {speedABRecipeOutput{Grid: grid, PostProcess: "nope"}, false},
			"unknown metric": {speedABRecipeOutput{Grid: grid,
				SLAFilters: []speedABSLAFilter{{Metric: "p99_ms", Op: "<="}}}, false},
			"non-finite threshold": {speedABRecipeOutput{Grid: grid,
				SLAFilters: []speedABSLAFilter{{Metric: "p90_ms", Op: "<=", Threshold: math.NaN()}}}, false},
			"unknown op": {speedABRecipeOutput{Grid: grid,
				SLAFilters: []speedABSLAFilter{{Metric: "p90_ms", Op: "<"}}}, false},
		} {
			_, err := expandSpeedABRecipe(stubSpeedABRecipe{out: tc.out}, pinned)
			if (err == nil) != tc.ok {
				t.Errorf("%s: err = %v, want ok=%v", name, err, tc.ok)
			}
		}
	})

	t.Run("recipe error is wrapped with its name", func(t *testing.T) {
		boom := errors.New("boom")
		_, err := expandSpeedABRecipe(stubSpeedABRecipe{err: boom}, pinned)
		if !errors.Is(err, boom) || !strings.Contains(err.Error(), `recipe "stub"`) {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("post process hook gates on sla filters", func(t *testing.T) {
		plan, err := compileSpeedABRecipe("min-samples-under-p90-sla", pinned)
		if err != nil {
			t.Fatal(err)
		}
		report := speedab.Report{Schema: speedab.Schema, Verdict: "NET_TRUE", Comparisons: []speedab.Comparison{
			{WorkClass: "grind", Fast: speedab.Arm{P90MS: 400}, Verdict: "NET_TRUE", Reason: "ok"},
			{WorkClass: "interactive", Fast: speedab.Arm{P90MS: 900}, Verdict: "NET_TRUE", Reason: "ok"},
		}}
		got, err := applySpeedABPostProcess(plan, report)
		if err != nil {
			t.Fatal(err)
		}
		if got.Verdict != "NOT_YET" || got.Comparisons[0].Verdict != "NET_TRUE" || got.Comparisons[1].Verdict != "NOT_YET" {
			t.Fatalf("gated report = %+v", got)
		}
		if !strings.Contains(got.Comparisons[1].Reason, "p90_ms <= 500") {
			t.Fatalf("reason = %q", got.Comparisons[1].Reason)
		}
		if report.Verdict != "NET_TRUE" || report.Comparisons[1].Verdict != "NET_TRUE" {
			t.Fatal("the hook mutated its input report")
		}

		report.Comparisons[1].Fast.P90MS = 500
		if got, _ := applySpeedABPostProcess(plan, report); got.Verdict != "NET_TRUE" {
			t.Fatalf("a report meeting the sla was demoted: %+v", got)
		}

		noHook, err := compileSpeedABRecipe("arm-grid", pinned)
		if err != nil {
			t.Fatal(err)
		}
		report.Comparisons[1].Fast.P90MS = 900
		if got, err := applySpeedABPostProcess(noHook, report); err != nil || got.Verdict != "NET_TRUE" {
			t.Fatalf("a plan with no hook changed the report: %+v err=%v", got, err)
		}
		if _, err := applySpeedABPostProcess(speedABRecipeOutput{PostProcess: "nope"}, report); err == nil {
			t.Fatal("an unknown hook was applied")
		}
	})
}
