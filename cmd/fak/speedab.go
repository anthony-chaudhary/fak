package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/speedab"
)

func runSpeedAB(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("speed-ab", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifest := fs.String("manifest", "", "captured speed A/B manifest JSON")
	ramp := fs.Bool("ramp", false, "interpret --manifest as an in-run adaptive scale document (ramp_until_fail)")
	if fs.Parse(argv) != nil {
		return 2
	}
	if *manifest == "" {
		fmt.Fprintln(stderr, "speed-ab: --manifest is required")
		return 2
	}
	b, err := os.ReadFile(*manifest)
	if err != nil {
		fmt.Fprintln(stderr, "speed-ab:", err)
		return 1
	}
	if *ramp {
		in, err := parseAdaptiveScaleInput(b)
		if err != nil {
			fmt.Fprintln(stderr, "speed-ab:", err)
			return 2
		}
		res, err := adaptiveScaleRamp(in.Windows, in.SLA)
		if err != nil {
			fmt.Fprintln(stderr, "speed-ab:", err)
			return 2
		}
		if err := writeAdaptiveScaleJSONL(stdout, res.Events); err != nil {
			fmt.Fprintln(stderr, "speed-ab:", err)
			return 1
		}
		out, _ := json.Marshal(res)
		fmt.Fprintln(stdout, string(out))
		return 0
	}
	var m speedab.Manifest
	if json.Unmarshal(b, &m) != nil {
		fmt.Fprintln(stderr, "speed-ab: invalid manifest")
		return 2
	}
	r := speedab.Grade(m)
	out, _ := json.MarshalIndent(r, "", "  ")
	fmt.Fprintln(stdout, string(out))
	if r.Verdict != "NET_TRUE" {
		return 3
	}
	return 0
}
func cmdSpeedAB(argv []string) { os.Exit(runSpeedAB(os.Stdout, os.Stderr, argv)) }

// Speed A/B recipes: named presets that compile to a canonical plan.
//
// (SearchRecipe, SearchRecipeContext, SearchRecipeOutput and its
// exactly-one-branch validator, resolve_concurrency_bounds), pinned at
// in idiomatic Go against the speed A/B run shape; no upstream code is
// copied. Study note: fak-private docs/research/STUDY-aiperf-2026-11-09.md,
// candidate aiperf-ops-5.
//
// Why the name stops at the CLI boundary: a preset that survives into stored
// configuration makes every reader of that configuration recipe-aware, and a
// later change to the preset silently re-means old records. Compiling the
// name away leaves one canonical shape downstream, and speedABRecipeOutput
// has no field that could carry the name.

// speedABRecipeContext is what a recipe may read while expanding. Recipes
// treat it as read-only.
type speedABRecipeContext struct {
	// Model, Account and Revision pin both arms of every comparison.
	Model    string
	Account  string
	Revision string
	// SLATargets holds recipe-specific thresholds keyed by short name, such
	// as "p90_sla_ms".
	SLATargets map[string]float64
	// Overrides holds recipe-specific integer knobs, such as "samples_min".
	Overrides map[string]int
}

// speedABSLAFilter admits a comparison when the fast arm's Metric satisfies
// Op against Threshold. Op is "<=" or ">=".
type speedABSLAFilter struct {
	Metric    string  `json:"metric"`
	Op        string  `json:"op"`
	Threshold float64 `json:"threshold"`
}

// speedABSearch is a bounded search over one integer parameter, Lo < Hi.
type speedABSearch struct {
	Parameter string `json:"parameter"`
	Lo        int    `json:"lo"`
	Hi        int    `json:"hi"`
	Objective string `json:"objective"`
	Maximize  bool   `json:"maximize"`
}

// speedABScenario is one pre-flattened run to collect: a named arm with its
// pins already applied.
type speedABScenario struct {
	Name      string `json:"name"`
	WorkClass string `json:"work_class"`
	Speed     string `json:"speed"`
	Model     string `json:"model"`
	Account   string `json:"account"`
	Revision  string `json:"revision"`
}

// speedABRecipeOutput is the canonical plan a recipe compiles to. Exactly
// one of Search, Grid and Scenarios is set; the rest of the plan rides
// alongside whichever branch that is.
type speedABRecipeOutput struct {
	Search    *speedABSearch      `json:"adaptive_search,omitempty"`
	Grid      map[string][]string `json:"sweep_parameters,omitempty"`
	Scenarios []speedABScenario   `json:"scenarios,omitempty"`

	SLAFilters []speedABSLAFilter `json:"sla_filters,omitempty"`
	// PostProcess names a hook in speedABPostProcessors run over the graded
	// report; empty means none.
	PostProcess string `json:"post_process,omitempty"`
}

// validate counts an empty grid or scenario list as unset: it would compile
// to zero runs and reads the same as nil once serialized.
func (o speedABRecipeOutput) validate() error {
	search, grid, scenarios := o.Search != nil, len(o.Grid) > 0, len(o.Scenarios) > 0
	set := 0
	for _, branch := range []bool{search, grid, scenarios} {
		if branch {
			set++
		}
	}
	if set != 1 {
		return fmt.Errorf("recipe output needs exactly one of adaptive_search, sweep_parameters, scenarios (got adaptive_search=%t, sweep_parameters=%t, scenarios=%t)",
			search, grid, scenarios)
	}
	for _, f := range o.SLAFilters {
		if _, ok := speedABArmMetric(speedab.Arm{}, f.Metric); !ok {
			return fmt.Errorf("sla filter names unknown metric %q", f.Metric)
		}
		if f.Op != "<=" && f.Op != ">=" {
			return fmt.Errorf("sla filter on %q has unknown op %q", f.Metric, f.Op)
		}
		if math.IsNaN(f.Threshold) || math.IsInf(f.Threshold, 0) {
			return fmt.Errorf("sla filter on %q has a non-finite threshold", f.Metric)
		}
	}
	if o.PostProcess != "" {
		if _, ok := speedABPostProcessors[o.PostProcess]; !ok {
			return fmt.Errorf("unknown post-process hook %q", o.PostProcess)
		}
	}
	return nil
}

// speedABRecipe is the preset contract: a name the CLI accepts, a one-line
// description, and an expansion into the canonical plan. Expand returns an
// error when a required input is missing or the context conflicts with the
// recipe's assumptions.
type speedABRecipe interface {
	Name() string
	Description() string
	Expand(ctx speedABRecipeContext) (speedABRecipeOutput, error)
}

type speedABBuiltinRecipe struct {
	name        string
	description string
	expand      func(speedABRecipeContext) (speedABRecipeOutput, error)
}

func (r speedABBuiltinRecipe) Name() string        { return r.name }
func (r speedABBuiltinRecipe) Description() string { return r.description }
func (r speedABBuiltinRecipe) Expand(ctx speedABRecipeContext) (speedABRecipeOutput, error) {
	return r.expand(ctx)
}

var (
	speedABWorkClasses = []string{"grind", "interactive"}
	speedABSpeeds      = []string{"fast", "standard"}
)

var speedABRecipes = []speedABRecipe{
	speedABBuiltinRecipe{
		name:        "paired-arms",
		description: "one pinned fast and standard run per work class",
		expand: func(ctx speedABRecipeContext) (speedABRecipeOutput, error) {
			if ctx.Model == "" || ctx.Account == "" || ctx.Revision == "" {
				return speedABRecipeOutput{}, errors.New("model, account and revision pins are required")
			}
			var out speedABRecipeOutput
			for _, class := range speedABWorkClasses {
				for _, speed := range speedABSpeeds {
					out.Scenarios = append(out.Scenarios, speedABScenario{
						Name: class + "-" + speed, WorkClass: class, Speed: speed,
						Model: ctx.Model, Account: ctx.Account, Revision: ctx.Revision,
					})
				}
			}
			return out, nil
		},
	},
	speedABBuiltinRecipe{
		name:        "arm-grid",
		description: "the full work-class by speed grid",
		expand: func(speedABRecipeContext) (speedABRecipeOutput, error) {
			return speedABRecipeOutput{Grid: map[string][]string{
				"work_class": slices.Clone(speedABWorkClasses),
				"speed":      slices.Clone(speedABSpeeds),
			}}, nil
		},
	},
	speedABBuiltinRecipe{
		name:        "min-samples-under-p90-sla",
		description: "fewest turnaround samples whose fast arm still meets a p90 latency SLA",
		expand: func(ctx speedABRecipeContext) (speedABRecipeOutput, error) {
			sla, ok := ctx.SLATargets["p90_sla_ms"]
			if !ok || !(sla > 0) {
				return speedABRecipeOutput{}, errors.New("a positive p90_sla_ms target is required")
			}
			// Grading refuses a run with fewer than two turnaround samples.
			lo, hi, err := speedABResolveBounds(ctx.Overrides, "samples_min", "samples_max", 2, 32)
			if err != nil {
				return speedABRecipeOutput{}, err
			}
			return speedABRecipeOutput{
				Search:      &speedABSearch{Parameter: "samples", Lo: lo, Hi: hi, Objective: "samples"},
				SLAFilters:  []speedABSLAFilter{{Metric: "p90_ms", Op: "<=", Threshold: sla}},
				PostProcess: "sla-gate",
			}, nil
		},
	},
}

// speedABResolveBounds reads a lower and upper override, falling back to the
// recipe's defaults, and rejects inverted or empty bounds here so the user
// sees the cause instead of a failure deep in whatever consumes the range.
func speedABResolveBounds(overrides map[string]int, loKey, hiKey string, defaultLo, defaultHi int) (int, int, error) {
	lo, hi := defaultLo, defaultHi
	if v, ok := overrides[loKey]; ok {
		lo = v
	}
	if v, ok := overrides[hiKey]; ok {
		hi = v
	}
	if lo >= hi {
		return 0, 0, fmt.Errorf("%s (%d) must be < %s (%d)", loKey, lo, hiKey, hi)
	}
	return lo, hi, nil
}

// compileSpeedABRecipe is the CLI boundary: it resolves a preset name,
// expands it, and returns only the canonical plan.
func compileSpeedABRecipe(name string, ctx speedABRecipeContext) (speedABRecipeOutput, error) {
	known := make([]string, 0, len(speedABRecipes))
	for _, r := range speedABRecipes {
		if r.Name() == name {
			return expandSpeedABRecipe(r, ctx)
		}
		known = append(known, r.Name())
	}
	sort.Strings(known)
	return speedABRecipeOutput{}, fmt.Errorf("unknown speed-ab recipe %q (known: %s)", name, strings.Join(known, ", "))
}

// expandSpeedABRecipe expands one recipe and refuses an output that breaks
// the plan contract, so a faulty recipe fails here and not downstream.
func expandSpeedABRecipe(r speedABRecipe, ctx speedABRecipeContext) (speedABRecipeOutput, error) {
	out, err := r.Expand(ctx)
	if err == nil {
		err = out.validate()
	}
	if err != nil {
		return speedABRecipeOutput{}, fmt.Errorf("recipe %q: %w", r.Name(), err)
	}
	return out, nil
}

func speedABArmMetric(a speedab.Arm, metric string) (float64, bool) {
	switch metric {
	case "p50_ms":
		return a.P50MS, true
	case "p90_ms":
		return a.P90MS, true
	case "tool_calls_per_hour":
		return a.ToolCallsPerHour, true
	default:
		return 0, false
	}
}

func (f speedABSLAFilter) admits(a speedab.Arm) bool {
	v, ok := speedABArmMetric(a, f.Metric)
	if !ok {
		return false
	}
	if f.Op == ">=" {
		return v >= f.Threshold
	}
	return f.Op == "<=" && v <= f.Threshold
}

// speedABPostProcessors is the closed set of post-process hooks a recipe may
// name. A hook returns a new report and leaves its input untouched.
var speedABPostProcessors = map[string]func(speedab.Report, []speedABSLAFilter) speedab.Report{
	// sla-gate withdraws NET_TRUE from any comparison whose fast arm misses
	// an SLA filter: faster than standard is not enough when the recipe set
	// an absolute bar.
	"sla-gate": func(r speedab.Report, filters []speedABSLAFilter) speedab.Report {
		out := r
		out.Comparisons = slices.Clone(r.Comparisons)
		for i, c := range out.Comparisons {
			if c.Verdict != "NET_TRUE" {
				continue
			}
			for _, f := range filters {
				if !f.admits(c.Fast) {
					out.Comparisons[i].Verdict = "NOT_YET"
					out.Comparisons[i].Reason = fmt.Sprintf("fast arm missed sla %s %s %g", f.Metric, f.Op, f.Threshold)
					out.Verdict = "NOT_YET"
					break
				}
			}
		}
		return out
	},
}

// applySpeedABPostProcess runs the plan's post-process hook, if any, over a
// graded report.
func applySpeedABPostProcess(plan speedABRecipeOutput, r speedab.Report) (speedab.Report, error) {
	if plan.PostProcess == "" {
		return r, nil
	}
	hook, ok := speedABPostProcessors[plan.PostProcess]
	if !ok {
		return r, fmt.Errorf("unknown post-process hook %q", plan.PostProcess)
	}
	return hook(r, plan.SLAFilters), nil
}
