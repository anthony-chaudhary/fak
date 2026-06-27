package nightrun

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"
)

// nightrun_witness_test.go covers the curated-witness registry hygiene fixes (#989-#992):
// the explicit Manual no-auto-run marker, ReqWeights gating on the serve/throughput
// witnesses, the detached-run build-cache preflight, and the per-row heavy timeouts.

// --- #989: explicit Manual marker is authoritative over the heuristic ----------------

// TestManualMarkerIsAuthoritative pins that a Task flagged Manual is never auto-runnable
// even when its Run reads as a clean, executable command — the `script.sh   # comment`
// shape that has no placeholder and no arrow, which the heuristic alone would exec every
// sweep and record as a spurious failure (the #989 residual gap).
func TestManualMarkerIsAuthoritative(t *testing.T) {
	bareRecipe := "experiments/agent-live/run.sh   # the committed recipe"
	// Heuristic alone (Manual unset) judges this bare recipe auto-runnable — the gap.
	if !(Task{Run: bareRecipe}).autoRunnable() {
		t.Fatalf("precondition: a bare `script.sh # comment` should be heuristic-auto-runnable; the test would prove nothing otherwise")
	}
	// The explicit marker overrides it.
	if (Task{Run: bareRecipe, Manual: true}).autoRunnable() {
		t.Errorf("a Manual task must NOT be auto-runnable regardless of how its Run reads (%q)", bareRecipe)
	}
	// Manual also wins for an already-concrete command (a row an operator pins as manual).
	if (Task{Run: "echo ok", Manual: true}).autoRunnable() {
		t.Error("Manual:true must override an otherwise-runnable command")
	}
}

// TestRegistryRecipeWitnessesAreManual asserts every curated witness whose Run is a human
// recipe (placeholder, prose arrow, or bare script+comment) carries Manual:true, so it is
// surfaced to an operator but never auto-run — the registry intent is authoritative, not
// a heuristic accident.
func TestRegistryRecipeWitnessesAreManual(t *testing.T) {
	wantManual := map[string]bool{
		"witness-glm52-native-load-10min":         true, // <glm-5.2.gguf> placeholder
		"witness-glm52-vllm-throughput-parity":    true, // bare `run.sh   # comment`
		"witness-terminalbench-live-credentialed": true, // <official-suite> placeholder
		"witness-a100-qwen-serve-first-run":       true, // prose arrow recipe
	}
	for _, task := range witnessTasks() {
		if wantManual[task.ID] && !task.Manual {
			t.Errorf("recipe witness %q must carry Manual:true (Run=%q)", task.ID, task.Run)
		}
		// A Manual row must also be non-auto-runnable end to end (the loop skips it).
		if task.Manual && task.autoRunnable() {
			t.Errorf("witness %q is Manual but autoRunnable() returned true (Run=%q)", task.ID, task.Run)
		}
	}
}

// TestManualWitnessSurfacedInPlan pins #989's discoverability criterion: a Manual recipe
// witness is still RANKED feasible (surfaced by plan/next for a human to run by hand), it
// is only declined at the auto-RUN gate. A feasible-but-manual task must appear in the
// ranking, not be filtered out of the plan.
func TestManualWitnessSurfacedInPlan(t *testing.T) {
	caps := Capabilities{Box: "a100", GPU: "cuda", Weights: true, Net: true, Creds: map[string]bool{}}
	now := mustTime(t, "2026-06-27T00:00:00Z")
	manual := Task{ID: "witness-recipe", Value: ValueFrontier, Requires: []Requirement{ReqCUDA, ReqWeights}, Run: "run.sh # recipe", Manual: true}
	ranked := Rank([]Task{manual}, caps, nil, now)
	if len(ranked) != 1 || !ranked[0].Feasible {
		t.Fatalf("a Manual witness the box can satisfy must still be ranked feasible (surfaced by plan), got %+v", ranked)
	}
}

// --- #990: serve/throughput witnesses gate on ReqWeights -----------------------------

// TestServeWitnessesInfeasibleWithoutWeights pins that the live serve/throughput witnesses
// declare ReqWeights, so a GPU box with weights=no reports them INFEASIBLE with the precise
// "needs local model weights" reason instead of feasible-but-failing, and a GPU box with
// weights=yes finds them feasible.
func TestServeWitnessesInfeasibleWithoutWeights(t *testing.T) {
	weightless := Capabilities{Box: "a100-bare", GPU: "cuda", Weights: false, Net: true, Creds: map[string]bool{}}
	weighted := Capabilities{Box: "a100", GPU: "cuda", Weights: true, Net: true, Creds: map[string]bool{}}

	byID := map[string]Task{}
	for _, task := range witnessTasks() {
		byID[task.ID] = task
	}
	for _, id := range []string{"witness-a100-qwen-serve-first-run", "witness-glm52-vllm-throughput-parity"} {
		task, ok := byID[id]
		if !ok {
			t.Fatalf("registry no longer has %q", id)
		}
		ok, why := weightless.Satisfies(task)
		if ok {
			t.Errorf("%q must be INFEASIBLE on a weightless GPU box", id)
		}
		if !strings.Contains(why, "needs local model weights") {
			t.Errorf("%q infeasibility reason = %q, want the 'needs local model weights' reason", id, why)
		}
		// On a GPU box WITH weights (and net, for the throughput row) it is feasible.
		netBox := weighted
		netBox.Net = true
		if ok, why := netBox.Satisfies(task); !ok {
			t.Errorf("%q must be feasible on a weights=yes GPU box, got infeasible: %q", id, why)
		}
	}
}

// --- #991: detached-run build-cache preflight ----------------------------------------

// TestPreflightGoCacheBranches drives the build-cache preflight across the three cases:
// an explicit GOCACHE inherits unchanged; a derivable default (HOME present) inherits
// unchanged; neither present provisions a per-run default under the build dir with an
// actionable remediation — the detached/minimal-env case (#991).
func TestPreflightGoCacheBranches(t *testing.T) {
	buildDir := filepath.FromSlash("/run/build")

	// 1. Explicit GOCACHE wins — nothing to override.
	st := preflightGoCache(func(k string) string {
		if k == "GOCACHE" {
			return "/cache/explicit"
		}
		return ""
	}, func() (string, error) { return "", errors.New("no user cache dir") }, buildDir)
	if !st.Usable || st.Default != "" || st.Remediation != "" {
		t.Errorf("explicit GOCACHE: want usable+no-override, got %+v", st)
	}

	// 2. No GOCACHE but a derivable default (UserCacheDir resolves) — inherit, no override.
	st = preflightGoCache(func(string) string { return "" },
		func() (string, error) { return "/home/u/.cache", nil }, buildDir)
	if !st.Usable || st.Default != "" || st.Remediation != "" {
		t.Errorf("derivable default: want usable+no-override, got %+v", st)
	}

	// 3. Neither set nor derivable — provision a per-run default and surface remediation.
	st = preflightGoCache(func(string) string { return "" },
		func() (string, error) { return "", errors.New("$HOME is not defined") }, buildDir)
	if !st.Usable {
		t.Error("detached case must still be usable (we provision a default so benches build)")
	}
	if st.Default != filepath.Join(buildDir, "gocache") {
		t.Errorf("per-run default = %q, want %q", st.Default, filepath.Join(buildDir, "gocache"))
	}
	if !strings.Contains(st.Remediation, "GOCACHE") || !strings.Contains(st.Remediation, "HOME") {
		t.Errorf("remediation must name HOME and GOCACHE, got %q", st.Remediation)
	}
}

// TestResolveGoCacheSetsEnvOnlyWhenProvisioned pins the cache wiring: buildEnv() inherits
// the ambient environment (nil) when the preflight provisioned nothing, and forces GOCACHE
// only when a per-run default was created.
func TestResolveGoCacheSetsEnvOnlyWhenProvisioned(t *testing.T) {
	// No per-run default → inherit (nil env).
	c := newGoRunCache()
	if env := c.buildEnv(); env != nil {
		t.Errorf("buildEnv with no provisioned cache must inherit (nil), got %v", env)
	}
	// A provisioned default → GOCACHE forced in the returned env.
	c.gocache = filepath.FromSlash("/run/build/gocache")
	env := c.buildEnv()
	found := false
	for _, kv := range env {
		if kv == "GOCACHE="+c.gocache {
			found = true
		}
	}
	if !found {
		t.Errorf("buildEnv with a provisioned cache must set GOCACHE=%s, env=%v", c.gocache, env)
	}
}

// --- #992: heavy witnesses carry an explicit per-task timeout -------------------------

// TestHeavyWitnessesCarryTimeout pins that the cold-load / serve / throughput witnesses
// declare an explicit TimeoutSec larger than the offline-lane default, so a healthy cold
// load is not truncated and recorded as a spurious timeout (#992).
func TestHeavyWitnessesCarryTimeout(t *testing.T) {
	byID := map[string]Task{}
	for _, task := range witnessTasks() {
		byID[task.ID] = task
	}
	for _, id := range []string{
		"witness-glm52-native-load-10min",
		"witness-glm52-vllm-throughput-parity",
		"witness-a100-qwen-serve-first-run",
	} {
		task, ok := byID[id]
		if !ok {
			t.Fatalf("registry no longer has %q", id)
		}
		if task.TimeoutSec <= DefaultTaskTimeoutSec {
			t.Errorf("heavy witness %q TimeoutSec = %d, must exceed the offline default %d (a cold load needs headroom)",
				id, task.TimeoutSec, DefaultTaskTimeoutSec)
		}
		if task.timeout().Seconds() != float64(task.TimeoutSec) {
			t.Errorf("%q timeout() must honor the explicit TimeoutSec", id)
		}
	}
}
