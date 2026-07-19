package dispatchtick

import (
	"strings"
	"testing"
)

// footprintBaseline returns a SPAWN_OK preflight with real headroom (cap 5, live 2)
// so a footprint fold that binds is a visible cap reduction.
func footprintBaseline(t *testing.T) PreflightResult {
	t.Helper()
	in := preflightInput()
	in.MaxWorkers = 5
	in.Kernel = KernelCheck{Alive: IntPtr(2), Target: IntPtr(9), Verdict: "FILLING"}
	res := EvaluatePreflight(in)
	if res.Verdict != PreflightOKVerdict || res.Cap != 5 || res.Live != 2 || res.Headroom != 3 {
		t.Fatalf("baseline = %s cap/live/headroom=%d/%d/%d, want SPAWN_OK 5/2/3", res.Verdict, res.Cap, res.Live, res.Headroom)
	}
	return res
}

// coldFootprintBaseline returns a SPAWN_OK preflight for a COLD fleet (0 live, cap 5)
// so the cold-start floor is a visible probe allowance rather than a deadlock.
func coldFootprintBaseline(t *testing.T) PreflightResult {
	t.Helper()
	in := preflightInput()
	in.MaxWorkers = 5
	in.Kernel = KernelCheck{Alive: IntPtr(0), Target: IntPtr(9), Verdict: "FILLING"}
	res := EvaluatePreflight(in)
	if res.Verdict != PreflightOKVerdict || res.Cap != 5 || res.Live != 0 || res.Headroom != 5 {
		t.Fatalf("cold baseline = %s cap/live/headroom=%d/%d/%d, want SPAWN_OK 5/0/5", res.Verdict, res.Cap, res.Live, res.Headroom)
	}
	return res
}

// footprintCheckCeiling builds a check whose priced ceiling is exactly n workers
// (threads-bound), keeping every other dimension unconfigured.
func footprintCheckCeiling(n int) FootprintCheck {
	return FootprintCheck{
		PerWorker: WorkerFootprint{Threads: 100},
		Budgets:   HostFootprintBudgets{MaxThreads: 100 * n},
	}
}

func TestFootprintRefusalWireTokenPinned(t *testing.T) {
	// The wire token is #3600's structured refusal reason: renaming the Go symbol
	// must never change the string peers and the sweep route on.
	if RefuseHostFootprint != "REFUSE_HOST_FOOTPRINT" {
		t.Fatalf("wire token = %q, want REFUSE_HOST_FOOTPRINT", RefuseHostFootprint)
	}
}

func TestWorkerFootprintMaxWorkersLimitingDimension(t *testing.T) {
	// The acceptance table: given per-worker footprint samples + host budgets,
	// MaxWorkers returns the correct limiting-dimension ceiling.
	measured := WorkerFootprint{Handles: 250, Threads: 200, WorkingSetMB: 1500, Conhosts: 2}
	cases := []struct {
		name     string
		fp       WorkerFootprint
		budgets  HostFootprintBudgets
		bound    bool
		ceiling  int
		limiting string
	}{
		{
			name:     "handles bind",
			fp:       measured,
			budgets:  HostFootprintBudgets{MaxHandles: 1000, MaxThreads: 4000, MaxWorkingSetMB: 32000, MaxConhosts: 20},
			bound:    true,
			ceiling:  4, // min(1000/250=4, 4000/200=20, 32000/1500=21, 20/2=10)
			limiting: "handles",
		},
		{
			name:     "threads bind",
			fp:       measured,
			budgets:  HostFootprintBudgets{MaxHandles: 10000, MaxThreads: 600, MaxWorkingSetMB: 32000, MaxConhosts: 20},
			bound:    true,
			ceiling:  3, // 600/200
			limiting: "threads",
		},
		{
			name:     "working set binds",
			fp:       measured,
			budgets:  HostFootprintBudgets{MaxHandles: 10000, MaxThreads: 4000, MaxWorkingSetMB: 3200, MaxConhosts: 20},
			bound:    true,
			ceiling:  2, // 3200/1500 truncates: a budget that holds 2.1 workers holds 2
			limiting: "ws_mb",
		},
		{
			name:     "conhost binds",
			fp:       measured,
			budgets:  HostFootprintBudgets{MaxHandles: 10000, MaxThreads: 4000, MaxWorkingSetMB: 32000, MaxConhosts: 6},
			bound:    true,
			ceiling:  3, // 6/2
			limiting: "conhost",
		},
		{
			name:     "tie names the canonical first dimension",
			fp:       WorkerFootprint{Handles: 100, Threads: 100},
			budgets:  HostFootprintBudgets{MaxHandles: 300, MaxThreads: 300},
			bound:    true,
			ceiling:  3,
			limiting: "handles",
		},
		{
			name:     "only one budget configured",
			fp:       measured,
			budgets:  HostFootprintBudgets{MaxThreads: 1000},
			bound:    true,
			ceiling:  5, // 1000/200; unconfigured budgets never bind
			limiting: "threads",
		},
		{
			name:     "unmeasured dimension cannot bind",
			fp:       WorkerFootprint{Threads: 200},
			budgets:  HostFootprintBudgets{MaxHandles: 1, MaxThreads: 1000},
			bound:    true,
			ceiling:  5, // the 1-handle budget has no measured per-worker cost: never divide by a guess
			limiting: "threads",
		},
		{
			name:     "over-budget worker prices a zero ceiling",
			fp:       WorkerFootprint{Threads: 500},
			budgets:  HostFootprintBudgets{MaxThreads: 400},
			bound:    true,
			ceiling:  0,
			limiting: "threads",
		},
		{
			name:    "no budgets disables the term",
			fp:      measured,
			budgets: HostFootprintBudgets{},
			bound:   false,
		},
		{
			name:    "no measurement disables the term",
			fp:      WorkerFootprint{},
			budgets: HostFootprintBudgets{MaxHandles: 1000, MaxThreads: 4000},
			bound:   false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.fp.MaxWorkers(tc.budgets)
			if got.Bound != tc.bound {
				t.Fatalf("bound = %v, want %v (%+v)", got.Bound, tc.bound, got)
			}
			if !tc.bound {
				if got.MaxWorkers != 0 || got.Limiting != "" || len(got.Components) != 0 {
					t.Fatalf("unbound ceiling must stay zero-valued; got %+v", got)
				}
				return
			}
			if got.MaxWorkers != tc.ceiling || got.Limiting != tc.limiting {
				t.Fatalf("ceiling/limiting = %d/%q, want %d/%q (components %v)", got.MaxWorkers, got.Limiting, tc.ceiling, tc.limiting, got.Components)
			}
			if got.Components[tc.limiting] != tc.ceiling {
				t.Fatalf("components[%s] = %d, want %d", tc.limiting, got.Components[tc.limiting], tc.ceiling)
			}
		})
	}
}

func TestMeasureWorkerFootprintRoundsUp(t *testing.T) {
	// A conservative per-worker price yields a conservative (lower) ceiling, so the
	// rolling mean rounds UP instead of letting fractional cost disappear.
	got := MeasureWorkerFootprint([]FootprintSample{
		{Handles: 10, Threads: 201, WorkingSetMB: 1499, Conhosts: 1},
		{Handles: 11, Threads: 200, WorkingSetMB: 1500, Conhosts: 2},
	})
	want := WorkerFootprint{Handles: 11, Threads: 201, WorkingSetMB: 1500, Conhosts: 2}
	if got != want {
		t.Fatalf("measured footprint = %+v, want %+v", got, want)
	}
	if zero := MeasureWorkerFootprint(nil); zero != (WorkerFootprint{}) {
		t.Fatalf("no samples must measure nothing (disable the term); got %+v", zero)
	}
}

func TestFootprintSampleFromProcsSumsWorkerTree(t *testing.T) {
	// One worker's attributed process tree: dimensions sum (nil pointers contribute
	// zero) and the console hosts are counted as their own dimension.
	handles := 300
	ws := 900
	got := FootprintSampleFromProcs([]ProcInfo{
		{PID: 10, Name: "claude", Threads: IntPtr(60), Handles: &handles, WorkingSetMB: &ws},
		{PID: 11, Name: "conhost.exe", Threads: IntPtr(6)},
		{PID: 12, Name: "OpenConsole", Threads: IntPtr(5)},
		{PID: 13, Name: "git"}, // all dimensions unmeasured
	})
	want := FootprintSample{Handles: 300, Threads: 71, WorkingSetMB: 900, Conhosts: 2}
	if got != want {
		t.Fatalf("sample = %+v, want %+v", got, want)
	}
}

func TestApplyHostFootprintCeilingOverCeilingRefuses(t *testing.T) {
	// At/over the ceiling the admission function refuses with the structured
	// REFUSE_HOST_FOOTPRINT reason: 2 live >= ceiling 2 (< cap 5).
	got := ApplyHostFootprintCeiling(footprintBaseline(t), footprintCheckCeiling(2))
	if got.OK {
		t.Fatalf("at-ceiling must refuse (the sweep stops on !ok); got ok=true verdict=%s", got.Verdict)
	}
	if got.Verdict != RefuseHostFootprint {
		t.Fatalf("verdict = %s, want %s", got.Verdict, RefuseHostFootprint)
	}
	if got.Cap != 2 || got.Headroom != 0 {
		t.Fatalf("cap/headroom = %d/%d, want the ceiling 2/0", got.Cap, got.Headroom)
	}
	if got.CapTerms.EffectiveCap != 2 || got.CapTerms.Limiting != "footprint" {
		t.Fatalf("cap_terms effective/limiting = %d/%q, want 2/footprint", got.CapTerms.EffectiveCap, got.CapTerms.Limiting)
	}
	if !strings.Contains(got.Reason, "limiting dimension threads") {
		t.Fatalf("reason must name the limiting dimension; got %q", got.Reason)
	}
	if v := got.Map()["verdict"]; v != RefuseHostFootprint {
		t.Fatalf("map verdict = %v, want %s", v, RefuseHostFootprint)
	}
}

func TestApplyHostFootprintCeilingOversubscribedHostRefuses(t *testing.T) {
	// Live already OVER the ceiling (2 live, ceiling 1): the refusal reports the
	// truthful negative headroom instead of pretending the box has room.
	got := ApplyHostFootprintCeiling(footprintBaseline(t), footprintCheckCeiling(1))
	if got.OK || got.Verdict != RefuseHostFootprint {
		t.Fatalf("over-ceiling must refuse; got ok=%v verdict=%s", got.OK, got.Verdict)
	}
	if got.Cap != 1 || got.Headroom != -1 {
		t.Fatalf("cap/headroom = %d/%d, want 1/-1", got.Cap, got.Headroom)
	}
}

func TestApplyHostFootprintCeilingBelowCeilingAdmits(t *testing.T) {
	// Below the ceiling admission is unchanged (still SPAWN_OK); a ceiling tighter
	// than the existing cap becomes the visible cap term so headroom reads truthfully.
	got := ApplyHostFootprintCeiling(footprintBaseline(t), footprintCheckCeiling(4))
	if !got.OK || got.Verdict != PreflightOKVerdict {
		t.Fatalf("below-ceiling must stay SPAWN_OK; got ok=%v verdict=%s", got.OK, got.Verdict)
	}
	if got.Cap != 4 || got.Headroom != 2 {
		t.Fatalf("cap/headroom = %d/%d, want 4/2", got.Cap, got.Headroom)
	}
	if got.CapTerms.EffectiveCap != 4 || got.CapTerms.Limiting != "footprint" {
		t.Fatalf("cap_terms effective/limiting = %d/%q, want 4/footprint", got.CapTerms.EffectiveCap, got.CapTerms.Limiting)
	}
}

func TestApplyHostFootprintCeilingAtOrAboveCapIsNoOp(t *testing.T) {
	// No regression below the ceiling: a ceiling at/above the existing cap leaves
	// the preflight untouched (the term never manufactures or raises capacity).
	base := footprintBaseline(t)
	got := ApplyHostFootprintCeiling(base, footprintCheckCeiling(5))
	if !sameAdmission(got, base) {
		t.Fatalf("ceiling at cap must be a no-op; got %+v, want %+v", got, base)
	}
	got = ApplyHostFootprintCeiling(base, footprintCheckCeiling(9))
	if !sameAdmission(got, base) {
		t.Fatalf("ceiling above cap must be a no-op; got %+v, want %+v", got, base)
	}
}

func TestApplyHostFootprintCeilingZeroValueIsNoOp(t *testing.T) {
	// The zero-value check (nothing wired) never lowers the cap, so a caller that
	// wires nothing keeps the existing thread-gate behavior byte-for-byte.
	base := footprintBaseline(t)
	if got := ApplyHostFootprintCeiling(base, FootprintCheck{}); !sameAdmission(got, base) {
		t.Fatalf("zero-value check must be a no-op; got %+v, want %+v", got, base)
	}
	// Budgets without a measurement must also abstain: never divide by a guess.
	unmeasured := FootprintCheck{Budgets: HostFootprintBudgets{MaxThreads: 1000}}
	if got := ApplyHostFootprintCeiling(base, unmeasured); !sameAdmission(got, base) {
		t.Fatalf("budget-without-measurement must be a no-op; got %+v, want %+v", got, base)
	}
}

func TestApplyHostFootprintCeilingColdStartFloorKeepsMinimalSpawn(t *testing.T) {
	// A zero priced ceiling (a bloated rolling sample) must not freeze a cold fleet
	// at a zero cap: the floor keeps one probe SPAWN_OK so the measurement that
	// imposed the ceiling can be refreshed.
	overBudget := FootprintCheck{
		PerWorker: WorkerFootprint{Threads: 500},
		Budgets:   HostFootprintBudgets{MaxThreads: 400}, // 400/500 prices a ceiling of 0
	}
	got := ApplyHostFootprintCeiling(coldFootprintBaseline(t), overBudget)
	if !got.OK || got.Verdict != PreflightOKVerdict {
		t.Fatalf("cold fleet under a zero ceiling must stay SPAWN_OK; got ok=%v verdict=%s", got.OK, got.Verdict)
	}
	if got.Cap != DefaultFootprintMinWorkers || got.Headroom != DefaultFootprintMinWorkers {
		t.Fatalf("cap/headroom = %d/%d, want the cold-start floor %d/%d", got.Cap, got.Headroom, DefaultFootprintMinWorkers, DefaultFootprintMinWorkers)
	}
	if got.CapTerms.EffectiveCap != DefaultFootprintMinWorkers || got.CapTerms.Limiting != "footprint" {
		t.Fatalf("cap_terms effective/limiting = %d/%q, want %d/footprint", got.CapTerms.EffectiveCap, got.CapTerms.Limiting, DefaultFootprintMinWorkers)
	}
}

func TestApplyHostFootprintCeilingDoesNotOverridePriorRefusal(t *testing.T) {
	// The footprint term binds only when it is the SOLE binding term: a preflight
	// that already refused keeps its higher-precedence verdict untouched.
	in := preflightInput()
	in.MaxWorkers = 2
	in.Kernel = KernelCheck{Alive: IntPtr(5), Target: IntPtr(9), Verdict: "OVER_TARGET"}
	atCap := EvaluatePreflight(in)
	if atCap.Verdict != PreflightRefuseAtCap {
		t.Fatalf("precondition: verdict = %s, want REFUSE_AT_CAP", atCap.Verdict)
	}
	got := ApplyHostFootprintCeiling(atCap, footprintCheckCeiling(1))
	if !sameAdmission(got, atCap) {
		t.Fatalf("footprint term must not override a prior refusal; got %+v, want %+v", got, atCap)
	}
}

func TestFootprintVarsExposesCeilingTriple(t *testing.T) {
	// The /debug/vars payload is the {live_workers, worker_ceiling,
	// limiting_dimension} triple the shell publishes for operators and the placer.
	vars := footprintCheckCeiling(3).Vars(2)
	if vars["live_workers"] != 2 || vars["worker_ceiling"] != 3 || vars["limiting_dimension"] != "threads" {
		t.Fatalf("vars = %v, want live 2 / ceiling 3 / threads", vars)
	}
	// Unbound (nothing wired): a nil ceiling, not a huge one -- "no ceiling
	// configured" must stay distinguishable from "ceiling is large".
	vars = FootprintCheck{}.Vars(4)
	if vars["live_workers"] != 4 || vars["worker_ceiling"] != nil || vars["limiting_dimension"] != "" {
		t.Fatalf("unbound vars = %v, want live 4 / nil ceiling / empty dimension", vars)
	}
}
