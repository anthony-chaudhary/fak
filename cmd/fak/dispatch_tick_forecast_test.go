package main

import (
	"io"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// TestDispatchPreflightFoldsForecastFloor is the #3775 live-producer witness for the #3368
// two-timescale worker floor. The slow predictive loop (a target issue-throughput rate an
// operator/scheduler writes to FAK_FLEET_TARGET_IPH) is turned into a worker FLOOR through
// fleetcap's Little's-law forecast; dispatchPreflight folds it into the input the FAST
// reactive tick (the kernel lease target) clamps UP to. It proves the four properties the
// issue's first checkable step names against the LIVE preflight, closing the gap the seam
// commit (6c64b83e6) left open ("no live producer"):
//
//   - a rising forecast RAISES the floor over a soft reactive dip (the lease target lags), so
//     capacity is pre-warmed for a ramp the reactive tick has not yet seen (limiting=="floor");
//   - the floor is bounded by the HARD ceilings (seat inventory here), never overbooking;
//   - an unset or malformed FAK_FLEET_TARGET_IPH is a total no-op (byte-identical payload);
//   - it composes via max with the operator setpoint floor (#4165), the higher floor winning.
//
// The tick-lead / never-dip-below invariant of the pure fold is unit-witnessed in
// dispatchtick.TestEvaluatePreflightForecastFloorRaisesReactiveTick; this test witnesses the
// live shell producer that finally emits a forecast into it.
func TestDispatchPreflightFoldsForecastFloor(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))

	// Pin every ambient knob the cap fold reads so the ceilings below are exact and hermetic
	// against the host's FAK_HOST_*/FAK_SESSIONS_PER_ACCOUNT calibration: 3 claude roster
	// accounts x 4 sessions -> seat total 12 (the binding HARD ceiling this test exercises);
	// the built-in per-worker host budgets over Cores=64/RAM=128000/threads=1000 -> host cap
	// 32 (never binding here); an empty stall dir -> the churn fold abstains; no setpoint.
	t.Setenv(dispatchtick.SessionsPerAccountEnv, "4")
	t.Setenv("FAK_HOST_CORES_PER_WORKER", "2")
	t.Setenv("FAK_HOST_RAM_MB_PER_WORKER", "1500")
	t.Setenv("FAK_HOST_THREADS_PER_CORE", "400")
	t.Setenv("FAK_HOST_THREADS_PER_WORKER", "200")
	t.Setenv("FAK_STALL_DIR", t.TempDir())
	t.Setenv(dispatchtick.SetpointConcurrencyEnv, "")
	t.Setenv("FAK_FLEET_SESSION_MIN", "") // default 10-minute median session (W)

	// A soft reactive dip: the kernel lease target sits at 4 while only 2 workers are live, so
	// the reactive min() alone would cap admits at 4. os procs are stubbed to 0 by the helper.
	dispatchRunExternalJSON = func(root string, _ time.Duration, name string, args ...string) (map[string]any, error) {
		return map[string]any{"alive": float64(2), "target": float64(4), "verdict": "FILLING"}, nil
	}
	// live (2) > leased (0) trips the #3109 unattributed-live census; stub the process-table
	// scan so the test spawns no real PowerShell and the worklist stays empty.
	oldRows := dispatchProbeWorkerProcessRows
	dispatchProbeWorkerProcessRows = func() ([]dispatchCodexProcessRow, error) { return nil, nil }
	t.Cleanup(func() { dispatchProbeWorkerProcessRows = oldRows })
	oldResources := dispatchBuildHostResources
	dispatchBuildHostResources = func(dispatchtick.ProcGuardInput) dispatchtick.HostResources {
		cores, ram, threads := 64, 128000, 1000
		return dispatchtick.HostResources{Cores: &cores, FreeRAMMB: &ram, TotalThreads: &threads}
	}
	t.Cleanup(func() { dispatchBuildHostResources = oldResources })

	root := t.TempDir()
	// configured max 20 leaves room for the seat ceiling (12) to be the binding hard limit.
	preflight := func(t *testing.T) map[string]any {
		t.Helper()
		got, err := dispatchPreflight(root, io.Discard, 20, "engineering", "claude")
		if err != nil {
			t.Fatalf("dispatchPreflight: %v", err)
		}
		return got
	}
	capTerms := func(t *testing.T, out map[string]any) map[string]any {
		t.Helper()
		terms, ok := out["cap_terms"].(map[string]any)
		if !ok {
			t.Fatalf("cap_terms missing or not a map: %v", out["cap_terms"])
		}
		return terms
	}

	// Unset forecast: the term is inert. cap = min(configured 20, lease 4, host 32, seat 12) = 4,
	// the reactive lease binds, no floor, no forecast_floor field -- byte-identical to before.
	t.Setenv("FAK_FLEET_TARGET_IPH", "")
	out := preflight(t)
	if out["verdict"] != dispatchtick.PreflightOKVerdict || out["cap"] != 4 {
		t.Fatalf("baseline verdict/cap = %v/%v, want SPAWN_OK/4; payload=%v", out["verdict"], out["cap"], out)
	}
	terms := capTerms(t, out)
	if terms["limiting"] != "lease" || terms["worker_floor"] != 0 {
		t.Fatalf("baseline cap_terms = %v, want limiting=lease worker_floor=0", terms)
	}
	if _, ok := out["forecast_floor"]; ok {
		t.Fatalf("unset forecast must not attach a forecast_floor field; payload=%v", out)
	}

	// Malformed forecast: parses to 0 -- identical to unset, so a garbled write can never arm a
	// wrong floor.
	t.Setenv("FAK_FLEET_TARGET_IPH", "bananas")
	out = preflight(t)
	if out["cap"] != 4 {
		t.Fatalf("malformed forecast cap = %v, want untouched 4; payload=%v", out["cap"], out)
	}
	if _, ok := out["forecast_floor"]; ok {
		t.Fatalf("malformed forecast must not attach a forecast_floor field; payload=%v", out)
	}

	// An active forecast that trails the reactive demand must not perturb the decision. Six
	// issues/hour over a 10-minute session forecasts one worker, below the lease cap of 4: the
	// forecast stays observable, but the lease remains limiting and cap/verdict stay unchanged.
	t.Setenv("FAK_FLEET_TARGET_IPH", "6")
	out = preflight(t)
	if out["verdict"] != dispatchtick.PreflightOKVerdict || out["cap"] != 4 {
		t.Fatalf("non-leading forecast verdict/cap = %v/%v, want SPAWN_OK/4; payload=%v", out["verdict"], out["cap"], out)
	}
	terms = capTerms(t, out)
	if terms["limiting"] != "lease" || terms["worker_floor"] != 1 {
		t.Fatalf("non-leading cap_terms = %v, want limiting=lease worker_floor=1", terms)
	}

	// Pre-warm: a target of 30 issues/hour over a 10-min median session forecasts
	// ceil(30 * 10/60) = 5 workers. The floor lifts the effective cap from the reactive 4 to 5
	// a tick before the lease target catches up -- limiting flips to "floor".
	t.Setenv("FAK_FLEET_TARGET_IPH", "30")
	out = preflight(t)
	if out["verdict"] != dispatchtick.PreflightOKVerdict || out["cap"] != 5 {
		t.Fatalf("pre-warm verdict/cap = %v/%v, want SPAWN_OK/5; payload=%v", out["verdict"], out["cap"], out)
	}
	terms = capTerms(t, out)
	if terms["limiting"] != "floor" || terms["worker_floor"] != 5 {
		t.Fatalf("pre-warm cap_terms = %v, want limiting=floor worker_floor=5", terms)
	}
	ff, ok := out["forecast_floor"].(map[string]any)
	if !ok {
		t.Fatalf("active forecast must attach forecast_floor; payload=%v", out)
	}
	if ff["target_iph"] != 30.0 || ff["session_min"] != fleetForecastDefaultSessionMinutes || ff["required_workers"] != 5 {
		t.Fatalf("pre-warm forecast_floor = %v, want target_iph=30 session_min=10 required_workers=5", ff)
	}

	// Rising forecast keeps leading: a target of 60 issues/hour forecasts 10 workers, so the
	// floor lifts the cap further to 10 while the reactive lease still sits at 4.
	t.Setenv("FAK_FLEET_TARGET_IPH", "60")
	out = preflight(t)
	if out["cap"] != 10 {
		t.Fatalf("rising forecast cap = %v, want 10; payload=%v", out["cap"], out)
	}
	terms = capTerms(t, out)
	if terms["limiting"] != "floor" || terms["worker_floor"] != 10 {
		t.Fatalf("rising cap_terms = %v, want limiting=floor worker_floor=10", terms)
	}

	// Never overbooks: a target of 180 issues/hour forecasts ceil(180 * 10/60) = 30 workers,
	// far past the seat pool. The applied floor is clamped by the HARD seat ceiling (12), so the
	// cap lands at 12 -- pre-warming beats a soft reactive dip without overbooking the seats. The
	// forecast_floor payload still reports the RAW 30 so an operator sees the ceiling bit.
	t.Setenv("FAK_FLEET_TARGET_IPH", "180")
	out = preflight(t)
	if out["cap"] != 12 {
		t.Fatalf("overbook-guard cap = %v, want seat-clamped 12; payload=%v", out["cap"], out)
	}
	terms = capTerms(t, out)
	if terms["limiting"] != "floor" || terms["worker_floor"] != 12 {
		t.Fatalf("overbook-guard cap_terms = %v, want limiting=floor worker_floor=12 (seat-bounded)", terms)
	}
	ff, _ = out["forecast_floor"].(map[string]any)
	if ff["required_workers"] != 30 {
		t.Fatalf("overbook-guard forecast_floor = %v, want raw required_workers=30", ff)
	}

	// A shorter median session (W) needs fewer standing workers for the same rate: target 60,
	// session 5 min forecasts ceil(60 * 5/60) = 5, so the FAK_FLEET_SESSION_MIN knob moves the
	// floor as expected.
	t.Setenv("FAK_FLEET_TARGET_IPH", "60")
	t.Setenv("FAK_FLEET_SESSION_MIN", "5")
	out = preflight(t)
	if out["cap"] != 5 {
		t.Fatalf("session-knob cap = %v, want 5; payload=%v", out["cap"], out)
	}
	terms = capTerms(t, out)
	if terms["worker_floor"] != 5 {
		t.Fatalf("session-knob worker_floor = %v, want 5", terms["worker_floor"])
	}
	t.Setenv("FAK_FLEET_SESSION_MIN", "") // restore default for the composition case

	// Composition with the operator setpoint (#4165): both are FLOORS and compose via max. A
	// grow setpoint of 8 and a forecast of 5 (target 30) -> the higher floor (8) wins, and BOTH
	// surface so an operator can see each producer's contribution.
	t.Setenv(dispatchtick.SetpointConcurrencyEnv, "8")
	t.Setenv("FAK_FLEET_TARGET_IPH", "30")
	out = preflight(t)
	if out["cap"] != 8 {
		t.Fatalf("compose cap = %v, want max(setpoint 8, forecast 5)=8; payload=%v", out["cap"], out)
	}
	terms = capTerms(t, out)
	if terms["limiting"] != "floor" || terms["worker_floor"] != 8 {
		t.Fatalf("compose cap_terms = %v, want limiting=floor worker_floor=8", terms)
	}
	if _, ok := out["setpoint"].(map[string]any); !ok {
		t.Fatalf("compose must still surface the setpoint plan; payload=%v", out)
	}
	if ff, ok := out["forecast_floor"].(map[string]any); !ok || ff["required_workers"] != 5 {
		t.Fatalf("compose must still surface the forecast floor (required_workers=5); payload=%v", out["forecast_floor"])
	}
}
