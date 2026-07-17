package main

import (
	"io"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// TestDispatchPreflightFoldsOperatorSetpoint is the #4165 live-wiring witness for the
// #4036 operator concurrency setpoint: dispatchPreflight reads FAK_DISPATCH_SETPOINT,
// folds it through ReconcileSetpoint, and the plan ACTUALLY moves the admitted cap --
// a setpoint below live shrinks admits to the post-drain target (via the #4038
// contraction term), a setpoint above live raises the cap toward the level over a
// soft reactive dip (via the ceiling-bounded #3368 floor), and an unset or malformed
// setpoint leaves the payload byte-identical to before the knob existed.
func TestDispatchPreflightFoldsOperatorSetpoint(t *testing.T) {
	oldBuildResources := dispatchBuildHostResources
	dispatchBuildHostResources = func(dispatchtick.ProcGuardInput) dispatchtick.HostResources {
		return dispatchtick.HostResources{Cores: dispatchtick.IntPtr(32), FreeRAMMB: dispatchtick.IntPtr(128000), TotalThreads: dispatchtick.IntPtr(500)}
	}
	t.Cleanup(func() { dispatchBuildHostResources = oldBuildResources })
	withDispatchJSONHelper(t, dispatchHappyHelper(t))

	// Pin every ambient knob the cap fold reads so the arithmetic below is exact:
	// 3 claude roster accounts x 4 sessions -> seat total 12 (never the binding term
	// for the caps asserted here); an empty stall dir -> the churn fold abstains.
	t.Setenv(dispatchtick.SessionsPerAccountEnv, "4")
	t.Setenv("FAK_STALL_DIR", t.TempDir())

	// The kernel probe drives the live count (live = max(kernel alive, os procs);
	// os procs are stubbed to 0 by withDispatchJSONHelper). Mutable so the grow
	// phase can re-point the fleet at a different live/target level.
	kernelAlive, kernelTarget := 5, 8
	dispatchRunExternalJSON = func(root string, _ time.Duration, name string, args ...string) (map[string]any, error) {
		return map[string]any{"alive": float64(kernelAlive), "target": float64(kernelTarget), "verdict": "FILLING"}, nil
	}
	// live > leased trips the #3109 unattributed-live census; stub the process-table
	// scan so the test spawns no real PowerShell and the worklist stays empty.
	oldRows := dispatchProbeWorkerProcessRows
	dispatchProbeWorkerProcessRows = func() ([]dispatchCodexProcessRow, error) { return nil, nil }
	t.Cleanup(func() { dispatchProbeWorkerProcessRows = oldRows })

	root := t.TempDir()
	preflight := func(t *testing.T) map[string]any {
		t.Helper()
		got, err := dispatchPreflight(root, io.Discard, 8, "engineering", "claude")
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

	// Unset setpoint: the inactive plan is a total no-op. cap = min(configured 8,
	// lease target 8, host 32, seat 12) = 8, live 5, spawn OK, no setpoint field.
	t.Setenv(dispatchtick.SetpointConcurrencyEnv, "")
	out := preflight(t)
	if out["verdict"] != dispatchtick.PreflightOKVerdict || out["cap"] != 8 {
		t.Fatalf("baseline verdict/cap = %v/%v, want SPAWN_OK/8; payload=%v", out["verdict"], out["cap"], out)
	}
	if _, ok := out["setpoint"]; ok {
		t.Fatalf("unset setpoint must not attach a setpoint field; payload=%v", out)
	}

	// Malformed setpoint: parses to the inactive sentinel -- identical to unset, so
	// a garbled setpoint write can never accidentally drain the fleet.
	t.Setenv(dispatchtick.SetpointConcurrencyEnv, "bananas")
	out = preflight(t)
	if out["cap"] != 8 {
		t.Fatalf("malformed setpoint cap = %v, want untouched 8; payload=%v", out["cap"], out)
	}
	if _, ok := out["setpoint"]; ok {
		t.Fatalf("malformed setpoint must not attach a setpoint field; payload=%v", out)
	}

	// Drain: setpoint 2 below live 5 shrinks ADMITS to the post-drain target -- the
	// contraction term caps the effective cap at 2, so with 5 live the preflight
	// refuses at cap (no new worker lands on capacity being reclaimed; the 3 surplus
	// workers drain as they finish, never killed).
	t.Setenv(dispatchtick.SetpointConcurrencyEnv, "2")
	out = preflight(t)
	if out["verdict"] != dispatchtick.PreflightRefuseAtCap || out["cap"] != 2 {
		t.Fatalf("drain verdict/cap = %v/%v, want REFUSE_AT_CAP/2; payload=%v", out["verdict"], out["cap"], out)
	}
	terms := capTerms(t, out)
	if terms["limiting"] != "contraction" || terms["effective_cap"] != 2 {
		t.Fatalf("drain cap_terms = %v, want limiting=contraction effective_cap=2", terms)
	}
	sp, ok := out["setpoint"].(map[string]any)
	if !ok {
		t.Fatalf("active setpoint must attach the plan; payload=%v", out)
	}
	if sp["mode"] != "drain" || sp["contraction_target"] != 2 || sp["draining"] != 3 {
		t.Fatalf("drain setpoint plan = %v, want mode=drain contraction_target=2 draining=3", sp)
	}

	// Grow: the fleet sits at live 2 under a soft reactive dip (lease target 4 would
	// cap admits at 4); a setpoint of 6 raises the effective cap toward the level via
	// the ceiling-bounded worker floor -- 6 fits under the hard ceilings (configured 8,
	// host 32, seat 12), so the cap lands exactly at the setpoint and admits grow.
	kernelAlive, kernelTarget = 2, 4
	t.Setenv(dispatchtick.SetpointConcurrencyEnv, "6")
	out = preflight(t)
	if out["verdict"] != dispatchtick.PreflightOKVerdict || out["cap"] != 8 {
		t.Fatalf("grow verdict/cap = %v/%v, want SPAWN_OK/8; payload=%v", out["verdict"], out["cap"], out)
	}
	terms = capTerms(t, out)
	if terms["limiting"] != "configured" || terms["worker_floor"] != 6 {
		t.Fatalf("grow cap_terms = %v, want limiting=configured worker_floor=6", terms)
	}
	sp, ok = out["setpoint"].(map[string]any)
	if !ok || sp["mode"] != "grow" || sp["desired_cap"] != 6 {
		t.Fatalf("grow setpoint plan = %v, want mode=grow desired_cap=6", out["setpoint"])
	}

	// Grow never overbooks: a setpoint of 40 is clamped by the hard ceilings -- the
	// binding one here is the configured max (8), not the operator's ask.
	t.Setenv(dispatchtick.SetpointConcurrencyEnv, "40")
	out = preflight(t)
	if out["cap"] != 8 {
		t.Fatalf("overbook-guard cap = %v, want ceiling-clamped 8; payload=%v", out["cap"], out)
	}
}
