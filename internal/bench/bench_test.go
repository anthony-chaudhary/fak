package bench

import (
	"context"
	"testing"

	// The built-in driver list: blank-importing it runs every leaf's init() so
	// the real kernel (engine, vdso, adjudicator, blob backend, ctx-MMU, ...) is
	// fully wired before kernel.New runs inside RunArm/Run.
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

const smokeTrace = "../../testdata/tau2/tau2-smoke.json"

// Unit 79: LoadTrace reads the frozen smoke fixture and WorkloadHash is stable.
func TestLoadTrace_AndStableWorkloadHash(t *testing.T) {
	tr, err := LoadTrace(smokeTrace)
	if err != nil {
		t.Fatalf("LoadTrace(%q): %v", smokeTrace, err)
	}
	if tr == nil {
		t.Fatal("LoadTrace returned nil trace")
	}
	if tr.SliceID == "" {
		t.Error("expected a non-empty slice_id from the fixture")
	}
	if len(tr.Calls) == 0 {
		t.Fatal("expected the smoke fixture to carry calls")
	}

	// Stable across two calls (same fixture, same trace value).
	h1 := tr.WorkloadHash()
	h2 := tr.WorkloadHash()
	if h1 == "" {
		t.Fatal("WorkloadHash returned empty string")
	}
	if h1 != h2 {
		t.Errorf("WorkloadHash not stable across calls: %q != %q", h1, h2)
	}

	// And stable across a fresh load of the SAME file (hash is a pure function of
	// the trace content, not load identity).
	tr2, err := LoadTrace(smokeTrace)
	if err != nil {
		t.Fatalf("re-LoadTrace: %v", err)
	}
	if got := tr2.WorkloadHash(); got != h1 {
		t.Errorf("WorkloadHash differs across reloads: %q != %q", got, h1)
	}
}

// Units 80,81: RunArm with the vDSO on yields VDSOHits>0 and P50Ns>0; with the
// vDSO off yields VDSOHits==0 (the --vdso ablation actually changes the path).
func TestRunArm_VDSOAblationChangesPath(t *testing.T) {
	ctx := context.Background()
	tr, err := LoadTrace(smokeTrace)
	if err != nil {
		t.Fatalf("LoadTrace: %v", err)
	}

	on, err := RunArm(ctx, tr, "mock", true, "on")
	if err != nil {
		t.Fatalf("RunArm(vdso on): %v", err)
	}
	if on.Label != "on" {
		t.Errorf("on arm label = %q, want %q", on.Label, "on")
	}
	if on.Calls != len(tr.Calls) {
		t.Errorf("on arm Calls = %d, want %d", on.Calls, len(tr.Calls))
	}
	if on.VDSOHits <= 0 {
		t.Errorf("on arm VDSOHits = %d, want > 0 (fast path should serve calculate/list_all_airports/cache)", on.VDSOHits)
	}
	if on.P50Ns <= 0 {
		t.Errorf("on arm P50Ns = %d, want > 0 (calibration loop measures real time)", on.P50Ns)
	}

	off, err := RunArm(ctx, tr, "mock", false, "off")
	if err != nil {
		t.Fatalf("RunArm(vdso off): %v", err)
	}
	if off.VDSOHits != 0 {
		t.Errorf("off arm VDSOHits = %d, want 0 (--vdso=off must skip the fast path)", off.VDSOHits)
	}
	if off.P50Ns <= 0 {
		t.Errorf("off arm P50Ns = %d, want > 0", off.P50Ns)
	}

	// The ablation must actually change the path: the off arm cannot have more
	// vDSO hits than the on arm, and the on arm must have served something the off
	// arm did not.
	if on.VDSOHits <= off.VDSOHits {
		t.Errorf("expected on.VDSOHits (%d) > off.VDSOHits (%d)", on.VDSOHits, off.VDSOHits)
	}
}

// Unit 86 (+ the Options/Run struct surface): a full Run with BinPath:"" (no
// spawned baseline) returns a populated report with both arms, the provenance
// workload hash set, and the honest live-seam RED flag. The smoke trace runs
// without error.
func TestRun_BothArmsPopulated_NoSpawn(t *testing.T) {
	ctx := context.Background()
	tr, err := LoadTrace(smokeTrace)
	if err != nil {
		t.Fatalf("LoadTrace: %v", err)
	}

	rep, err := Run(ctx, tr, Options{EngineID: "mock", EngineModel: "mock-offline"})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if rep == nil {
		t.Fatal("Run returned a nil report")
	}

	// Both arms populated.
	if rep.On.Label != "vdso_on" {
		t.Errorf("On arm label = %q, want %q", rep.On.Label, "vdso_on")
	}
	if rep.Off.Label != "vdso_off" {
		t.Errorf("Off arm label = %q, want %q", rep.Off.Label, "vdso_off")
	}
	if rep.On.Calls != len(tr.Calls) || rep.Off.Calls != len(tr.Calls) {
		t.Errorf("arm call counts = on:%d off:%d, want %d each", rep.On.Calls, rep.Off.Calls, len(tr.Calls))
	}
	if rep.On.VDSOHits <= 0 {
		t.Errorf("On arm VDSOHits = %d, want > 0", rep.On.VDSOHits)
	}
	if rep.Off.VDSOHits != 0 {
		t.Errorf("Off arm VDSOHits = %d, want 0", rep.Off.VDSOHits)
	}

	// Provenance stamped, workload hash set and matching the trace.
	if rep.Provenance.WorkloadHash == "" {
		t.Error("Provenance.WorkloadHash is empty")
	}
	if rep.Provenance.AppVersion == "" {
		t.Error("Provenance.AppVersion is empty")
	}
	if got, want := rep.Provenance.WorkloadHash, tr.WorkloadHash(); got != want {
		t.Errorf("Provenance.WorkloadHash = %q, want %q", got, want)
	}
	if rep.Provenance.SliceID != tr.SliceID {
		t.Errorf("Provenance.SliceID = %q, want %q", rep.Provenance.SliceID, tr.SliceID)
	}
	if rep.Provenance.EngineModel != "mock-offline" {
		t.Errorf("Provenance.EngineModel = %q, want %q", rep.Provenance.EngineModel, "mock-offline")
	}

	// BinPath:"" => no spawned baseline recorded (RED).
	if rep.Baseline.Calls != 0 {
		t.Errorf("expected no spawned baseline with BinPath:\"\", got Calls=%d", rep.Baseline.Calls)
	}

	// Honest live-seam RED flag when no transcript was supplied.
	if rep.LiveSeam != "live_seam_unverified" {
		t.Errorf("LiveSeam = %q, want %q", rep.LiveSeam, "live_seam_unverified")
	}
}
