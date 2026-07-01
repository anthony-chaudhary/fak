package ablate

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/metrics"

	// Blank-import the built-in driver list so the real kernel (engine, vdso,
	// adjudicator, blob backend, ctx-MMU, …) is wired before kernel.New runs inside
	// bench.RunArm — the same setup internal/bench's own test uses.
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

const smokeTrace = "../../testdata/tau2/tau2-smoke.json"

// A vDSO sweep over the smoke trace yields the two-arm matrix (all-off + vdso), every
// arm bound to the SAME workload hash, with the delta isolated to the knob: the
// vdso-off baseline serves nothing from the fast path, the vdso-on arm serves > 0.
func TestSweep_VDSO_NArmGuardAndIsolatedDelta(t *testing.T) {
	ctx := context.Background()
	tr, err := bench.LoadTrace(smokeTrace)
	if err != nil {
		t.Fatalf("LoadTrace(%q): %v", smokeTrace, err)
	}

	configs, err := BuildSweep([]string{FeatureVDSO})
	if err != nil {
		t.Fatalf("BuildSweep: %v", err)
	}
	if len(configs) != 2 {
		t.Fatalf("BuildSweep([vdso]) = %d arms, want 2 (all-off, vdso)", len(configs))
	}

	rep, err := Sweep(ctx, tr, "mock", "mock-offline", configs, "all-off")
	if err != nil {
		t.Fatalf("Sweep: %v", err)
	}
	if len(rep.Runs) != 2 {
		t.Fatalf("report has %d arms, want 2", len(rep.Runs))
	}

	// The report binds the trace's workload hash, and every arm shares it (the N-arm
	// identical-workload guard).
	if rep.WorkloadHash != tr.WorkloadHash() {
		t.Errorf("report WorkloadHash = %q, want %q", rep.WorkloadHash, tr.WorkloadHash())
	}
	for _, run := range rep.Runs {
		if run.WorkloadHash != rep.WorkloadHash {
			t.Errorf("arm %q WorkloadHash = %q, want %q", run.ArmID, run.WorkloadHash, rep.WorkloadHash)
		}
		if run.Arm.Calls != len(tr.Calls) {
			t.Errorf("arm %q Calls = %d, want %d", run.ArmID, run.Arm.Calls, len(tr.Calls))
		}
	}

	off := rep.ArmByID("all-off")
	on := rep.ArmByID("vdso")
	if off == nil || on == nil {
		t.Fatalf("expected arms all-off and vdso, got %+v", armIDs(rep))
	}
	if off.Features[FeatureVDSO] != "off" {
		t.Errorf("all-off descriptor vdso = %q, want off", off.Features[FeatureVDSO])
	}
	if on.Features[FeatureVDSO] != "on" {
		t.Errorf("vdso descriptor vdso = %q, want on", on.Features[FeatureVDSO])
	}
	if off.Arm.VDSOHits != 0 {
		t.Errorf("all-off arm VDSOHits = %d, want 0 (fast path skipped)", off.Arm.VDSOHits)
	}
	if on.Arm.VDSOHits <= 0 {
		t.Errorf("vdso arm VDSOHits = %d, want > 0 (fast path served repeats)", on.Arm.VDSOHits)
	}
	if on.Arm.VDSOHits <= off.Arm.VDSOHits {
		t.Errorf("expected vdso arm hits (%d) > all-off hits (%d) — the ablation must change the path",
			on.Arm.VDSOHits, off.Arm.VDSOHits)
	}
	if on.MechanismSavings.FakVDSOAvoidedCalls <= off.MechanismSavings.FakVDSOAvoidedCalls {
		t.Errorf("expected vdso arm avoided calls (%d) > all-off avoided calls (%d)",
			on.MechanismSavings.FakVDSOAvoidedCalls, off.MechanismSavings.FakVDSOAvoidedCalls)
	}

	// Provenance is stamped and self-describing.
	if rep.Provenance.GeneratedBy != "fak/internal/ablate" {
		t.Errorf("GeneratedBy = %q, want fak/internal/ablate", rep.Provenance.GeneratedBy)
	}
	if rep.Provenance.SliceID != tr.SliceID {
		t.Errorf("SliceID = %q, want %q", rep.Provenance.SliceID, tr.SliceID)
	}
	if len(rep.JSON()) == 0 {
		t.Error("JSON() returned empty")
	}
}

func TestRunOneArm_CompressorReportsFakTokenEquivOnly(t *testing.T) {
	tr, err := bench.LoadTrace(smokeTrace)
	if err != nil {
		t.Fatalf("LoadTrace(%q): %v", smokeTrace, err)
	}
	cfg := FeatureConfig{Name: FeatureCompressor}
	cfg.apply(FeatureCompressor, true)

	run, err := RunOneArm(context.Background(), tr, "mock", cfg)
	if err != nil {
		t.Fatalf("RunOneArm: %v", err)
	}
	if run.MechanismSavings.FakCompactionShedTokens == 0 {
		t.Fatalf("compressor arm did not report compaction-shed tokens: %+v", run.MechanismSavings)
	}
	if run.FakTokenEquiv() <= 0 {
		t.Fatalf("compressor arm fak_tokeq = %v, want > 0", run.FakTokenEquiv())
	}
	if run.ProviderTokenEquiv() != 0 {
		t.Fatalf("compressor arm provider_tokeq = %v, want 0 (fak-owned slice only)", run.ProviderTokenEquiv())
	}
}

func TestAblationRunJSONIncludesOwnerTokenEquiv(t *testing.T) {
	run := AblationRun{
		ArmID: "split",
		MechanismSavings: gateway.MechanismSavings{
			ProviderPromptCacheReadTokenEquiv:         900,
			ProviderPromptCacheWritePremiumTokenEquiv: -50,
			FakCompactionShedTokens:                   300,
			FakKVPrefixReusedTokens:                   400,
		},
	}
	b, err := json.Marshal(run)
	if err != nil {
		t.Fatalf("marshal AblationRun: %v", err)
	}
	s := string(b)
	for _, want := range []string{`"provider_tokeq":850`, `"fak_tokeq":700`, `"total_tokeq":1550`, `"mechanism_savings":`} {
		if !strings.Contains(s, want) {
			t.Fatalf("AblationRun JSON missing %q: %s", want, s)
		}
	}
}

// The N-arm guard refuses a report whose arms ran different workloads — the same
// fail-closed contract metrics.Report.Validate enforces for the 2-arm bench.
func TestValidate_RefusesMismatchedWorkloadHash(t *testing.T) {
	rep := &Report{
		WorkloadHash: "hashA",
		Baseline:     "a",
		Runs: []AblationRun{
			{ArmID: "a", WorkloadHash: "hashA"},
			{ArmID: "b", WorkloadHash: "hashB"}, // diverged — must be refused
		},
	}
	if err := rep.Validate(); err == nil {
		t.Fatal("Validate accepted arms with different workload hashes; want refusal")
	}

	// Same hash everywhere validates.
	rep.Runs[1].WorkloadHash = "hashA"
	if err := rep.Validate(); err != nil {
		t.Fatalf("Validate refused a matched-hash report: %v", err)
	}

	// A baseline naming an absent arm is refused.
	rep.Baseline = "ghost"
	if err := rep.Validate(); err == nil {
		t.Fatal("Validate accepted a baseline that is not among the arms; want refusal")
	}
}

// BuildSweep fails loud on a feature the harness cannot sweep, collapses duplicates,
// and rejects an empty request — so a typo never silently measures nothing. As of
// rung 2 the env-gated features (normgate, radix, …) ARE known (they sweep through the
// subprocess path), so only a genuine non-feature is refused.
func TestBuildSweep_UnknownAndDuplicate(t *testing.T) {
	if _, err := BuildSweep([]string{"nope-not-a-feature"}); err == nil {
		t.Error("BuildSweep([nope]) should fail on a token that is no known feature")
	}
	if _, err := BuildSweep(nil); err == nil {
		t.Error("BuildSweep(nil) should fail with no features to sweep")
	}
	configs, err := BuildSweep([]string{"vdso", "vdso"})
	if err != nil {
		t.Fatalf("BuildSweep([vdso,vdso]): %v", err)
	}
	if len(configs) != 2 { // all-off + a single deduped vdso arm (no all-on for one feature)
		t.Errorf("duplicate vdso collapsed to %d arms, want 2", len(configs))
	}
}

// Sweep rejects duplicate arm names (a misconfigured matrix) before running anything.
func TestSweep_RejectsDuplicateArmNames(t *testing.T) {
	tr, err := bench.LoadTrace(smokeTrace)
	if err != nil {
		t.Fatalf("LoadTrace: %v", err)
	}
	dup := []FeatureConfig{{Name: "x"}, {Name: "x", VDSO: true}}
	if _, err := Sweep(context.Background(), tr, "mock", "mock-offline", dup, ""); err == nil {
		t.Error("Sweep accepted duplicate arm names; want refusal")
	}
}

// Descriptor reports only the knobs this rung actually applies, so the artifact never
// claims an ablation it did not perform.
func TestDescriptor_ListsOnlyAppliedKnobs(t *testing.T) {
	d := FeatureConfig{Name: "vdso", VDSO: true}.Descriptor()
	if got := FeatureKeys(d); len(got) != 1 || got[0] != FeatureVDSO {
		t.Errorf("Descriptor keys = %v, want [%s] (rung 1 applies only vdso)", got, FeatureVDSO)
	}
	if d[FeatureVDSO] != "on" {
		t.Errorf("vdso=%q, want on", d[FeatureVDSO])
	}
}

// compile-time: AblationRun embeds the canonical metrics.Arm (no schema fork).
var _ = metrics.Arm{}

func armIDs(r *Report) []string {
	out := make([]string, 0, len(r.Runs))
	for _, run := range r.Runs {
		out = append(out, run.ArmID)
	}
	return out
}
