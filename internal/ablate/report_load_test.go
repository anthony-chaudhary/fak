package ablate

import (
	"os"
	"path/filepath"
	"testing"
)

// committedVDSOReport is a real AblationReport artifact committed under experiments/,
// addressed relative to this package dir. It is exactly the shape --out writes, so it
// is the round-trip fixture for the read path.
const committedVDSOReport = "../../experiments/ablate/tau2-smoke-vdso-ablation.json"

// committedCrossAgentReport is a DIFFERENT-schema experiment artifact (the cross-agent
// harness: no runs[]). LoadReport must reject it rather than return an empty report.
const committedCrossAgentReport = "../../experiments/ablate/cross-agent-pong-opus.json"

// LoadReport reads a committed N-arm AblationReport back into a *Report that carries its
// arms, its stored baseline, and a workload hash that passes the identical-workload guard
// — the invariant that lets a saved sweep be re-rendered without re-running it.
func TestLoadReportRoundTripsCommittedArtifact(t *testing.T) {
	rep, err := LoadReport(committedVDSOReport)
	if err != nil {
		t.Fatalf("LoadReport(%s): %v", committedVDSOReport, err)
	}
	if len(rep.Runs) == 0 {
		t.Fatalf("loaded report has no arms")
	}
	if rep.Baseline != "all-off" {
		t.Errorf("baseline = %q, want all-off", rep.Baseline)
	}
	if rep.WorkloadHash == "" {
		t.Error("loaded report has empty workload hash")
	}
	if rep.ArmByID(rep.Baseline) == nil {
		t.Errorf("baseline arm %q not among loaded arms", rep.Baseline)
	}
	// A round-tripped report must still satisfy the guard it was written under.
	if err := rep.Validate(); err != nil {
		t.Errorf("Validate on loaded report: %v", err)
	}
}

// A committed cross-agent artifact is a different schema (no runs[]); the reader fails
// loud instead of surfacing an empty, misleading report.
func TestUnmarshalReportRejectsNonArmSchema(t *testing.T) {
	data, err := os.ReadFile(committedCrossAgentReport)
	if err != nil {
		t.Fatalf("read %s: %v", committedCrossAgentReport, err)
	}
	if _, err := UnmarshalReport(data); err == nil {
		t.Fatalf("UnmarshalReport accepted a non-arm (cross-agent) artifact; want an error")
	}
}

// A byte body with no arms is rejected with the same fail-loud reason.
func TestUnmarshalReportRejectsEmpty(t *testing.T) {
	if _, err := UnmarshalReport([]byte(`{"workload_hash":"x"}`)); err == nil {
		t.Fatal("UnmarshalReport accepted a report with no runs[]; want an error")
	}
}

// A round-trip through Report.JSON must reload byte-for-byte-equivalent arms.
func TestLoadReportRoundTripsJSONWriteBack(t *testing.T) {
	orig, err := LoadReport(committedVDSOReport)
	if err != nil {
		t.Fatalf("LoadReport: %v", err)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "rep.json")
	if err := os.WriteFile(path, orig.JSON(), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	reloaded, err := LoadReport(path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if len(reloaded.Runs) != len(orig.Runs) || reloaded.WorkloadHash != orig.WorkloadHash || reloaded.Baseline != orig.Baseline {
		t.Fatalf("round-trip drifted: reloaded {%d arms, wh=%s, base=%s} != orig {%d arms, wh=%s, base=%s}",
			len(reloaded.Runs), reloaded.WorkloadHash, reloaded.Baseline,
			len(orig.Runs), orig.WorkloadHash, orig.Baseline)
	}
}

// A missing path is an ordinary I/O error, not a panic.
func TestLoadReportMissingPath(t *testing.T) {
	if _, err := LoadReport(filepath.Join(t.TempDir(), "nope.json")); err == nil {
		t.Fatal("LoadReport of a missing path returned nil error")
	}
}
