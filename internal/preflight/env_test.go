package preflight

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/testroute"
)

func TestPlanEnvPreflightWindowsHost(t *testing.T) {
	rep := PlanEnvPreflight(EnvProbe{
		GOOS: "windows",
		Test: testroute.Probe{GOOS: "windows", NativeTestAllowed: false, WSLPresent: true},
	})
	if rep.Schema != EnvPreflightSchema {
		t.Fatalf("schema = %q, want %q", rep.Schema, EnvPreflightSchema)
	}
	if rep.Verdict != EnvVerdictClear {
		t.Fatalf("verdict = %q, want %q", rep.Verdict, EnvVerdictClear)
	}
	if rep.TestRoute.Kind != testroute.KindWSL {
		t.Fatalf("test route = %q, want %q", rep.TestRoute.Kind, testroute.KindWSL)
	}
	if rep.GitShell.Shell != GitShellPowerShell {
		t.Fatalf("git shell = %q, want %q", rep.GitShell.Shell, GitShellPowerShell)
	}
	if !hasHazardKind(rep.InteractiveHazards, "bash_git_hang") {
		t.Fatalf("windows hazards missing bash_git_hang: %+v", rep.InteractiveHazards)
	}
	if !hasHazardKind(rep.InteractiveHazards, "git_commit_editor") {
		t.Fatalf("hazards missing git_commit_editor: %+v", rep.InteractiveHazards)
	}
}

func TestPlanEnvPreflightNativeHost(t *testing.T) {
	rep := PlanEnvPreflight(EnvProbe{
		GOOS: "linux",
		Test: testroute.Probe{GOOS: "linux", NativeTestAllowed: true},
	})
	if rep.TestRoute.Kind != testroute.KindNative {
		t.Fatalf("test route = %q, want %q", rep.TestRoute.Kind, testroute.KindNative)
	}
	if rep.GitShell.Shell != GitShellBash {
		t.Fatalf("git shell = %q, want %q", rep.GitShell.Shell, GitShellBash)
	}
	if hasHazardKind(rep.InteractiveHazards, "bash_git_hang") {
		t.Fatalf("non-windows hazards should not carry bash_git_hang: %+v", rep.InteractiveHazards)
	}
}

func TestPlanEnvPreflightLeaseOverlap(t *testing.T) {
	rep := PlanEnvPreflight(EnvProbe{
		GOOS:  "windows",
		Test:  testroute.Probe{GOOS: "windows", WSLPresent: true},
		Paths: []string{"internal/foo/**"},
		LiveLeases: []LeaseObservation{
			{ID: "far", Holder: "peer-a", Tree: []string{"internal/bar/**"}},
			{ID: "near", Holder: "peer-b", Tree: []string{"internal/foo/widget.go"}},
		},
	})
	if rep.Verdict != EnvVerdictLeased {
		t.Fatalf("verdict = %q, want %q", rep.Verdict, EnvVerdictLeased)
	}
	if len(rep.LiveLeases) != 1 || rep.LiveLeases[0].ID != "near" {
		t.Fatalf("live leases = %+v, want exactly the overlapping lease %q", rep.LiveLeases, "near")
	}
}

func TestPlanEnvPreflightGlobalLeaseOverlapsEverything(t *testing.T) {
	rep := PlanEnvPreflight(EnvProbe{
		Paths:      []string{"internal/foo/**"},
		LiveLeases: []LeaseObservation{{ID: "global", Holder: "peer", Tree: nil}},
	})
	if rep.Verdict != EnvVerdictLeased {
		t.Fatalf("verdict = %q, want %q for a tree-less (global) lease", rep.Verdict, EnvVerdictLeased)
	}
}

func TestPlanEnvPreflightNoPathsReportsAllLeasesButStaysClear(t *testing.T) {
	rep := PlanEnvPreflight(EnvProbe{
		LiveLeases: []LeaseObservation{
			{ID: "b", Holder: "peer-b", Tree: []string{"internal/bar/**"}},
			{ID: "a", Holder: "peer-a", Tree: []string{"internal/foo/**"}},
		},
	})
	if rep.Verdict != EnvVerdictClear {
		t.Fatalf("verdict = %q, want %q with no declared paths", rep.Verdict, EnvVerdictClear)
	}
	got := []string{rep.LiveLeases[0].ID, rep.LiveLeases[1].ID}
	if !reflect.DeepEqual(got, []string{"a", "b"}) {
		t.Fatalf("live leases = %v, want all leases sorted by ID", got)
	}
}

func hasHazardKind(hs []EnvHazard, kind string) bool {
	for _, h := range hs {
		if h.Kind == kind {
			return true
		}
	}
	return false
}
