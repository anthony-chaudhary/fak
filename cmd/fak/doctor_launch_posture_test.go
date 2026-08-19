package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func postureByName(t *testing.T, report launchPostureReport, name string) launchPostureMechanism {
	t.Helper()
	for _, mechanism := range report.Mechanisms {
		if mechanism.Name == name {
			return mechanism
		}
	}
	t.Fatalf("mechanism %q missing from %+v", name, report.Mechanisms)
	return launchPostureMechanism{}
}

func TestLaunchPostureNativeServeInNonFakRepository(t *testing.T) {
	repo := t.TempDir()
	if err := os.WriteFile(filepath.Join(repo, "benchmark.txt"), []byte("third-party fixture\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	report, err := deriveLaunchPosture(launchPostureOptions{
		entrypoint: "serve", provider: "native", workspace: repo, native: true, nativeCodeTools: true,
		outputProfile: agentDefaultOutputStyle, workProfile: agentDefaultWorkProfile,
		compactHistory: gateway.DefaultCompactHistoryBudget, elideStaleReads: true, deferColdTools: true, vcacheAnchor: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := postureByName(t, report, "bounded-code-tools"); got.State != "active" || report.Workspace != repo {
		t.Fatalf("code tools = %+v, workspace=%q", got, report.Workspace)
	}
	if report.Harness != "" {
		t.Fatalf("native serve reported phantom harness %q", report.Harness)
	}
	for _, name := range []string{"compact-history", "stale-read-elision", "cold-tool-deferral", "vcache-anchor"} {
		if got := postureByName(t, report, name); got.State != "inert" || !strings.Contains(got.Reason, "in-kernel") {
			t.Fatalf("%s = %+v", name, got)
		}
	}
}

func TestLaunchPostureGuardClaudeDefaultsActive(t *testing.T) {
	report, err := deriveLaunchPosture(launchPostureOptions{entrypoint: "guard", harness: "claude", workspace: t.TempDir(), nativeCodeTools: true, outputProfile: agentDefaultOutputStyle, workProfile: agentDefaultWorkProfile, compactHistory: gateway.DefaultCompactHistoryBudget, elideStaleReads: true, deferColdTools: true, vcacheAnchor: true})
	if err != nil {
		t.Fatal(err)
	}
	if report.Wire != shrinkWireAnthropicPassthrough {
		t.Fatalf("wire = %q", report.Wire)
	}
	for _, name := range []string{"caveman-response-profile", "ponytail-work-profile", "compact-history", "stale-read-elision", "cold-tool-deferral", "vcache-anchor"} {
		if got := postureByName(t, report, name); got.State != "active" {
			t.Fatalf("%s = %+v", name, got)
		}
	}
	if got := postureByName(t, report, "bounded-code-tools"); got.State != "active" || got.Disable == "" {
		t.Fatalf("code tool diagnostic = %+v", got)
	}
}

func TestLaunchPostureGuardCodexNamesUnsupportedProfilesAndInertWire(t *testing.T) {
	report, err := deriveLaunchPosture(launchPostureOptions{entrypoint: "guard", harness: "codex", provider: "openai", baseURL: "https://api.openai.example", workspace: t.TempDir(), nativeCodeTools: true, outputProfile: agentDefaultOutputStyle, workProfile: agentDefaultWorkProfile, compactHistory: gateway.DefaultCompactHistoryBudget, elideStaleReads: true, deferColdTools: true, vcacheAnchor: true})
	if err != nil {
		t.Fatal(err)
	}
	if got := postureByName(t, report, "caveman-response-profile"); got.State != "unsupported" || got.Action == "" {
		t.Fatalf("output profile = %+v", got)
	}
	if got := postureByName(t, report, "compact-history"); got.State != "inert" || got.Action == "" {
		t.Fatalf("compaction = %+v", got)
	}
}

func TestLaunchPostureAnthropicServeHasActiveShrinkStack(t *testing.T) {
	report, err := deriveLaunchPosture(launchPostureOptions{entrypoint: "serve", provider: "anthropic", workspace: t.TempDir(), nativeCodeTools: true, outputProfile: agentDefaultOutputStyle, workProfile: agentDefaultWorkProfile, compactHistory: gateway.DefaultCompactHistoryBudget, elideStaleReads: true, deferColdTools: true, vcacheAnchor: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"compact-history", "stale-read-elision", "cold-tool-deferral", "vcache-anchor"} {
		if got := postureByName(t, report, name); got.State != "active" {
			t.Fatalf("%s = %+v", name, got)
		}
	}
}

func TestLaunchPostureDisabledOverridesAreExplicit(t *testing.T) {
	report, err := deriveLaunchPosture(launchPostureOptions{entrypoint: "guard", harness: "claude", workspace: t.TempDir(), outputProfile: "full", workProfile: "standard"})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"bounded-code-tools", "caveman-response-profile", "ponytail-work-profile", "compact-history", "stale-read-elision", "cold-tool-deferral", "vcache-anchor"} {
		if got := postureByName(t, report, name); got.State != "disabled" && name != "bounded-code-tools" {
			t.Fatalf("%s = %+v", name, got)
		}
	}
}

func TestRunDoctorLaunchPostureJSONIsStableAndComplete(t *testing.T) {
	var out, errOut bytes.Buffer
	rc := runDoctorLaunchPosture(&out, &errOut, []string{"--entrypoint", "guard", "--harness", "claude", "--workspace", t.TempDir(), "--json"})
	if rc != 0 || errOut.Len() != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errOut.String())
	}
	var report launchPostureReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Schema != launchPostureSchema || len(report.Mechanisms) != 7 || report.Summary["active"] != 7 {
		t.Fatalf("report = %+v", report)
	}
}
