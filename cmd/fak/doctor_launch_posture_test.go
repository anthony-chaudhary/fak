package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/vcachecalibration"
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
		compactHistory: gateway.DefaultCompactHistoryBudget, ctxViewBudget: agent.DefaultCtxViewBudget, elideStaleReads: true, deferColdTools: true, vcacheAnchor: true,
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
	for _, name := range []string{"compact-history", "cold-tool-deferral", "vcache-anchor"} {
		if got := postureByName(t, report, name); got.State != "inert" || !strings.Contains(got.Reason, "in-kernel") {
			t.Fatalf("%s = %+v", name, got)
		}
	}
	if got := postureByName(t, report, "stale-read-elision"); got.State != "active" || !strings.Contains(got.Reason, "decoded provider-neutral") {
		t.Fatalf("stale-read-elision = %+v", got)
	}
}

func TestLaunchPostureGuardClaudeDefaultsActive(t *testing.T) {
	report, err := deriveLaunchPosture(launchPostureOptions{entrypoint: "guard", harness: "claude", workspace: t.TempDir(), nativeCodeTools: true, outputProfile: agentDefaultOutputStyle, workProfile: agentDefaultWorkProfile, compactHistory: gateway.DefaultCompactHistoryBudget, ctxViewBudget: agent.DefaultCtxViewBudget, elideStaleReads: true, deferColdTools: true, vcacheAnchor: true})
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

func TestLaunchPostureGuardCodexNamesActiveProfilesAndInertWire(t *testing.T) {
	report, err := deriveLaunchPosture(launchPostureOptions{entrypoint: "guard", harness: "codex", provider: "openai", baseURL: "https://api.openai.example", workspace: t.TempDir(), nativeCodeTools: true, outputProfile: agentDefaultOutputStyle, workProfile: agentDefaultWorkProfile, compactHistory: gateway.DefaultCompactHistoryBudget, ctxViewBudget: agent.DefaultCtxViewBudget, elideStaleReads: true, deferColdTools: true, vcacheAnchor: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"caveman-response-profile", "ponytail-work-profile"} {
		if got := postureByName(t, report, name); got.State != "active" || !strings.Contains(got.Reason, "developer_instructions") {
			t.Fatalf("%s = %+v", name, got)
		}
	}
	if got := postureByName(t, report, "compact-history"); got.State != "inert" || got.Action == "" {
		t.Fatalf("anthropic compaction = %+v", got)
	}
	if got := postureByName(t, report, "decoded-context-view"); got.State != "active" || !strings.Contains(got.Reason, "OpenAI-compatible") {
		t.Fatalf("provider-neutral context view = %+v", got)
	}
}

func TestLaunchPostureAnthropicServeHasActiveShrinkStack(t *testing.T) {
	report, err := deriveLaunchPosture(launchPostureOptions{entrypoint: "serve", provider: "anthropic", workspace: t.TempDir(), nativeCodeTools: true, outputProfile: agentDefaultOutputStyle, workProfile: agentDefaultWorkProfile, compactHistory: gateway.DefaultCompactHistoryBudget, ctxViewBudget: agent.DefaultCtxViewBudget, elideStaleReads: true, deferColdTools: true, vcacheAnchor: true})
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
	if rc != 1 || errOut.Len() != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errOut.String())
	}
	var report launchPostureReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Schema != launchPostureSchema || len(report.Mechanisms) != 10 || report.Summary["active"] != 9 || report.Summary["inert"] != 1 {
		t.Fatalf("report = %+v", report)
	}
}

func TestLaunchPostureFlagsMissingVCacheCalibration(t *testing.T) {
	t.Setenv(ledgerRootEnv, t.TempDir())
	report, err := deriveLaunchPosture(launchPostureOptions{entrypoint: "guard", harness: "codex", provider: "openai", workspace: t.TempDir(), compactHistory: gateway.DefaultCompactHistoryBudget, ctxViewBudget: agent.DefaultCtxViewBudget, elideStaleReads: true, deferColdTools: true, vcacheAnchor: true})
	if err != nil {
		t.Fatal(err)
	}
	got := postureByName(t, report, "vcache-calibration")
	if got.State != "inert" || got.Action == "" || report.OK {
		t.Fatalf("calibration=%+v report.OK=%v", got, report.OK)
	}
}

func TestLaunchPostureAcceptsFreshSteeringVCacheCalibration(t *testing.T) {
	root := t.TempDir()
	t.Setenv(ledgerRootEnv, root)
	path := nightrunLedgerPath(vcachecalibration.DefaultCalibrationRel)
	row := vcachecalibration.ProviderCalibration{
		Schema: vcachecalibration.CalibrationSchema, TS: time.Now().UTC().Format(time.RFC3339Nano),
		Provider: "openai", Model: "gpt-test", Source: "test", Turns: 2, Predictions: 1,
		TrueWarm: 1, StaleAfterDays: 7, MinPrefixTokens: 2048, MinPrefixMeasured: true,
	}
	if err := vcachecalibration.AppendCalibration(path, row); err != nil {
		t.Fatal(err)
	}
	report, err := deriveLaunchPosture(launchPostureOptions{entrypoint: "guard", harness: "codex", provider: "openai", workspace: t.TempDir(), compactHistory: gateway.DefaultCompactHistoryBudget, ctxViewBudget: agent.DefaultCtxViewBudget, elideStaleReads: true, deferColdTools: true, vcacheAnchor: true})
	if err != nil {
		t.Fatal(err)
	}
	got := postureByName(t, report, "vcache-calibration")
	if got.State != "active" || !strings.Contains(got.Reason, "steering") {
		t.Fatalf("calibration=%+v", got)
	}
}

func TestLaunchPostureFlagsFreshObservationOnlyCalibration(t *testing.T) {
	root := t.TempDir()
	t.Setenv(ledgerRootEnv, root)
	path := nightrunLedgerPath(vcachecalibration.DefaultCalibrationRel)
	row := vcachecalibration.ProviderCalibration{
		Schema: vcachecalibration.CalibrationSchema, TS: time.Now().UTC().Format(time.RFC3339Nano),
		Provider: "openai", Model: "gpt-test", Source: "test", Turns: 2, Predictions: 1, TrueWarm: 1,
	}
	if err := vcachecalibration.AppendCalibration(path, row); err != nil {
		t.Fatal(err)
	}
	report, err := deriveLaunchPosture(launchPostureOptions{entrypoint: "guard", harness: "codex", provider: "openai", workspace: t.TempDir(), compactHistory: gateway.DefaultCompactHistoryBudget, ctxViewBudget: agent.DefaultCtxViewBudget, elideStaleReads: true, deferColdTools: true, vcacheAnchor: true})
	if err != nil {
		t.Fatal(err)
	}
	got := postureByName(t, report, "vcache-calibration")
	if got.State != "inert" || !strings.Contains(got.Reason, "observational only") || report.OK {
		t.Fatalf("calibration=%+v report.OK=%v", got, report.OK)
	}
}

func TestLaunchPostureDecodedContextViewOptOut(t *testing.T) {
	report, err := deriveLaunchPosture(launchPostureOptions{entrypoint: "guard", harness: "codex", provider: "openai", workspace: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	got := postureByName(t, report, "decoded-context-view")
	if got.State != "disabled" || got.Action == "" {
		t.Fatalf("decoded context-view opt-out = %+v", got)
	}
}

func TestLaunchPostureOpenAINamesDecodedStaleReadsAndCacheSignalsActive(t *testing.T) {
	report, err := deriveLaunchPosture(launchPostureOptions{
		entrypoint: "guard", harness: "codex", provider: "openai", baseURL: "https://api.openai.com/v1",
		workspace: t.TempDir(), nativeCodeTools: true, outputProfile: "caveman:medium", workProfile: "ponytail:medium",
		compactHistory: 32000, ctxViewBudget: agent.DefaultCtxViewBudget, elideStaleReads: true, deferColdTools: true, vcacheAnchor: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	stale := postureByName(t, report, "stale-read-elision")
	if stale.State != "active" || !strings.Contains(stale.Reason, "decoded provider-neutral") {
		t.Fatalf("stale-read posture = %+v", stale)
	}
	signals := postureByName(t, report, "vcache-signals")
	if signals.State != "active" || !strings.Contains(signals.Reason, "normalized provider usage") {
		t.Fatalf("vcache signals posture = %+v", signals)
	}
	cold := postureByName(t, report, "cold-tool-deferral")
	if cold.State != "inert" {
		t.Fatalf("cold-tool posture must remain honest without an OpenAI discovery seam: %+v", cold)
	}
}

func TestLaunchPosturePassthroughNamesClientOwnedTools(t *testing.T) {
	report, err := deriveLaunchPosture(launchPostureOptions{
		entrypoint: "serve", provider: "openai", baseURL: "https://api.openai.com/v1", workspace: t.TempDir(), nativeCodeTools: true,
		outputProfile: "full", workProfile: "standard", ctxViewBudget: agent.DefaultCtxViewBudget, elideStaleReads: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	tools := postureByName(t, report, "bounded-code-tools")
	if tools.State != "inert" || !strings.Contains(tools.Reason, "owned native loop") {
		t.Fatalf("passthrough tools posture = %+v", tools)
	}
}
