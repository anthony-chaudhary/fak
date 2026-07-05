package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleet"
)

func TestLabTargetResolvesReadyGLMTarget(t *testing.T) {
	env := writeLabTargetEnv(t, labTargetEnvOpts{
		readiness: fleet.LabReadyForDevWork,
		report: fleet.Report{
			State: fleet.StateLive,
			Inference: &fleet.InferenceStats{
				Status: fleet.InferenceReady,
				Engine: "sglang",
				Model:  "glm-5.2",
				Reason: "v1-models",
			},
		},
		targets: []labTargetConfig{{
			Alias:   "@lab/glm-5.2",
			BaseURL: "http://127.0.0.1:18181/v1",
			Model:   "glm-5.2",
			BoxID:   "box-a",
		}},
	})
	t.Setenv("FAK_LAB_READINESS", env.readinessPath)
	t.Setenv("FAK_FLEET_REPORTS", env.reportsDir)
	t.Setenv("FAK_LAB_TARGETS", env.targetsPath)

	var out bytes.Buffer
	rc := runLab(&out, io.Discard, []string{"target", "@lab/glm-5.2", "--json"})
	if rc != 0 {
		t.Fatalf("lab target should resolve, got rc=%d out=%s", rc, out.String())
	}
	var res labTargetResolution
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("lab target did not emit JSON: %v\n%s", err, out.String())
	}
	if res.Status != fleet.LabReadyForDevWork || res.Alias != "@lab/glm-5.2" || res.Model != "glm-5.2" || res.BoxID != "box-a" {
		t.Fatalf("resolution = %+v", res)
	}
	if strings.Contains(out.String(), "127.0.0.1") || strings.Contains(out.String(), "18181") {
		t.Fatalf("target JSON must stay scrubbed and not print local coordinates:\n%s", out.String())
	}

	base, err := resolveGuardRemoteServe("@lab/glm-5.2")
	if err != nil {
		t.Fatalf("guard remote alias resolve: %v", err)
	}
	if base != "http://127.0.0.1:18181" {
		t.Fatalf("guard remote alias base = %q, want bare local tunnel base", base)
	}
}

func TestLabTargetRefusesMissingReadiness(t *testing.T) {
	env := writeLabTargetEnv(t, labTargetEnvOpts{
		noReadiness: true,
		report:      readyGLMReport(),
		targets: []labTargetConfig{{
			Alias: "@lab/glm-5.2", BaseURL: "http://localhost:18181", Model: "glm-5.2", BoxID: "box-a",
		}},
	})
	t.Setenv("FAK_LAB_READINESS", env.readinessPath)
	t.Setenv("FAK_FLEET_REPORTS", env.reportsDir)
	t.Setenv("FAK_LAB_TARGETS", env.targetsPath)

	var stderr bytes.Buffer
	rc := runLab(io.Discard, &stderr, []string{"target", "@lab/glm-5.2"})
	if rc != 1 {
		t.Fatalf("missing readiness should fail closed, got rc=%d", rc)
	}
	if !strings.Contains(stderr.String(), "LAB_READINESS_NOT_READY") {
		t.Fatalf("missing readiness error should be typed, got:\n%s", stderr.String())
	}
}

func TestLabTargetUsesPerTargetRoster(t *testing.T) {
	root := t.TempDir()
	readinessPath := filepath.Join(root, "lab-readiness.json")
	reportsDir := filepath.Join(root, "reports")
	targetsPath := filepath.Join(root, "lab-targets.json")
	rosterPath := filepath.Join(root, "private-roster.json")

	rec := fleet.NewLabReadiness("gpu-server", fleet.LabReadyForDevWork, "admit-lab-backed-dispatch", "scrubbed-fleet-report", time.Now())
	if err := writeIndentedJSONFile(readinessPath, rec); err != nil {
		t.Fatalf("write readiness: %v", err)
	}
	ro := fleet.Roster{Schema: fleet.RosterSchema, Boxes: []fleet.Box{{
		ID: "private-a", Class: "a100x8", Group: "lab",
	}}}
	if err := writeIndentedJSONFile(rosterPath, ro); err != nil {
		t.Fatalf("write roster: %v", err)
	}
	if err := fleet.WriteReport(reportsDir, "private-a", readyGLMReport()); err != nil {
		t.Fatalf("write private report: %v", err)
	}
	doc := labTargetsFile{Schema: labTargetsSchema, Targets: []labTargetConfig{{
		Alias:   "@lab/glm-5.2",
		BaseURL: "http://127.0.0.1:18181",
		Model:   "glm-5.2",
		Roster:  rosterPath,
	}}}
	if err := writeIndentedJSONFile(targetsPath, doc); err != nil {
		t.Fatalf("write targets: %v", err)
	}
	t.Setenv("FAK_LAB_READINESS", readinessPath)
	t.Setenv("FAK_FLEET_REPORTS", reportsDir)
	t.Setenv("FAK_LAB_TARGETS", targetsPath)

	var out bytes.Buffer
	rc := runLab(&out, io.Discard, []string{"target", "@lab/glm-5.2", "--json"})
	if rc != 0 {
		t.Fatalf("lab target should resolve with per-target roster, got rc=%d out=%s", rc, out.String())
	}
	var res labTargetResolution
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("lab target did not emit JSON: %v\n%s", err, out.String())
	}
	if res.BoxID != "" || res.Evidence != "scrubbed-fleet-report" {
		t.Fatalf("resolution = %+v", res)
	}
	if strings.Contains(out.String(), rosterPath) || strings.Contains(out.String(), "private-a") || strings.Contains(out.String(), "18181") {
		t.Fatalf("target JSON must not print local/private coordinates:\n%s", out.String())
	}
}

func TestLabTargetRefusesMissingConfig(t *testing.T) {
	env := writeLabTargetEnv(t, labTargetEnvOpts{
		readiness: fleet.LabReadyForDevWork,
		report:    readyGLMReport(),
		noTargets: true,
	})
	t.Setenv("FAK_LAB_READINESS", env.readinessPath)
	t.Setenv("FAK_FLEET_REPORTS", env.reportsDir)
	t.Setenv("FAK_LAB_TARGETS", env.targetsPath)

	var stderr bytes.Buffer
	rc := runLab(io.Discard, &stderr, []string{"target", "@lab/glm-5.2"})
	if rc != 1 {
		t.Fatalf("missing target config should fail closed, got rc=%d", rc)
	}
	if !strings.Contains(stderr.String(), "LAB_TARGET_CONFIG_MISSING") {
		t.Fatalf("missing target config error should be typed, got:\n%s", stderr.String())
	}
}

func TestLabTargetRefusesStaleOrNonUsefulReports(t *testing.T) {
	for _, tc := range []struct {
		name   string
		report fleet.Report
		stale  bool
	}{
		{name: "warming", report: fleet.Report{State: fleet.StateLive, Inference: &fleet.InferenceStats{Status: fleet.InferenceWarming, Model: "glm-5.2"}}},
		{name: "degraded", report: fleet.Report{State: fleet.StateLive, Inference: &fleet.InferenceStats{Status: fleet.InferenceDegraded, Model: "glm-5.2", Reason: "route-degraded"}}},
		{name: "model-mismatch", report: fleet.Report{State: fleet.StateLive, Inference: &fleet.InferenceStats{Status: fleet.InferenceReady, Model: "qwen", Reason: "v1-models"}}},
		{name: "stale-ready", report: readyGLMReport(), stale: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			env := writeLabTargetEnv(t, labTargetEnvOpts{
				readiness: fleet.LabReadyForDevWork,
				report:    tc.report,
				stale:     tc.stale,
				targets: []labTargetConfig{{
					Alias: "@lab/glm-5.2", BaseURL: "http://localhost:18181", Model: "glm-5.2", BoxID: "box-a",
				}},
			})
			t.Setenv("FAK_LAB_READINESS", env.readinessPath)
			t.Setenv("FAK_FLEET_REPORTS", env.reportsDir)
			t.Setenv("FAK_LAB_TARGETS", env.targetsPath)

			var stderr bytes.Buffer
			rc := runLab(io.Discard, &stderr, []string{"target", "@lab/glm-5.2"})
			if rc != 1 {
				t.Fatalf("non-useful report should fail closed, got rc=%d", rc)
			}
			if !strings.Contains(stderr.String(), "LAB_TARGET_REPORT_NOT_USEFUL") {
				t.Fatalf("report refusal should be typed, got:\n%s", stderr.String())
			}
		})
	}
}

func TestResolveGuardRemoteServeKeepsLiteralHostPath(t *testing.T) {
	got, err := resolveGuardRemoteServe("labbox:8082")
	if err != nil {
		t.Fatalf("literal remote serve should still parse: %v", err)
	}
	if got != "http://labbox:8082" {
		t.Fatalf("literal remote serve = %q", got)
	}
}

type labTargetEnvOpts struct {
	readiness   string
	noReadiness bool
	report      fleet.Report
	stale       bool
	targets     []labTargetConfig
	noTargets   bool
}

type labTargetEnv struct {
	root          string
	reportsDir    string
	readinessPath string
	targetsPath   string
}

func writeLabTargetEnv(t *testing.T, opts labTargetEnvOpts) labTargetEnv {
	t.Helper()
	root := t.TempDir()
	env := labTargetEnv{
		root:          root,
		reportsDir:    filepath.Join(root, "reports"),
		readinessPath: filepath.Join(root, "lab-readiness.json"),
		targetsPath:   filepath.Join(root, "lab-targets.json"),
	}
	if !opts.noReadiness {
		status := opts.readiness
		if status == "" {
			status = fleet.LabReadyForDevWork
		}
		rec := fleet.NewLabReadiness("gpu-server", status, "admit-lab-backed-dispatch", "scrubbed-fleet-report", time.Now())
		if err := writeIndentedJSONFile(env.readinessPath, rec); err != nil {
			t.Fatalf("write readiness: %v", err)
		}
	}
	if opts.report.State != "" || opts.report.Inference != nil {
		if err := fleet.WriteReport(env.reportsDir, "box-a", opts.report); err != nil {
			t.Fatalf("write report: %v", err)
		}
		if opts.stale {
			old := time.Now().Add(-(time.Duration(fleet.DefaultStaleSec)*time.Second + time.Minute))
			p := filepath.Join(env.reportsDir, "box-a.json")
			if err := os.Chtimes(p, old, old); err != nil {
				t.Fatalf("make report stale: %v", err)
			}
		}
	}
	if !opts.noTargets {
		targets := opts.targets
		if len(targets) == 0 {
			targets = []labTargetConfig{{Alias: "@lab/glm-5.2", BaseURL: "http://localhost:18181", Model: "glm-5.2", BoxID: "box-a"}}
		}
		doc := labTargetsFile{Schema: labTargetsSchema, Targets: targets}
		if err := writeIndentedJSONFile(env.targetsPath, doc); err != nil {
			t.Fatalf("write targets: %v", err)
		}
	}
	return env
}

func readyGLMReport() fleet.Report {
	return fleet.Report{
		State:     fleet.StateLive,
		Inference: &fleet.InferenceStats{Status: fleet.InferenceReady, Engine: "sglang", Model: "glm-5.2", Reason: "v1-models"},
	}
}
