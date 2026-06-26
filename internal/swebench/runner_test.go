package swebench

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestDeepSWERunnerExternalAdapterJSON(t *testing.T) {
	t.Setenv("FAK_DEEPSWE_RUNNER", os.Args[0])
	t.Setenv("FAK_DEEPSWE_RUNNER_ARGS", "-test.run=TestDeepSWEHelperProcess")
	t.Setenv("FAK_DEEPSWE_HELPER", "1")
	t.Setenv("FAK_SANDBOX_ENV_ALLOW", "FAK_DEEPSWE_HELPER")

	runner := deepSWERunner{cfg: RunConfig{Model: "DeepSWE-Preview", MaxSteps: 7}}
	pred, err := runner.RunInstance(context.Background(), Instance{
		InstanceID: "django__django-10000",
		Repo:       "django/django",
		BaseCommit: "abc123",
	})
	if err != nil {
		t.Fatalf("RunInstance returned error: %v", err)
	}
	if pred.InstanceID != "django__django-10000" {
		t.Fatalf("InstanceID = %q", pred.InstanceID)
	}
	if pred.ModelNameOrPath != "DeepSWE-Preview" {
		t.Fatalf("ModelNameOrPath = %q", pred.ModelNameOrPath)
	}
	if !strings.Contains(pred.ModelPatch, "diff --git") {
		t.Fatalf("ModelPatch does not look like a diff: %q", pred.ModelPatch)
	}
}

func TestDeepSWERunnerMasksParentEnvButKeepsExplicit(t *testing.T) {
	t.Setenv("FAK_DEEPSWE_RUNNER", os.Args[0])
	t.Setenv("FAK_DEEPSWE_RUNNER_ARGS", "-test.run=TestDeepSWEHelperProcess")
	t.Setenv("FAK_DEEPSWE_HELPER", "1")
	t.Setenv("FAK_DEEPSWE_HELPER_MODE", "envcheck")
	t.Setenv("FAK_SANDBOX_ENV_ALLOW", "FAK_DEEPSWE_HELPER,FAK_DEEPSWE_HELPER_MODE")
	t.Setenv("FAK_PARENT_SECRET", "do-not-cross")

	runner := deepSWERunner{cfg: RunConfig{Model: "DeepSWE-Preview", MaxSteps: 7}}
	if _, err := runner.RunInstance(context.Background(), Instance{
		InstanceID: "django__django-10000",
		Repo:       "django/django",
		BaseCommit: "abc123",
	}); err != nil {
		t.Fatalf("RunInstance returned error: %v", err)
	}
}

func TestDeepSWERunnerUnconfiguredFailsClosed(t *testing.T) {
	t.Setenv("FAK_DEEPSWE_RUNNER", "")
	t.Setenv("FAK_DEEPSWE_RUNNER_ARGS", "")

	runner := deepSWERunner{cfg: RunConfig{Model: "DeepSWE-Preview"}}
	_, err := runner.RunInstance(context.Background(), Instance{InstanceID: "django__django-10000"})
	if err == nil {
		t.Fatal("RunInstance succeeded without an adapter")
	}
	if strings.Contains(err.Error(), "not wired") {
		t.Fatalf("error still reports placeholder path: %v", err)
	}
	if !strings.Contains(err.Error(), "adapter not configured") {
		t.Fatalf("error does not explain adapter configuration: %v", err)
	}
}

func TestRunMarksTimeoutInstances(t *testing.T) {
	dir := t.TempDir()
	difficulty := filepath.Join(dir, "difficulty.json")
	if err := os.WriteFile(difficulty, []byte(`{"django__django-10000":"<15min"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_DEEPSWE_RUNNER", os.Args[0])
	t.Setenv("FAK_DEEPSWE_RUNNER_ARGS", "-test.run=TestDeepSWEHelperProcess")
	t.Setenv("FAK_DEEPSWE_HELPER", "1")
	t.Setenv("FAK_DEEPSWE_HELPER_MODE", "sleep")
	t.Setenv("FAK_SANDBOX_ENV_ALLOW", "FAK_DEEPSWE_HELPER,FAK_DEEPSWE_HELPER_MODE")

	res, err := Run(context.Background(), RunConfig{
		Runner:     RunnerDeepSWE,
		Filter:     "full",
		Limit:      1,
		Timeout:    10 * time.Millisecond,
		OutputDir:  filepath.Join(dir, "out"),
		Difficulty: difficulty,
	})
	if err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if res.Meta.DoneInstances != 0 || res.Meta.Failed != 1 {
		t.Fatalf("run counts = done %d failed %d, want 0/1", res.Meta.DoneInstances, res.Meta.Failed)
	}
	if len(res.Meta.InstanceMeta) != 1 {
		t.Fatalf("instance meta rows = %d, want 1", len(res.Meta.InstanceMeta))
	}
	row := res.Meta.InstanceMeta[0]
	if row.Status != "timeout" {
		t.Fatalf("timeout instance status = %q, want timeout (error %q)", row.Status, row.Error)
	}
	if len(res.Predictions) != 1 || res.Predictions[0].InstanceID != "django__django-10000" || res.Predictions[0].ModelPatch != "" {
		t.Fatalf("timeout should still emit one dummy prediction for grading: %+v", res.Predictions)
	}
}

func TestDeepSWEHelperProcess(t *testing.T) {
	if os.Getenv("FAK_DEEPSWE_HELPER") != "1" {
		return
	}
	var req deepSWERequest
	if err := json.NewDecoder(os.Stdin).Decode(&req); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if req.Schema != "fak.swebench.deepswe-request.v1" || req.Instance.InstanceID == "" {
		fmt.Fprintf(os.Stderr, "bad request: %+v\n", req)
		os.Exit(3)
	}
	if os.Getenv("FAK_DEEPSWE_HELPER_MODE") == "envcheck" {
		if os.Getenv("FAK_PARENT_SECRET") != "" {
			fmt.Fprintln(os.Stderr, "parent secret crossed sandbox env")
			os.Exit(5)
		}
		if os.Getenv("FAK_SWEBENCH_INSTANCE_ID") != req.Instance.InstanceID {
			fmt.Fprintln(os.Stderr, "explicit instance env missing")
			os.Exit(6)
		}
		if os.Getenv("FAK_DEEPSWE_MODEL") != req.Model {
			fmt.Fprintln(os.Stderr, "explicit model env missing")
			os.Exit(7)
		}
	}
	if os.Getenv("FAK_DEEPSWE_HELPER_MODE") == "sleep" {
		time.Sleep(time.Hour)
	}
	patch := "diff --git a/example.py b/example.py\n--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-old\n+new\n"
	resp := deepSWEResponse{
		InstanceID:      req.Instance.InstanceID,
		ModelNameOrPath: req.Model,
		ModelPatch:      patch,
	}
	if err := json.NewEncoder(os.Stdout).Encode(resp); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(4)
	}
	os.Exit(0)
}
