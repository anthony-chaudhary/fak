package swebench

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestDeepSWERunnerExternalAdapterJSON(t *testing.T) {
	t.Setenv("FAK_DEEPSWE_RUNNER", os.Args[0])
	t.Setenv("FAK_DEEPSWE_RUNNER_ARGS", "-test.run=TestDeepSWEHelperProcess")
	t.Setenv("FAK_DEEPSWE_HELPER", "1")

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
