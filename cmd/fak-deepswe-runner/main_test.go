package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestFixturePrediction(t *testing.T) {
	req := `{
  "schema": "fak.swebench.deepswe-request.v1",
  "runner": "deepswe",
  "model": "DeepSWE-Preview-fixture",
  "max_steps": 7,
  "instance": {
    "instance_id": "django__django-12345",
    "repo": "django/django",
    "base_commit": "abc123"
  }
}`
	var stdout, stderr bytes.Buffer
	rc := run([]string{"--fixture"}, strings.NewReader(req), &stdout, &stderr)
	if rc != 0 {
		t.Fatalf("run rc=%d stderr=%s", rc, stderr.String())
	}
	var pred struct {
		InstanceID      string `json:"instance_id"`
		ModelNameOrPath string `json:"model_name_or_path"`
		ModelPatch      string `json:"model_patch"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &pred); err != nil {
		t.Fatalf("stdout is not prediction JSON: %v\n%s", err, stdout.String())
	}
	if pred.InstanceID != "django__django-12345" {
		t.Fatalf("instance_id=%q", pred.InstanceID)
	}
	if pred.ModelNameOrPath != "DeepSWE-Preview-fixture" {
		t.Fatalf("model_name_or_path=%q", pred.ModelNameOrPath)
	}
	if !strings.Contains(pred.ModelPatch, "diff --git") {
		t.Fatalf("model_patch is not a diff: %q", pred.ModelPatch)
	}
	if !strings.Contains(pred.ModelPatch, "not a benchmark score") {
		t.Fatalf("fixture patch should carry the no-score fence: %q", pred.ModelPatch)
	}
}

func TestFixtureFlagRequired(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := run(nil, strings.NewReader(`{}`), &stdout, &stderr)
	if rc != 2 {
		t.Fatalf("run rc=%d, want 2", rc)
	}
	if !strings.Contains(stderr.String(), "--fixture is required") {
		t.Fatalf("stderr missing fixture explanation: %s", stderr.String())
	}
}

func TestBadSchemaRejected(t *testing.T) {
	req := `{"schema":"wrong","runner":"deepswe","instance":{"instance_id":"x"}}`
	var stdout, stderr bytes.Buffer
	rc := run([]string{"--fixture"}, strings.NewReader(req), &stdout, &stderr)
	if rc != 2 {
		t.Fatalf("run rc=%d, want 2", rc)
	}
	if !strings.Contains(stderr.String(), "bad schema") {
		t.Fatalf("stderr missing schema error: %s", stderr.String())
	}
}
