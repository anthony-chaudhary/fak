package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessartifact"
)

func TestHarnessLifecycleSelfcheckTypedStale(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lifecycle.json")
	receipt := harnessartifact.ModelLifecycleReceipt{
		Declaration: harnessartifact.LifecycleIdentity{ID: "decl-1"}, Artifact: harnessartifact.LifecycleIdentity{ID: "artifact-1"}, Runtime: harnessartifact.LifecycleIdentity{ID: "runtime-1"}, Admission: harnessartifact.LifecycleIdentity{ID: "admission-1"}, Process: harnessartifact.LifecycleIdentity{ID: "process-1"}, Readiness: harnessartifact.LifecycleIdentity{ID: "ready-1"}, Stop: harnessartifact.LifecycleIdentity{ID: "stop-1"}, HealthURL: "http://user:selfchecksecret@127.0.0.1/health?token=selfchecksecret", State: "ready",
	}
	if err := harnessartifact.WriteModelLifecycleReceipt(path, receipt); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runHarnessLifecycleSelfcheck(&stdout, &stderr, []string{"--lifecycle-receipt", path, "--expected-declaration", "decl-2"})
	captured := stdout.String() + stderr.String()
	if code == 0 || !strings.Contains(captured, "LIFECYCLE_RECEIPT_STALE") {
		t.Fatalf("code=%d output=%s", code, captured)
	}
	if strings.Contains(captured, "selfchecksecret") {
		t.Fatalf("secret leaked: %s", captured)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), "selfchecksecret") {
		t.Fatal("fixture lost injected secret")
	}
}
