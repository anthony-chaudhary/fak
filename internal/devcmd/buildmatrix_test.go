package devcmd

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestBuildMatrixDryRun(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunBuildMatrix(&stdout, &stderr, []string{"--dry-run"})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d. stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "[PLAN] default (linux/amd64") {
		t.Errorf("output missing default linux/amd64 plan:\n%s", out)
	}
	if !strings.Contains(out, "[PLAN] default (windows/amd64") {
		t.Errorf("output missing default windows/amd64 plan:\n%s", out)
	}
	if !strings.Contains(out, "[PLAN] wip_sessionfleet") {
		t.Errorf("output missing wip_sessionfleet plan:\n%s", out)
	}
}

func TestBuildMatrixJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunBuildMatrix(&stdout, &stderr, []string{"--dry-run", "--json"})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d. stderr: %s", code, stderr.String())
	}

	var report BuildMatrixReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("unmarshal report: %v", err)
	}
	if report.Schema != "fak.build_matrix_report.v1" {
		t.Errorf("expected schema fak.build_matrix_report.v1, got %s", report.Schema)
	}
	if report.Total == 0 {
		t.Error("expected non-zero total plans")
	}
	if !report.OK {
		t.Error("expected report.OK == true in dry-run")
	}
}

func TestBuildMatrixEmitGHAMatrix(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunBuildMatrix(&stdout, &stderr, []string{"--emit-gha-matrix"})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d. stderr: %s", code, stderr.String())
	}

	var parsed struct {
		Include []struct {
			Target   string `json:"target"`
			GOOS     string `json:"goos"`
			GOARCH   string `json:"goarch"`
			Variant  string `json:"variant"`
			Tags     string `json:"tags"`
			Advisory bool   `json:"advisory"`
		} `json:"include"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &parsed); err != nil {
		t.Fatalf("unmarshal GHA matrix: %v", err)
	}
	if len(parsed.Include) == 0 {
		t.Fatal("expected non-empty include list in GHA matrix")
	}

	foundDefaultLinux := false
	for _, item := range parsed.Include {
		if item.Variant == "default" && item.Target == "linux/amd64" {
			foundDefaultLinux = true
			if item.GOOS != "linux" || item.GOARCH != "amd64" {
				t.Errorf("mismatched GOOS/GOARCH: %s/%s", item.GOOS, item.GOARCH)
			}
		}
	}
	if !foundDefaultLinux {
		t.Error("GHA matrix missing default linux/amd64 entry")
	}
}

func TestBuildMatrixTargetFilter(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunBuildMatrix(&stdout, &stderr, []string{"--dry-run", "--target", "darwin/arm64"})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d. stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "darwin/arm64") {
		t.Errorf("expected darwin/arm64 in output:\n%s", out)
	}
	if strings.Contains(out, "linux/amd64") {
		t.Errorf("did not expect linux/amd64 in output with filter:\n%s", out)
	}
}
