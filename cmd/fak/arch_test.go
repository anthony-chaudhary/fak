package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/archcheck"
)

func TestArchUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runArch(&out, &errOut, []string{"--help"})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	if !strings.Contains(out.String(), "fak arch check") {
		t.Fatalf("usage output missing check command: %s", out.String())
	}
}

func TestArchCheckPackageJSON(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runArchCheck(&out, &errOut, []string{"--package", "internal/agentquery", "--json"})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0, stderr = %s", code, errOut.String())
	}

	var res archcheck.CheckResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("unmarshal json: %v, raw: %s", err, out.String())
	}
	if !res.OK {
		t.Fatalf("expected internal/agentquery to be clean, got: %+v", res.Violations)
	}
	if res.CheckedPackages != 1 {
		t.Fatalf("CheckedPackages = %d, want 1", res.CheckedPackages)
	}
}

func TestArchCheckMineClean(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runArchCheck(&out, &errOut, []string{"--mine", "--json"})
	if code != 0 && code != 1 {
		t.Fatalf("unexpected exit code %d, stderr: %s", code, errOut.String())
	}
	var res archcheck.CheckResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("unmarshal json: %v", err)
	}
}
