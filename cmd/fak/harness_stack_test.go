package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHarnessStackResolveAllowText(t *testing.T) {
	manifest := stackManifestFixture(t, false)
	var stdout, stderr bytes.Buffer
	code := runHarnessCommand(&stdout, &stderr, []string{"stack", "resolve", "--manifest", manifest})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	for _, want := range []string{"ALLOW stack for coding@1", "selected: 2 components", "harness:ponytail@r8", "model:coder@sha256:111"} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, stdout.String())
		}
	}
}

func TestHarnessStackResolveRefusalExitThree(t *testing.T) {
	manifest := stackManifestFixture(t, true)
	var stdout, stderr bytes.Buffer
	code := runHarnessCommand(&stdout, &stderr, []string{"stack", "resolve", "--manifest", manifest})
	if code != 3 {
		t.Fatalf("code=%d, want 3; stderr=%s", code, stderr.String())
	}
	for _, want := range []string{"REFUSE stack for coding@1", "blocker: device.cuda.sm80", "chain: harness:ponytail@r8 -> model:coder@sha256:111 -> device.cuda.sm80"} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, stdout.String())
		}
	}
}

func TestHarnessStackResolveJSONReceiptSchema(t *testing.T) {
	manifest := stackManifestFixture(t, false)
	var stdout, stderr bytes.Buffer
	code := runHarnessCommand(&stdout, &stderr, []string{"stack", "resolve", "--manifest", manifest, "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var receipt struct {
		Schema   string `json:"schema"`
		Status   string `json:"status"`
		Workload string `json:"workload"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, stdout.String())
	}
	if receipt.Schema != "fak-stack-receipt/1" || receipt.Status != "allow" || receipt.Workload != "coding@1" {
		t.Fatalf("receipt = %+v", receipt)
	}
}

func TestHarnessStackResolveRejectsMalformedInput(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(path, []byte(`{"schema":"wrong"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runHarnessCommand(&stdout, &stderr, []string{"stack", "resolve", "--manifest", path})
	if code != 1 || !strings.Contains(stderr.String(), "want \"fak-stack-manifest/1\"") {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
}

func TestHarnessStackResolveMatchesDemoFixtures(t *testing.T) {
	for _, tc := range []struct {
		name       string
		fixture    string
		wantCode   int
		wantStatus string
		wantChain  []string
	}{
		{name: "allow", fixture: "coding-stack.json", wantCode: 0, wantStatus: "allow"},
		{name: "refuse", fixture: "awq-sm75-unsat.json", wantCode: 3, wantStatus: "refuse", wantChain: []string{"harness:ponytail@r8", "model:coder-awq@sha256:111", "backend:awq@1.4.2", "kernel:awq-fast@0.9", "device.cuda.sm80"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fixture := filepath.Join("..", "..", "internal", "stackresolve", "testdata", tc.fixture)
			var stdout, stderr bytes.Buffer
			code := runHarnessCommand(&stdout, &stderr, []string{"stack", "resolve", "--manifest", fixture, "--json"})
			if code != tc.wantCode {
				t.Fatalf("code=%d want=%d stderr=%s", code, tc.wantCode, stderr.String())
			}
			var receipt struct {
				Schema   string `json:"schema"`
				Status   string `json:"status"`
				Conflict *struct {
					Chain []string `json:"chain"`
				} `json:"conflict"`
			}
			if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
				t.Fatal(err)
			}
			if receipt.Schema != "fak-stack-receipt/1" || receipt.Status != tc.wantStatus {
				t.Fatalf("receipt = %+v", receipt)
			}
			if len(tc.wantChain) > 0 && (receipt.Conflict == nil || strings.Join(receipt.Conflict.Chain, " -> ") != strings.Join(tc.wantChain, " -> ")) {
				t.Fatalf("conflict = %+v, want chain %v", receipt.Conflict, tc.wantChain)
			}
		})
	}
}

func TestHarnessStackResolveUsage(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runHarnessCommand(&stdout, &stderr, []string{"stack"}); code != 2 || !strings.Contains(stderr.String(), "harness stack resolve") {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
}

func stackManifestFixture(t *testing.T, refuse bool) string {
	t.Helper()
	device := "device.cuda.sm80"
	components := `
    {"id":"harness:ponytail@r8","kind":"harness","version":"r8","relations":[{"kind":"requires","target":"model.coder","evidence":{"authority":"maintainer","source":"kit-lock"}}],"evidence":{"authority":"maintainer","source":"kit-lock"}},
    {"id":"model:coder@sha256:111","kind":"model","version":"sha256:111","provides":["model.coder"],"evidence":{"authority":"publisher","source":"model-card"}}`
	if refuse {
		components = `
    {"id":"harness:ponytail@r8","kind":"harness","version":"r8","relations":[{"kind":"requires","target":"model.coder","evidence":{"authority":"maintainer","source":"kit-lock"}}],"evidence":{"authority":"maintainer","source":"kit-lock"}},
    {"id":"model:coder@sha256:111","kind":"model","version":"sha256:111","provides":["model.coder"],"relations":[{"kind":"requires","target":"` + device + `","evidence":{"authority":"runtime","source":"compat-table","tier":"observed"}}],"evidence":{"authority":"publisher","source":"model-card"}}`
	}
	raw := `{"schema":"fak-stack-manifest/1","workload":"coding@1","roots":["harness:ponytail@r8"],"components":[` + components + `]}`
	path := filepath.Join(t.TempDir(), "stack.json")
	if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}
