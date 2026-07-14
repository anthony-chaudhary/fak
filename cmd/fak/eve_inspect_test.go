package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/evebridge"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// writeEveApp lays down a minimal source-tree Eve app fixture and returns its root.
func writeEveApp(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"agent/agent.ts":                                "export default {};",
		"agent/tools/greet.ts":                          "//",
		"agent/connections/crm.ts":                      "//",
		"agent/subagents/researcher/agent.ts":           "//",
		"agent/subagents/researcher/tools/summarize.ts": "//",
		"agent/schedules/daily-report.md":               "cron",
		"agent/sandbox/sandbox.ts":                      "//",
		"agent/sandbox/workspace/notes.md":              "seed",
	}
	for rel, content := range files {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// TestEveInspectCLIJSONDeterministic is the issue's CLI witness: the fixture
// app produces a green, byte-identical `fak eve inspect --json` manifest on
// repeated runs.
func TestEveInspectCLIJSONDeterministic(t *testing.T) {
	root := writeEveApp(t)
	run := func() (int, []byte, string) {
		var stdout, stderr bytes.Buffer
		code := runEveInspect(&stdout, &stderr, []string{"--json", root})
		return code, stdout.Bytes(), stderr.String()
	}
	code, first, stderrText := run()
	if code != 0 {
		t.Fatalf("expected green inspect, got exit %d (stderr: %s, stdout: %s)", code, stderrText, first)
	}
	var m evebridge.InspectManifest
	if err := json.Unmarshal(first, &m); err != nil {
		t.Fatalf("stdout is not a JSON manifest (%v):\n%s", err, first)
	}
	if m.Schema != evebridge.SchemaInspect || len(m.Tools) != 2 || len(m.SandboxMounts) != 1 {
		t.Fatalf("unexpected manifest shape: %+v", m)
	}
	if _, second, _ := run(); !bytes.Equal(first, second) {
		t.Fatal("two runs over the same tree emitted different manifests")
	}
}

// TestEveInspectCLIPolicyDraft: --policy-draft emits a manifest the real
// `fak policy --check` load path (policy.ParseRuntime) accepts.
func TestEveInspectCLIPolicyDraft(t *testing.T) {
	root := writeEveApp(t)
	var stdout, stderr bytes.Buffer
	if code := runEveInspect(&stdout, &stderr, []string{"--policy-draft", root}); code != 0 {
		t.Fatalf("expected a policy draft, got exit %d (stderr: %s)", code, stderr.String())
	}
	rt, err := policy.ParseRuntime(stdout.Bytes())
	if err != nil {
		t.Fatalf("fak policy --check load path rejected the emitted draft: %v", err)
	}
	if !rt.Adjudicator.Allow["greet"] || rt.Adjudicator.Allow["crm__anything"] {
		t.Fatalf("draft must allow authored tools and default-deny connection operations, allow=%v", rt.Adjudicator.Allow)
	}
}

// TestEveInspectCLIFailsClosedAndUsage: a non-eve root exits fail-closed with
// the typed reason in the JSON; a missing root is an IO error; extra args are usage.
func TestEveInspectCLIFailsClosedAndUsage(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runEveInspect(&stdout, &stderr, []string{t.TempDir()})
	if code != eveInspectFailed {
		t.Fatalf("a non-eve root should exit %d, got %d", eveInspectFailed, code)
	}
	if !strings.Contains(stdout.String(), evebridge.CodeLayoutUnsupported) {
		t.Fatalf("expected %s in the JSON report, got %s", evebridge.CodeLayoutUnsupported, stdout.String())
	}

	stdout.Reset()
	if code := runEveInspect(&stdout, &stderr, []string{filepath.Join(t.TempDir(), "absent")}); code != 1 {
		t.Fatalf("a missing root should be exit 1, got %d", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("an IO error must not emit a report, got %q", stdout.String())
	}

	if code := runEveInspect(&stdout, &stderr, []string{"a", "b"}); code != 2 {
		t.Fatalf("two positional args should be usage exit 2, got %d", code)
	}
}
