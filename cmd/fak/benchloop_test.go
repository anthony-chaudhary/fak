package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBenchLoopStatusAndNext(t *testing.T) {
	root := t.TempDir()
	writeBenchLoopFile(t, filepath.Join(root, "BENCHMARK-AUTHORITY.md"), "**Last updated:** 2026-06-29\n")
	benchDir := filepath.Join(root, "experiments", "benchmark")
	if err := os.MkdirAll(benchDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeBenchLoopJSON(t, filepath.Join(benchDir, "catalog.json"), map[string]any{"runs": []map[string]any{{
		"run_id": "run-a", "machine_id": "box-a", "model": "smollm2",
		"precision": "q8", "timestamp": "2026-06-29T00:00:00Z",
	}}})
	ledgerDir := filepath.Join(root, "docs", "nightrun")
	if err := os.MkdirAll(ledgerDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeBenchLoopFile(t, filepath.Join(ledgerDir, "collected.jsonl"),
		`{"schema":"fak-nightrun-collect/1","date":"2026-06-30","box":"ci-box","task_id":"bench-ablate","value":"smoke","command":"fak ablate --sweep vdso","outcome":"collected","generated_at":"2026-06-30T00:00:00Z"}`+"\n")

	t.Setenv("FAK_BOX_ID", "ci-box")
	t.Setenv("FAK_OFFLINE", "1")

	var out, errb bytes.Buffer
	if code := runBenchLoop(&out, &errb, []string{"status", "--workspace", root, "--now", "2026-06-30T00:00:00Z"}); code != 0 {
		t.Fatalf("status code=%d stderr=%s", code, errb.String())
	}
	for _, want := range []string{"benchmark super-loop status", "registry:", "catalog:", "next action:"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("status missing %q:\n%s", want, out.String())
		}
	}

	out.Reset()
	errb.Reset()
	if code := runBenchLoop(&out, &errb, []string{"next", "--workspace", root, "--now", "2026-06-30T00:00:00Z", "--json"}); code != 0 {
		t.Fatalf("next code=%d stderr=%s", code, errb.String())
	}
	var action struct {
		Kind    string `json:"kind"`
		Command string `json:"command"`
	}
	if err := json.Unmarshal(out.Bytes(), &action); err != nil {
		t.Fatalf("next json: %v\n%s", err, out.String())
	}
	if action.Kind == "" || action.Command == "" {
		t.Fatalf("next action incomplete: %+v", action)
	}
}

func TestBenchLoopWalk(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runBenchLoop(&out, &errb, []string{"walk"}); code != 0 {
		t.Fatalf("walk code=%d stderr=%s", code, errb.String())
	}
	for _, want := range []string{"benchmark super-loop walk", "fak bench-loop status", "fak bench-loop run --apply --loop"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("walk missing %q:\n%s", want, out.String())
		}
	}
}

func TestWorkspaceOrRepoRoot(t *testing.T) {
	if got, want := workspaceOrRepoRoot(""), repoRoot(); got != want {
		t.Fatalf("empty workspace: got %q, want repo root %q", got, want)
	}
	abs := t.TempDir()
	if got := workspaceOrRepoRoot(abs); got != abs {
		t.Fatalf("absolute workspace: got %q, want %q", got, abs)
	}
	rel := filepath.Join("some", "rel")
	got := workspaceOrRepoRoot(rel)
	if !filepath.IsAbs(got) {
		t.Fatalf("relative workspace not absolutized: %q", got)
	}
	if !strings.HasSuffix(got, rel) {
		t.Fatalf("absolutized workspace %q lost the relative tail %q", got, rel)
	}
}

func TestParsePinnedTime(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want string
	}{
		{"2026-06-30T05:00:00+02:00", "2026-06-30T03:00:00Z"}, // RFC3339, normalized to UTC
		{"20260630T030000Z", "2026-06-30T03:00:00Z"},          // compact
		{"2026-06-30", "2026-06-30T00:00:00Z"},                // date-only
	} {
		got, err := parsePinnedTimeFlag("--now", tc.in)
		if err != nil {
			t.Fatalf("parsePinnedTimeFlag(%q): %v", tc.in, err)
		}
		if got.Format("2006-01-02T15:04:05Z07:00") != tc.want {
			t.Fatalf("parsePinnedTimeFlag(%q) = %s, want %s", tc.in, got, tc.want)
		}
	}
	if _, err := parsePinnedTimeFlag("--now", "yesterday"); err == nil {
		t.Fatal("bad stamp: want error, got nil")
	} else if msg := err.Error(); !strings.Contains(msg, "--now") || !strings.Contains(msg, "YYYY-MM-DD") {
		t.Fatalf("error message lost flag name or format hint: %s", msg)
	}
}

func writeBenchLoopJSON(t *testing.T, path string, v any) {
	t.Helper()
	raw, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	writeBenchLoopFile(t, path, string(raw)+"\n")
}

func writeBenchLoopFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
