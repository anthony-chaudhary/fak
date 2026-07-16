package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func runValidateJSON(t *testing.T, argv []string) (validateResult, int, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := runValidate(&stdout, &stderr, argv)
	var res validateResult
	if stdout.Len() > 0 {
		if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
			t.Fatalf("invalid JSON: %v\n%s", err, stdout.String())
		}
	}
	return res, code, stderr.String()
}

func TestValidateRequiresExplicitMine(t *testing.T) {
	_, code, stderr := runValidateJSON(t, []string{"--json"})
	if code != 2 || !bytes.Contains([]byte(stderr), []byte("at least one --mine")) {
		t.Fatalf("code=%d stderr=%q", code, stderr)
	}
}

func TestValidateCommittedTipPlusOnlyMine(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	repo, git := seedCIPreflightRepo(t)
	commitFiles(t, repo, git, "clean", map[string]string{
		"go.mod": cleanGoMod,
		"p/p.go": cleanGoFile,
		"p/p_test.go": `package p

import "testing"

func TestAdd(t *testing.T) {
	if Add(1, 2) != 3 {
		t.Fatal("bad")
	}
}
`,
		"peer/peer.go": "package peer\n\nfunc OK() {}\n",
	})
	// Caller-owned change is valid; unrelated tracked peer WIP is intentionally uncompilable.
	if err := os.WriteFile(filepath.Join(repo, "p", "p.go"), []byte("package p\n\n// Add returns a + b.\nfunc Add(a, b int) int {\n\treturn a + b + 0\n}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "peer", "peer.go"), []byte("package peer\n\nfunc Broken( {\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	res, code, stderr := runValidateJSON(t, []string{"--root", repo, "--mine", "p/p.go", "--json"})
	if code != 0 || !res.OK {
		t.Fatalf("code=%d stderr=%q result=%+v", code, stderr, res)
	}
	if len(res.Mine) != 1 || res.Mine[0] != "p/p.go" {
		t.Fatalf("mine=%v", res.Mine)
	}
	if len(res.Tested) == 0 {
		t.Fatalf("expected affected package test selection")
	}
}

func TestValidateIgnoresUnformattedPeerWIP(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	repo, git := seedCIPreflightRepo(t)
	commitFiles(t, repo, git, "clean", map[string]string{"go.mod": cleanGoMod, "p/p.go": cleanGoFile, "peer/peer.go": "package peer\n\nfunc OK() {}\n"})
	if err := os.WriteFile(filepath.Join(repo, "p", "p.go"), []byte(cleanGoFile), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "peer", "peer.go"), []byte("package peer\nfunc Peer( ){ }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	res, code, stderr := runValidateJSON(t, []string{"--root", repo, "--mine", "p/p.go", "--json"})
	if code != 0 || !res.OK {
		t.Fatalf("code=%d stderr=%q result=%+v", code, stderr, res)
	}
}

func TestValidateReportsMineFailure(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	repo, git := seedCIPreflightRepo(t)
	commitFiles(t, repo, git, "clean", map[string]string{"go.mod": cleanGoMod, "p/p.go": cleanGoFile})
	if err := os.WriteFile(filepath.Join(repo, "p", "p.go"), []byte("package p\n\nfunc Broken( {\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	res, code, _ := runValidateJSON(t, []string{"--root", repo, "--mine", "p/p.go", "--json"})
	if code != 1 || res.OK {
		t.Fatalf("code=%d result=%+v", code, res)
	}
	found := false
	for _, failure := range res.Failures {
		if failure.Step == "build" {
			found = true
		}
	}
	if !found {
		t.Fatalf("failures=%+v; want build failure", res.Failures)
	}
}

func TestValidateIncludesReverseDependencyTests(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	repo, git := seedCIPreflightRepo(t)
	commitFiles(t, repo, git, "clean", map[string]string{
		"go.mod":                    "module validate.test\n\ngo 1.26\n",
		"lib/lib.go":                "package lib\n\nfunc Value() int { return 1 }\n",
		"consumer/consumer.go":      "package consumer\n\nimport \"validate.test/lib\"\n\nfunc Value() int { return lib.Value() }\n",
		"consumer/consumer_test.go": "package consumer\n\nimport \"testing\"\n\nfunc TestValue(t *testing.T) {\n\tif Value() != 1 { t.Fatal(\"contract changed\") }\n}\n",
	})
	if err := os.WriteFile(filepath.Join(repo, "lib", "lib.go"), []byte("package lib\n\nfunc Value() int { return 2 }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	res, code, _ := runValidateJSON(t, []string{"--root", repo, "--mine", "lib/lib.go", "--json"})
	if code != 1 || res.OK {
		t.Fatalf("code=%d result=%+v", code, res)
	}
	if !validateContains(res.Tested, "validate.test/consumer") {
		t.Fatalf("tested=%v; reverse dependency absent", res.Tested)
	}
	found := false
	for _, failure := range res.Failures {
		if failure.Step == "test" {
			found = true
		}
	}
	if !found {
		t.Fatalf("failures=%+v; want affected test failure", res.Failures)
	}
}

func TestNormalizeMinePathsRejectsEscape(t *testing.T) {
	root := t.TempDir()
	if _, err := normalizeMinePaths(root, []string{"../peer.go"}); err == nil {
		t.Fatal("expected repo escape refusal")
	}
	if _, err := normalizeMinePaths(root, []string{"."}); err == nil {
		t.Fatal("expected repo-root refusal")
	}
}

func validateContains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
