package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/studymonitor"
)

func TestStudyInventoryJSONMapsLocalCheckout(t *testing.T) {
	root := t.TempDir()
	writeStudyInventoryFixture(t, root, "README.md", "# demo\n")
	writeStudyInventoryFixture(t, root, "cmd/app/main.go", "package main\nfunc main() {}\n")

	var stdout, stderr bytes.Buffer
	code := RunStudyInventory(&stdout, &stderr, []string{
		"--root", root,
		"--repository", "owner/repo",
		"--revision", "abc123",
		"--observed-at", "2026-08-25T00:00:00Z",
		"--json",
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report studymonitor.InventoryMap
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode output: %v\n%s", err, stdout.String())
	}
	if report.Schema != studymonitor.InventoryMapSchema || report.Repository != "owner/repo" || report.IndexedRevision != "abc123" {
		t.Fatalf("report identity = %+v", report)
	}
	if report.Totals.RuntimeFiles != 1 {
		t.Fatalf("totals = %+v, want one runtime file", report.Totals)
	}
	if !strings.Contains(report.CompletenessNote, "still requires non-tree study artifacts") {
		t.Fatalf("completeness note = %q", report.CompletenessNote)
	}
}

func TestStudyInventoryWritesMarkdown(t *testing.T) {
	root := t.TempDir()
	writeStudyInventoryFixture(t, root, "README.md", "# demo\n")
	out := filepath.Join(t.TempDir(), "inventory.md")

	var stdout, stderr bytes.Buffer
	code := RunStudyInventory(&stdout, &stderr, []string{
		"--root", root,
		"--repository", "owner/repo",
		"--revision", "abc123",
		"--observed-at", "2026-08-25T00:00:00Z",
		"--out", out,
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "wrote study inventory map") {
		t.Fatalf("stdout = %q, want write confirmation", stdout.String())
	}
	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "# Study inventory: owner/repo") || !strings.Contains(string(data), "## Completeness Critic") {
		t.Fatalf("markdown output missing expected sections:\n%s", string(data))
	}
}

func writeStudyInventoryFixture(t *testing.T, root, rel, text string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestStudyInventoryParseExcludeSubmoduleFlags(t *testing.T) {
	got, err := parseExcludeSubmoduleFlags([]string{"vendor/K=private SSH-only URL", " 3rdparty/M = docs=only "})
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{"vendor/K": "private SSH-only URL", "3rdparty/M": "docs=only"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("parsed = %v, want %v", got, want)
	}
	for _, bad := range [][]string{{"vendor/K"}, {"vendor/K="}, {"=reason"}, {"vendor/K=a", "vendor/K=b"}} {
		if _, err := parseExcludeSubmoduleFlags(bad); err == nil {
			t.Fatalf("parseExcludeSubmoduleFlags(%q) accepted, want refusal", bad)
		}
	}
}

func TestStudyInventoryExcludeSubmoduleFlagRecordsExclusions(t *testing.T) {
	root := t.TempDir()
	writeStudyInventoryFixture(t, root, ".gitmodules", "[submodule \"K\"]\n\tpath = vendor/K\n[submodule \"M\"]\n\tpath = 3rdparty/M\n[submodule \"P\"]\n\tpath = 3rdparty/P\n")
	writeStudyInventoryFixture(t, root, "cmd/main.go", "package main\n")
	writeStudyInventoryFixture(t, root, "3rdparty/P/csrc/gemm.cu", "__global__ void gemm() {}\n")
	base := []string{"--root", root, "--repository", "owner/repo", "--revision", "abc", "--observed-at", "2026-10-22T00:00:00Z", "--json"}

	var stdout, stderr bytes.Buffer
	code := RunStudyInventory(&stdout, &stderr, append(append([]string(nil), base...),
		"--exclude-submodule", "vendor/K=private SSH-only URL", "--exclude-submodule", "3rdparty/M=docs only"))
	if code != 0 {
		t.Fatalf("study-inventory code = %d, stderr:\n%s", code, stderr.String())
	}
	var report studymonitor.InventoryMap
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	wantExcluded := []studymonitor.InventoryExcludedSubmodule{
		{Path: "3rdparty/M", Reason: "docs only"},
		{Path: "vendor/K", Reason: "private SSH-only URL"},
	}
	if !reflect.DeepEqual(report.ExcludedSubmodules, wantExcluded) || len(report.UnpopulatedSubmodules) != 0 {
		t.Fatalf("excluded = %+v unpopulated = %v, want both exclusions and nothing unpopulated", report.ExcludedSubmodules, report.UnpopulatedSubmodules)
	}
	if !report.KernelInventoryRequired() {
		t.Fatalf("report = %+v, want exclusions to keep kernel_inventory required", report)
	}

	for _, tt := range []struct {
		flag string
		code int
	}{
		{flag: "vendor/K", code: 2},
		{flag: "vendor/K=", code: 2},
		{flag: "vendor/Undeclared=gone", code: 1},
		{flag: "3rdparty/P=too large to fetch", code: 1},
	} {
		stdout.Reset()
		stderr.Reset()
		got := RunStudyInventory(&stdout, &stderr, append(append([]string(nil), base...), "--exclude-submodule", tt.flag))
		if got != tt.code {
			t.Fatalf("--exclude-submodule %q code = %d, want %d; stderr:\n%s", tt.flag, got, tt.code, stderr.String())
		}
	}
}
