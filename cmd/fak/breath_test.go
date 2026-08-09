package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
	"github.com/anthony-chaudhary/fak/internal/promptlint/breath"
)

func TestBreathIsCatalogedAsADevVerb(t *testing.T) {
	tier, ok := devindex.TierOf("breath")
	if !ok || tier != devindex.TierDev {
		t.Fatalf("TierOf(breath) = %q, %v; want %q, true", tier, ok, devindex.TierDev)
	}
	var catalog devindex.Catalog
	verb, ok := catalog.VerbByName("breath")
	if !ok || verb.Doc != "docs/ONE-BREATH-CONTRACT.md" || !strings.Contains(verb.Synopsis, "counted-ratchet") {
		t.Fatalf("breath catalog entry = %+v, %v", verb, ok)
	}
}

func TestBreathAdvisoryAndGateUseTheCountedFloor(t *testing.T) {
	root := newBreathRepo(t, map[string]string{
		"docs/explainers/clean.md": "# Clean\n\n> **In one breath:** A cache keeps an answer. The next call reuses it.\n\n**One line:** precise.\n",
		"docs/explainers/old.md":   "# Old\n\nNo summary.\n",
		"docs/explainers/new.md":   "# New\n\nNo summary.\n",
	})
	base := breath.FormatBaseline([]breath.Finding{{
		Kind: breath.BreathMissing, Path: "docs/explainers/old.md",
	}})
	writeBreathFile(t, root, breath.BaselineFile, base)

	var stdout, stderr bytes.Buffer
	code := runBreath(&stdout, &stderr, []string{"--root", root, "--floor", "0", "--json"})
	if code != 0 {
		t.Fatalf("advisory exit=%d stderr=%s", code, stderr.String())
	}
	var report breathReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode report: %v\n%s", err, stdout.String())
	}
	if report.Verdict != "ADVISORY" || !report.Advisory || report.NewFindings != 1 {
		t.Fatalf("report = %+v, want one advisory finding above the floor", report)
	}
	if report.Census.Pages != 3 || report.Census.Conforming != 1 || report.Census.Missing != 2 || report.Census.Failing != 0 {
		t.Fatalf("census = %+v, want pages=3 conforming=1 missing=2 failing=0", report.Census)
	}
	if report.Census.Notice != breath.ScopeNotice {
		t.Fatal("JSON report dropped the judgement-half scope notice")
	}

	stdout.Reset()
	stderr.Reset()
	code = runBreath(&stdout, &stderr, []string{"--root", root, "--floor", "0", "--gate"})
	if code != 1 || !strings.Contains(stdout.String(), "GROWING") || !strings.Contains(stdout.String(), "docs/explainers/new.md") {
		t.Fatalf("gate exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}

	// Once the measured debt is counted, the same corpus is explicitly not
	// growing. The report must retain all three census buckets and denominator;
	// a bare "clean" would overclaim what the ratchet knows.
	writeBreathFile(t, root, breath.BaselineFile, breath.FormatBaseline(report.Findings))
	stdout.Reset()
	stderr.Reset()
	code = runBreath(&stdout, &stderr, []string{"--root", root, "--floor", "0", "--gate"})
	if code != 0 || !strings.Contains(stdout.String(), "NOT_GROWING") ||
		!strings.Contains(stdout.String(), "3 pages: 1 conforming, 0 failing, 2 missing") {
		t.Fatalf("counted gate exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
}

func TestBreathEmitBaselineIsStableAndBypassesTheOldFloor(t *testing.T) {
	root := newBreathRepo(t, map[string]string{
		"docs/explainers/a.md": "# A\n\nNo summary.\n",
	})
	var first, second, stderr bytes.Buffer
	args := []string{"--root", root, "--floor", "0", "--emit-baseline"}
	if code := runBreath(&first, &stderr, args); code != 0 {
		t.Fatalf("first emit exit=%d stderr=%s", code, stderr.String())
	}
	if code := runBreath(&second, &stderr, args); code != 0 {
		t.Fatalf("second emit exit=%d stderr=%s", code, stderr.String())
	}
	if first.String() != second.String() {
		t.Fatal("baseline output changed across identical runs")
	}
	if !strings.Contains(first.String(), "BREATH_MISSING\tdocs/explainers/a.md\t1") {
		t.Fatalf("baseline omitted counted missing finding:\n%s", first.String())
	}
}

func newBreathRepo(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	for rel, body := range files {
		writeBreathFile(t, root, rel, body)
	}
	cmd := exec.Command("git", "-C", root, "init", "--quiet")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, out)
	}
	cmd = exec.Command("git", "-C", root, "add", "docs/explainers")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git add: %v: %s", err, out)
	}
	return root
}

func writeBreathFile(t *testing.T, root, rel, body string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
