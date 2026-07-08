package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/growthgate"
)

// TestGrowthgateUsage: an unknown flag is a usage error (exit 2).
func TestGrowthgateUsage(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runGrowthgate(&out, &errb, []string{"--nope"}); rc != 2 {
		t.Fatalf("bad flag rc = %d, want 2", rc)
	}
}

// TestGrowthgateCleanDir: a directory whose only append-only file is tiny stays
// under every budget — verdict ok, exit 0, and the "no offenders" line prints.
func TestGrowthgateCleanDir(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "small.jsonl"), []byte("{\"a\":1}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if rc := runGrowthgate(&out, &errb, []string{dir}); rc != 0 {
		t.Fatalf("clean dir rc = %d, want 0 (stderr=%s)", rc, errb.String())
	}
	if !strings.Contains(out.String(), "no offenders") {
		t.Errorf("clean dir output missing 'no offenders':\n%s", out.String())
	}
}

// TestGatherGrowthArtifacts: the gatherer picks up only the growth-prone
// suffixes, records real sizes, and prunes .git.
func TestGatherGrowthArtifacts(t *testing.T) {
	dir := t.TempDir()
	writes := map[string]string{
		"a.jsonl":         "one\ntwo\n",
		"b.log":           "log line\n",
		"c.err":           "err\n",
		"skip.txt":        "not a growth class\n",
		".git/objects.db": "should be pruned\n",
	}
	for rel, body := range writes {
		p := filepath.Join(dir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	arts, err := gatherGrowthArtifacts(dir, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]bool{}
	for _, a := range arts {
		got[filepath.Base(a.Path)] = true
		if a.Size <= 0 {
			t.Errorf("artifact %s has non-positive size %d", a.Path, a.Size)
		}
	}
	if len(arts) != 3 {
		t.Fatalf("gathered %d artifacts, want 3: %v", len(arts), got)
	}
	for _, want := range []string{"a.jsonl", "b.log", "c.err"} {
		if !got[want] {
			t.Errorf("gatherer missed %s", want)
		}
	}
	if got["skip.txt"] {
		t.Error("gatherer should skip .txt")
	}
	if got["objects.db"] {
		t.Error("gatherer should prune .git")
	}
}

// TestGrowthReportEnvelope pins the garden-facing top-level envelope: an ACTION
// census reports ok=false/verdict=ACTION (so a non-gating garden member folds it
// to advisory "action"), a clean census reports ok=true/verdict=OK, and the full
// classification is always carried under "report".
func TestGrowthReportEnvelope(t *testing.T) {
	action := growthgate.Report{Verdict: growthgate.SevAction, Scanned: 2, TotalBytes: 200 << 20,
		Findings: []growthgate.Finding{{Path: "x", Class: growthgate.ClassDosMetrics, Size: 200 << 20, Severity: growthgate.SevAction}}}
	env := growthReport([]string{"."}, action)
	if env["ok"] != false || env["verdict"] != "ACTION" || env["finding"] != "unbounded_growth_action" {
		t.Errorf("action envelope wrong: ok=%v verdict=%v finding=%v", env["ok"], env["verdict"], env["finding"])
	}
	for _, k := range []string{"ok", "verdict", "finding", "reason", "next_action", "report", "roots", "schema"} {
		if _, present := env[k]; !present {
			t.Errorf("envelope missing top-level key %q", k)
		}
	}

	clean := growthgate.Report{Verdict: growthgate.SevWatch, Scanned: 1}
	env = growthReport([]string{"."}, clean)
	if env["ok"] != true || env["verdict"] != "OK" {
		t.Errorf("watch/clean envelope should be ok=true/OK, got ok=%v verdict=%v", env["ok"], env["verdict"])
	}
}

// TestGrowthgateReapActuator drives the reaper end-to-end against real files: a
// dry-run touches nothing; --apply deletes the one COLD/ACTION/disposable file and
// leaves the HOT log and the WAL untouched; the ledger records every decision.
func TestGrowthgateReapActuator(t *testing.T) {
	dir := t.TempDir()
	mk := func(name string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte("x\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	coldLog := mk("cold.log")   // disposable + COLD + ACTION → reap
	hotLog := mk("hot.log")     // disposable + HOT → protected
	walFile := mk("lane.jsonl") // lane-journal (non-disposable) + COLD → protected

	rep := growthgate.Report{
		Verdict: growthgate.SevAction,
		Findings: []growthgate.Finding{
			{Path: coldLog, Class: growthgate.ClassDispatchLog, Size: 99, Severity: growthgate.SevAction, Hot: false},
			{Path: hotLog, Class: growthgate.ClassDispatchLog, Size: 99, Severity: growthgate.SevAction, Hot: true},
			{Path: walFile, Class: growthgate.ClassLaneJournal, Size: 99, Severity: growthgate.SevAction, Hot: false},
		},
	}

	// Dry-run: nothing deleted.
	var out, errb bytes.Buffer
	if rc := runGrowthgateReap(&out, &errb, rep, false, "", false); rc != 0 {
		t.Fatalf("dry-run rc = %d, want 0 (stderr=%s)", rc, errb.String())
	}
	if !strings.Contains(out.String(), "DRY-RUN") {
		t.Errorf("dry-run output missing DRY-RUN banner:\n%s", out.String())
	}
	for _, p := range []string{coldLog, hotLog, walFile} {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("dry-run must not delete %s: %v", p, err)
		}
	}

	// Apply with a ledger.
	ledger := filepath.Join(dir, "reap.jsonl")
	out.Reset()
	errb.Reset()
	if rc := runGrowthgateReap(&out, &errb, rep, true, ledger, false); rc != 0 {
		t.Fatalf("apply rc = %d, want 0 (stderr=%s)", rc, errb.String())
	}
	if _, err := os.Stat(coldLog); !os.IsNotExist(err) {
		t.Errorf("apply should delete the COLD disposable log, but it remains (%v)", err)
	}
	if _, err := os.Stat(hotLog); err != nil {
		t.Errorf("apply must NOT delete the HOT log: %v", err)
	}
	if _, err := os.Stat(walFile); err != nil {
		t.Errorf("apply must NOT delete the WAL: %v", err)
	}
	data, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatalf("ledger not written: %v", err)
	}
	if !strings.Contains(string(data), "\"action\":\"reaped\"") {
		t.Errorf("ledger missing a reaped decision:\n%s", data)
	}
}

// TestIsGrowthCandidate pins the suffix pre-filter.
func TestIsGrowthCandidate(t *testing.T) {
	for _, y := range []string{"x.jsonl", "X.LOG", "run.err", "a.b.jsonl"} {
		if !isGrowthCandidate(y) {
			t.Errorf("isGrowthCandidate(%q) = false, want true", y)
		}
	}
	for _, n := range []string{"x.txt", "x.json", "x.go", "log"} {
		if isGrowthCandidate(n) {
			t.Errorf("isGrowthCandidate(%q) = true, want false", n)
		}
	}
}
