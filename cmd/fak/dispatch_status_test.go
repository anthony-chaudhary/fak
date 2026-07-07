package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// plantLiveResolveWorker writes the resolve-*.log/.pid/.lease-tree.json sidecars a
// spawned worker leaves behind, pinned to THIS process's pid so the liveness gate
// (dispatchPIDAlive) passes hermetically — the same fixture shape the tick tests use.
func plantLiveResolveWorker(t *testing.T, runsDir string, issue int, lane string, tree []string) string {
	t.Helper()
	stem := filepath.Join(runsDir, fmt.Sprintf("resolve-%d-20000101-000000", issue))
	// The first log line must carry lane=<lane>; the body must clear the banner-noop
	// floor so it is classified as real live work, not a reapable stub.
	body := fmt.Sprintf("worker start lane=%s issue=%d\n%s\n", lane, issue, strings.Repeat("streamed work line\n", 40))
	if err := os.WriteFile(stem+".log", []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stem+".pid", []byte(fmt.Sprint(os.Getpid())), 0o644); err != nil {
		t.Fatal(err)
	}
	if len(tree) > 0 {
		raw, _ := json.Marshal(tree)
		if err := os.WriteFile(stem+dispatchLeaseTreeSidecarSuffix, raw, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return stem
}

func TestDispatchStatusReportsLiveWorkers(t *testing.T) {
	runsDir := t.TempDir()
	plantLiveResolveWorker(t, runsDir, 1406, "tools", []string{"tools/**"})
	plantLiveResolveWorker(t, runsDir, 2634, "tools", nil)

	out, errb, code := runDispatchAt("status", "--runs-dir", runsDir, "--json")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	var snap dispatchStatusSnapshot
	if err := json.Unmarshal([]byte(out), &snap); err != nil {
		t.Fatalf("decode json: %v\n%s", err, out)
	}
	if snap.Schema != dispatchStatusSchema {
		t.Fatalf("schema = %q, want %q", snap.Schema, dispatchStatusSchema)
	}
	if snap.LiveWorkerCount != 2 {
		t.Fatalf("live_worker_count = %d, want 2", snap.LiveWorkerCount)
	}
	if got := fmt.Sprint(snap.IssuesInFlight); got != "[1406 2634]" {
		t.Fatalf("issues_in_flight = %s, want [1406 2634]", got)
	}
	if got := fmt.Sprint(snap.LanesHeld); got != "[tools]" {
		t.Fatalf("lanes_held = %s, want [tools]", got)
	}
	// The lease-tree sidecar rides through to the snapshot for the collision view.
	var found bool
	for _, w := range snap.Workers {
		if w.Issue == 1406 {
			found = true
			if w.PID != os.Getpid() {
				t.Fatalf("worker #1406 pid = %d, want %d", w.PID, os.Getpid())
			}
			if fmt.Sprint(w.Tree) != "[tools/**]" {
				t.Fatalf("worker #1406 tree = %v, want [tools/**]", w.Tree)
			}
		}
	}
	if !found {
		t.Fatalf("worker #1406 missing from snapshot: %+v", snap.Workers)
	}
}

func TestDispatchStatusEmptyRunsDir(t *testing.T) {
	runsDir := t.TempDir()

	// Human card names the empty state without erroring.
	out, errb, code := runDispatchAt("status", "--runs-dir", runsDir)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	if !strings.Contains(out, "no live issue-resolution workers") {
		t.Fatalf("human card missing empty note:\n%s", out)
	}

	// JSON snapshot is a well-formed zero-worker payload.
	jout, _, jcode := runDispatchAt("status", "--runs-dir", runsDir, "--json")
	if jcode != 0 {
		t.Fatalf("json exit=%d", jcode)
	}
	var snap dispatchStatusSnapshot
	if err := json.Unmarshal([]byte(jout), &snap); err != nil {
		t.Fatalf("decode json: %v", err)
	}
	if snap.LiveWorkerCount != 0 || len(snap.Workers) != 0 {
		t.Fatalf("want zero workers, got %+v", snap)
	}
}

func TestDispatchStatusMarkdownParity(t *testing.T) {
	runsDir := t.TempDir()
	plantLiveResolveWorker(t, runsDir, 1406, "tools", nil)

	out, errb, code := runDispatchAt("status", "--runs-dir", runsDir, "--markdown")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	// Markdown output carries the same live count and a table row for the worker.
	if !strings.Contains(out, "### dispatch status — 1 live worker(s)") {
		t.Fatalf("markdown header missing:\n%s", out)
	}
	if !strings.Contains(out, "| #1406 | tools |") {
		t.Fatalf("markdown table row missing:\n%s", out)
	}
}

func TestDispatchStatusRejectsJSONAndMarkdown(t *testing.T) {
	_, _, code := runDispatchAt("status", "--json", "--markdown")
	if code != 2 {
		t.Fatalf("exit=%d, want 2 for conflicting output flags", code)
	}
}
