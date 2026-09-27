package main

// Smoke tests for `fak session-audit reconcile` (#10673) against the committed
// regression fixture in internal/codexlifecycle/testdata/reconcile/issue-10673.
// The fixture reproduces the audited started-vs-terminal gap: 7 starts, 3
// observed terminals, a nonzero raw delta the fold's synthesized terminals
// fully explain (residual 0). Reports are aggregate counts only.

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
)

func reconcileFixtureRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", "..", "internal", "codexlifecycle", "testdata", "reconcile", "issue-10673"))
	if err != nil {
		t.Fatal(err)
	}
	return root
}

// reconcileClosedWindow pins the fixture as a closed past window. Live vs
// process_death for an open final start is decided by file mtime against the wall
// clock (default --fresh-mins 120), and a fresh checkout (CI, a validator's clean
// copy) stamps the fixture with a just-now mtime, which typed the open start Live
// and left residual_unaccounted=1. --fresh-mins 0 disables the freshness window so
// the fold is a function of the committed bytes alone, not of when they were checked out.
var reconcileClosedWindow = []string{"--fresh-mins", "0"}

func TestSessionAuditReconcileJSONFixtureSmoke(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runSessionAuditReconcile(&stdout, &stderr, append([]string{"--root", reconcileFixtureRoot(t), "--json"}, reconcileClosedWindow...))
	if rc != 0 {
		t.Fatalf("rc = %d, stderr: %s", rc, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		`"schema": "fak-codex-reconcile/1"`,
		`"task_started": 7`,
		`"task_complete": 3`,
		`"turn_aborted": 1`,
		`"raw_unaccounted": 3`,
		`"process_death": 1`,
		`"residual_unaccounted": 0`,
		`"all_starts_typed": true`,
		`"scanned": 4`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("json output missing %s", want)
		}
	}
	// NO CONTENT IN REPORTS: a conformance-named fixture carries no prompts or
	// message bodies, but the guard stays so a future fixture edit that adds
	// one fails here instead of leaking through the CLI surface.
	for _, forbidden := range []string{"last_agent_message", "prompt"} {
		if strings.Contains(out, forbidden) {
			t.Errorf("reconcile report leaked %q", forbidden)
		}
	}
}

func TestSessionAuditReconcileTextFixtureSmoke(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runSessionAuditReconcile(&stdout, &stderr, append([]string{"--root", reconcileFixtureRoot(t)}, reconcileClosedWindow...))
	if rc != 0 {
		t.Fatalf("rc = %d, stderr: %s", rc, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"reconcile",
		"task_started=7",
		"raw_unaccounted=3",
		"complete=3",
		"superseded=2",
		"process_death=1",
		"residual_unaccounted=0",
		"all_starts_typed=true",
		"provider fak 0.144.4",
		"provider openai 0.142.2",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("text output missing %q:\n%s", want, out)
		}
	}
}

func TestSessionAuditReconcileUsageListsSubcommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if rc := runSessionAudit(&stdout, &stderr, []string{"help"}); rc != 0 {
		t.Fatalf("rc = %d", rc)
	}
	if !strings.Contains(stderr.String()+stdout.String(), "session-audit reconcile") {
		t.Error("session-audit usage does not list the reconcile subcommand")
	}
}
