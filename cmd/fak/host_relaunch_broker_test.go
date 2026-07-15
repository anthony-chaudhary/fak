package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hostresurrect"
)

func TestHostRelaunchBrokerExpandsTildeSpool(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	dir := filepath.Join(home, "spool")
	cwd := t.TempDir()
	req := hostresurrect.Request{Schema: hostresurrect.Schema, EventID: "evt", Session: "g1", CWD: cwd, Command: []string{"claude", "--resume", "g1"}, ResumeHandle: "g1"}
	if _, err := hostresurrect.Enqueue(dir, req); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	if rc := runHostRelaunchBroker(&out, &stderr, []string{"--dir", "~/spool", "--dry-run"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(out.String(), "wt.exe") {
		t.Fatalf("expanded spool was not drained: %q", out.String())
	}
}

func TestHostRelaunchBrokerDryRunDrainsTypedSpool(t *testing.T) {
	dir := t.TempDir()
	cwd := t.TempDir()
	req := hostresurrect.Request{Schema: hostresurrect.Schema, EventID: "evt", Session: "g1", CWD: cwd, Command: []string{"claude", "--resume", "g1"}, ResumeHandle: "g1"}
	if _, err := hostresurrect.Enqueue(dir, req); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	if rc := runHostRelaunchBroker(&out, &stderr, []string{"--dir", dir, "--dry-run"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	got := out.String()
	for _, w := range []string{"wt.exe", "new-tab", cwd, "claude", "--resume", "g1"} {
		if !strings.Contains(got, w) {
			t.Fatalf("%q missing %q", got, w)
		}
	}
	pending, _ := hostresurrect.Pending(dir)
	if len(pending) != 1 {
		t.Fatal("dry run consumed request")
	}
}
