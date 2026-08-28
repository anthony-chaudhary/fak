package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestResumeHoldAcceptsIDFirstFlags(t *testing.T) {
	for _, args := range [][]string{{"session-1", "--state", "stopped", "--reason", "operator intent"}, {"--state", "stopped", "--reason", "operator intent", "session-1"}} {
		dir := t.TempDir()
		args = append([]string{"--reg-dir", dir}, args...)
		var out, err bytes.Buffer
		if rc := runResumeHold(&out, &err, args); rc != 0 {
			t.Fatalf("%v rc=%d err=%s", args, rc, err.String())
		}
		b, e := os.ReadFile(filepath.Join(dir, "resume_drivestate.jsonl"))
		if e != nil {
			t.Fatal(e)
		}
		var row map[string]any
		if e = json.Unmarshal(bytes.TrimSpace(b), &row); e != nil {
			t.Fatal(e)
		}
		if row["state"] != "stopped" || row["reason"] != "operator intent" {
			t.Fatalf("row=%v", row)
		}
	}
}
func TestResumeHoldRejectsExtraPositionals(t *testing.T) {
	var out, err bytes.Buffer
	if rc := runResumeHold(&out, &err, []string{"one", "two", "--reg-dir", t.TempDir()}); rc != 2 {
		t.Fatalf("rc=%d", rc)
	}
}
