package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hostresurrect"
)

func TestHostRelaunchBrokerDryRunDrainsTypedSpool(t *testing.T) {
	dir := t.TempDir()
	req := hostresurrect.Request{Schema: hostresurrect.Schema, EventID: "evt", Session: "g1", CWD: `C:\work\repo`, Command: []string{"claude", "--resume", "g1"}, ResumeHandle: "g1"}
	if _, err := hostresurrect.Enqueue(dir, req); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	if rc := runHostRelaunchBroker(&out, &stderr, []string{"--dir", dir, "--dry-run"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	got := out.String()
	for _, w := range []string{"wt.exe", "new-tab", `C:\\work\\repo`, "claude", "--resume", "g1"} {
		if !strings.Contains(got, w) {
			t.Fatalf("%q missing %q", got, w)
		}
	}
	pending, _ := hostresurrect.Pending(dir)
	if len(pending) != 1 {
		t.Fatal("dry run consumed request")
	}
}
