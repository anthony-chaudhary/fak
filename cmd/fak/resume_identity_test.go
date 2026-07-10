package main

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// writeIdentityFixture writes an append-only identity store into regDir and returns it.
func writeIdentityFixture(t *testing.T, regDir, content string) {
	t.Helper()
	if err := os.WriteFile(resume.IdentityLedgerPath(regDir), []byte(content), 0o644); err != nil {
		t.Fatalf("write identity fixture: %v", err)
	}
}

// TestRunResumeIdentity drives the read-side verb (#4115) over a fixture store: it resolves
// either direction, carries provenance, emits JSON on demand, and exits non-zero on no match or
// a missing argument — never inventing a join.
func TestRunResumeIdentity(t *testing.T) {
	regDir := t.TempDir()
	writeIdentityFixture(t, regDir,
		`{"uuid":"u-abc","trace":"t-xyz","handle":"sess-7","account":"worker-a","via":"guard SessionStart"}
{"uuid":"u-two","trace":"t-two"}
`)

	t.Run("uuid resolves to its trace with provenance", func(t *testing.T) {
		var out, errb bytes.Buffer
		if code := runResumeIdentity(&out, &errb, []string{"--reg-dir", regDir, "u-abc"}); code != 0 {
			t.Fatalf("exit = %d, want 0; stderr=%s", code, errb.String())
		}
		got := out.String()
		if !strings.Contains(got, "uuid u-abc -> trace t-xyz") {
			t.Fatalf("output %q missing the resolved join", got)
		}
		if !strings.Contains(got, "handle=sess-7") || !strings.Contains(got, "account=worker-a") {
			t.Fatalf("output %q missing provenance", got)
		}
	})

	t.Run("trace resolves back to its uuid", func(t *testing.T) {
		var out, errb bytes.Buffer
		if code := runResumeIdentity(&out, &errb, []string{"--reg-dir", regDir, "t-xyz"}); code != 0 {
			t.Fatalf("exit = %d, want 0", code)
		}
		if !strings.Contains(out.String(), "trace t-xyz -> uuid u-abc") {
			t.Fatalf("output %q missing the reverse join", out.String())
		}
	})

	t.Run("--json emits the structured match", func(t *testing.T) {
		var out, errb bytes.Buffer
		if code := runResumeIdentity(&out, &errb, []string{"--reg-dir", regDir, "--json", "u-abc"}); code != 0 {
			t.Fatalf("exit = %d, want 0", code)
		}
		var got map[string]any
		if err := json.Unmarshal(out.Bytes(), &got); err != nil {
			t.Fatalf("bad JSON %q: %v", out.String(), err)
		}
		if got["matched"] != true || got["paired"] != "t-xyz" || got["direction"] != "uuid->trace" {
			t.Fatalf("json = %v, want matched uuid->trace t-xyz", got)
		}
	})

	t.Run("unknown id exits 4", func(t *testing.T) {
		var out, errb bytes.Buffer
		if code := runResumeIdentity(&out, &errb, []string{"--reg-dir", regDir, "nope"}); code != 4 {
			t.Fatalf("exit = %d, want 4 on no join", code)
		}
		if !strings.Contains(errb.String(), "no identity join") {
			t.Fatalf("stderr %q missing the no-join message", errb.String())
		}
	})

	t.Run("unknown id --json emits matched:false and exits 4", func(t *testing.T) {
		var out, errb bytes.Buffer
		if code := runResumeIdentity(&out, &errb, []string{"--reg-dir", regDir, "--json", "nope"}); code != 4 {
			t.Fatalf("exit = %d, want 4", code)
		}
		var got map[string]any
		if err := json.Unmarshal(out.Bytes(), &got); err != nil {
			t.Fatalf("bad JSON %q: %v", out.String(), err)
		}
		if got["matched"] != false {
			t.Fatalf("json = %v, want matched:false", got)
		}
	})

	t.Run("missing argument is a usage error", func(t *testing.T) {
		var out, errb bytes.Buffer
		if code := runResumeIdentity(&out, &errb, []string{"--reg-dir", regDir}); code != 2 {
			t.Fatalf("exit = %d, want 2 usage", code)
		}
	})
}

// TestRwLoadIdentity asserts the watchdog-side loader folds the store into the traceByUUID
// direction it needs, and fails open (nil) on a missing store.
func TestRwLoadIdentity(t *testing.T) {
	regDir := t.TempDir()

	if got := rwLoadIdentity(regDir); got != nil {
		t.Fatalf("rwLoadIdentity on missing store = %v, want nil (fail-open)", got)
	}

	writeIdentityFixture(t, regDir,
		`{"uuid":"u1","trace":"t1"}
{"uuid":"u2","trace":"t2"}
{"uuid":"u1","trace":"t3"}
`)
	got := rwLoadIdentity(regDir)
	if got["u1"] != "t3" || got["u2"] != "t2" {
		t.Fatalf("rwLoadIdentity = %v, want u1->t3 (last wins), u2->t2", got)
	}
}
