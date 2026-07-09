package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardsessions"
)

// seedGuardSessionIndex writes a fixed index under a temp reg dir and points --reg-dir at
// it, returning the reg dir. The two rows share a "g12" handle prefix so an ambiguity path
// is exercisable.
func seedGuardSessionIndex(t *testing.T) string {
	t.Helper()
	reg := t.TempDir()
	body := strings.Join([]string{
		`{"schema":"fak.guard-session.v1","handle":"g12aaaa1","trace_id":"issue-2200","agent":"claude","pid":111,"cwd":"/w/a","audit":"a.jsonl","started_utc":"2026-07-06T02:00:00Z"}`,
		`{"schema":"fak.guard-session.v1","handle":"g12bbbb2","trace_id":"issue-2201","agent":"codex","pid":222,"cwd":"/w/b","audit":"b.jsonl","started_utc":"2026-07-06T01:00:00Z"}`,
		`{"schema":"fak.guard-session.v1","handle":"gdeadbee","trace_id":"guard","agent":"claude","pid":333,"cwd":"/w/c","started_utc":"2026-07-06T00:00:00Z"}`,
		"",
	}, "\n")
	if err := os.WriteFile(guardsessions.IndexPath(reg), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return reg
}

func TestRunGuardSessionsListsNewestFirst(t *testing.T) {
	reg := seedGuardSessionIndex(t)
	var out, errb bytes.Buffer
	if rc := runGuardSessions(&out, &errb, []string{"--reg-dir", reg}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	body := out.String()
	// Newest (02:00 issue-2200 / g12aaaa1) must appear before the oldest (guard / gdeadbee).
	iNew := strings.Index(body, "g12aaaa1")
	iOld := strings.Index(body, "gdeadbee")
	if iNew < 0 || iOld < 0 || iNew > iOld {
		t.Fatalf("sessions not newest-first:\n%s", body)
	}
	if !strings.Contains(body, "HANDLE") || !strings.Contains(body, "reference one with") {
		t.Fatalf("list missing header/footer:\n%s", body)
	}
}

func TestRunGuardSessionsEmptyIndex(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runGuardSessions(&out, &errb, []string{"--reg-dir", t.TempDir()}); rc != 0 {
		t.Fatalf("rc=%d", rc)
	}
	if !strings.Contains(out.String(), "no recorded guard sessions") {
		t.Fatalf("empty index message missing:\n%s", out.String())
	}
}

func TestRunGuardSessionsResolvePrefixUnique(t *testing.T) {
	reg := seedGuardSessionIndex(t)
	var out, errb bytes.Buffer
	// "gdead" is a unique handle prefix.
	if rc := runGuardSessions(&out, &errb, []string{"--reg-dir", reg, "gdead"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	body := out.String()
	if !strings.Contains(body, "guard session gdeadbee") || !strings.Contains(body, "trace_id: guard") {
		t.Fatalf("resolved row not rendered:\n%s", body)
	}
}

func TestRunGuardSessionsResolveTracePrefixUnique(t *testing.T) {
	reg := seedGuardSessionIndex(t)
	var out, errb bytes.Buffer
	// "issue-2201" is a unique trace id.
	if rc := runGuardSessions(&out, &errb, []string{"--reg-dir", reg, "issue-2201"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "g12bbbb2") {
		t.Fatalf("trace-prefix resolve missed:\n%s", out.String())
	}
}

func TestRunGuardSessionsAmbiguousExit3(t *testing.T) {
	reg := seedGuardSessionIndex(t)
	var out, errb bytes.Buffer
	// "g12" matches two handles → ambiguous, exit 3.
	rc := runGuardSessions(&out, &errb, []string{"--reg-dir", reg, "g12"})
	if rc != 3 {
		t.Fatalf("ambiguous resolve rc=%d, want 3; stderr=%s", rc, errb.String())
	}
	if !strings.Contains(errb.String(), "ambiguous") {
		t.Fatalf("ambiguity not reported:\n%s", errb.String())
	}
}

func TestRunGuardSessionsNoMatchExit1(t *testing.T) {
	reg := seedGuardSessionIndex(t)
	var out, errb bytes.Buffer
	rc := runGuardSessions(&out, &errb, []string{"--reg-dir", reg, "zzz-nope"})
	if rc != 1 {
		t.Fatalf("no-match rc=%d, want 1", rc)
	}
	if !strings.Contains(errb.String(), "no guard session matches") {
		t.Fatalf("no-match message missing:\n%s", errb.String())
	}
}

func TestRunGuardSessionsJSONResolve(t *testing.T) {
	reg := seedGuardSessionIndex(t)
	var out, errb bytes.Buffer
	if rc := runGuardSessions(&out, &errb, []string{"--reg-dir", reg, "--json", "gdead"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	var row guardsessions.Row
	if err := json.Unmarshal(out.Bytes(), &row); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if row.Handle != "gdeadbee" || row.TraceID != "guard" {
		t.Fatalf("resolved json row = %+v", row)
	}
}

// TestRecordGuardSessionIndexRoundTrips proves the guard-start recording helper writes a
// row the query surface can then read back and resolve by its handle.
func TestRecordGuardSessionIndexRoundTrips(t *testing.T) {
	reg := t.TempDir()
	t.Setenv("FLEET_REG_DIR", reg) // resolveSweepRegDir("") reads this
	handle := recordGuardSessionIndex("issue-9999", "claude", filepath.Join("x", "audit.jsonl"), "nonce-xyz", time.Unix(1_700_000_000, 0))
	if handle == "" {
		t.Fatal("recordGuardSessionIndex returned empty handle")
	}
	var out, errb bytes.Buffer
	if rc := runGuardSessions(&out, &errb, []string{"--reg-dir", reg, handle}); rc != 0 {
		t.Fatalf("resolve recorded handle rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "trace_id: issue-9999") {
		t.Fatalf("recorded session did not round-trip:\n%s", out.String())
	}
}
