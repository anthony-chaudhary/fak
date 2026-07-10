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
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/resume"
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

// TestSessionLsIdentity (A5, #4116): the guard-session surface joins each session's trace to
// its transcript UUID from the A1/A2 identity store, and renders a dash when there is no join
// yet — never a blank cell or another session's UUID.
func TestSessionLsIdentity(t *testing.T) {
	reg := t.TempDir()
	idx := strings.Join([]string{
		`{"schema":"fak.guard-session.v1","handle":"s1aaaaaa","trace_id":"trace-mapped","agent":"claude","pid":101,"cwd":"/w/a","audit":"a.jsonl","started_utc":"2026-07-08T02:00:00Z"}`,
		`{"schema":"fak.guard-session.v1","handle":"s2bbbbbb","trace_id":"trace-unmapped","agent":"codex","pid":102,"cwd":"/w/b","audit":"b.jsonl","started_utc":"2026-07-08T01:00:00Z"}`,
		"",
	}, "\n")
	if err := os.WriteFile(guardsessions.IndexPath(reg), []byte(idx), 0o644); err != nil {
		t.Fatal(err)
	}
	// The identity store joins only trace-mapped -> uuid-XYZ; trace-unmapped has no join.
	id := `{"ts":"2026-07-08T00:00:00Z","uuid":"uuid-XYZ","trace":"trace-mapped","account":".claude-a","via":"guard-sessionstart"}` + "\n"
	if err := os.WriteFile(resume.IdentityLedgerPath(reg), []byte(id), 0o644); err != nil {
		t.Fatal(err)
	}

	// uuidDetail pulls the value off the `  uuid: <x>` detail line, robust to column padding.
	uuidDetail := func(body string) string {
		for _, ln := range strings.Split(body, "\n") {
			if s := strings.TrimSpace(ln); strings.HasPrefix(s, "uuid:") {
				return strings.TrimSpace(strings.TrimPrefix(s, "uuid:"))
			}
		}
		return ""
	}

	// List: a UUID column exists and the joined UUID shows for the mapped trace.
	var out, errb bytes.Buffer
	if rc := runGuardSessions(&out, &errb, []string{"--reg-dir", reg}); rc != 0 {
		t.Fatalf("list rc=%d stderr=%s", rc, errb.String())
	}
	if list := out.String(); !strings.Contains(list, "UUID") || !strings.Contains(list, "uuid-XYZ") {
		t.Fatalf("list missing UUID column / joined uuid:\n%s", list)
	}

	// Resolve the mapped session: the detail names the transcript UUID.
	out.Reset()
	errb.Reset()
	if rc := runGuardSessions(&out, &errb, []string{"--reg-dir", reg, "s1aaaaaa"}); rc != 0 {
		t.Fatalf("resolve mapped rc=%d stderr=%s", rc, errb.String())
	}
	if got := uuidDetail(out.String()); got != "uuid-XYZ" {
		t.Fatalf("mapped detail uuid = %q, want uuid-XYZ:\n%s", got, out.String())
	}

	// Resolve the unmapped session: the uuid line is a dash, and never leaks the other UUID.
	out.Reset()
	errb.Reset()
	if rc := runGuardSessions(&out, &errb, []string{"--reg-dir", reg, "s2bbbbbb"}); rc != 0 {
		t.Fatalf("resolve unmapped rc=%d stderr=%s", rc, errb.String())
	}
	if got := uuidDetail(out.String()); got != "-" {
		t.Fatalf("unmapped detail uuid = %q, want a dash:\n%s", got, out.String())
	}
	if strings.Contains(out.String(), "uuid-XYZ") {
		t.Fatalf("unmapped session leaked the mapped UUID:\n%s", out.String())
	}
}

func TestMaybeRecordGuardSessionIndexProductionSeam(t *testing.T) {
	orig := guardSessionIndexRecorder
	defer func() { guardSessionIndexRecorder = orig }()
	var trace, agent, audit, nonce string
	guardSessionIndexRecorder = func(tid, a, ap, n string, started time.Time) string {
		trace, agent, audit, nonce = tid, a, ap, n
		return "handle"
	}
	j := journal.OpenMemory()
	got := maybeRecordGuardSessionIndex(j, "trace-1", []string{"codex", "exec"}, time.Unix(1700000000, 0))
	if got != "handle" || trace != "trace-1" || agent != "codex" || audit != j.Path() || nonce == "" {
		t.Fatalf("got=%q trace=%q agent=%q audit=%q nonce=%q", got, trace, agent, audit, nonce)
	}
	if got := maybeRecordGuardSessionIndex(nil, "trace", []string{"codex"}, time.Now()); got != "" {
		t.Fatalf("nil audit recorded %q", got)
	}
}
