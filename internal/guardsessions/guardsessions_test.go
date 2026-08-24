package guardsessions

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestHandleIsDeterministicAndDistinct(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	t1 := time.Unix(1_700_000_001, 0)
	// Same trace + same instant → same handle (deterministic).
	if a, b := Handle("guard", t0), Handle("guard", t0); a != b {
		t.Fatalf("handle not deterministic: %s != %s", a, b)
	}
	// Same default trace id but different start instants → distinct handles (so two
	// sessions reusing the default "guard" trace on one box don't collide).
	if a, b := Handle("guard", t0), Handle("guard", t1); a == b {
		t.Fatalf("distinct start instants produced the same handle: %s", a)
	}
	// The handle has the git-short-sha feel: a "g" prefix + 8 hex.
	h := Handle("some-trace", t0)
	if len(h) != 9 || h[0] != 'g' { //boundarylint:ignore CHANGE_DETECTOR_TEST the handle is a fixed g+8hex (9-char) format
		t.Fatalf("handle %q is not the g+8hex form", h)
	}
}

func TestRecordThenLoadFoldsLatestPerHandleNewestFirst(t *testing.T) {
	dir := t.TempDir()
	base := time.Unix(1_700_000_000, 0)
	older := NewRow("trace-old", "claude", 100, "/w/a", "audit-a.jsonl", "nonce-a", base)
	newer := NewRow("trace-new", "codex", 200, "/w/b", "audit-b.jsonl", "nonce-b", base.Add(time.Hour))
	for _, r := range []Row{older, newer} {
		if err := Record(dir, r); err != nil {
			t.Fatalf("record: %v", err)
		}
	}
	// A re-record of the older handle (same handle, updated pid) must WIN in the fold.
	olderUpdated := older
	olderUpdated.PID = 999
	if err := Record(dir, olderUpdated); err != nil {
		t.Fatalf("re-record: %v", err)
	}

	rows := Load(dir)
	if len(rows) != 2 {
		t.Fatalf("rows = %d, want 2 (folded per handle): %+v", len(rows), rows)
	}
	// Newest start first: the "newer" session leads.
	if rows[0].Handle != newer.Handle {
		t.Fatalf("row[0] = %s, want newest %s", rows[0].Handle, newer.Handle)
	}
	// The re-record won: the older handle carries the updated pid.
	if rows[1].Handle != older.Handle || rows[1].PID != 999 {
		t.Fatalf("folded older row = %+v, want pid 999", rows[1])
	}
}

func TestLoadSkipsForeignAndMalformedLines(t *testing.T) {
	dir := t.TempDir()
	path := IndexPath(dir)
	body := "" +
		`{"schema":"fak.guard-session.v1","handle":"gaaaa1111","trace_id":"t1","started_utc":"2026-07-06T00:00:00Z"}` + "\n" +
		"not json at all\n" +
		`{"schema":"some.other.v1","handle":"gbbbb2222","trace_id":"t2"}` + "\n" + // wrong schema
		`{"schema":"fak.guard-session.v1","handle":"","trace_id":"t3"}` + "\n" + // no handle
		"\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	rows := Load(dir)
	if len(rows) != 1 || rows[0].Handle != "gaaaa1111" {
		t.Fatalf("rows = %+v, want only the one well-formed guard-session row", rows)
	}
}

func TestLoadMissingIndexIsEmptyNotError(t *testing.T) {
	if rows := Load(t.TempDir()); rows != nil {
		t.Fatalf("missing index = %+v, want nil", rows)
	}
}

func TestResolveExactThenPrefixThenAmbiguous(t *testing.T) {
	rows := []Row{
		{Schema: Schema, Handle: "g12345678", TraceID: "issue-2200", StartedAt: "2026-07-06T02:00:00Z"},
		{Schema: Schema, Handle: "g12abcdef", TraceID: "issue-2201", StartedAt: "2026-07-06T01:00:00Z"},
		{Schema: Schema, Handle: "gdeadbeef", TraceID: "guard", StartedAt: "2026-07-06T00:00:00Z"},
	}

	// Exact handle wins.
	if r := Resolve(rows, "g12345678"); r.Matched != 1 || r.Row.TraceID != "issue-2200" {
		t.Fatalf("exact handle: %+v", r)
	}
	// Exact trace id wins.
	if r := Resolve(rows, "guard"); r.Matched != 1 || r.Row.Handle != "gdeadbeef" {
		t.Fatalf("exact trace: %+v", r)
	}
	// Unambiguous prefix of the handle.
	if r := Resolve(rows, "gdead"); r.Matched != 1 || r.Row.Handle != "gdeadbeef" {
		t.Fatalf("handle prefix: %+v", r)
	}
	// Unambiguous prefix of the trace id.
	if r := Resolve(rows, "issue-2201"); r.Matched != 1 || r.Row.Handle != "g12abcdef" {
		t.Fatalf("trace prefix: %+v", r)
	}
	// Ambiguous prefix: "g12" matches two handles.
	if r := Resolve(rows, "g12"); r.Matched != 2 || len(r.Candidates) != 2 {
		t.Fatalf("ambiguous prefix should match 2: %+v", r)
	}
	// No match.
	if r := Resolve(rows, "zzz"); r.Matched != 0 {
		t.Fatalf("no match expected: %+v", r)
	}
	// Empty query matches nothing.
	if r := Resolve(rows, "  "); r.Matched != 0 {
		t.Fatalf("empty query should match nothing: %+v", r)
	}
}

// TestResolveExactBeatsLongerPrefix proves a full id resolves to ITSELF even when it is
// also a prefix of a longer handle (the exact-first rule).
func TestResolveExactBeatsLongerPrefix(t *testing.T) {
	rows := []Row{
		{Schema: Schema, Handle: "g123", TraceID: "t-a", StartedAt: "2026-07-06T00:00:00Z"},
		{Schema: Schema, Handle: "g1234567", TraceID: "t-b", StartedAt: "2026-07-06T00:00:01Z"},
	}
	r := Resolve(rows, "g123")
	if r.Matched != 1 || r.Row.Handle != "g123" {
		t.Fatalf("exact handle g123 should win over the longer g1234567: %+v", r)
	}
}

func TestIndexPath(t *testing.T) {
	got := IndexPath(filepath.Join("reg", "dir"))
	if want := filepath.Join("reg", "dir", IndexFileName); got != want {
		t.Fatalf("IndexPath = %q, want %q", got, want)
	}
}

func TestLiveInteractiveReadbackAndCleanExit(t *testing.T) {
	dir := t.TempDir()
	oldWindow, hadWindow := os.LookupEnv("WT_WINDOW")
	oldTab, hadTab := os.LookupEnv("WT_TAB_ID")
	t.Cleanup(func() {
		if hadWindow {
			_ = os.Setenv("WT_WINDOW", oldWindow)
		} else {
			_ = os.Unsetenv("WT_WINDOW")
		}
		if hadTab {
			_ = os.Setenv("WT_TAB_ID", oldTab)
		} else {
			_ = os.Unsetenv("WT_TAB_ID")
		}
	})
	_ = os.Setenv("WT_WINDOW", "main")
	_ = os.Setenv("WT_TAB_ID", "tab-7")

	start := time.Date(2026, 7, 14, 20, 0, 0, 0, time.UTC)
	var rows []Row
	for i := 0; i < 3; i++ {
		r := NewInteractiveRow("trace-"+string(rune('a'+i)), "claude", 100+i,
			filepath.Join(dir, string(rune('a'+i))), filepath.Join(dir, "audit.jsonl"), "", start.Add(time.Duration(i)*time.Second),
			[]string{"claude", "--continue"})
		if err := Record(dir, r); err != nil {
			t.Fatalf("Record row %d: %v", i, err)
		}
		rows = append(rows, r)
	}
	live := LiveInteractive(Load(dir))
	if len(live) != 3 {
		t.Fatalf("live rows = %d, want 3: %+v", len(live), live)
	}
	for _, r := range live {
		if r.CWD == "" || len(r.Command) != 2 || r.ResumeHandle == "" || r.WindowID != "main" || r.TabID != "tab-7" {
			t.Fatalf("incomplete relaunch row: %+v", r)
		}
	}

	if err := Record(dir, rows[1].Ended(start.Add(time.Minute))); err != nil {
		t.Fatal(err)
	}
	live = LiveInteractive(Load(dir))
	if len(live) != 2 {
		t.Fatalf("live rows after clean exit = %d, want 2: %+v", len(live), live)
	}
	for _, r := range live {
		if r.Handle == rows[1].Handle {
			t.Fatalf("cleanly-ended row remained live: %+v", r)
		}
	}
}

func TestLiveInteractiveExcludesDispatcherAndIncompleteRows(t *testing.T) {
	legacy := NewRow("legacy", "claude", 1, `C:\work`, "", "", time.Now())
	incomplete := NewInteractiveRow("bad", "claude", 2, "", "", "", time.Now(), nil)
	if got := LiveInteractive([]Row{legacy, incomplete}); len(got) != 0 {
		t.Fatalf("LiveInteractive = %+v, want none", got)
	}
}

// countIndexLines returns the number of non-blank lines in the index file at path.
func countIndexLines(t *testing.T, path string) int {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	n := 0
	for _, ln := range strings.Split(string(b), "\n") {
		if strings.TrimSpace(ln) != "" {
			n++
		}
	}
	return n
}

// writeSupersededIndex writes a raw append-only index at path holding `handles` distinct
// handles, each re-recorded across enough rounds that the raw file exceeds the compaction
// byte gate. The latest row per handle carries the final round as its PID, so a correct fold
// keeps exactly that row per handle. Any extra lines (blank/malformed/foreign) are appended
// verbatim after the valid rows. Returns the folded rows LoadFile should yield.
func writeSupersededIndex(t *testing.T, path string, handles int, extra ...string) {
	t.Helper()
	base := time.Unix(1_700_000_000, 0)
	seeds := make([]Row, handles)
	for i := 0; i < handles; i++ {
		seeds[i] = NewRow(fmt.Sprintf("trace-%d", i), "claude", 0, "/w/some/padded/working/dir", "audit-journal.jsonl", "nonce-value", base.Add(time.Duration(i)*time.Second))
	}
	var buf strings.Builder
	bytesWritten := 0
	round := 0
	for bytesWritten <= compactMinBytes+8192 {
		for i := 0; i < handles; i++ {
			r := seeds[i]
			r.PID = round // the latest round wins the fold
			b, err := json.Marshal(r)
			if err != nil {
				t.Fatal(err)
			}
			buf.Write(b)
			buf.WriteByte('\n')
			bytesWritten += len(b) + 1
		}
		round++
	}
	for _, e := range extra {
		buf.WriteString(e)
		buf.WriteByte('\n')
	}
	if err := os.WriteFile(path, []byte(buf.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestCompactFileFoldsSupersededRowsToOneRowPerHandle proves a file with many superseded
// rows per handle compacts to exactly one row per handle and that LoadFile returns the
// IDENTICAL folded set before and after — the losslessness invariant.
func TestCompactFileFoldsSupersededRowsToOneRowPerHandle(t *testing.T) {
	dir := t.TempDir()
	path := IndexPath(dir)
	const handles = 4
	writeSupersededIndex(t, path, handles)

	before := LoadFile(path)
	if len(before) != handles {
		t.Fatalf("folded before = %d, want %d", len(before), handles)
	}
	rawBefore := countIndexLines(t, path)
	if rawBefore <= handles {
		t.Fatalf("test setup did not create superseded rows: %d raw lines for %d handles", rawBefore, handles)
	}

	dropped, err := CompactFile(path)
	if err != nil {
		t.Fatalf("CompactFile: %v", err)
	}
	if dropped != rawBefore-handles {
		t.Fatalf("dropped = %d, want %d", dropped, rawBefore-handles)
	}

	after := LoadFile(path)
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("folded set changed across compaction:\n before=%+v\n after =%+v", before, after)
	}
	if got := countIndexLines(t, path); got != handles {
		t.Fatalf("compacted file has %d lines, want one per handle (%d)", got, handles)
	}
}

// TestCompactFileUnderGateLeavesBytesUntouched proves a small file below the size gate is
// left BYTE-for-BYTE untouched (only an os.Stat is spent).
func TestCompactFileUnderGateLeavesBytesUntouched(t *testing.T) {
	dir := t.TempDir()
	path := IndexPath(dir)
	base := time.Unix(1_700_000_000, 0)
	for i := 0; i < 3; i++ {
		if err := Record(dir, NewRow(fmt.Sprintf("t-%d", i), "claude", i, "/w", "a.jsonl", "n", base.Add(time.Duration(i)*time.Second))); err != nil {
			t.Fatal(err)
		}
	}
	pre, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	dropped, err := CompactFile(path)
	if err != nil {
		t.Fatalf("CompactFile: %v", err)
	}
	if dropped != 0 {
		t.Fatalf("dropped = %d, want 0 under the gate", dropped)
	}
	post, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(pre, post) {
		t.Fatalf("small file was rewritten under the gate")
	}
}

// TestCompactFileDropsMalformedAndForeignLikeLoadFile proves the rewrite physically drops a
// malformed line and a wrong-schema line EXACTLY as LoadFile already drops them on read, and
// that the folded set is unchanged across the rewrite.
func TestCompactFileDropsMalformedAndForeignLikeLoadFile(t *testing.T) {
	dir := t.TempDir()
	path := IndexPath(dir)
	const handles = 3
	const malformed = "not json at all"
	const foreign = `{"schema":"some.other.v1","handle":"gforeign0","trace_id":"tf"}`
	writeSupersededIndex(t, path, handles, malformed, foreign, "")

	before := LoadFile(path)
	if len(before) != handles {
		t.Fatalf("folded before = %d, want %d", len(before), handles)
	}
	if _, err := CompactFile(path); err != nil {
		t.Fatalf("CompactFile: %v", err)
	}
	after := LoadFile(path)
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("folded set changed across compaction:\n before=%+v\n after =%+v", before, after)
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	text := string(body)
	if strings.Contains(text, malformed) {
		t.Fatalf("malformed line survived the rewrite: %q", text)
	}
	if strings.Contains(text, "some.other.v1") {
		t.Fatalf("foreign-schema line survived the rewrite: %q", text)
	}
	if got := countIndexLines(t, path); got != handles {
		t.Fatalf("compacted file has %d lines, want %d (one per valid handle)", got, handles)
	}
}

// TestRecordCompactionTriggerNeverFailsRecord drives Record past the size gate and proves
// (1) the best-effort compaction trigger never returns an error that fails a guard launch,
// and (2) the trigger actually fires — the folded index ends far smaller than the raw
// append count, while Load still returns every distinct handle.
func TestRecordCompactionTriggerNeverFailsRecord(t *testing.T) {
	dir := t.TempDir()
	base := time.Unix(1_700_000_000, 0)
	const handles = 3
	appended := 0
	cum := 0
	for cum < 2*compactMinBytes {
		for i := 0; i < handles; i++ {
			r := NewRow(fmt.Sprintf("trace-%d", i), "claude", appended, "/w/some/padded/working/dir", "audit-journal.jsonl", "nonce-value", base.Add(time.Duration(i)*time.Second))
			if err := Record(dir, r); err != nil {
				t.Fatalf("Record must never fail on its compaction trigger: %v", err)
			}
			b, _ := json.Marshal(r)
			cum += len(b) + 1
			appended++
		}
	}
	lines := countIndexLines(t, IndexPath(dir))
	if lines >= appended {
		t.Fatalf("index not compacted by the Record trigger: %d lines for %d appends", lines, appended)
	}
	if rows := Load(dir); len(rows) != handles {
		t.Fatalf("folded rows = %d, want %d distinct handles", len(rows), handles)
	}
}

func TestNewInteractiveRowHostRecoveryIsExplicit(t *testing.T) {
	t.Setenv("FAK_HOST_RECOVERY_SESSION", "")
	row := NewInteractiveRow("trace", "codex", 1, t.TempDir(), "", "", time.Now(), []string{"codex"})
	if row.HostRecovery {
		t.Fatal("host recovery unexpectedly defaulted on without a session match")
	}
	t.Setenv("FAK_HOST_RECOVERY_SESSION", "current")
	row = NewInteractiveRow("trace", "codex", 1, t.TempDir(), "", "", time.Now(), []string{"codex"})
	if !row.HostRecovery {
		t.Fatal("current session was not opted in")
	}
}
