package guardsessions

import (
	"os"
	"path/filepath"
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
