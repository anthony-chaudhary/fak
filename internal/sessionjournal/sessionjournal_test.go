package sessionjournal

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func mustTime(t *testing.T, s string) time.Time {
	t.Helper()
	tm, err := time.Parse(time.RFC3339, s)
	if err != nil {
		t.Fatalf("parse %q: %v", s, err)
	}
	return tm.UTC()
}

// TestClassify pins the boot-epoch verdict — the load-bearing crash detector.
func TestClassify(t *testing.T) {
	boot := mustTime(t, "2026-07-09T21:00:00Z")
	now := mustTime(t, "2026-07-09T21:30:00Z")
	cfg := ClassifyConfig{
		Now:        now,
		BootTime:   boot,
		StaleAfter: 15 * time.Minute,
		PIDAlive:   func(pid int) bool { return pid == 111 }, // 111 alive, everything else dead
	}
	cases := []struct {
		name       string
		s          Session
		wantStatus Status
		wantReason string
	}{
		{
			name:       "clean close is definitive",
			s:          Session{ID: "a", StartedAt: mustTime(t, "2026-07-09T20:00:00Z"), Closed: true},
			wantStatus: StatusClosed, wantReason: ReasonCleanExit,
		},
		{
			name:       "started before current boot -> machine reboot",
			s:          Session{ID: "b", StartedAt: mustTime(t, "2026-07-09T20:00:00Z"), PID: 111, LastSeen: now},
			wantStatus: StatusCrashed, wantReason: ReasonMachineReboot,
		},
		{
			name:       "same boot but pid dead -> crashed",
			s:          Session{ID: "c", StartedAt: mustTime(t, "2026-07-09T21:05:00Z"), PID: 222, LastSeen: now},
			wantStatus: StatusCrashed, wantReason: ReasonPIDDead,
		},
		{
			name:       "same boot, pid alive, stale beat -> stale",
			s:          Session{ID: "d", StartedAt: mustTime(t, "2026-07-09T21:05:00Z"), PID: 111, LastSeen: mustTime(t, "2026-07-09T21:10:00Z")},
			wantStatus: StatusStale, wantReason: ReasonStaleBeat,
		},
		{
			name:       "same boot, pid alive, fresh -> live",
			s:          Session{ID: "e", StartedAt: mustTime(t, "2026-07-09T21:05:00Z"), PID: 111, LastSeen: mustTime(t, "2026-07-09T21:29:00Z")},
			wantStatus: StatusLive, wantReason: ReasonLive,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Classify([]Session{tc.s}, cfg)
			if len(got) != 1 {
				t.Fatalf("want 1 verdict, got %d", len(got))
			}
			if got[0].Status != tc.wantStatus || got[0].Reason != tc.wantReason {
				t.Fatalf("got (%s,%s), want (%s,%s)", got[0].Status, got[0].Reason, tc.wantStatus, tc.wantReason)
			}
		})
	}
}

// TestClassifyUnknownBootSkipsReboot: with no boot time, a session that looks pre-boot must
// NOT be called MACHINE_REBOOT — the fold degrades to PID / stale / live, never a false crash.
func TestClassifyUnknownBoot(t *testing.T) {
	now := mustTime(t, "2026-07-09T21:30:00Z")
	cfg := ClassifyConfig{Now: now, BootTime: time.Time{}, StaleAfter: 15 * time.Minute, PIDAlive: func(int) bool { return true }}
	s := Session{ID: "x", StartedAt: mustTime(t, "2020-01-01T00:00:00Z"), PID: 5, LastSeen: now}
	got := Classify([]Session{s}, cfg)
	if got[0].Status != StatusLive {
		t.Fatalf("unknown boot must not crash-classify; got %s/%s", got[0].Status, got[0].Reason)
	}
}

// TestFoldEventsReopenClearsClose: open -> close -> open (a resumed handle) must fold to a
// live (not closed) session with the latest start, and last_seen at the max event time.
func TestFoldEventsReopen(t *testing.T) {
	events := []Event{
		{Schema: Schema, Kind: KindOpen, ID: "s1", TS: "2026-07-09T20:00:00Z", CWD: "C:/work/a", PID: 10},
		{Schema: Schema, Kind: KindClose, ID: "s1", TS: "2026-07-09T20:10:00Z", Reason: "done"},
		{Schema: Schema, Kind: KindOpen, ID: "s1", TS: "2026-07-09T20:20:00Z", CWD: "C:/work/b", PID: 11},
		{Schema: Schema, Kind: KindBeat, ID: "s1", TS: "2026-07-09T20:25:00Z"},
	}
	sessions := FoldEvents(events)
	if len(sessions) != 1 {
		t.Fatalf("want 1 session, got %d", len(sessions))
	}
	s := sessions[0]
	if s.Closed {
		t.Fatalf("reopen must clear the closed flag")
	}
	if s.StartedAt != mustTime(t, "2026-07-09T20:20:00Z") {
		t.Fatalf("start should be the latest open, got %s", s.StartedAt)
	}
	if s.LastSeen != mustTime(t, "2026-07-09T20:25:00Z") {
		t.Fatalf("last_seen should be the max event time, got %s", s.LastSeen)
	}
	if s.CWD != "C:/work/b" || s.PID != 11 {
		t.Fatalf("provenance should reflect the latest open, got cwd=%s pid=%d", s.CWD, s.PID)
	}
}

// TestFoldEventsClose: open -> beat -> close folds to a closed session.
func TestFoldEventsClose(t *testing.T) {
	events := []Event{
		{Schema: Schema, Kind: KindOpen, ID: "s2", TS: "2026-07-09T20:00:00Z"},
		{Schema: Schema, Kind: KindBeat, ID: "s2", TS: "2026-07-09T20:05:00Z"},
		{Schema: Schema, Kind: KindClose, ID: "s2", TS: "2026-07-09T20:10:00Z", Reason: "graceful"},
	}
	s := FoldEvents(events)[0]
	if !s.Closed || s.CloseReason != "graceful" {
		t.Fatalf("want closed/graceful, got closed=%v reason=%q", s.Closed, s.CloseReason)
	}
}

// TestAppendParseRoundTrip: appended events read back; a garbage line and a foreign-schema
// line are skipped, not fatal.
func TestAppendParseRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "j.jsonl")
	if err := Append(path, Event{Kind: KindOpen, ID: "r1", TS: "2026-07-09T20:00:00Z"}); err != nil {
		t.Fatal(err)
	}
	if err := Append(path, Event{Kind: KindBeat, ID: "r1", TS: "2026-07-09T20:01:00Z"}); err != nil {
		t.Fatal(err)
	}
	// A torn line and a foreign-schema line must not corrupt the fold.
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = f.WriteString("{not json\n{\"schema\":\"other.v1\",\"id\":\"z\"}\n")
	_ = f.Close()

	evs := LoadFile(path)
	if len(evs) != 2 {
		t.Fatalf("want 2 valid events, got %d", len(evs))
	}
	if evs[0].ID != "r1" || evs[0].Schema != Schema {
		t.Fatalf("schema should be stamped on append, got %q/%q", evs[0].ID, evs[0].Schema)
	}
}

// TestFoldEventsDriveRoundTrip: an open event carrying a drive block survives
// Append -> LoadFile -> FoldEvents and lands on the folded Session with equal values.
func TestFoldEventsDriveRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "j.jsonl")
	drive := &DriveCarry{TurnsLeft: 3, TokensLeft: 12000, SpendMicroCentsLeft: 450_000_000, Generation: 2, ObjectivePinID: "obj-7"}
	if err := Append(path, Event{Kind: KindOpen, ID: "d1", TS: "2026-07-08T10:00:00Z", CWD: "C:/work/fak", Drive: drive}); err != nil {
		t.Fatal(err)
	}
	sessions := FoldEvents(LoadFile(path))
	if len(sessions) != 1 {
		t.Fatalf("want 1 session, got %d", len(sessions))
	}
	got := sessions[0].Drive
	if got == nil {
		t.Fatalf("drive carry lost through Append/LoadFile/FoldEvents")
	}
	if *got != *drive {
		t.Fatalf("drive round-trip mismatch: got %+v, want %+v", *got, *drive)
	}
	// The folded Session must own its own copy, not alias the event pointer.
	if got == drive {
		t.Fatalf("folded Session.Drive aliases the appended pointer; want a copy")
	}
}

// TestFoldEventsDriveNewestWins: a reopen (or beat) carrying a newer drive replaces the
// older carry — last non-nil write wins, exactly like the scalar provenance fields.
func TestFoldEventsDriveNewestWins(t *testing.T) {
	events := []Event{
		{Schema: Schema, Kind: KindOpen, ID: "d2", TS: "2026-07-09T20:00:00Z", Drive: &DriveCarry{TurnsLeft: 9, Generation: 1}},
		{Schema: Schema, Kind: KindBeat, ID: "d2", TS: "2026-07-09T20:05:00Z", Drive: &DriveCarry{TurnsLeft: 4, Generation: 2}},
	}
	s := FoldEvents(events)[0]
	if s.Drive == nil || s.Drive.TurnsLeft != 4 || s.Drive.Generation != 2 {
		t.Fatalf("newest drive should win, got %+v", s.Drive)
	}
}

// TestFoldEventsDriveNilUnchanged: an event with no drive folds to a nil Drive (today's
// behavior preserved), and a later nil-drive beat never clobbers an earlier carry.
func TestFoldEventsDriveNilUnchanged(t *testing.T) {
	// No drive anywhere -> nil.
	s := FoldEvents([]Event{{Schema: Schema, Kind: KindOpen, ID: "d3", TS: "2026-07-09T20:00:00Z"}})[0]
	if s.Drive != nil {
		t.Fatalf("absent drive should fold to nil, got %+v", s.Drive)
	}
	// A carry followed by a nil-drive beat retains the carry (nil never clobbers).
	s2 := FoldEvents([]Event{
		{Schema: Schema, Kind: KindOpen, ID: "d4", TS: "2026-07-09T20:00:00Z", Drive: &DriveCarry{TurnsLeft: 5}},
		{Schema: Schema, Kind: KindBeat, ID: "d4", TS: "2026-07-09T20:05:00Z"},
	})[0]
	if s2.Drive == nil || s2.Drive.TurnsLeft != 5 {
		t.Fatalf("nil-drive beat must not clobber an earlier carry, got %+v", s2.Drive)
	}
}

func TestDefaultPathEnvOverride(t *testing.T) {
	t.Setenv(EnvPath, "C:/tmp/custom-journal.jsonl")
	if got := DefaultPath(); got != "C:/tmp/custom-journal.jsonl" {
		t.Fatalf("env override not honored, got %q", got)
	}
}

// TestBootIDStableWithinBoot: two instants in the same 60s bucket share an id; instants in
// different buckets differ; a zero time yields the empty id.
func TestBootID(t *testing.T) {
	if BootID(time.Time{}) != "" {
		t.Fatal("zero boot time must yield empty id")
	}
	a := mustTime(t, "2026-07-09T21:00:05Z")
	b := mustTime(t, "2026-07-09T21:00:55Z")
	c := mustTime(t, "2026-07-09T21:02:05Z")
	if BootID(a) != BootID(b) {
		t.Fatalf("same-minute instants must share a boot id: %s vs %s", BootID(a), BootID(b))
	}
	if BootID(a) == BootID(c) {
		t.Fatalf("instants >60s apart should differ: %s == %s", BootID(a), BootID(c))
	}
}

// TestBootTimeReasonable: on the test host BootTime is either unknown (degraded) or a plausible
// past instant — never in the future. Loose on purpose so it holds on any CI OS.
func TestBootTimeReasonable(t *testing.T) {
	now := time.Now().UTC()
	bt, src := BootTime(now)
	if src == "unknown" {
		if !bt.IsZero() {
			t.Fatalf("unknown source must carry a zero time, got %s", bt)
		}
		return
	}
	if bt.After(now) {
		t.Fatalf("boot time %s is after now %s (src=%s)", bt, now, src)
	}
	if bt.Before(now.Add(-2 * 365 * 24 * time.Hour)) {
		t.Fatalf("boot time %s implausibly far in the past (src=%s)", bt, src)
	}
}
