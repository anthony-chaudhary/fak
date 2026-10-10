package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/snapshot"
)

// fak-test:runtime fast est=50ms lane=default
func TestServeSessionStateCorruptFileQuarantinedAndServeContinues(t *testing.T) {
	good := func(t *testing.T) []byte {
		t.Helper()
		src := session.NewTable()
		src.Transition("sess-a", session.Paused, "")
		p := filepath.Join(t.TempDir(), "good.snap")
		dumpServeSessions(src, p)
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	wrongKind := func(t *testing.T) []byte {
		t.Helper()
		p := filepath.Join(t.TempDir(), "turn.snap")
		if err := snapshot.WriteFile(p, snapshot.KindTurn, "t", []int{1}, nil, 1); err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	for _, tc := range []struct {
		name  string
		bytes func(t *testing.T) []byte
	}{
		{"truncated", func(t *testing.T) []byte { b := good(t); return b[:len(b)/2] }},
		{"concatenated", func(t *testing.T) []byte { b := good(t); return append(append([]byte{}, b...), b...) }},
		{"torn-tail", func(t *testing.T) []byte { b := good(t); return append(append([]byte{}, b...), b[len(b)/3:]...) }},
		{"digest-mismatch", func(t *testing.T) []byte {
			return bytes.Replace(good(t), []byte(`"sess-a"`), []byte(`"sess-b"`), 1)
		}},
		{"wrong-kind", wrongKind},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "session-state.snap")
			bad := tc.bytes(t)
			if err := os.WriteFile(path, bad, 0o644); err != nil {
				t.Fatal(err)
			}

			tbl := session.NewTable()
			if err := restoreServeSessions(tbl, path); err != nil {
				t.Fatalf("corrupt --session-state must be quarantined, not fatal: %v", err)
			}
			if n := len(tbl.Snapshot()); n != 0 {
				t.Fatalf("restored %d session(s) from a corrupt file, want empty state", n)
			}
			if _, err := os.Stat(path); !os.IsNotExist(err) {
				t.Fatalf("corrupt file still at %s (stat err=%v), want it moved aside", path, err)
			}
			moved, _ := filepath.Glob(path + ".corrupt-*")
			if len(moved) != 1 {
				t.Fatalf("quarantine files = %v, want exactly one %s.corrupt-<unix>", moved, path)
			}
			kept, err := os.ReadFile(moved[0])
			if err != nil || !bytes.Equal(kept, bad) {
				t.Fatalf("quarantined bytes differ from the corrupt original (err=%v)", err)
			}

			tbl.Transition("sess-next", session.Throttled, "after-quarantine")
			dumpServeSessions(tbl, path)
			next := session.NewTable()
			if err := restoreServeSessions(next, path); err != nil {
				t.Fatalf("restart after quarantine: %v", err)
			}
			if got := next.Get("sess-next").Run; got != session.Throttled {
				t.Fatalf("restart after quarantine restored %v, want Throttled", got)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestServeSessionStateUnreadablePathStillFailsLoud(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session-state.snap")
	if err := os.Mkdir(path, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := restoreServeSessions(session.NewTable(), path); err == nil {
		t.Fatal("an unreadable --session-state path must fail loud, got nil")
	}
	if fi, err := os.Stat(path); err != nil || !fi.IsDir() {
		t.Fatalf("an I/O failure must not be quarantined; %s changed (err=%v)", path, err)
	}
}
