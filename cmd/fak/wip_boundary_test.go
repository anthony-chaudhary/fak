package main

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/doomloop"
	"github.com/anthony-chaudhary/fak/internal/wipref"
)

func assertWipRef(t *testing.T, dir, session string) string {
	t.Helper()
	raw, err := gitWipOut(context.Background(), dir, nil, "rev-parse", wipref.SessionRef(session))
	if err != nil {
		t.Fatalf("missing checkpoint ref: %v", err)
	}
	return strings.TrimSpace(string(raw))
}

func dirtyWipRepo(t *testing.T) (string, string) {
	t.Helper()
	dir, file := wipTestRepo(t)
	if err := os.WriteFile(file, []byte("risky boundary edit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir, file
}

func TestGuardStopHookAutoCheckpointsWIP(t *testing.T) {
	dir, _ := dirtyWipRepo(t)
	t.Chdir(dir)
	t.Setenv("CLAUDE_CODE_SESSION_ID", "stop-session")
	t.Setenv(guardStopsLedgerEnv, filepath.Join(t.TempDir(), "stops.jsonl"))
	_ = runGuardStopHook(io.Discard, bytes.NewBufferString(`{}`), []string{"--mode", "off"})
	assertWipRef(t, dir, "stop-session")
}

func TestDoomloopNudgeAutoCheckpointsWIP(t *testing.T) {
	dir, _ := dirtyWipRepo(t)
	t.Chdir(dir)
	_, _, err := applyCorrection(t.TempDir(), "doom-session", doomloop.Result{Correction: doomloop.CorrectNudge, BurningFlatStreak: 3}, true, 1)
	if err != nil {
		t.Fatal(err)
	}
	assertWipRef(t, dir, "doom-session")
}
