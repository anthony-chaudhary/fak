package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionmine"
)

func runSessionIndexHealth(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("session-history status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	home, _ := os.UserHomeDir()
	index := fs.String("index", defaultHistoryIndexPath(), "path to fak-sessionmine-index/1")
	codex := fs.String("codex-root", filepath.Join(home, ".codex", "sessions"), "Codex session JSONL root")
	claude := fs.String("claude-root", filepath.Join(home, ".claude", "projects"), "Claude project JSONL root")
	asJSON := fs.Bool("json", false, "emit fak-session-history-status/1 JSON")
	nowUnix := fs.Int64("now", 0, "clock for freshness math (0=current time)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *index == "" {
		fmt.Fprintln(stderr, "session-history status: --index is required")
		return 2
	}
	now := time.Now().UTC()
	if *nowUnix > 0 {
		now = time.Unix(*nowUnix, 0).UTC()
	}
	status := sessionmine.InspectIndexHealthWithOptions(sessionmine.IndexHealthOptions{IndexPath: *index, CodexRoot: *codex, ClaudeRoot: *claude, Now: now})
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, status, "session-history status")
	}
	fmt.Fprint(stdout, sessionmine.RenderIndexHealth(status))
	return 0
}
