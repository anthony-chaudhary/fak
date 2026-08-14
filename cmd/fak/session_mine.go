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

func runSessionMine(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("session-mine", flag.ContinueOnError)
	fs.SetOutput(stderr)
	home, _ := os.UserHomeDir()
	codex := fs.String("codex-root", filepath.Join(home, ".codex", "sessions"), "Codex session JSONL root")
	claude := fs.String("claude-root", filepath.Join(home, ".claude", "projects"), "Claude project JSONL root")
	days := fs.Int("days", 7, "scan files modified in the last N days (0 scans all)")
	minSupport := fs.Int("min-support", 2, "minimum distinct session support")
	limit := fs.Int("limit", 25, "maximum ranked candidates")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *days < 0 || *minSupport < 1 || *limit < 1 {
		fmt.Fprintln(stderr, "session-mine: --days must be >= 0; --min-support and --limit must be >= 1")
		return 2
	}
	var since time.Time
	if *days > 0 {
		since = time.Now().Add(-time.Duration(*days) * 24 * time.Hour)
	}
	report, err := sessionmine.Mine(sessionmine.Options{CodexRoot: *codex, ClaudeRoot: *claude, Since: since, MinSupport: *minSupport, Limit: *limit})
	if err != nil {
		fmt.Fprintf(stderr, "session-mine: %v\n", err)
		return 1
	}
	if err := sessionmine.WriteJSON(stdout, report); err != nil {
		fmt.Fprintf(stderr, "session-mine: %v\n", err)
		return 1
	}
	return 0
}
