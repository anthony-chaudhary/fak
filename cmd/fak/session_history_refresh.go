package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionmine"
)

func defaultHistoryIndexPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".fak", "session-history", "index.json")
}

func runSessionHistoryRefresh(ctx context.Context, stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("session-history refresh", flag.ContinueOnError)
	fs.SetOutput(stderr)
	home, _ := os.UserHomeDir()
	index := fs.String("index", defaultHistoryIndexPath(), "durable history index path")
	codex := fs.String("codex-root", filepath.Join(home, ".codex", "sessions"), "Codex session JSONL root")
	claude := fs.String("claude-root", filepath.Join(home, ".claude", "projects"), "Claude project JSONL root")
	days := fs.Int("days", 30, "files modified in the last N days (0 scans all)")
	interval := fs.Duration("interval", 15*time.Minute, "recurring refresh interval")
	maxRuns := fs.Int("max-runs", 0, "stop after N refreshes (0 runs until canceled)")
	once := fs.Bool("once", false, "refresh once and exit")
	minSupport := fs.Int("min-support", 2, "minimum candidate session support")
	limit := fs.Int("limit", 25, "maximum ranked candidates")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *days < 0 || *maxRuns < 0 || *minSupport < 1 || *limit < 1 || (!*once && *interval <= 0) {
		fmt.Fprintln(stderr, "session-history refresh: invalid bounds")
		return 2
	}
	var since time.Time
	if *days > 0 {
		since = time.Now().Add(-time.Duration(*days) * 24 * time.Hour)
	}
	runs := *maxRuns
	if *once {
		runs = 1
	}
	err := sessionmine.RefreshIndex(ctx, sessionmine.RefreshOptions{Mine: sessionmine.Options{CodexRoot: *codex, ClaudeRoot: *claude, Since: since, MinSupport: *minSupport, Limit: *limit}, IndexPath: *index, Interval: *interval, MaxRuns: runs}, func(receipt sessionmine.RefreshReceipt) error { return sessionmine.WriteJSON(stdout, receipt) })
	if err != nil {
		if ctx.Err() != nil {
			return 0
		}
		fmt.Fprintf(stderr, "session-history refresh: %v\n", err)
		return 1
	}
	return 0
}
