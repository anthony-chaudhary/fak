package main

import (
	"context"
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/sessionmine"
)

func runSessionHistory(stdout, stderr io.Writer, args []string) int {
	if len(args) > 0 && args[0] == "benchmark" {
		return runSessionHistoryBenchmark(context.Background(), stdout, stderr, args[1:])
	}
	if len(args) > 0 && args[0] == "refresh" {
		return runSessionHistoryRefresh(context.Background(), stdout, stderr, args[1:])
	}
	if len(args) > 0 && args[0] == "status" {
		return runSessionIndexHealth(stdout, stderr, args[1:])
	}
	fs := flag.NewFlagSet("session-history", flag.ContinueOnError)
	fs.SetOutput(stderr)
	index := fs.String("index", defaultHistoryIndexPath(), "path to fak-sessionmine-index/1")
	provider := fs.String("provider", "", "filter by provider")
	minErrors := fs.Int("min-errors", 0, "minimum tool errors")
	limit := fs.Int("limit", 25, "maximum ranked sessions")
	sessionID := fs.String("session", "", "drill into one session ID")
	tool := fs.String("tool", "", "filter by exact normalized trajectory step")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *index == "" || *minErrors < 0 || *limit < 1 {
		fmt.Fprintln(stderr, "session-history: --index is required; --min-errors must be >= 0; --limit must be >= 1")
		return 2
	}
	report, err := sessionmine.ExploreIndex(*index, sessionmine.HistoryOptions{Provider: *provider, MinErrors: *minErrors, Limit: *limit, SessionID: *sessionID, Tool: *tool})
	if err != nil {
		fmt.Fprintf(stderr, "session-history: %v\n", err)
		return 1
	}
	if err := sessionmine.WriteJSON(stdout, report); err != nil {
		fmt.Fprintf(stderr, "session-history: %v\n", err)
		return 1
	}
	return 0
}
