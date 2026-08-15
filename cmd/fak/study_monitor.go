package main

import (
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/studymonitor"
)

const defaultStudyMonitorRegistry = "docs/research/monitored-repositories.json"

func runStudyMonitor(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("study-monitor", flag.ContinueOnError)
	fs.SetOutput(stderr)
	registryPath := fs.String("registry", defaultStudyMonitorRegistry, "path to the monitored repository registry")
	dueDays := fs.Int("due-days", 14, "mark a repository due after this many days without a check")
	asOf := fs.String("as-of", "", "report date in YYYY-MM-DD (defaults to today)")
	jsonOutput := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || *dueDays < 1 {
		fmt.Fprintln(stderr, "usage: fak study-monitor [--registry PATH] [--due-days N] [--as-of YYYY-MM-DD] [--json]")
		return 2
	}
	now := time.Now().UTC()
	if *asOf != "" {
		parsed, err := time.Parse("2006-01-02", *asOf)
		if err != nil {
			fmt.Fprintf(stderr, "study-monitor: invalid --as-of: %v\n", err)
			return 2
		}
		now = parsed
	}
	registry, err := studymonitor.Read(*registryPath)
	if err != nil {
		fmt.Fprintf(stderr, "study-monitor: %v\n", err)
		return 1
	}
	report := studymonitor.BuildReport(*registryPath, registry, now, *dueDays)
	if *jsonOutput {
		if err := studymonitor.WriteJSON(stdout, report); err != nil {
			fmt.Fprintf(stderr, "study-monitor: %v\n", err)
			return 1
		}
		return 0
	}
	studymonitor.RenderHuman(stdout, report)
	return 0
}
