package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"path/filepath"
)

func runMicroDogfoodReadiness(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak micro collapse readiness", flag.ContinueOnError)
	fs.SetOutput(stderr)
	minimum := fs.Int("minimum", 5, "minimum durable post-default launch receipts required")
	jsonOut := fs.Bool("json", false, "emit the readiness receipt as JSON")
	runsDir := fs.String("runs-dir", filepath.Join(repoRoot(), ".dispatch-runs"), "dispatch run archive")
	if !parseFlags(fs, argv) {
		return 2
	}
	r := assessRepoPulseCohort(*runsDir, *minimum)
	if *jsonOut {
		_ = json.NewEncoder(stdout).Encode(r)
	} else {
		fmt.Fprintf(stdout, "repo-pulse cohort %s: post_launches=%d minimum=%d - %s\n", r.Verdict, r.PostLaunches, r.Minimum, r.Reason)
	}
	if r.Verdict != "ready" {
		return 3
	}
	return 0
}
