package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
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

type dispatchBlockerReceipt struct {
	Action  string `json:"action"`
	Verdict string `json:"verdict"`
	Reason  string `json:"reason"`
	Backend string `json:"backend"`
}

// foldLatestDispatchBlocker explains a thin cohort from the newest dispatch tick
// without treating a refusal as a launch sample. A malformed newest receipt is
// surfaced as unreadable rather than falling back to older, potentially stale state.
func foldLatestDispatchBlocker(dir string, readiness *repoPulseCohortReadiness) {
	matches, err := filepath.Glob(filepath.Join(dir, "last-resolve-tick*.json"))
	if err != nil || len(matches) == 0 {
		return
	}
	type candidate struct {
		path string
		mod  int64
	}
	candidates := make([]candidate, 0, len(matches))
	for _, path := range matches {
		info, statErr := os.Stat(path)
		if statErr == nil {
			candidates = append(candidates, candidate{path: path, mod: info.ModTime().UnixNano()})
		}
	}
	if len(candidates) == 0 {
		return
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].mod == candidates[j].mod {
			return candidates[i].path < candidates[j].path
		}
		return candidates[i].mod > candidates[j].mod
	})
	latest := candidates[0].path
	readiness.DispatchEvidence = filepath.Base(latest)
	raw, readErr := os.ReadFile(latest)
	var receipt dispatchBlockerReceipt
	if readErr != nil || json.Unmarshal(raw, &receipt) != nil {
		readiness.DispatchBlocker = "dispatch-evidence-unreadable"
		readiness.NextAction = "inspect " + latest + " before retrying dispatch"
		return
	}
	if strings.EqualFold(strings.TrimSpace(receipt.Action), "refused") || strings.HasPrefix(strings.ToUpper(strings.TrimSpace(receipt.Verdict)), "REFUSE") {
		readiness.DispatchBlocker = strings.TrimSpace(receipt.Verdict)
		if readiness.DispatchBlocker == "" {
			readiness.DispatchBlocker = "dispatch-refused"
		}
		switch strings.ToUpper(readiness.DispatchBlocker) {
		case "REFUSE_NO_SEAT":
			readiness.NextAction = "wait for a leased worker to exit, then run fak dispatch tick --backend " + fallback(receipt.Backend, "codex")
		default:
			readiness.NextAction = "resolve the latest dispatch refusal, then run fak dispatch tick --backend " + fallback(receipt.Backend, "codex")
		}
	}
}

func fallback(value, defaultValue string) string {
	if strings.TrimSpace(value) == "" {
		return defaultValue
	}
	return strings.TrimSpace(value)
}
