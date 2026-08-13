package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type dispatchRepoPulseTotals struct {
	Launches         int
	SavedTokens      int64
	ToolTurnsSkipped int64
	JournalRows      int64
	DuplicateRows    int
}

// foldDispatchRepoPulseReceipts folds one durable spawn sidecar per launch. The
// sidecar filename is the launch identity; duplicate JSON with the same pid/issue
// key is counted once so retries cannot manufacture savings.
func foldDispatchRepoPulseReceipts(dir string) dispatchRepoPulseTotals {
	var out dispatchRepoPulseTotals
	seen := map[string]bool{}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return out
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			continue
		}
		var row struct {
			Schema    string `json:"schema"`
			Issue     int    `json:"issue"`
			PID       int    `json:"pid"`
			RepoPulse struct {
				Schema           string `json:"schema"`
				SavedTokens      int64  `json:"saved_tokens"`
				ToolTurnsSkipped int64  `json:"tool_turns_skipped"`
				JournalRows      int64  `json:"journal_rows"`
			} `json:"repo_pulse_receipt"`
		}
		if json.Unmarshal(b, &row) != nil || row.RepoPulse.Schema != "fak-dispatch-repo-pulse-receipt/1" {
			continue
		}
		key := entry.Name()
		if row.PID != 0 || row.Issue != 0 {
			key = fmt.Sprintf("%d/%d", row.Issue, row.PID)
		}
		if seen[key] {
			out.DuplicateRows++
			continue
		}
		seen[key] = true
		out.Launches++
		out.SavedTokens += row.RepoPulse.SavedTokens
		out.ToolTurnsSkipped += row.RepoPulse.ToolTurnsSkipped
		out.JournalRows += row.RepoPulse.JournalRows
	}
	return out
}

func writeRepoPulseMetrics(w *promWriter, dir string) {
	t := foldDispatchRepoPulseReceipts(dir)
	w.gauge("fak_fleet_repo_pulse_launches_total", "Dispatched coding-agent launches carrying a durable governed repo-pulse receipt.", float64(t.Launches))
	w.gauge("fak_fleet_repo_pulse_context_tokens_saved_total", "Estimated parent context tokens avoided by default-on governed repository orientation; not provider billing.", float64(t.SavedTokens))
	w.gauge("fak_fleet_repo_pulse_tool_turns_skipped_total", "Parent-visible tool turns avoided by default-on governed repository orientation.", float64(t.ToolTurnsSkipped))
	w.gauge("fak_fleet_repo_pulse_journal_rows_total", "Governed child-call journal rows represented by repository orientation receipts.", float64(t.JournalRows))
	w.gauge("fak_fleet_repo_pulse_duplicate_receipts_dropped_total", "Duplicate launch receipts excluded from cumulative repository-orientation savings.", float64(t.DuplicateRows))
}

type repoPulseCohortReadiness struct {
	Verdict      string `json:"verdict"`
	PostLaunches int    `json:"post_launches"`
	Minimum      int    `json:"minimum"`
	Reason       string `json:"reason"`
}

func assessRepoPulseCohort(dir string, minimum int) repoPulseCohortReadiness {
	if minimum < 1 {
		minimum = 1
	}
	t := foldDispatchRepoPulseReceipts(dir)
	r := repoPulseCohortReadiness{Verdict: "not-yet", PostLaunches: t.Launches, Minimum: minimum}
	if t.Launches < minimum {
		r.Reason = fmt.Sprintf("need %d more durable post-default launch receipt(s) before outcome comparison", minimum-t.Launches)
		return r
	}
	r.Verdict = "ready"
	r.Reason = "post-default cohort meets the sample floor; match a pre-default cohort before claiming throughput change"
	return r
}
