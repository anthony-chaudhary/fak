package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// TestRunScheduledOpenCodeRunReceiptCarriesJob pins that every fak-opencode-run/1
// outcome row names its routine. Before, only the fak-cron-fire/1 slot row carried
// "job", so 0 of 1457 live outcome rows were attributable to a routine.
func TestRunScheduledOpenCodeRunReceiptCarriesJob(t *testing.T) {
	cases := []struct {
		name    string
		windows string
		unix    string
		outcome string
	}{
		{"succeeded", `echo {"session_id": "ses_job_ok"}`, `echo '{"session_id": "ses_job_ok"}'`, "succeeded"},
		{"failed", `echo {"session_id": "ses_job_fail"} & exit 3`, `echo '{"session_id": "ses_job_fail"}'; exit 3`, "failed"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ledger := filepath.Join(t.TempDir(), "ledger.jsonl")
			cmdArgs := []string{"sh", "-c", tc.unix}
			if runtime.GOOS == "windows" {
				cmdArgs = []string{"cmd", "/c", tc.windows}
			}
			var stdout, stderr bytes.Buffer
			_, _ = RunScheduledOpenCode(ScheduledOpenCodeOptions{
				Job:      "ops-evergreen-backlog-00",
				Ledger:   ledger,
				Interval: 15 * time.Minute,
				Timeout:  5 * time.Second,
				RunID:    "run-job-" + tc.name,
				Command:  cmdArgs,
				Stdout:   &stdout,
				Stderr:   &stderr,
			})
			b, err := os.ReadFile(ledger)
			if err != nil {
				t.Fatalf("read ledger: %v", err)
			}
			var found bool
			for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
				var row map[string]any
				if err := json.Unmarshal([]byte(line), &row); err != nil {
					t.Fatalf("ledger row not JSON: %v", err)
				}
				if row["schema"] != "fak-opencode-run/1" {
					continue
				}
				found = true
				if row["outcome"] != tc.outcome {
					t.Fatalf("outcome = %v, want %s", row["outcome"], tc.outcome)
				}
				if row["job"] != "ops-evergreen-backlog-00" {
					t.Fatalf("run receipt job = %v, want ops-evergreen-backlog-00", row["job"])
				}
			}
			if !found {
				t.Fatalf("no fak-opencode-run/1 row in ledger")
			}
		})
	}
}
