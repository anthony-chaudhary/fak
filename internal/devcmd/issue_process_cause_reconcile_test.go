package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func writeProcessCauseEvent(t *testing.T, body, state string, labels ...string) string {
	return writeProcessCauseEventFor(t, "owner/repo", 17, body, state, labels...)
}

func writeProcessCauseEventFor(t *testing.T, repo string, number int, body, state string, labels ...string) string {
	t.Helper()
	event := map[string]any{
		"action":     "opened",
		"repository": map[string]any{"full_name": repo},
		"issue": map[string]any{"number": number, "body": body, "state": state, "labels": func() []map[string]string {
			out := make([]map[string]string, len(labels))
			for i, label := range labels {
				out[i] = map[string]string{"name": label}
			}
			return out
		}()},
	}
	raw, err := json.Marshal(event)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "event.json")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func processCauseLiveJSON(t *testing.T, body, state string, labels ...string) string {
	t.Helper()
	live := map[string]any{
		"body":  body,
		"state": state,
		"labels": func() []map[string]string {
			out := make([]map[string]string, len(labels))
			for i, label := range labels {
				out[i] = map[string]string{"name": label}
			}
			return out
		}(),
	}
	raw, err := json.Marshal(live)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func recordingProcessCauseRunner(t *testing.T, calls *[][]string, liveBody, liveState string, liveLabels ...string) issueProcessCauseRunner {
	t.Helper()
	liveJSON := processCauseLiveJSON(t, liveBody, liveState, liveLabels...)
	return func(args []string) (string, string, bool) {
		*calls = append(*calls, append([]string(nil), args...))
		if len(args) >= 2 && args[0] == "issue" && args[1] == "view" {
			return liveJSON, "", true
		}
		return "", "", true
	}
}

func TestIssueProcessCauseReconcileValidReplacesStaleWithoutReopening(t *testing.T) {
	body := "### Development process cause\n\nProcess cause: concurrency\n\n### Concurrency detail\n\nProcess cause detail: shared-state\n"
	labels := []string{"bug", "process-cause:none", "process-cause:model-failure", processCauseRepairLabel}
	path := writeProcessCauseEventFor(t, "fork-owner/arbitrary-repo", 941,
		body, "closed", labels...)
	var calls [][]string
	runner := recordingProcessCauseRunner(t, &calls, body, "closed", labels...)
	var out, errb bytes.Buffer
	code := runIssueProcessCauseReconcileWith(&out, &errb, []string{"--event-file", path, "--json"}, runner)
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	want := [][]string{
		{"issue", "view", "941", "--repo", "fork-owner/arbitrary-repo", "--json", "body,state,labels"},
		{"label", "create", "process-cause:concurrency", "--repo", "fork-owner/arbitrary-repo", "--force", "--color", "0E8A16", "--description", "Declared development-process cause"},
		{"issue", "edit", "941", "--repo", "fork-owner/arbitrary-repo", "--add-label", "process-cause:concurrency", "--remove-label", "process-cause:model-failure", "--remove-label", "process-cause:none", "--remove-label", processCauseRepairLabel},
	}
	if !reflect.DeepEqual(calls, want) {
		t.Fatalf("commands mismatch:\n got: %#v\nwant: %#v", calls, want)
	}
}

func TestIssueProcessCauseReconcileMissingNeverDefaultsToUnknown(t *testing.T) {
	body := "### What happened\n\nNo declaration.\n"
	path := writeProcessCauseEvent(t, body, "open")
	var calls [][]string
	runner := recordingProcessCauseRunner(t, &calls, body, "open")
	var out, errb bytes.Buffer
	if code := runIssueProcessCauseReconcileWith(&out, &errb, []string{"--event-file", path}, runner); code != 3 {
		t.Fatalf("code=%d, want 3; stderr=%s", code, errb.String())
	}
	joined := commandText(calls)
	if strings.Contains(joined, "issue close") || strings.Contains(joined, "issue reopen") {
		t.Fatalf("reconciler changed issue state:\n%s", joined)
	}
	if strings.Contains(joined, "process-cause:unknown") {
		t.Fatalf("missing declaration was silently labeled unknown:\n%s", joined)
	}
	for _, want := range []string{"--add-label needs-process-cause", "missing Process cause declaration"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("commands missing %q:\n%s", want, joined)
		}
	}
}

func TestIssueProcessCauseReconcileInvalidLabelsWithoutClosing(t *testing.T) {
	body := "### Development process cause\n\nProcess cause: concurrency\n"
	path := writeProcessCauseEvent(t, body, "open", "process-cause:none")
	var calls [][]string
	runner := recordingProcessCauseRunner(t, &calls, body, "open", "process-cause:none")
	var out, errb bytes.Buffer
	code := runIssueProcessCauseReconcileWith(&out, &errb, []string{"--event-file", path}, runner)
	if code != 3 {
		t.Fatalf("code=%d, want 3; stderr=%s", code, errb.String())
	}
	joined := commandText(calls)
	for _, want := range []string{"--add-label needs-process-cause", "--remove-label process-cause:none", "issue comment 17", "requires Process cause detail"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("commands missing %q:\n%s", want, joined)
		}
	}
}

func TestIssueProcessCauseReconcileUsesLiveIssueInsteadOfStaleEvent(t *testing.T) {
	tests := []struct {
		name        string
		eventBody   string
		eventState  string
		eventLabels []string
		liveBody    string
		liveState   string
		liveLabels  []string
		wantLabel   string
	}{
		{
			name:        "stale invalid event cannot close repaired valid issue",
			eventBody:   "missing declaration",
			eventState:  "open",
			eventLabels: []string{"process-cause:none"},
			liveBody:    "Process cause: model-failure\n",
			liveState:   "open",
			liveLabels:  []string{"process-cause:model-failure"},
			wantLabel:   "process-cause:model-failure",
		},
		{
			name:        "stale valid event cannot restore old label",
			eventBody:   "Process cause: verification-gap\n",
			eventState:  "open",
			eventLabels: []string{"process-cause:verification-gap"},
			liveBody:    "Process cause: harness-failure\n",
			liveState:   "open",
			liveLabels:  []string{"process-cause:harness-failure"},
			wantLabel:   "process-cause:harness-failure",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			const repo = "another-owner/live-repo"
			const number = 812
			path := writeProcessCauseEventFor(t, repo, number, tc.eventBody, tc.eventState, tc.eventLabels...)
			var calls [][]string
			runner := recordingProcessCauseRunner(t, &calls, tc.liveBody, tc.liveState, tc.liveLabels...)
			var out, errb bytes.Buffer
			if code := runIssueProcessCauseReconcileWith(&out, &errb, []string{"--event-file", path}, runner); code != 0 {
				t.Fatalf("code=%d stderr=%s", code, errb.String())
			}
			want := [][]string{
				{"issue", "view", "812", "--repo", repo, "--json", "body,state,labels"},
				{"label", "create", tc.wantLabel, "--repo", repo, "--force", "--color", "0E8A16", "--description", "Declared development-process cause"},
			}
			if !reflect.DeepEqual(calls, want) {
				t.Fatalf("stale event influenced commands:\n got: %#v\nwant: %#v", calls, want)
			}
		})
	}
}

func TestIssueProcessCauseReconcileDryRunDoesNotMutate(t *testing.T) {
	path := writeProcessCauseEvent(t, "Process cause: none\n", "open")
	called := false
	runner := func(args []string) (string, string, bool) { called = true; return "", "", true }
	var out, errb bytes.Buffer
	if code := runIssueProcessCauseReconcileWith(&out, &errb, []string{"--event-file", path, "--dry-run", "--json"}, runner); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	if called {
		t.Fatal("dry-run invoked runner")
	}
	if !strings.Contains(out.String(), "process-cause:none") {
		t.Fatalf("dry-run omitted planned label: %s", out.String())
	}
}

func commandText(commands [][]string) string {
	var rows []string
	for _, command := range commands {
		rows = append(rows, strings.Join(command, " "))
	}
	return strings.Join(rows, "\n")
}
