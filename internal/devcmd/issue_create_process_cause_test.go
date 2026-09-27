package devcmd

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestIssueCreateRejectsMissingOrInvalidProcessCauseBeforeGH(t *testing.T) {
	validContractWithoutCause := validDispatchableIssueBody("compute", []string{"internal/compute/kernel.go"}, 2)
	for _, tc := range []struct {
		name string
		args []string
	}{
		{
			name: "normal-dry-run-missing",
			args: []string{"--title", "bug(compute): wrong token", "--body", validContractWithoutCause, "--dry-run"},
		},
		{
			name: "raw-body-missing",
			args: []string{"--title", "administrative issue", "--body", "plain body", "--raw-body"},
		},
		{
			name: "raw-body-invalid",
			args: []string{"--title", "administrative issue", "--body", "Process cause: flaky-ci", "--raw-body"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			called := false
			runner := func(args []string) (string, string, bool) {
				called = true
				return "https://example.test/issues/1", "", true
			}
			var stdout, stderr bytes.Buffer
			if code := runIssueCreateWithCleanScrubNoDefault(&stdout, &stderr, tc.args, runner); code == 0 {
				t.Fatalf("invalid process cause unexpectedly succeeded: stdout=%s", stdout.String())
			}
			if called {
				t.Fatal("gh runner was called before process-cause validation")
			}
			if got := strings.ToLower(stderr.String()); !strings.Contains(got, "process cause") {
				t.Fatalf("stderr does not identify process cause: %q", stderr.String())
			}
		})
	}
}

func TestIssueCreateDryRunAddsExactlyOneProcessCauseLabel(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runIssueCreateWithCleanScrubNoDefault(&stdout, &stderr, []string{
		"--title", "bug(runtime): wrong token",
		"--body", "Process cause: model-failure",
		"--raw-body", "--dry-run", "--json",
	}, func(args []string) (string, string, bool) {
		t.Fatalf("dry run invoked gh: %v", args)
		return "", "", false
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var result issueCreateResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("decode result: %v\n%s", err, stdout.String())
	}
	var processLabels []string
	for _, label := range result.Labels {
		if strings.HasPrefix(label, "process-cause:") {
			processLabels = append(processLabels, label)
		}
	}
	if len(processLabels) != 1 || processLabels[0] != "process-cause:model-failure" {
		t.Fatalf("process labels=%v all labels=%v, want exactly [process-cause:model-failure]", processLabels, result.Labels)
	}
}

func TestIssueCreateRejectsConflictingExplicitProcessCauseLabel(t *testing.T) {
	called := false
	runner := func(args []string) (string, string, bool) {
		called = true
		return "https://example.test/issues/1", "", true
	}
	var stdout, stderr bytes.Buffer
	code := runIssueCreateWithCleanScrubNoDefault(&stdout, &stderr, []string{
		"--title", "bug(runtime): wrong token",
		"--body", "Process cause: handoff-failure",
		"--raw-body",
		"--labels", "bug,process-cause:scoping-failure",
	}, runner)
	if code == 0 {
		t.Fatalf("conflicting process-cause label unexpectedly succeeded: stdout=%s", stdout.String())
	}
	if called {
		t.Fatal("gh runner was called before conflicting-label validation")
	}
	got := strings.ToLower(stderr.String())
	if !strings.Contains(got, "process-cause") || !strings.Contains(got, "conflict") {
		t.Fatalf("stderr does not explain conflicting process-cause label: %q", stderr.String())
	}
}

func TestIssueCreateRevalidatesProcessCauseAfterBodyAdditions(t *testing.T) {
	for _, tc := range []struct {
		name  string
		flags []string
	}{
		{name: "dry-run", flags: []string{"--dry-run"}},
		{name: "discoverability-bypass", flags: []string{"--no-audit-discoverability"}},
		{name: "non-dispatchable-bypass", flags: []string{"--allow-non-dispatchable"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			called := false
			args := []string{
				"--title", "administrative issue",
				"--body", "Process cause: none",
				"--raw-body",
				"--category", "custom\n\nProcess cause: unknown",
				"--layer", "leaf",
			}
			args = append(args, tc.flags...)
			var stdout, stderr bytes.Buffer
			code := runIssueCreateWithCleanScrubNoDefault(&stdout, &stderr, args, func([]string) (string, string, bool) {
				called = true
				return "https://example.test/issues/1", "", true
			})
			if code == 0 || called {
				t.Fatalf("post-addition duplicate reached mutation: code=%d called=%v stdout=%s", code, called, stdout.String())
			}
			got := strings.ToLower(stderr.String())
			if !strings.Contains(got, "invalid process cause") || !strings.Contains(got, "duplicate") {
				t.Fatalf("stderr does not explain final-body duplicate: %q", stderr.String())
			}
		})
	}
}
