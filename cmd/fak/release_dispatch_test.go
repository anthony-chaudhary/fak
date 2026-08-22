package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestReleaseDispatchPreviewRendersExactGuardedWorkflowInputs(t *testing.T) {
	old := releaseDispatchRun
	t.Cleanup(func() { releaseDispatchRun = old })
	releaseDispatchRun = func(string, ...string) (string, error) {
		t.Fatal("preview must not call gh")
		return "", nil
	}

	var out, errb bytes.Buffer
	if code := runReleaseDispatch(&out, &errb, []string{"--json", "--ref", "main"}); code != 0 {
		t.Fatalf("runReleaseDispatch code=%d stderr=%s", code, errb.String())
	}
	var got releaseDispatchResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v\n%s", err, out.String())
	}
	want := []string{"gh", "workflow", "run", "release-cadence.yml", "--ref", "main", "-f", "dry_run=false", "-f", "require_ci_green=true", "-f", "force=false"}
	if !reflect.DeepEqual(got.Command, want) {
		t.Fatalf("command=%q want=%q", got.Command, want)
	}
	if !got.OK || !got.DryRun || got.Dispatched || got.DecisionOnly || !got.RequireCIGreen || got.Force {
		t.Fatalf("unexpected preview result: %+v", got)
	}
}

func TestRunReleaseRoutesOnDemandAliases(t *testing.T) {
	old := releaseDispatchRun
	t.Cleanup(func() { releaseDispatchRun = old })
	releaseDispatchRun = func(string, ...string) (string, error) {
		t.Fatal("preview aliases must not call gh")
		return "", nil
	}
	for _, alias := range []string{"dispatch", "ondemand", "on-demand"} {
		t.Run(alias, func(t *testing.T) {
			var out, errb bytes.Buffer
			if code := runRelease(&out, &errb, []string{alias, "--json"}); code != 0 {
				t.Fatalf("code=%d stderr=%s", code, errb.String())
			}
			var got releaseDispatchResult
			if err := json.Unmarshal(out.Bytes(), &got); err != nil || !got.OK || !got.DryRun {
				t.Fatalf("result=%+v decode=%v", got, err)
			}
		})
	}
}

func TestReleaseDispatchExecuteReturnsCheckableActionsURL(t *testing.T) {
	old := releaseDispatchRun
	t.Cleanup(func() { releaseDispatchRun = old })
	var calls [][]string
	releaseDispatchRun = func(name string, args ...string) (string, error) {
		calls = append(calls, append([]string{name}, args...))
		if len(args) > 1 && args[0] == "repo" {
			return "https://github.com/anthony-chaudhary/fak\n", nil
		}
		if len(args) > 1 && args[0] == "run" {
			return "https://github.com/anthony-chaudhary/fak/actions/runs/123\n", nil
		}
		return "", nil
	}

	var out, errb bytes.Buffer
	code := runReleaseDispatch(&out, &errb, []string{"--execute", "--json", "--plan-only", "--force"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var got releaseDispatchResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !got.OK || got.DryRun || !got.Dispatched || !got.DecisionOnly || !got.Force {
		t.Fatalf("unexpected execute result: %+v", got)
	}
	if got.ActionsURL != "https://github.com/anthony-chaudhary/fak/actions/workflows/release-cadence.yml" {
		t.Fatalf("actions_url=%q", got.ActionsURL)
	}
	if got.RunURL != "https://github.com/anthony-chaudhary/fak/actions/runs/123" {
		t.Fatalf("run_url=%q", got.RunURL)
	}
	if !strings.Contains(got.NextAction, "decision and plan") {
		t.Fatalf("next_action=%q", got.NextAction)
	}
	if len(calls) != 3 || !containsString(calls[0], "dry_run=true") || !containsString(calls[0], "force=true") {
		t.Fatalf("calls=%q", calls)
	}
}

func TestReleaseDispatchFailsClosedWhenGHRefuses(t *testing.T) {
	old := releaseDispatchRun
	t.Cleanup(func() { releaseDispatchRun = old })
	releaseDispatchRun = func(string, ...string) (string, error) { return "HTTP 403", errors.New("exit status 1") }

	var out, errb bytes.Buffer
	code := runReleaseDispatch(&out, &errb, []string{"--execute", "--json"})
	if code != 1 || !strings.Contains(errb.String(), "HTTP 403") {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var got releaseDispatchResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.OK || got.Dispatched {
		t.Fatalf("refusal reported success: %+v", got)
	}
}

func TestReleaseDispatchWaitFollowsExactRunToSuccess(t *testing.T) {
	oldRun, oldNow, oldSleep := releaseDispatchRun, releaseDispatchNow, releaseDispatchSleep
	t.Cleanup(func() { releaseDispatchRun, releaseDispatchNow, releaseDispatchSleep = oldRun, oldNow, oldSleep })
	now := time.Unix(100, 0)
	releaseDispatchNow = func() time.Time { return now }
	releaseDispatchSleep = func(time.Duration) { now = now.Add(time.Second) }
	var calls [][]string
	views := 0
	releaseDispatchRun = func(name string, args ...string) (string, error) {
		calls = append(calls, append([]string{name}, args...))
		switch args[0] {
		case "api":
			return `{"workflow_run_id":42,"run_url":"https://api.github.test/runs/42","html_url":"https://github.test/actions/runs/42"}`, nil
		case "repo":
			return "https://github.test/acme/fak\n", nil
		case "run":
			views++
			if views == 1 {
				return `{"databaseId":42,"url":"https://github.test/actions/runs/42","status":"in_progress","conclusion":""}`, nil
			}
			return `{"databaseId":42,"url":"https://github.test/actions/runs/42","status":"completed","conclusion":"success"}`, nil
		default:
			t.Fatalf("unexpected call: %q", args)
			return "", nil
		}
	}

	var out, errb bytes.Buffer
	if code := runReleaseDispatch(&out, &errb, []string{"--execute", "--wait", "--timeout", "5s", "--json", "--plan-only"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var got releaseDispatchResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !got.OK || got.RunID != 42 || got.RunURL != "https://github.test/actions/runs/42" || got.Status != "completed" || got.Conclusion != "success" || got.Verdict != "passed" {
		t.Fatalf("result=%+v", got)
	}
	if len(calls) != 4 || !containsString(calls[0], "return_run_details=true") || !containsString(calls[2], "42") || !containsString(calls[3], "42") {
		t.Fatalf("calls=%q", calls)
	}
}

func TestReleaseDispatchWaitReportsTerminalFailure(t *testing.T) {
	old := releaseDispatchRun
	t.Cleanup(func() { releaseDispatchRun = old })
	releaseDispatchRun = func(_ string, args ...string) (string, error) {
		switch args[0] {
		case "api":
			return `{"workflow_run_id":77,"run_url":"api","html_url":"https://github.test/actions/runs/77"}`, nil
		case "repo":
			return "", errors.New("unavailable")
		case "run":
			return `{"databaseId":77,"url":"https://github.test/actions/runs/77","status":"completed","conclusion":"failure"}`, nil
		}
		return "", nil
	}

	var out, errb bytes.Buffer
	if code := runReleaseDispatch(&out, &errb, []string{"--execute", "--wait", "--json"}); code != 1 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var got releaseDispatchResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.OK || got.RunID != 77 || got.Status != "completed" || got.Conclusion != "failure" || got.Verdict != "failed" {
		t.Fatalf("result=%+v", got)
	}
}

func TestReleaseDispatchWaitTimesOut(t *testing.T) {
	oldRun, oldNow, oldSleep := releaseDispatchRun, releaseDispatchNow, releaseDispatchSleep
	t.Cleanup(func() { releaseDispatchRun, releaseDispatchNow, releaseDispatchSleep = oldRun, oldNow, oldSleep })
	now := time.Unix(100, 0)
	releaseDispatchNow = func() time.Time { return now }
	releaseDispatchSleep = func(time.Duration) { now = now.Add(2 * time.Second) }
	releaseDispatchRun = func(_ string, args ...string) (string, error) {
		switch args[0] {
		case "api":
			return `{"workflow_run_id":88,"run_url":"api","html_url":"https://github.test/actions/runs/88"}`, nil
		case "repo":
			return "", nil
		case "run":
			return `{"databaseId":88,"url":"https://github.test/actions/runs/88","status":"queued","conclusion":""}`, nil
		}
		return "", nil
	}

	var out, errb bytes.Buffer
	if code := runReleaseDispatch(&out, &errb, []string{"--execute", "--wait", "--timeout", "1s", "--json"}); code != 1 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var got releaseDispatchResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.OK || got.RunID != 88 || got.Status != "queued" || got.Verdict != "timed_out" {
		t.Fatalf("result=%+v", got)
	}
}

func TestReleaseDispatchWaitRejectsMismatchedRun(t *testing.T) {
	old := releaseDispatchRun
	t.Cleanup(func() { releaseDispatchRun = old })
	releaseDispatchRun = func(_ string, args ...string) (string, error) {
		switch args[0] {
		case "api":
			return `{"workflow_run_id":99,"run_url":"api","html_url":"https://github.test/actions/runs/99"}`, nil
		case "repo":
			return "", nil
		case "run":
			return `{"databaseId":100,"url":"https://github.test/actions/runs/100","status":"completed","conclusion":"success"}`, nil
		}
		return "", nil
	}

	var out, errb bytes.Buffer
	if code := runReleaseDispatch(&out, &errb, []string{"--execute", "--wait", "--json"}); code != 1 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var got releaseDispatchResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.OK || got.RunID != 99 || got.Verdict != "github_refused" {
		t.Fatalf("result=%+v", got)
	}
}

func TestReleaseDispatchWaitReportsCancellation(t *testing.T) {
	old := releaseDispatchRun
	t.Cleanup(func() { releaseDispatchRun = old })
	releaseDispatchRun = func(_ string, args ...string) (string, error) {
		switch args[0] {
		case "api":
			return `{"workflow_run_id":78,"run_url":"api","html_url":"https://github.test/actions/runs/78"}`, nil
		case "repo":
			return "", nil
		case "run":
			return `{"databaseId":78,"url":"https://github.test/actions/runs/78","status":"completed","conclusion":"cancelled"}`, nil
		}
		return "", nil
	}

	var out, errb bytes.Buffer
	if code := runReleaseDispatch(&out, &errb, []string{"--execute", "--wait", "--json"}); code != 1 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var got releaseDispatchResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.OK || got.Status != "completed" || got.Conclusion != "cancelled" || got.Verdict != "cancelled" {
		t.Fatalf("result=%+v", got)
	}
}
