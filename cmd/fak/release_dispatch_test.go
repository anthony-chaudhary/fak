package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
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
