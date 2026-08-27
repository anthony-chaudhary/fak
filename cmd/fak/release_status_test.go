package main

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestReleaseStatusCIDiagnosisUsesDecisionSelectedExactRun(t *testing.T) {
	oldJSON, oldText := releaseStatusRunGHJSON, releaseStatusRunGHText
	t.Cleanup(func() { releaseStatusRunGHJSON, releaseStatusRunGHText = oldJSON, oldText })
	calls := []string{}
	releaseStatusRunGHJSON = func(_ string, _ time.Duration, name string, args ...string) (any, error) {
		calls = append(calls, name+" "+strings.Join(args, " "))
		return map[string]any{"jobs": []any{map[string]any{
			"databaseId": float64(333), "name": "unit", "status": "completed", "conclusion": "failure",
			"steps": []any{map[string]any{"name": "Go test", "conclusion": "failure"}},
		}}}, nil
	}
	releaseStatusRunGHText = func(_ string, _ time.Duration, name string, args ...string) ([]byte, error) {
		calls = append(calls, name+" "+strings.Join(args, " "))
		return []byte("--- FAIL: TestAligned (0.00s)\nFAIL github.com/anthony-chaudhary/fak/internal/aligned 0.01s\n"), nil
	}
	decision := map[string]any{
		"ci_source": "fast", "blockers": []any{"CI_BASE_RED"},
		"ci_signal": map[string]any{
			"status": "red", "workflow": "custom-fast.yml",
			"latest_trunk_ci": map[string]any{"database_id": float64(222), "head_sha": "abc1234", "url": "https://example/run/222", "conclusion": "failure"},
		},
	}
	context := map[string]any{
		"ci_fast":    map[string]any{"workflow": "later-fast.yml", "latest_trunk_ci": map[string]any{"database_id": float64(999), "head_sha": "later", "conclusion": "failure"}},
		"ci_on_head": map[string]any{"workflow": "whole-custom.yml", "latest_trunk_ci": map[string]any{"database_id": float64(111), "conclusion": "failure"}},
	}
	got := releaseStatusCIDiagnosis("/repo", decision, context)
	if got["workflow"] != "custom-fast.yml" || releaseStatusInt(releaseStatusMap(got["run"])["database_id"]) != 222 {
		t.Fatalf("diagnosed wrong signal/run: %+v", got)
	}
	if got["snapshot_source"] != "decision" {
		t.Fatalf("snapshot source = %v, want decision", got["snapshot_source"])
	}
	for _, call := range calls {
		if strings.Contains(call, "run list") || strings.Contains(call, "ci.yml") || strings.Contains(call, "999") || !strings.Contains(call, "222") {
			t.Fatalf("diagnosis drifted from selected exact run: %q", call)
		}
	}
	summary := releaseStatusMap(got["summary"])
	if releaseStatusInt(summary["package_count"]) != 1 || releaseStatusInt(summary["test_count"]) != 1 {
		t.Fatalf("summary = %+v", summary)
	}
}

func TestReleaseStatusCIDiagnosisContextFallbackIsExplicit(t *testing.T) {
	workflow, run, source, snapshot := releaseStatusEffectiveCIRun(
		map[string]any{"ci_source": "whole"},
		map[string]any{"ci_on_head": map[string]any{"workflow": "legacy-ci.yml", "latest_trunk_ci": map[string]any{"database_id": float64(17)}}},
	)
	if workflow != "legacy-ci.yml" || releaseStatusInt(run["database_id"]) != 17 || source != "whole" || snapshot != "context_fallback" {
		t.Fatalf("fallback = workflow:%q run:%v source:%q snapshot:%q", workflow, run, source, snapshot)
	}
}

func TestReleaseStatusCIDiagnosisMissingRunIDNeverQueriesZero(t *testing.T) {
	oldJSON := releaseStatusRunGHJSON
	t.Cleanup(func() { releaseStatusRunGHJSON = oldJSON })
	called := false
	releaseStatusRunGHJSON = func(_ string, _ time.Duration, _ string, _ ...string) (any, error) {
		called = true
		return nil, fmt.Errorf("unexpected gh call")
	}
	decision := map[string]any{
		"ci_source": "fast", "blockers": []any{"CI_BASE_RED"},
		"ci_signal": map[string]any{"status": "red", "workflow": "ci-fast.yml", "latest_trunk_ci": map[string]any{"conclusion": "failure"}},
	}
	got := releaseStatusCIDiagnosis("/repo", decision, nil)
	if called || got["kind"] != "missing_run_id" || got["status"] == "not_failed" {
		t.Fatalf("missing-id diagnosis = %+v called=%v", got, called)
	}
}

func TestReleaseStatusCIDiagnosisTypesTimedOutAndStartupFailure(t *testing.T) {
	oldJSON, oldText := releaseStatusRunGHJSON, releaseStatusRunGHText
	t.Cleanup(func() { releaseStatusRunGHJSON, releaseStatusRunGHText = oldJSON, oldText })
	releaseStatusRunGHJSON = func(_ string, _ time.Duration, _ string, _ ...string) (any, error) {
		return map[string]any{"jobs": []any{}}, nil
	}
	releaseStatusRunGHText = func(_ string, _ time.Duration, _ string, _ ...string) ([]byte, error) {
		return nil, fmt.Errorf("unexpected log call")
	}
	for i, conclusion := range []string{"timed_out", "startup_failure"} {
		decision := map[string]any{
			"ci_source": "fast", "blockers": []any{"CI_BASE_RED"},
			"ci_signal": map[string]any{"status": "red", "workflow": "ci-fast.yml", "latest_trunk_ci": map[string]any{"database_id": float64(700 + i), "conclusion": conclusion}},
		}
		got := releaseStatusCIDiagnosis("/repo", decision, nil)
		if got["kind"] != conclusion || got["status"] == "not_failed" {
			t.Errorf("%s diagnosis = %+v", conclusion, got)
		}
	}
}

func TestReleaseStatusNextActionLeadsWithWorkCounts(t *testing.T) {
	decision := map[string]any{"decision": "hold", "blockers": []any{"CI_BASE_RED"}}
	diagnosis := map[string]any{"workflow": "custom-fast.yml", "summary": map[string]any{"package_count": 12, "test_count": 14}}
	got := releaseStatusNextAction(decision, map[string]any{}, map[string]any{"clean": true}, diagnosis, map[string]any{})
	wantPrefix := fmt.Sprintf("%d package(s), %d unique test(s)", 12, 14)
	if detail := releaseStatusString(got["detail"]); !strings.HasPrefix(detail, wantPrefix) {
		t.Fatalf("detail = %q, want prefix %q", detail, wantPrefix)
	}
	if detail := releaseStatusString(got["detail"]); !strings.Contains(detail, "fak release status --json") {
		t.Fatalf("human next action omits JSON work-unit command: %q", detail)
	}
}
