package main

import (
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

const releaseStatusCICapturedRunID int64 = 33047582124

func TestReleaseStatusCICapturedRunDecomposesIntoPackageWorkUnits(t *testing.T) {
	logText := releaseStatusCITestFixture(t)
	jobs := []releaseStatusCIJob{{
		Name:       "build · vet · test (no -race, fast release gate)",
		Conclusion: "failure",
		Steps: []releaseStatusCIStep{{
			Name:       "go test ./... (no -race)",
			Conclusion: "failure",
		}},
	}}
	lines := releaseStatusCIAttachFailedSteps(releaseStatusCIParseLog(logText), jobs)
	units, truncated := releaseStatusCIWorkUnits(lines)
	if truncated {
		t.Fatal("captured fixture unexpectedly exceeded diagnosis bounds")
	}
	wantUnitKeys := map[string]bool{}
	for _, line := range lines {
		if match := releaseStatusCIPkgFailRE.FindStringSubmatch(line.Message); len(match) == 2 && releaseStatusCIValidPackage(match[1]) {
			wantUnitKeys[line.Job+"\x00"+line.Step+"\x00"+match[1]] = true
		}
	}
	if len(units) != len(wantUnitKeys) {
		t.Fatalf("work units = %d, want one per unique failed job/step/package relation (%d)", len(units), len(wantUnitKeys))
	}
	failedTests := 0
	for _, unit := range units {
		failedTests += len(unit.Tests)
		if unit.Package == "" || unit.Target == "" {
			t.Fatalf("unit missing package/target: %#v", unit)
		}
		if unit.Step != "go test ./... (no -race)" {
			t.Fatalf("unit %s step = %q, want exact failed step", unit.Package, unit.Step)
		}
		if !strings.HasPrefix(unit.Reproduce, "fak test ") || !strings.Contains(unit.Reproduce, " -count=1") {
			t.Fatalf("unit %s reproduce command is not host-aware: %q", unit.Package, unit.Reproduce)
		}
	}
	if failedTests != 121 {
		t.Fatalf("named failing tests = %d, want 121 top-level tests", failedTests)
	}
	if len(units) < 10 {
		t.Fatalf("captured proof yielded only %d work units, want at least 10", len(units))
	}

	worker := releaseStatusCIFindUnit(units, releaseStatusCIModulePath+"/internal/workerworktree")
	if worker == nil {
		t.Fatal("captured run missing internal/workerworktree work unit")
	}
	if !containsString(worker.Tests, "TestLandReadbackVerifyRefusesWhenPathSweptByRace") {
		t.Fatalf("workerworktree tests missing exact named failure: %#v", worker.Tests)
	}
	if !strings.Contains(worker.Reproduce, "fak test ./internal/workerworktree") ||
		!strings.Contains(worker.Reproduce, "TestLandReadbackVerifyRefusesWhenPathSweptByRace") {
		t.Fatalf("workerworktree reproduce command = %q", worker.Reproduce)
	}
}

func TestReleaseStatusCISelectsDecisionWorkflowAndExactRun(t *testing.T) {
	t.Setenv("FAK_RELEASE_FAST_CI_WORKFLOW", "legacy-override.yml")
	contextPayload := map[string]any{
		"ci_fast": map[string]any{
			"workflow": "ci-fast.yml",
			"latest_trunk_ci": map[string]any{
				"database_id": releaseStatusCICapturedRunID,
				"conclusion":  "failure",
			},
		},
		"ci_on_head": map[string]any{
			"workflow": "ci.yml",
			"latest_trunk_ci": map[string]any{
				"database_id": int64(111),
				"conclusion":  "failure",
			},
		},
	}
	fast := releaseStatusSelectCITarget(map[string]any{"ci_source": "fast"}, contextPayload)
	if fast.Source != "fast" || fast.Workflow != "ci-fast.yml" || releaseStatusCIRunID(fast.Run) != releaseStatusCICapturedRunID {
		t.Fatalf("fast target = %#v", fast)
	}
	whole := releaseStatusSelectCITarget(map[string]any{"ci_source": "whole"}, contextPayload)
	if whole.Source != "whole" || whole.Workflow != "ci.yml" || releaseStatusCIRunID(whole.Run) != 111 {
		t.Fatalf("whole target = %#v", whole)
	}
	ancestor := releaseStatusSelectCITarget(map[string]any{"ci_source": "fast+ancestor"}, contextPayload)
	if ancestor.Source != "fast" || ancestor.Workflow != "ci-fast.yml" || releaseStatusCIRunID(ancestor.Run) != releaseStatusCICapturedRunID {
		t.Fatalf("fast+ancestor target = %#v", ancestor)
	}
	unknown := releaseStatusSelectCITarget(map[string]any{"ci_source": "mystery"}, contextPayload)
	if unknown.Source != "mystery" || unknown.Workflow != "" || len(unknown.Run) != 0 {
		t.Fatalf("unknown source must fail closed, got %#v", unknown)
	}
	fallback := releaseStatusSelectCITarget(map[string]any{"ci_source": "fast"}, map[string]any{
		"ci_fast": map[string]any{"latest_trunk_ci": map[string]any{"database_id": int64(222)}},
	})
	if fallback.Workflow != "legacy-override.yml" {
		t.Fatalf("shared release workflow override did not reach status fallback: %#v", fallback)
	}
}

func TestReleaseStatusCIDiagnosisReadsOnlyDecisionSelectedRun(t *testing.T) {
	oldJSON := releaseStatusRunGHJSON
	oldText := releaseStatusRunGHText
	t.Cleanup(func() {
		releaseStatusRunGHJSON = oldJSON
		releaseStatusRunGHText = oldText
	})

	var jsonCalls [][]string
	releaseStatusRunGHJSON = func(_ string, _ time.Duration, name string, args ...string) (any, error) {
		call := append([]string{name}, args...)
		jsonCalls = append(jsonCalls, call)
		if strings.Join(call, " ") != "gh run view 33047582124 --json databaseId,name,workflowName,displayTitle,headBranch,headSha,status,conclusion,url,event,createdAt,updatedAt,jobs" {
			t.Fatalf("unexpected gh JSON call: %v", call)
		}
		return releaseStatusCIRunViewFixture(), nil
	}
	var textCalls [][]string
	releaseStatusRunGHText = func(_ string, _ time.Duration, name string, args ...string) (string, bool, error) {
		call := append([]string{name}, args...)
		textCalls = append(textCalls, call)
		if strings.Join(call, " ") != "gh run view 33047582124 --log-failed" {
			t.Fatalf("unexpected gh text call: %v", call)
		}
		return releaseStatusCITestFixture(t), false, nil
	}

	decision := map[string]any{
		"blockers":  []any{"CI_BASE_RED"},
		"ci_source": "fast",
	}
	contextPayload := map[string]any{
		"ci_fast": map[string]any{
			"workflow": "ci-fast.yml",
			"latest_trunk_ci": map[string]any{
				"database_id": releaseStatusCICapturedRunID,
				"conclusion":  "failure",
			},
		},
		"ci_on_head": map[string]any{
			"workflow": "ci.yml",
			"latest_trunk_ci": map[string]any{
				"database_id": int64(111),
				"conclusion":  "failure",
			},
		},
	}
	diagnosis := releaseStatusCIDiagnosis(t.TempDir(), decision, contextPayload)
	if diagnosis["workflow"] != "ci-fast.yml" || releaseStatusInt64(releaseStatusMap(diagnosis["run"])["database_id"]) != releaseStatusCICapturedRunID {
		t.Fatalf("diagnosis selected wrong workflow/run: %#v", diagnosis)
	}
	if diagnosis["work_unit_count"] != 35 || diagnosis["failed_test_count"] != 121 {
		t.Fatalf("diagnosis counts = units %v tests %v", diagnosis["work_unit_count"], diagnosis["failed_test_count"])
	}
	if len(jsonCalls) != 1 || len(textCalls) != 1 {
		t.Fatalf("gh calls = json %d text %d, want one metadata + one failed-log read", len(jsonCalls), len(textCalls))
	}
}

func TestReleaseStatusCISkipsGitHubReadWithoutRedBlocker(t *testing.T) {
	oldJSON := releaseStatusRunGHJSON
	oldText := releaseStatusRunGHText
	t.Cleanup(func() {
		releaseStatusRunGHJSON = oldJSON
		releaseStatusRunGHText = oldText
	})
	releaseStatusRunGHJSON = func(string, time.Duration, string, ...string) (any, error) {
		t.Fatal("green/non-red status must not query gh JSON")
		return nil, nil
	}
	releaseStatusRunGHText = func(string, time.Duration, string, ...string) (string, bool, error) {
		t.Fatal("green/non-red status must not read gh logs")
		return "", false, nil
	}

	got := releaseStatusCIDiagnosis(t.TempDir(), map[string]any{"blockers": []any{}}, nil)
	if got["status"] != releaseStatusCIStatusNotRequired {
		t.Fatalf("status = %v, want %s", got["status"], releaseStatusCIStatusNotRequired)
	}
}

func TestReleaseStatusCIMissingExactRunRefusesFallbackSelection(t *testing.T) {
	oldJSON := releaseStatusRunGHJSON
	oldText := releaseStatusRunGHText
	t.Cleanup(func() {
		releaseStatusRunGHJSON = oldJSON
		releaseStatusRunGHText = oldText
	})
	releaseStatusRunGHJSON = func(string, time.Duration, string, ...string) (any, error) {
		t.Fatal("missing exact run must not query a newer workflow run")
		return nil, nil
	}
	releaseStatusRunGHText = func(string, time.Duration, string, ...string) (string, bool, error) {
		t.Fatal("missing exact run must not read arbitrary logs")
		return "", false, nil
	}

	got := releaseStatusCIDiagnosis(t.TempDir(), map[string]any{
		"blockers":  []any{"CI_BASE_RED"},
		"ci_source": "fast",
	}, map[string]any{
		"ci_fast": map[string]any{
			"workflow":        "ci-fast.yml",
			"latest_trunk_ci": map[string]any{"conclusion": "failure"},
		},
	})
	if got["status"] != "unavailable" || !strings.Contains(releaseStatusString(got["reason"]), "exact run") {
		t.Fatalf("diagnosis = %#v, want exact-run unavailable refusal", got)
	}
}

func TestReleaseStatusCIBillingTextSurvivesMissingJobsAndLogError(t *testing.T) {
	oldJSON := releaseStatusRunGHJSON
	oldText := releaseStatusRunGHText
	t.Cleanup(func() {
		releaseStatusRunGHJSON = oldJSON
		releaseStatusRunGHText = oldText
	})
	releaseStatusRunGHJSON = func(string, time.Duration, string, ...string) (any, error) {
		return map[string]any{
			"databaseId": releaseStatusCICapturedRunID,
			"conclusion": "startup_failure",
			"jobs":       []any{},
		}, nil
	}
	releaseStatusRunGHText = func(_ string, _ time.Duration, _ string, args ...string) (string, bool, error) {
		if containsString(args, "--log-failed") {
			return "job was not started because recent account payments have failed", false, errors.New("no job log")
		}
		return "", false, nil
	}

	got := releaseStatusCIDiagnosis(t.TempDir(), map[string]any{
		"blockers":  []any{"CI_BASE_RED"},
		"ci_source": "fast",
	}, map[string]any{
		"ci_fast": map[string]any{
			"workflow": "ci-fast.yml",
			"latest_trunk_ci": map[string]any{
				"database_id": releaseStatusCICapturedRunID,
				"conclusion":  "startup_failure",
			},
		},
	})
	if got["action"] != "fix_ci_billing" || !releaseStatusCIHasCause(releaseStatusAnySlice(got["causes"]), "workflow_admission_billing") {
		t.Fatalf("billing diagnosis = %#v", got)
	}
}

func TestReleaseStatusCIClassifiesRequiredFailureFamilies(t *testing.T) {
	cases := map[string]struct {
		step string
		log  string
	}{
		"workflow_admission_billing": {log: "job was not started because recent account payments have failed"},
		"checkout":                   {step: "Run actions/checkout@v4"},
		"toolchain_setup":            {step: "Run actions/setup-go@v5"},
		"artifact_transfer":          {step: "cache Go build/test artifacts"},
		"compile_build":              {step: "build"},
		"formatting":                 {step: "gofmt check"},
		"vet":                        {step: "vet"},
		"architecture_boundary":      {step: "architest tier gate"},
		"go_package_tests":           {step: "go test ./... (no -race)"},
		"race_test_timeout":          {step: "go test -race ./...", log: "panic: test timed out after 10m0s"},
		"claims":                     {step: "claims lint"},
		"repository_hygiene":         {step: "repository hygiene"},
		"leakage_security":           {step: "secret leakage gate"},
		"generated_artifact_drift":   {log: "docs block drifted; run fak workflow-audit --write-doc"},
		"commit_provenance":          {log: "commit witness is unwitnessed; run dos verify"},
		"unknown":                    {step: "opaque custom gate", log: "exit 7"},
	}
	for want, tc := range cases {
		t.Run(want, func(t *testing.T) {
			kinds := releaseStatusCICauseKinds("job", tc.step, tc.log)
			if !containsString(kinds, want) {
				t.Fatalf("kinds = %#v, want %q", kinds, want)
			}
		})
	}
}

func TestReleaseStatusCIKeepsRepeatedPackageStepsSeparateAndRaceAware(t *testing.T) {
	logText := strings.Join([]string{
		"job\tgo test ./... (no -race)\t2026-08-27T01:00:00Z --- FAIL: TestFast (0.00s)",
		"job\tgo test ./... (no -race)\t2026-08-27T01:00:01Z FAIL\tgithub.com/anthony-chaudhary/fak/internal/example\t0.01s",
		"job\tgo test -race ./...\t2026-08-27T01:00:02Z --- FAIL: TestRace (0.00s)",
		"job\tgo test -race ./...\t2026-08-27T01:00:03Z FAIL\tgithub.com/anthony-chaudhary/fak/internal/example\t0.01s",
	}, "\n")
	units, truncated := releaseStatusCIWorkUnits(releaseStatusCIParseLog(logText))
	if truncated || len(units) != 2 {
		t.Fatalf("units=%#v truncated=%v, want two separate units", units, truncated)
	}
	var fast, race *releaseStatusCIWorkUnit
	for i := range units {
		if units[i].Race {
			race = &units[i]
		} else {
			fast = &units[i]
		}
	}
	if fast == nil || race == nil {
		t.Fatalf("missing fast/race split: %#v", units)
	}
	if containsString(fast.Argv, "-race") || !containsString(race.Argv, "-race") {
		t.Fatalf("race argv not mode-aware: fast=%#v race=%#v", fast.Argv, race.Argv)
	}
	if !reflect.DeepEqual(fast.Tests, []string{"TestFast"}) || !reflect.DeepEqual(race.Tests, []string{"TestRace"}) {
		t.Fatalf("tests merged across steps: fast=%#v race=%#v", fast.Tests, race.Tests)
	}
}

func TestReleaseStatusCIRejectsCommandInjectionTokens(t *testing.T) {
	logText := strings.Join([]string{
		"job\tstep\t2026-08-27T01:00:00Z --- FAIL: TestSafe;RemoveItem (0.00s)",
		"job\tstep\t2026-08-27T01:00:01Z FAIL\tgithub.com/anthony-chaudhary/fak/internal/example;Remove-Item\t0.01s",
	}, "\n")
	units, _ := releaseStatusCIWorkUnits(releaseStatusCIParseLog(logText))
	if len(units) != 0 {
		t.Fatalf("forged package/test tokens produced executable work units: %#v", units)
	}

	command, argv := releaseStatusCIReproduceCommand("./internal/example", []string{"TestSafe"}, false)
	if command != releaseStatusCICommandText(argv) || strings.Contains(strings.Join(argv, "\x00"), "Remove") || !containsString(argv, "./internal/example") || !containsString(argv, "^(?:TestSafe)$") {
		t.Fatalf("structured argv is not a safe render of the admitted package/test relation: command=%q argv=%#v", command, argv)
	}
	if command != "fak test ./internal/example -- -run '^(?:TestSafe)$' -count=1" {
		t.Fatalf("safe rendered command = %q", command)
	}
}

func TestReleaseStatusCIParserStripsANSIAndCRLF(t *testing.T) {
	logText := "job\tstep\t2026-08-27T01:00:00Z \x1b[31m--- FAIL: TestSafe (0.00s)\x1b[0m\r\n" +
		"job\tstep\t2026-08-27T01:00:01Z FAIL\tgithub.com/anthony-chaudhary/fak/internal/example\t0.01s\r\n"
	units, truncated := releaseStatusCIWorkUnits(releaseStatusCIParseLog(logText))
	if truncated || len(units) != 1 || !reflect.DeepEqual(units[0].Tests, []string{"TestSafe"}) {
		t.Fatalf("ANSI/CRLF parse = units %#v truncated=%v", units, truncated)
	}
}

func TestReleaseStatusCILimitedBufferCapsWhileDraining(t *testing.T) {
	var buf releaseStatusCILimitedBuffer
	buf.limit = 5
	if n, err := buf.Write([]byte("abcdefghij")); err != nil || n != 10 {
		t.Fatalf("Write = %d, %v", n, err)
	}
	got, truncated := buf.Result()
	if got != "abcde" || !truncated {
		t.Fatalf("limited result = %q truncated=%v", got, truncated)
	}
}

func TestReleaseStatusCINonGoFailureKeepsTypedInspectionFallback(t *testing.T) {
	target := releaseStatusCITarget{
		Source:   "fast",
		Workflow: "ci-fast.yml",
		Run:      map[string]any{"database_id": int64(77), "conclusion": "failure"},
	}
	view := map[string]any{
		"databaseId":   int64(77),
		"workflowName": "ci-fast",
		"conclusion":   "failure",
		"jobs": []any{
			map[string]any{
				"name":       "checkout",
				"conclusion": "failure",
				"steps": []any{
					map[string]any{"name": "Run actions/checkout@v4", "conclusion": "failure"},
				},
			},
		},
	}
	diagnosis := releaseStatusBuildCIDiagnosis(target, view, "", "")
	if diagnosis["work_unit_count"] != 0 {
		t.Fatalf("work units = %v, want zero", diagnosis["work_unit_count"])
	}
	cause := releaseStatusCIFirstCause(diagnosis)
	if cause["kind"] != "checkout" || cause["inspect_command"] != "gh run view 77 --log-failed" {
		t.Fatalf("cause = %#v, want checkout + safe inspection command", cause)
	}
}

func TestReleaseStatusCIJSONKeepsCompatibilityFields(t *testing.T) {
	target := releaseStatusCITarget{
		Source:   "fast",
		Workflow: "ci-fast.yml",
		Run:      map[string]any{"database_id": releaseStatusCICapturedRunID, "conclusion": "failure"},
	}
	diagnosis := releaseStatusBuildCIDiagnosis(target, releaseStatusCIRunViewFixture(), releaseStatusCITestFixture(t), "")
	encoded, err := json.Marshal(diagnosis)
	if err != nil {
		t.Fatalf("marshal diagnosis: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(encoded, &got); err != nil {
		t.Fatalf("unmarshal diagnosis: %v", err)
	}
	for _, field := range []string{"status", "scope", "kind", "action", "detail", "reason", "run", "causes", "work_units"} {
		if _, ok := got[field]; !ok {
			t.Fatalf("compatibility/additive field %q missing: %s", field, encoded)
		}
	}
	run := releaseStatusMap(got["run"])
	for _, field := range []string{"databaseId", "headSha", "displayTitle", "database_id", "head_sha", "display_title"} {
		if _, ok := run[field]; !ok {
			t.Fatalf("run compatibility field %q missing: %#v", field, run)
		}
	}
	unit := releaseStatusCIFirstWorkUnit(got)
	if len(releaseStatusStringSlice(unit["reproduce_argv"])) == 0 {
		t.Fatalf("work unit lacks structured reproduce argv: %#v", unit)
	}
}

func TestReleaseStatusCIRenderAndNextActionLeadWithWorkUnit(t *testing.T) {
	target := releaseStatusCITarget{
		Source:   "fast",
		Workflow: "ci-fast.yml",
		Run:      map[string]any{"database_id": releaseStatusCICapturedRunID, "conclusion": "failure"},
	}
	diagnosis := releaseStatusBuildCIDiagnosis(target, releaseStatusCIRunViewFixture(), releaseStatusCITestFixture(t), "")
	action := releaseStatusNextAction(
		map[string]any{"decision": "hold", "blockers": []any{"CI_BASE_RED"}},
		map[string]any{},
		map[string]any{"clean": true},
		diagnosis,
		map[string]any{},
	)
	if action["kind"] != "fix_ci" || !strings.Contains(releaseStatusString(action["detail"]), "fak test ./") {
		t.Fatalf("next action = %#v, want first host-aware reproduce command", action)
	}

	status := map[string]any{
		"rolling": map[string]any{
			"last_tag":          "v1.2.3",
			"commits_since_tag": 9,
			"decision":          map[string]any{"decision": "hold", "reason": "CI red"},
			"ci_diagnosis":      diagnosis,
		},
		"stable":         map[string]any{"latest_stable": nil},
		"next_action":    action,
		"branch_regime":  map[string]any{},
		"shadow_cutover": map[string]any{},
	}
	rendered := renderReleaseStatus(status)
	diagnosisAt := strings.Index(rendered, "ci diagnosis:")
	lastTagAt := strings.Index(rendered, "last tag:")
	nextActionAt := strings.Index(rendered, "next action:")
	if diagnosisAt < 0 || lastTagAt < 0 || nextActionAt < 0 || diagnosisAt > lastTagAt || diagnosisAt > nextActionAt {
		t.Fatalf("actionable diagnosis did not lead the human surface:\n%s", rendered)
	}
	if !strings.Contains(rendered, "work unit: ./") || !strings.Contains(rendered, "use --json for the complete diagnosis") {
		t.Fatalf("rendered diagnosis missing bounded work-unit cards:\n%s", rendered)
	}
	if !strings.Contains(rendered, "cause: commit_provenance") {
		t.Fatalf("mixed non-Go cause was hidden by work-unit rendering:\n%s", rendered)
	}
}

func releaseStatusCITestFixture(t *testing.T) string {
	t.Helper()
	data, err := os.ReadFile("testdata/release-status-ci-run-33047582124.txt")
	if err != nil {
		t.Fatalf("read captured CI fixture: %v", err)
	}
	return string(data)
}

func releaseStatusCIRunViewFixture() map[string]any {
	return map[string]any{
		"databaseId":   releaseStatusCICapturedRunID,
		"name":         "ci-fast",
		"workflowName": "ci-fast",
		"headSha":      "c222c119b4f12d780e61030cefabadf508eaa265",
		"status":       "completed",
		"conclusion":   "failure",
		"url":          "https://github.com/anthony-chaudhary/fak/actions/runs/33047582124",
		"jobs": []any{
			map[string]any{
				"name":       "build · vet · test (no -race, fast release gate)",
				"conclusion": "failure",
				"steps": []any{
					map[string]any{"name": "go test ./... (no -race)", "conclusion": "failure"},
				},
			},
		},
	}
}

func releaseStatusCIFindUnit(units []releaseStatusCIWorkUnit, pkg string) *releaseStatusCIWorkUnit {
	for i := range units {
		if units[i].Package == pkg {
			return &units[i]
		}
	}
	return nil
}
