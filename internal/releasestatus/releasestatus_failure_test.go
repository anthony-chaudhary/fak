package releasestatus

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestReleaseStatusFailurePathsAndRefusals exercises the failure paths, empty inputs,
// corrupted log handling, and structured refusal channels for Actionable CI base-red diagnosis.
// It verifies that all returned errors and refusal messages are non-nil, distinct, and contain
// actionable recovery hints (e.g. naming the expected format or required path).
func TestReleaseStatusFailurePathsAndRefusals(t *testing.T) {
	t.Run("EmptyLogInput", func(t *testing.T) {
		// 1. Empty string log produces 0 work units without panic or allocation leaks.
		emptyUnits := ParseGoTestWorkUnits("")
		if len(emptyUnits) != 0 {
			t.Fatalf("ParseGoTestWorkUnits(\"\") returned %d units, want 0", len(emptyUnits))
		}

		// Whitespace-only log also yields 0 work units.
		wsUnits := ParseGoTestWorkUnits("   \n\t  \n  ")
		if len(wsUnits) != 0 {
			t.Fatalf("ParseGoTestWorkUnits(whitespace) returned %d units, want 0", len(wsUnits))
		}

		// DiagnoseCIFailure with empty log and no steps produces undifferentiated/inspect_ci status.
		run := CIRun{DatabaseID: 100, Conclusion: "failure"}
		diag := DiagnoseCIFailure("ci-fast.yml", run, nil, "")
		if diag.Status != "undifferentiated" || diag.Kind != "unknown" || diag.Action != "inspect_ci" {
			t.Fatalf("empty log diagnosis = %+v, want undifferentiated/unknown/inspect_ci", diag)
		}
		if diag.Primary != nil {
			t.Fatalf("expected nil Primary for empty log, got %+v", diag.Primary)
		}
		if len(diag.WorkUnits) != 0 {
			t.Fatalf("expected 0 work units for empty log, got %d", len(diag.WorkUnits))
		}
		wantDetail := "ci-fast.yml is red, but no named failed step or Go test failure was available"
		if diag.Detail != wantDetail {
			t.Fatalf("empty log detail = %q, want %q", diag.Detail, wantDetail)
		}

		// Folding the empty log diagnosis with CI_BASE_RED emits the actionable recovery next action.
		facts := Facts{
			Decision:    Decision{Decision: "hold", Blockers: []string{"CI_BASE_RED"}},
			CIDiagnosis: diag,
		}
		status := Fold(facts)
		if status.NextAction.Kind != "fix_ci" {
			t.Fatalf("status next_action.kind = %q, want fix_ci", status.NextAction.Kind)
		}
		if !strings.Contains(status.NextAction.Detail, "fix current main ci.yml failure before cutting a release") {
			t.Fatalf("status next_action.detail missing recovery hint: %q", status.NextAction.Detail)
		}
		if status.OK || status.Verdict != "ACTION" {
			t.Fatalf("status ok=%v verdict=%q, want false/ACTION", status.OK, status.Verdict)
		}
	})

	t.Run("NonExistentFilePath", func(t *testing.T) {
		// 2. Non-existent file path read error when attempting to load CI log.
		nonExistentLog := filepath.Join("testdata", "nonexistent-ci-failure-run.log")
		_, err := os.ReadFile(nonExistentLog)
		if err == nil {
			t.Fatal("expected error reading non-existent CI log file")
		}
		if !errors.Is(err, os.ErrNotExist) && !os.IsNotExist(err) {
			t.Fatalf("expected not-exist error, got %v", err)
		}
		if !strings.Contains(err.Error(), "nonexistent-ci-failure-run.log") {
			t.Fatalf("error missing path hint %q: %v", nonExistentLog, err)
		}

		// Non-existent committed evidence file path is detected with required path hint.
		stableTag := StableTag{
			Tag:           "stable/2026-06-stable",
			SHA:           "deadbeef1234",
			EvidenceFound: false,
		}
		ev := foldStableEvidence([]StableTag{stableTag})
		if ev.OK {
			t.Fatal("foldStableEvidence should report OK=false when evidence file is absent")
		}
		if len(ev.Failures) == 0 {
			t.Fatal("expected at least one evidence failure for missing file")
		}
		wantEvidencePath := "docs/stable-releases/2026-06-stable.md"
		if ev.Failures[0].EvidencePath != wantEvidencePath {
			t.Fatalf("evidence path = %q, want %q", ev.Failures[0].EvidencePath, wantEvidencePath)
		}
		wantDetail := "missing committed evidence file " + wantEvidencePath
		if ev.Failures[0].Detail != wantDetail {
			t.Fatalf("failure detail = %q, want %q", ev.Failures[0].Detail, wantDetail)
		}

		// Fold emits repair action naming the missing path.
		status := Fold(Facts{
			Decision: Decision{Decision: "hold"},
			Stable:   []StableTag{stableTag},
		})
		if status.NextAction.Kind != "repair_stable_evidence" {
			t.Fatalf("next_action.kind = %q, want repair_stable_evidence", status.NextAction.Kind)
		}
		if !strings.Contains(status.NextAction.Detail, wantEvidencePath) {
			t.Fatalf("repair detail missing path: %q", status.NextAction.Detail)
		}

		// Missing cadence workflow file path.
		cadence := foldCadence(CadenceText{Present: false, Path: ".github/workflows/release-cadence.yml"})
		if cadence.Present {
			t.Fatal("cadence Present should be false for absent workflow")
		}
		if cadence.Path != ".github/workflows/release-cadence.yml" {
			t.Fatalf("cadence path = %q, want .github/workflows/release-cadence.yml", cadence.Path)
		}
	})

	t.Run("CorruptedOrNoTestFailures", func(t *testing.T) {
		// 3. Log with corrupted / unparseable lines or hostile paths.
		malformedLogs := []struct {
			name string
			log  string
		}{
			{name: "incomplete_test_marker", log: "--- FAIL:\nFAIL\n"},
			{name: "test_without_package", log: "--- FAIL: TestDangling (0.01s)\n"},
			{name: "path_traversal", log: "--- FAIL: TestInjected (0.00s)\nFAIL ../../etc/shadow 0.01s\n"},
			{name: "prefix_traversal", log: "--- FAIL: TestInjected (0.00s)\nFAIL github.com/anthony-chaudhary/fak/../../etc/passwd 0.01s\n"},
			{name: "backslash_path", log: "--- FAIL: TestBackslash (0.00s)\nFAIL github.com/anthony-chaudhary/fak\\internal\\foo 0.01s\n"},
			{name: "external_module", log: "--- FAIL: TestExternal (0.00s)\nFAIL example.com/other/repo/pkg 0.01s\n"},
			{name: "binary_noise", log: "\x00\x01\xfe\xff\x80\nCompiler Warning: unexpected token\n"},
		}
		for _, tc := range malformedLogs {
			units := ParseGoTestWorkUnits(tc.log)
			if len(units) != 0 {
				t.Fatalf("case %s returned %d work units, want 0: %+v", tc.name, len(units), units)
			}
		}

		// Log with NO test failures (all passed).
		passingLog := "=== RUN TestOk\n--- PASS: TestOk (0.01s)\nPASS\nok github.com/anthony-chaudhary/fak/cmd/fak 0.02s\n"
		if units := ParseGoTestWorkUnits(passingLog); len(units) != 0 {
			t.Fatalf("passing log returned %d work units, want 0", len(units))
		}
		passDiag := DiagnoseCIFailure("ci.yml", CIRun{Conclusion: "failure"}, nil, passingLog)
		if passDiag.Status != "undifferentiated" || passDiag.Kind != "unknown" || passDiag.Action != "inspect_ci" {
			t.Fatalf("passing log red run diagnosis = %+v, want undifferentiated/inspect_ci", passDiag)
		}

		// Corrupted or unrecognized step names stay safely unknown with inspect_ci.
		corruptedStep := "mystery_unrecognized_custom_step_#!"
		kind, stage, action, repro := ClassifyCIFailedStep(corruptedStep)
		if kind != "unknown" || stage != "unknown" || action != "inspect_ci" || repro != nil {
			t.Fatalf("ClassifyCIFailedStep(%q) = (%q,%q,%q,%v), want unknown/unknown/inspect_ci/nil", corruptedStep, kind, stage, action, repro)
		}
		stepDiag := DiagnoseCIFailure("ci.yml", CIRun{Conclusion: "failure"}, []string{corruptedStep}, "")
		if stepDiag.Status != "diagnosed" || stepDiag.Kind != "unknown" || stepDiag.Action != "inspect_ci" {
			t.Fatalf("step diagnosis = %+v, want diagnosed/unknown/inspect_ci", stepDiag)
		}
		if !strings.Contains(stepDiag.Detail, corruptedStep) {
			t.Fatalf("step detail missing step name: %q", stepDiag.Detail)
		}
	})

	t.Run("ActionableRecoveryHintsAndDistinctErrors", func(t *testing.T) {
		// 4. Test that all returned errors/refusals across failure paths are non-nil,
		// distinct, and contain actionable recovery hints (naming format, required path, or recovery action).

		// Read error from non-existent file:
		_, readErr := os.ReadFile(filepath.Join("testdata", "nonexistent-error-witness.log"))
		if readErr == nil {
			t.Fatal("expected read error for non-existent file")
		}

		cases := []struct {
			name         string
			errString    string
			recoveryHint string
			pathOrFormat string
		}{
			{
				name:         "non_existent_file_io",
				errString:    readErr.Error(),
				recoveryHint: "nonexistent-error-witness.log",
				pathOrFormat: "testdata",
			},
			{
				name: "missing_committed_evidence_file",
				errString: foldStableEvidence([]StableTag{{
					Tag:           "stable/2026-06-stable",
					EvidenceFound: false,
				}}).Failures[0].Detail,
				recoveryHint: "missing committed evidence file",
				pathOrFormat: "docs/stable-releases/2026-06-stable.md",
			},
			{
				name: "missing_or_unparseable_frontmatter",
				errString: foldStableEvidence([]StableTag{{
					Tag:           "stable/2026-06-stable",
					EvidenceFound: true,
					Evidence:      map[string]string{},
				}}).Failures[0].Detail,
				recoveryHint: "missing or unparseable frontmatter",
				pathOrFormat: "frontmatter",
			},
			{
				name: "frontmatter_candidate_sha_missing",
				errString: foldStableEvidence([]StableTag{{
					Tag:           "stable/2026-06-stable",
					EvidenceFound: true,
					Evidence:      map[string]string{"underlying_version": "v1.0.0"},
				}}).Failures[0].Detail,
				recoveryHint: "missing",
				pathOrFormat: "candidate_sha",
			},
			{
				name: "frontmatter_candidate_sha_mismatch",
				errString: foldStableEvidence([]StableTag{{
					Tag:           "stable/2026-06-stable",
					SHA:           "deadbeef1234",
					EvidenceFound: true,
					Evidence:      map[string]string{"candidate_sha": "wrongsha", "underlying_version": "v1.0.0"},
				}}).Failures[0].Detail,
				recoveryHint: "does not match",
				pathOrFormat: "deadbeef1234",
			},
			{
				name: "frontmatter_underlying_version_not_semver",
				errString: foldStableEvidence([]StableTag{{
					Tag:           "stable/2026-06-stable",
					SHA:           "deadbeef1234",
					EvidenceFound: true,
					Evidence:      map[string]string{"candidate_sha": "deadbeef1234", "underlying_version": "1.0-invalid"},
				}}).Failures[0].Detail,
				recoveryHint: "is not vX.Y.Z",
				pathOrFormat: "vX.Y.Z",
			},
			{
				name: "underlying_rolling_tag_not_exist",
				errString: foldStableEvidence([]StableTag{{
					Tag:              "stable/2026-06-stable",
					SHA:              "deadbeef1234",
					EvidenceFound:    true,
					Evidence:         map[string]string{"candidate_sha": "deadbeef1234", "underlying_version": "v1.0.0"},
					UnderlyingExists: false,
				}}).Failures[0].Detail,
				recoveryHint: "does not exist",
				pathOrFormat: "v1.0.0",
			},
			{
				name: "frontmatter_codename_mismatch",
				errString: foldStableEvidence([]StableTag{{
					Tag:              "stable/2026-06-stable",
					SHA:              "deadbeef1234",
					EvidenceFound:    true,
					Evidence:         map[string]string{"candidate_sha": "deadbeef1234", "underlying_version": "v1.0.0", "codename": "wrong-codename"},
					UnderlyingExists: true,
					UnderlyingTagSHA: "deadbeef1234",
				}}).Failures[0].Detail,
				recoveryHint: "does not match",
				pathOrFormat: "2026-06-stable",
			},
			{
				name: "unreadable_stable_version",
				errString: foldStable(Facts{Stable: []StableTag{{
					Tag:     "stable/2026-06-stable",
					Version: "invalid-semver-token",
				}}}).Recommendation,
				recoveryHint: "inspect before promoting another stable",
				pathOrFormat: "VERSION",
			},
			{
				name:         "undifferentiated_ci_detail",
				errString:    DiagnoseCIFailure("ci-fast.yml", CIRun{Conclusion: "failure"}, nil, "").Detail,
				recoveryHint: "no named failed step or Go test failure was available",
				pathOrFormat: "ci-fast.yml",
			},
			{
				name: "workflow_unparseable_recovery",
				errString: foldNextAction(
					Decision{Decision: "hold", Blockers: []string{"WORKFLOW_UNPARSEABLE"}},
					StableSummary{Evidence: StableEvidence{OK: true}},
					DirtySummary{Clean: true},
					CIDiagnosis{},
					BranchRegime{},
				).Detail,
				recoveryHint: "repair",
				pathOrFormat: "YAML",
			},
			{
				name: "version_drift_recovery",
				errString: foldNextAction(
					Decision{Decision: "hold", Blockers: []string{"VERSION_DRIFT"}},
					StableSummary{Evidence: StableEvidence{OK: true}},
					DirtySummary{Clean: true},
					CIDiagnosis{},
					BranchRegime{},
				).Detail,
				recoveryHint: "reconcile",
				pathOrFormat: "VERSION",
			},
			{
				name: "dirty_worktree_recovery",
				errString: foldNextAction(
					Decision{Decision: "hold", Blockers: []string{"CI_BASE_RED"}},
					StableSummary{Evidence: StableEvidence{OK: true}},
					DirtySummary{Clean: false, ReleaseRelevantCount: 3},
					CIDiagnosis{},
					BranchRegime{},
				).Detail,
				recoveryHint: "commit, shelve, or remove",
				pathOrFormat: "release-relevant dirty path(s)",
			},
			{
				name: "branch_role_config_blocker",
				errString: FoldBranchRegime(BranchRegimeFacts{
					RoleError: "branch_roles config invalid: missing release_branch",
				}).NextAction,
				recoveryHint: "clear blocker(s)",
				pathOrFormat: "BRANCH_ROLE_CONFIG",
			},
			{
				name:         "ci_billing_wall_detail",
				errString:    DiagnoseCIFailure("ci-fast.yml", CIRun{Conclusion: "failure"}, nil, "payments have failed; check spending limit").Detail,
				recoveryHint: "operator attention",
				pathOrFormat: "billing, payments, or spending limits",
			},
			{
				name:         "ci_startup_failure_detail",
				errString:    DiagnoseCIFailure("ci-fast.yml", CIRun{Conclusion: "startup_failure"}, nil, "").Detail,
				recoveryHint: "inspect runner admission and workflow startup annotations",
				pathOrFormat: "workflow startup annotations",
			},
		}

		seen := make(map[string]string)
		for _, tc := range cases {
			// 1. Error string must be non-empty (non-nil).
			if strings.TrimSpace(tc.errString) == "" {
				t.Fatalf("case %s produced empty error string", tc.name)
			}

			// 2. All error strings must be distinct.
			if prior, exists := seen[tc.errString]; exists {
				t.Fatalf("error collision: case %q produced duplicate error string with case %q: %q", tc.name, prior, tc.errString)
			}
			seen[tc.errString] = tc.name

			// 3. Error must contain actionable recovery hint.
			if !strings.Contains(tc.errString, tc.recoveryHint) {
				t.Errorf("case %s error %q does not contain recovery hint %q", tc.name, tc.errString, tc.recoveryHint)
			}

			// 4. Error must name the expected format or required path.
			if !strings.Contains(tc.errString, tc.pathOrFormat) {
				t.Errorf("case %s error %q does not contain expected format/path %q", tc.name, tc.errString, tc.pathOrFormat)
			}
		}
	})
}
