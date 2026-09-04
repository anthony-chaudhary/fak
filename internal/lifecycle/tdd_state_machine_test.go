package lifecycle

import (
	"sync"
	"testing"
)

func TestTDDStateMachine(t *testing.T) {
	t.Run("PhaseTransitions", func(t *testing.T) {
		reproPath := "internal/lifecycle/repro_test.go"
		sm := NewTDDStateMachine(reproPath)

		if got := sm.CurrentPhase(); got != PhaseRed {
			t.Fatalf("expected initial phase %s, got %s", PhaseRed, got)
		}
		if PhaseRed.String() != "PHASE_RED_REPRO" {
			t.Errorf("expected string PHASE_RED_REPRO, got %s", PhaseRed.String())
		}
		if PhaseGreen.String() != "PHASE_GREEN_FIX" {
			t.Errorf("expected string PHASE_GREEN_FIX, got %s", PhaseGreen.String())
		}
		if PhaseVerified.String() != "PHASE_VERIFIED" {
			t.Errorf("expected string PHASE_VERIFIED, got %s", PhaseVerified.String())
		}

		// Cannot transition directly from PhaseRed to PhaseVerified
		if err := sm.RecordFixSuccess("all tests passed"); err == nil {
			t.Fatalf("expected error recording fix success while in PhaseRed")
		}

		// Transition to PhaseGreen
		failureEvidence := "--- FAIL: TestReproBug (0.01s)\n    repro_test.go:12: assertion failed"
		if err := sm.RecordReproductionFailure(reproPath, failureEvidence); err != nil {
			t.Fatalf("unexpected error recording repro failure: %v", err)
		}

		if got := sm.CurrentPhase(); got != PhaseGreen {
			t.Fatalf("expected phase %s after failure, got %s", PhaseGreen, got)
		}

		// Cannot re-record failure while in PhaseGreen
		if err := sm.RecordReproductionFailure(reproPath, failureEvidence); err == nil {
			t.Fatalf("expected error recording repro failure while in PhaseGreen")
		}

		// Transition to PhaseVerified
		successEvidence := "=== RUN   TestReproBug\n--- PASS: TestReproBug (0.01s)\nPASS"
		if err := sm.RecordFixSuccess(successEvidence); err != nil {
			t.Fatalf("unexpected error recording fix success: %v", err)
		}

		if got := sm.CurrentPhase(); got != PhaseVerified {
			t.Fatalf("expected phase %s after fix success, got %s", PhaseVerified, got)
		}

		// Cannot record fix success while in PhaseVerified
		if err := sm.RecordFixSuccess(successEvidence); err == nil {
			t.Fatalf("expected error recording fix success while in PhaseVerified")
		}
		// Cannot record repro failure while in PhaseVerified
		if err := sm.RecordReproductionFailure(reproPath, failureEvidence); err == nil {
			t.Fatalf("expected error recording repro failure while in PhaseVerified")
		}
	})

	t.Run("CanModifyFile_PhaseRed", func(t *testing.T) {
		sm := NewTDDStateMachine("tests/bug_test.go")

		testFiles := []string{
			"tests/bug_test.go",
			"pkg/service_test.go",
			"test_component.py",
			"component.test.ts",
			"tests/fixture.go",
		}
		for _, tf := range testFiles {
			allowed, reason := sm.CanModifyFile(tf)
			if !allowed {
				t.Errorf("expected %s to be allowed in PhaseRed, got blocked with reason: %s", tf, reason)
			}
			if reason != "" {
				t.Errorf("expected empty reason for allowed file %s, got: %s", tf, reason)
			}
		}

		nonTestFiles := []string{
			"pkg/service.go",
			"main.go",
			"internal/lifecycle/tdd_state_machine.go",
			"app.py",
			"index.ts",
		}
		expectedReason := "Phase Red requires adding a failing reproduction test before editing code."
		for _, ntf := range nonTestFiles {
			allowed, reason := sm.CanModifyFile(ntf)
			if allowed {
				t.Errorf("expected non-test file %s to be blocked in PhaseRed", ntf)
			}
			if reason != expectedReason {
				t.Errorf("expected reason %q for %s, got %q", expectedReason, ntf, reason)
			}
		}
	})

	t.Run("CanModifyFile_PhaseGreen", func(t *testing.T) {
		reproPath := "internal/lifecycle/repro_test.go"
		sm := NewTDDStateMachine(reproPath)
		if err := sm.RecordReproductionFailure(reproPath, "FAIL: expected red"); err != nil {
			t.Fatalf("failed to enter PhaseGreen: %v", err)
		}

		// Reproduction test must be frozen
		expectedFrozenReason := "Reproduction test is frozen during Phase Green fix."
		reproVariations := []string{
			reproPath,
			"./" + reproPath,
			"repro_test.go",
		}
		for _, p := range reproVariations {
			allowed, reason := sm.CanModifyFile(p)
			if allowed {
				t.Errorf("expected repro test %s to be blocked in PhaseGreen", p)
			}
			if reason != expectedFrozenReason {
				t.Errorf("expected reason %q for %s, got %q", expectedFrozenReason, p, reason)
			}
		}

		// Non-test files permitted in PhaseGreen
		implFiles := []string{
			"internal/lifecycle/tdd_state_machine.go",
			"cmd/fak/main.go",
			"pkg/service.go",
		}
		for _, f := range implFiles {
			allowed, reason := sm.CanModifyFile(f)
			if !allowed {
				t.Errorf("expected non-test file %s to be allowed in PhaseGreen, got: %s", f, reason)
			}
			if reason != "" {
				t.Errorf("expected empty reason for %s, got %q", f, reason)
			}
		}

		// Non-repro test files permitted
		allowed, reason := sm.CanModifyFile("internal/lifecycle/other_test.go")
		if !allowed {
			t.Errorf("expected other test file to be allowed in PhaseGreen, got: %s", reason)
		}
	})

	t.Run("CanModifyFile_PhaseVerified", func(t *testing.T) {
		reproPath := "repro_test.go"
		sm := NewTDDStateMachine(reproPath)
		_ = sm.RecordReproductionFailure(reproPath, "FAIL")
		_ = sm.RecordFixSuccess("PASS")

		expectedReason := "Task complete."
		files := []string{
			"main.go",
			"repro_test.go",
			"other_test.go",
			"README.md",
		}
		for _, f := range files {
			allowed, reason := sm.CanModifyFile(f)
			if allowed {
				t.Errorf("expected file %s to be blocked in PhaseVerified", f)
			}
			if reason != expectedReason {
				t.Errorf("expected reason %q for %s, got %q", expectedReason, f, reason)
			}
		}
	})

	t.Run("RecordReproductionFailure_Validation", func(t *testing.T) {
		sm := NewTDDStateMachine("repro_test.go")

		// Empty evidence rejected
		if err := sm.RecordReproductionFailure("repro_test.go", ""); err == nil {
			t.Errorf("expected error for empty failure evidence")
		}
		if err := sm.RecordReproductionFailure("repro_test.go", "   \t\n"); err == nil {
			t.Errorf("expected error for whitespace failure evidence")
		}

		// Phase remains PhaseRed when validation fails
		if sm.CurrentPhase() != PhaseRed {
			t.Errorf("expected phase to remain %s, got %s", PhaseRed, sm.CurrentPhase())
		}

		// Updating repro path on failure record
		if err := sm.RecordReproductionFailure("updated_repro_test.go", "FAIL: panic"); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if sm.CurrentPhase() != PhaseGreen {
			t.Errorf("expected phase %s, got %s", PhaseGreen, sm.CurrentPhase())
		}

		// Verify updated repro test is frozen in PhaseGreen
		allowed, _ := sm.CanModifyFile("updated_repro_test.go")
		if allowed {
			t.Errorf("expected updated repro test to be frozen")
		}
	})

	t.Run("RecordFixSuccess_Validation", func(t *testing.T) {
		sm := NewTDDStateMachine("repro_test.go")
		_ = sm.RecordReproductionFailure("repro_test.go", "FAIL")

		// Empty evidence rejected
		if err := sm.RecordFixSuccess(""); err == nil {
			t.Errorf("expected error for empty success evidence")
		}
		if err := sm.RecordFixSuccess("   \n"); err == nil {
			t.Errorf("expected error for whitespace success evidence")
		}

		if sm.CurrentPhase() != PhaseGreen {
			t.Errorf("expected phase to remain %s, got %s", PhaseGreen, sm.CurrentPhase())
		}

		if err := sm.RecordFixSuccess("PASS: all tests ok"); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if sm.CurrentPhase() != PhaseVerified {
			t.Errorf("expected phase %s, got %s", PhaseVerified, sm.CurrentPhase())
		}
	})

	t.Run("IsTestFile_Detection", func(t *testing.T) {
		tests := []struct {
			path     string
			expected bool
		}{
			{"", false},
			{".", false},
			{"foo.go", false},
			{"foo_test.go", true},
			{"internal/lifecycle/foo_test.go", true},
			{"internal/lifecycle/control.go", false},
			{"test_handler.py", true},
			{"handler_test.py", true},
			{"handler.py", false},
			{"component.test.ts", true},
			{"component.spec.ts", true},
			{"component.test.tsx", true},
			{"component.spec.tsx", true},
			{"component.test.js", true},
			{"component.spec.js", true},
			{"component.test.jsx", true},
			{"component.spec.jsx", true},
			{"component.test.mjs", true},
			{"component.spec.mjs", true},
			{"component.test.cjs", true},
			{"component.spec.cjs", true},
			{"component.ts", false},
			{"UserServiceTest.java", true},
			{"UserServiceTests.java", true},
			{"UserServiceTestCase.java", true},
			{"UserServiceTest.kt", true},
			{"UserServiceTests.kt", true},
			{"UserService.java", false},
			{"user_spec.rb", true},
			{"test_user.rb", true},
			{"user.rb", false},
			{"lib_test.rs", true},
			{"lib.rs", false},
			{"math_test.cc", true},
			{"math_test.cpp", true},
			{"math_unittest.cc", true},
			{"math_unittest.cpp", true},
			{"math_test.c", true},
			{"math.cpp", false},
			{"tests/helper.go", true},
			{"testdata/input.json", true},
			{"src/__tests__/app.ts", true},
		}

		for _, tc := range tests {
			got := IsTestFile(tc.path)
			if got != tc.expected {
				t.Errorf("IsTestFile(%q) = %v, want %v", tc.path, got, tc.expected)
			}
		}
	})

	t.Run("Concurrency_Safety", func(t *testing.T) {
		sm := NewTDDStateMachine("repro_test.go")
		var wg sync.WaitGroup

		for i := 0; i < 20; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_ = sm.CurrentPhase()
				_, _ = sm.CanModifyFile("foo.go")
				_, _ = sm.CanModifyFile("repro_test.go")
			}()
		}

		wg.Wait()
	})
}
