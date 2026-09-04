package adjudicator

import (
	"testing"
)

func TestImmunityPhase(t *testing.T) {
	t.Run("IsTestPath", func(t *testing.T) {
		tests := []struct {
			path string
			want bool
		}{
			// Go tests
			{"foo_test.go", true},
			{"internal/adjudicator/decide_test.go", true},
			{"/workspace/pkg/bar_test.go", true},
			{"_test.go", true},
			{"foo.go", false},
			{"test.go", false},
			{"testing.go", false},
			{"foo_test_helper.go", false},

			// TypeScript / JavaScript tests
			{"foo.test.ts", true},
			{"src/components/Button.test.ts", true},
			{"foo.spec.ts", true},
			{"src/components/Button.spec.ts", true},
			{"foo.test.tsx", true},
			{"bar.spec.tsx", true},
			{"foo.test.js", true},
			{"bar.spec.js", true},
			{"foo.test.jsx", true},
			{"bar.spec.jsx", true},
			{"foo.ts", false},
			{"foo.js", false},
			{"test.ts", false},
			{"spec.ts", false},
			{"testing.tsx", false},

			// Python tests
			{"test_models.py", true},
			{"src/test_views.py", true},
			{"models_test.py", true},
			{"src/views_test.py", true},
			{"test_.py", true},
			{"_test.py", true},
			{"test.py", false},
			{"helper.py", false},
			{"models.py", false},
			{"testing.py", false},

			// Java tests
			{"UserServiceTest.java", true},
			{"Test.java", true},
			{"com/example/OrderTest.java", true},
			{"UserServiceTests.java", true},
			{"UserService.java", false},
			{"Testing.java", false},
			{"Tester.java", false},

			// tests/ or testdata/ directories
			{"tests/helper.py", true},
			{"tests/fixtures/data.json", true},
			{"src/tests/helper.ts", true},
			{"tests/unit/test.sh", true},
			{"testdata/input.json", true},
			{"internal/adjudicator/testdata/cases.txt", true},
			{"pkg/testdata/sub/fixture.xml", true},
			{"tests", true},
			{"testdata", true},
			{"tests/", true},
			{"testdata/", true},
			{"foo/tests/bar.txt", true},
			{"foo/testdata/bar.txt", true},
			{"Tests/Helper.java", true},
			{"TestData/fixture.json", true},

			// Non-test directories / files
			{"my_tests/foo.go", false},
			{"tests_extra/foo.go", false},
			{"my_testdata/foo.go", false},
			{"tests.go", false},
			{"testdata.go", false},
			{"pkg/tests.txt", false},

			// Windows separators
			{`tests\helper.py`, true},
			{`src\components\Button.test.ts`, true},
			{`internal\adjudicator\testdata\sample.json`, true},
			{`src\models\user.go`, false},

			// Edge cases
			{"", false},
			{"   ", false},
			{".", false},
			{"/", false},
		}

		for _, tc := range tests {
			got := IsTestPath(tc.path)
			if got != tc.want {
				t.Errorf("IsTestPath(%q) = %v; want %v", tc.path, got, tc.want)
			}
		}
	})

	t.Run("RedPhaseEvaluation", func(t *testing.T) {
		redPhases := []string{
			"RED_REPRODUCTION",
			"phase_red",
			"repro_authoring",
			"red_reproduction",
			"PHASE_RED",
			"REPRO_AUTHORING",
			"  phase_red  ",
		}

		testFiles := []string{
			"internal/adjudicator/test_immunity_test.go",
			"src/Button.test.ts",
			"tests/repro.py",
			"com/example/OrderTest.java",
			"testdata/input.json",
		}

		nonTestFiles := []string{
			"internal/adjudicator/test_immunity.go",
			"src/Button.tsx",
			"main.py",
			"com/example/Order.java",
			"cmd/fak/main.go",
		}

		for _, phase := range redPhases {
			for _, tf := range testFiles {
				allowed, reason, affordance := CheckTestImmunityPhase(tf, phase)
				if !allowed {
					t.Errorf("CheckTestImmunityPhase(%q, %q) allowed = false; want true", tf, phase)
				}
				if reason != "" {
					t.Errorf("CheckTestImmunityPhase(%q, %q) reason = %q; want empty", tf, phase, reason)
				}
				if affordance != "" {
					t.Errorf("CheckTestImmunityPhase(%q, %q) affordance = %q; want empty", tf, phase, affordance)
				}
			}

			for _, ntf := range nonTestFiles {
				allowed, reason, affordance := CheckTestImmunityPhase(ntf, phase)
				if allowed {
					t.Errorf("CheckTestImmunityPhase(%q, %q) allowed = true; want false", ntf, phase)
				}
				if reason != ReasonImplementationLockedInRedPhase {
					t.Errorf("CheckTestImmunityPhase(%q, %q) reason = %q; want %q", ntf, phase, reason, ReasonImplementationLockedInRedPhase)
				}
				if affordance != AffordanceImplementationLockedInRedPhase {
					t.Errorf("CheckTestImmunityPhase(%q, %q) affordance = %q; want %q", ntf, phase, affordance, AffordanceImplementationLockedInRedPhase)
				}
			}
		}
	})

	t.Run("GreenPhaseEvaluation", func(t *testing.T) {
		greenPhases := []string{
			"GREEN_IMPLEMENTATION",
			"phase_green",
			"fix_implementation",
			"green_implementation",
			"PHASE_GREEN",
			"FIX_IMPLEMENTATION",
			"  phase_green  ",
		}

		testFiles := []string{
			"internal/adjudicator/test_immunity_test.go",
			"src/Button.test.ts",
			"tests/repro.py",
			"com/example/OrderTest.java",
			"testdata/input.json",
		}

		nonTestFiles := []string{
			"internal/adjudicator/test_immunity.go",
			"src/Button.tsx",
			"main.py",
			"com/example/Order.java",
			"cmd/fak/main.go",
		}

		for _, phase := range greenPhases {
			for _, ntf := range nonTestFiles {
				allowed, reason, affordance := CheckTestImmunityPhase(ntf, phase)
				if !allowed {
					t.Errorf("CheckTestImmunityPhase(%q, %q) allowed = false; want true", ntf, phase)
				}
				if reason != "" {
					t.Errorf("CheckTestImmunityPhase(%q, %q) reason = %q; want empty", ntf, phase, reason)
				}
				if affordance != "" {
					t.Errorf("CheckTestImmunityPhase(%q, %q) affordance = %q; want empty", ntf, phase, affordance)
				}
			}

			for _, tf := range testFiles {
				allowed, reason, affordance := CheckTestImmunityPhase(tf, phase)
				if allowed {
					t.Errorf("CheckTestImmunityPhase(%q, %q) allowed = true; want false", tf, phase)
				}
				if reason != ReasonTestImmuneInGreenPhase {
					t.Errorf("CheckTestImmunityPhase(%q, %q) reason = %q; want %q", tf, phase, reason, ReasonTestImmuneInGreenPhase)
				}
				if affordance != AffordanceTestImmuneInGreenPhase {
					t.Errorf("CheckTestImmunityPhase(%q, %q) affordance = %q; want %q", tf, phase, affordance, AffordanceTestImmuneInGreenPhase)
				}
			}
		}
	})

	t.Run("NeutralOrEmptyPhaseAllowsAll", func(t *testing.T) {
		otherPhases := []string{
			"",
			"   ",
			"neutral",
			"VERIFY_PHASE",
			"PHASE_DONE",
			"refactor",
			"other",
		}

		files := []string{
			"internal/adjudicator/test_immunity_test.go",
			"src/Button.test.ts",
			"tests/repro.py",
			"internal/adjudicator/test_immunity.go",
			"src/Button.tsx",
			"main.py",
		}

		for _, phase := range otherPhases {
			for _, f := range files {
				allowed, reason, affordance := CheckTestImmunityPhase(f, phase)
				if !allowed {
					t.Errorf("CheckTestImmunityPhase(%q, %q) allowed = false; want true", f, phase)
				}
				if reason != "" {
					t.Errorf("CheckTestImmunityPhase(%q, %q) reason = %q; want empty", f, phase, reason)
				}
				if affordance != "" {
					t.Errorf("CheckTestImmunityPhase(%q, %q) affordance = %q; want empty", f, phase, affordance)
				}
			}
		}
	})
}
