package adjudicator

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func testImmunityBasePolicy() Policy {
	return Policy{
		Allow: map[string]bool{
			"Edit":        true,
			"edit":        true,
			"Write":       true,
			"write":       true,
			"write_file":  true,
			"delete_file": true,
			"Bash":        true,
			"PowerShell":  true,
		},
	}
}

// TestImmunityRefusesTestFileWrites verifies that write/edit/delete proposals
// targeting *_test.go files are refused with TEST_TAMPER_REFUSED under implementation
// or default lanes (#10923).
func TestImmunityRefusesTestFileWrites(t *testing.T) {
	a := New(testImmunityBasePolicy())
	ctx := context.Background()

	cases := []struct {
		name string
		tool string
		args string
		meta map[string]string
	}{
		{
			name: "Edit targeting *_test.go under default lane",
			tool: "Edit",
			args: `{"filePath":"internal/gateway/gateway_test.go"}`,
			meta: nil,
		},
		{
			name: "Edit targeting *_test.go under implementation lane",
			tool: "Edit",
			args: `{"filePath":"internal/gateway/gateway_test.go"}`,
			meta: map[string]string{"lane": "implementation"},
		},
		{
			name: "Edit targeting *_test.go under adjudicator lane",
			tool: "Edit",
			args: `{"filePath":"internal/adjudicator/decide_test.go"}`,
			meta: map[string]string{"lane": "adjudicator"},
		},
		{
			name: "Edit snake_case file_path targeting *_test.go",
			tool: "Edit",
			args: `{"file_path":"internal/witness/witness_test.go"}`,
			meta: map[string]string{"lane": "witness"},
		},
		{
			name: "Write targeting *_test.go",
			tool: "Write",
			args: `{"filePath":"cmd/fak/main_test.go","content":"package main"}`,
			meta: map[string]string{"lane": "cmd"},
		},
		{
			name: "write_file targeting *_test.go",
			tool: "write_file",
			args: `{"path":"internal/policy/policy_test.go","content":"package policy"}`,
			meta: map[string]string{"lane": "policy"},
		},
		{
			name: "delete_file targeting *_test.go",
			tool: "delete_file",
			args: `{"path":"internal/engine/engine_test.go"}`,
			meta: map[string]string{"lane": "engine"},
		},
		{
			name: "Bash rm *_test.go",
			tool: "Bash",
			args: `{"command":"rm internal/adjudicator/decide_test.go"}`,
			meta: map[string]string{"lane": "adjudicator"},
		},
		{
			name: "Bash redirect to *_test.go",
			tool: "Bash",
			args: `{"command":"echo 'package decide' > internal/adjudicator/helper_test.go"}`,
			meta: map[string]string{"lane": "adjudicator"},
		},
		{
			name: "Bash git rm *_test.go",
			tool: "Bash",
			args: `{"command":"git rm internal/gateway/http_test.go"}`,
			meta: map[string]string{"lane": "gateway"},
		},
		{
			name: "PowerShell Remove-Item *_test.go",
			tool: "PowerShell",
			args: `{"command":"Remove-Item internal/adjudicator/decide_test.go"}`,
			meta: map[string]string{"lane": "adjudicator"},
		},
		{
			name: "PowerShell Set-Content *_test.go",
			tool: "PowerShell",
			args: `{"command":"Set-Content -Path internal/adjudicator/decide_test.go -Value 'x'"}`,
			meta: map[string]string{"lane": "adjudicator"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			call := inlineCall(tc.tool, tc.args)
			call.Meta = tc.meta
			v := a.Adjudicate(ctx, call)
			if v.Kind != abi.VerdictDeny {
				t.Fatalf("%s: expected VerdictDeny, got %v (reason=%s)",
					tc.name, v.Kind, abi.ReasonName(v.Reason))
			}
			if v.Reason != ReasonTestTamperRefused {
				t.Fatalf("%s: expected ReasonTestTamperRefused, got %v (%s)",
					tc.name, v.Reason, abi.ReasonName(v.Reason))
			}
			if abi.ReasonName(v.Reason) != "TEST_TAMPER_REFUSED" {
				t.Fatalf("%s: expected reason name TEST_TAMPER_REFUSED, got %s",
					tc.name, abi.ReasonName(v.Reason))
			}
		})
	}
}

// TestImmunityRefusesTestDataAndFixtures verifies that proposals targeting testdata,
// fixtures, and test configurations are refused (#10923).
func TestImmunityRefusesTestDataAndFixtures(t *testing.T) {
	a := New(testImmunityBasePolicy())
	ctx := context.Background()

	cases := []struct {
		name string
		tool string
		args string
	}{
		{
			name: "Write targeting testdata file",
			tool: "Write",
			args: `{"filePath":"testdata/poison.json","content":"{}"}`,
		},
		{
			name: "Edit targeting sub-package testdata",
			tool: "Edit",
			args: `{"filePath":"internal/adjudicator/testdata/baseline.json"}`,
		},
		{
			name: "Write targeting fixtures directory",
			tool: "Write",
			args: `{"filePath":"fixtures/sample.json","content":"{}"}`,
		},
		{
			name: "Write targeting .golden fixture",
			tool: "Write",
			args: `{"filePath":"internal/witness/output.golden","content":"x"}`,
		},
		{
			name: "Edit targeting jest.config.js",
			tool: "Edit",
			args: `{"filePath":"jest.config.js"}`,
		},
		{
			name: "Edit targeting pytest.ini",
			tool: "Edit",
			args: `{"filePath":"pytest.ini"}`,
		},
		{
			name: "Write targeting *.test.json",
			tool: "Write",
			args: `{"filePath":"config.test.json","content":"{}"}`,
		},
		{
			name: "Bash rm testdata directory",
			tool: "Bash",
			args: `{"command":"rm -rf testdata/coding_smoke"}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			call := inlineCall(tc.tool, tc.args)
			call.Meta = map[string]string{"lane": "implementation"}
			v := a.Adjudicate(ctx, call)
			if v.Kind != abi.VerdictDeny || v.Reason != ReasonTestTamperRefused {
				t.Fatalf("%s: expected Deny/TEST_TAMPER_REFUSED, got %v/%s",
					tc.name, v.Kind, abi.ReasonName(v.Reason))
			}
		})
	}
}

// TestImmunityAllowsImplementationFiles verifies that write/edit proposals targeting
// implementation files (non-test files) pass through successfully (#10923).
func TestImmunityAllowsImplementationFiles(t *testing.T) {
	a := New(testImmunityBasePolicy())
	ctx := context.Background()

	cases := []struct {
		name string
		tool string
		args string
		meta map[string]string
	}{
		{
			name: "Edit implementation file",
			tool: "Edit",
			args: `{"filePath":"internal/adjudicator/decide.go"}`,
			meta: map[string]string{"lane": "implementation"},
		},
		{
			name: "Write implementation file",
			tool: "Write",
			args: `{"filePath":"cmd/fak/main.go","content":"package main"}`,
			meta: map[string]string{"lane": "cmd"},
		},
		{
			name: "Bash echo to implementation file",
			tool: "Bash",
			args: `{"command":"echo 'package adjudicator' > internal/adjudicator/helper.go"}`,
			meta: map[string]string{"lane": "adjudicator"},
		},
		{
			name: "Bash running go test (test execution is not a file write)",
			tool: "Bash",
			args: `{"command":"go test ./internal/adjudicator/..."}`,
			meta: map[string]string{"lane": "adjudicator"},
		},
		{
			name: "Bash reading test file (cat is not a file write)",
			tool: "Bash",
			args: `{"command":"cat internal/adjudicator/decide_test.go"}`,
			meta: map[string]string{"lane": "adjudicator"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			call := inlineCall(tc.tool, tc.args)
			call.Meta = tc.meta
			v := a.Adjudicate(ctx, call)
			if v.Kind != abi.VerdictAllow {
				t.Fatalf("%s: expected VerdictAllow for implementation target, got %v (reason=%s)",
					tc.name, v.Kind, abi.ReasonName(v.Reason))
			}
		})
	}
}

// TestImmunityExemptsTestLanes verifies that explicitly designated test lanes
// are exempted from the test-immunity gate (#10923).
func TestImmunityExemptsTestLanes(t *testing.T) {
	a := New(testImmunityBasePolicy())
	ctx := context.Background()

	testLaneTokens := []string{
		"test", "tests", "testing", "qa", "eval", "evaluation",
		"benchmark", "benchmarks", "redteam", "test_lane", "test-lane",
		"testsuite", "test_suite", "test-suite",
		"test/adjudicator", "adjudicator-test", "gateway-qa",
	}

	for _, lane := range testLaneTokens {
		t.Run("lane="+lane, func(t *testing.T) {
			call := inlineCall("Edit", `{"filePath":"internal/adjudicator/decide_test.go"}`)
			call.Meta = map[string]string{"lane": lane}
			v := a.Adjudicate(ctx, call)
			if v.Kind != abi.VerdictAllow {
				t.Fatalf("lane %q should be exempt: got %v (reason=%s), want VerdictAllow",
					lane, v.Kind, abi.ReasonName(v.Reason))
			}
		})
	}

	t.Run("meta lane_type=test", func(t *testing.T) {
		call := inlineCall("Edit", `{"filePath":"internal/adjudicator/decide_test.go"}`)
		call.Meta = map[string]string{"lane_type": "test"}
		v := a.Adjudicate(ctx, call)
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("lane_type=test should be exempt: got %v (reason=%s), want VerdictAllow",
				v.Kind, abi.ReasonName(v.Reason))
		}
	})

	t.Run("context with test lane", func(t *testing.T) {
		laneCtx := ContextWithLane(ctx, "test")
		call := inlineCall("Edit", `{"filePath":"internal/adjudicator/decide_test.go"}`)
		call.Meta = nil
		v := a.Adjudicate(laneCtx, call)
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("ContextWithLane test should be exempt: got %v (reason=%s), want VerdictAllow",
				v.Kind, abi.ReasonName(v.Reason))
		}
	})

	t.Run("custom test lane in policy", func(t *testing.T) {
		p := testImmunityBasePolicy()
		p.TestLanes = []string{"custom-suite-lane"}
		customA := New(p)
		call := inlineCall("Edit", `{"filePath":"internal/adjudicator/decide_test.go"}`)
		call.Meta = map[string]string{"lane": "custom-suite-lane"}
		v := customA.Adjudicate(ctx, call)
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("custom policy test lane should be exempt: got %v (reason=%s), want VerdictAllow",
				v.Kind, abi.ReasonName(v.Reason))
		}
	})

	t.Run("policy DisableTestImmunity", func(t *testing.T) {
		p := testImmunityBasePolicy()
		p.DisableTestImmunity = true
		disabledA := New(p)
		call := inlineCall("Edit", `{"filePath":"internal/adjudicator/decide_test.go"}`)
		call.Meta = map[string]string{"lane": "implementation"}
		v := disabledA.Adjudicate(ctx, call)
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("DisableTestImmunity should allow test writes: got %v (reason=%s), want VerdictAllow",
				v.Kind, abi.ReasonName(v.Reason))
		}
	})
}

// TestImmunityDosReasonRegistered verifies that TEST_TAMPER_REFUSED is declared
// and bound in both abi and dos.toml (#10923).
func TestImmunityDosReasonRegistered(t *testing.T) {
	// ABI registration check
	if name := abi.ReasonName(ReasonTestTamperRefused); name != ReasonTestTamperRefusedName {
		t.Fatalf("abi.ReasonName: got %q, want %q", name, ReasonTestTamperRefusedName)
	}
	code, ok := abi.ReasonByName(ReasonTestTamperRefusedName)
	if !ok {
		t.Fatalf("abi.ReasonByName(%q) returned ok=false", ReasonTestTamperRefusedName)
	}
	if code != ReasonTestTamperRefused {
		t.Fatalf("abi.ReasonByName(%q) code mismatch: got %v, want %v",
			ReasonTestTamperRefusedName, code, ReasonTestTamperRefused)
	}

	// dos.toml registration check
	root := findRepoRoot(t)
	raw, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		t.Fatalf("read dos.toml: %v", err)
	}
	content := string(raw)

	header := "[reasons.TEST_TAMPER_REFUSED]"
	idx := strings.Index(content, header)
	if idx < 0 {
		t.Fatalf("dos.toml does not declare %s", header)
	}
	block := content[idx:]
	if end := strings.Index(block[len(header):], "\n["); end >= 0 {
		block = block[:len(header)+end]
	}

	if !strings.Contains(block, "refusal  = true") && !strings.Contains(block, "refusal = true") {
		t.Errorf("%s missing refusal = true", header)
	}
	if !strings.Contains(block, `category = "OPERATOR_GATE"`) {
		t.Errorf("%s missing category OPERATOR_GATE", header)
	}
	if !strings.Contains(block, "Tool call attempted to write or edit gating test files under an implementation lane") {
		t.Errorf("%s summary mismatch", header)
	}
	if !strings.Contains(block, "Modify only implementation files in your assigned lane; test modifications are forbidden") {
		t.Errorf("%s fix mismatch", header)
	}
	if !strings.Contains(block, `"internal/adjudicator"`) || !strings.Contains(block, `"internal/witness"`) {
		t.Errorf("%s see_also missing expected references", header)
	}
}

func findRepoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "dos.toml")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("could not find repo root containing dos.toml")
		}
		dir = parent
	}
}

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
