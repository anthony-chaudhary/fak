package dispatchtick

import "testing"

// src wraps test-func declarations into a minimal parseable Go file. Parse mode is 0, so
// unimported selectors (require., *testing.T) parse fine — no import block is needed.
func src(decls string) string {
	return "package sample\n\n" + decls + "\n"
}

// TestAddedTestsWitnessed proves both directions of the rung from synthetic diffs: a test
// whose body structurally cannot fail is denied, while every real failure mechanism (a
// Fatal/Error, testify, a helper passing t, a subtest, a panic) clears it. This is the
// #3364 first checkable step — it exercises the AST analyzer without touching the live
// dispatch path.
func TestAddedTestsWitnessed(t *testing.T) {
	cases := []struct {
		name  string
		files []AddedTestFile
		want  bool // want AddedTestsWitnessed == want
	}{
		{
			name:  "log_only_cannot_fail",
			files: []AddedTestFile{{Path: "x_test.go", Content: src(`func TestX(t *testing.T) { t.Log("ok") }`)}},
			want:  false,
		},
		{
			name:  "skip_only_cannot_fail",
			files: []AddedTestFile{{Path: "x_test.go", Content: src(`func TestX(t *testing.T) { t.Skip("todo") }`)}},
			want:  false,
		},
		{
			name:  "logf_and_helper_marker_only_cannot_fail",
			files: []AddedTestFile{{Path: "x_test.go", Content: src(`func TestX(t *testing.T) { t.Helper(); t.Parallel(); t.Logf("v=%d", 1) }`)}},
			want:  false,
		},
		{
			name:  "real_fatalf_assertion_can_fail",
			files: []AddedTestFile{{Path: "x_test.go", Content: src(`func TestX(t *testing.T) { got := 2; if got != 1 { t.Fatalf("got %d", got) } }`)}},
			want:  true,
		},
		{
			name:  "errorf_can_fail",
			files: []AddedTestFile{{Path: "x_test.go", Content: src(`func TestX(t *testing.T) { t.Log("start"); if false { t.Errorf("x") } }`)}},
			want:  true,
		},
		{
			name:  "testify_require_can_fail",
			files: []AddedTestFile{{Path: "x_test.go", Content: src(`func TestX(t *testing.T) { require.Equal(t, 1, 2) }`)}},
			want:  true,
		},
		{
			name:  "helper_passing_t_can_fail",
			files: []AddedTestFile{{Path: "x_test.go", Content: src(`func TestX(t *testing.T) { checkInvariant(t, 42) }`)}},
			want:  true,
		},
		{
			name:  "table_subtest_run_clears",
			files: []AddedTestFile{{Path: "x_test.go", Content: src(`func TestX(t *testing.T) { t.Run("a", func(t *testing.T) { t.Log("x") }) }`)}},
			want:  true,
		},
		{
			name:  "panic_can_fail",
			files: []AddedTestFile{{Path: "x_test.go", Content: src(`func TestX(t *testing.T) { panic("boom") }`)}},
			want:  true,
		},
		{
			name:  "non_test_func_ignored",
			files: []AddedTestFile{{Path: "x_test.go", Content: src(`func helperOnly(t *testing.T) { t.Log("x") }` + "\n" + `func TestReal(t *testing.T) { t.Fatal("always") }`)}},
			want:  true,
		},
		{
			name:  "non_test_file_ignored",
			files: []AddedTestFile{{Path: "prod.go", Content: src(`func TestX(t *testing.T) { t.Log("ok") }`)}},
			want:  true,
		},
		{
			name:  "no_added_tests_is_witnessed",
			files: nil,
			want:  true,
		},
		{
			name:  "unparseable_fails_open",
			files: []AddedTestFile{{Path: "x_test.go", Content: "package sample\nfunc TestX(t *testing.T) { this is (((not go"}},
			want:  true,
		},
		{
			name: "one_vacuous_among_real_trips_the_rung",
			files: []AddedTestFile{
				{Path: "a_test.go", Content: src(`func TestReal(t *testing.T) { if 1 != 2 { t.Fatal("x") } }`)},
				{Path: "b_test.go", Content: src(`func TestVacuous(t *testing.T) { t.Log("nothing") }`)},
			},
			want: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := AddedTestsWitnessed(tc.files); got != tc.want {
				t.Fatalf("AddedTestsWitnessed = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestAddedTestsRefusal proves the rung names the offending test with the structured
// reason, and stays silent when every added test can fail.
func TestAddedTestsRefusal(t *testing.T) {
	reason, path, fn := AddedTestsRefusal([]AddedTestFile{
		{Path: "pkg/thing_test.go", Content: src(`func TestNoop(t *testing.T) { t.Log("ok") }`)},
	})
	if reason != WitnessTestCannotFail {
		t.Fatalf("reason = %q, want %q", reason, WitnessTestCannotFail)
	}
	if path != "pkg/thing_test.go" || fn != "TestNoop" {
		t.Fatalf("offender = (%q, %q), want (pkg/thing_test.go, TestNoop)", path, fn)
	}

	reason, _, _ = AddedTestsRefusal([]AddedTestFile{
		{Path: "x_test.go", Content: src(`func TestReal(t *testing.T) { if 1 != 2 { t.Fatal("x") } }`)},
	})
	if reason != "" {
		t.Fatalf("reason = %q on a can-fail test, want empty", reason)
	}
}

// TestCommitWitnessedWithIntegrity proves the fold: the composite keep-bit downgrades a
// would-be-witnessed commit whose only new test cannot fail (with a legible reason), keeps
// a commit with a real assertion, and adds no new reason when the claim-vs-diff rung has
// already failed.
func TestCommitWitnessedWithIntegrity(t *testing.T) {
	logOnly := []AddedTestFile{{Path: "x_test.go", Content: src(`func TestX(t *testing.T) { t.Log("ok") }`)}}
	real := []AddedTestFile{{Path: "x_test.go", Content: src(`func TestX(t *testing.T) { if 1 != 2 { t.Fatal("x") } }`)}}

	// The gap this closes: today's CommitWitnessed clears the log-only commit because it
	// never reads the test body.
	if !CommitWitnessed("OK", WitnessOK) {
		t.Fatalf("precondition: CommitWitnessed(OK, %q) should be true", WitnessOK)
	}

	cases := []struct {
		name       string
		verdict    string
		witness    string
		tests      []AddedTestFile
		wantOK     bool
		wantReason string
	}{
		{"witnessed_but_test_cannot_fail", "OK", WitnessOK, logOnly, false, WitnessTestCannotFail},
		{"witnessed_and_real_assertion", "OK", WitnessOK, real, true, ""},
		{"claim_rung_already_failed_subject_only", "OK", "subject-only", logOnly, false, ""},
		{"claim_rung_already_failed_verdict", "CLAIM_UNWITNESSED", WitnessOK, logOnly, false, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ok, reason := CommitWitnessedWithIntegrity(tc.verdict, tc.witness, tc.tests)
			if ok != tc.wantOK || reason != tc.wantReason {
				t.Fatalf("CommitWitnessedWithIntegrity(%q,%q,...) = (%v,%q), want (%v,%q)",
					tc.verdict, tc.witness, ok, reason, tc.wantOK, tc.wantReason)
			}
		})
	}
}
