package witness

import (
	"strings"
	"testing"
)

func TestSymptomFailureExcerpt(t *testing.T) {
	ev := func(action, test, output string) string {
		out := `{"Action":"` + action + `","Package":"example.com/p"`
		if test != "" {
			out += `,"Test":"` + test + `"`
		}
		if output != "" {
			out += `,"Output":` + jsonQuote(output)
		}
		return out + "}"
	}
	join := func(lines ...string) string { return strings.Join(lines, "\n") }
	long := strings.Repeat("x", 300)

	cases := []struct {
		name string
		out  string
		want string
	}{
		{"empty", "", ""},
		{"all pass", join(
			ev("run", "TestA", ""),
			ev("output", "TestA", "=== RUN   TestA\n"),
			ev("pass", "TestA", ""),
			ev("pass", "", ""),
		), ""},
		{"first failing test with assertion line", join(
			ev("run", "TestA", ""),
			ev("output", "TestA", "=== RUN   TestA\n"),
			ev("output", "TestA", "    calc_test.go:5: got 2 want 3\n"),
			ev("output", "TestA", "    calc_test.go:6: second\n"),
			ev("output", "TestA", "--- FAIL: TestA (0.00s)\n"),
			ev("fail", "TestA", ""),
			ev("output", "TestB", "    b_test.go:9: other\n"),
			ev("fail", "TestB", ""),
			ev("output", "", "FAIL\n"),
			ev("fail", "", ""),
		), "FAIL TestA: calc_test.go:5: got 2 want 3"},
		{"failing test with only noise output", join(
			ev("output", "TestA", "=== RUN   TestA\n"),
			ev("output", "TestA", "=== PAUSE TestA\n"),
			ev("output", "TestA", "=== CONT  TestA\n"),
			ev("output", "TestA", "--- FAIL: TestA (0.00s)\n"),
			ev("output", "TestA", "FAIL\n"),
			ev("fail", "TestA", ""),
		), "FAIL TestA"},
		{"output of a later test is not attributed", join(
			ev("output", "TestB", "    b_test.go:9: other\n"),
			ev("fail", "TestA", ""),
			ev("fail", "TestB", ""),
		), "FAIL TestA"},
		{"whitespace collapsed", join(
			ev("output", "TestA", "\t got   2\t\twant 3  \n"),
			ev("fail", "TestA", ""),
		), "FAIL TestA: got 2 want 3"},
		{"compile error with column in non-JSON line", join(
			"# example.com/p [example.com/p.test]",
			"./calc_test.go:12:3: undefined: Calc",
			ev("output", "", "FAIL\texample.com/p [build failed]\n"),
		), "./calc_test.go:12:3: undefined: Calc"},
		{"compile error without column", join(
			"noise line",
			"p/x.go:7: syntax error",
		), "p/x.go:7: syntax error"},
		{"package-level build failed output", join(
			ev("start", "", ""),
			ev("output", "", "FAIL\texample.com/p [build failed]\n"),
			ev("fail", "", ""),
		), "FAIL example.com/p [build failed]"},
		{"setup failed", "FAIL\texample.com/p [setup failed]", "FAIL example.com/p [setup failed]"},
		{"panic line", join(
			"some preamble",
			"panic: runtime error: index out of range",
			"goroutine 1 [running]:",
		), "panic: runtime error: index out of range"},
		{"test-scoped compile-looking output ignored without fail", join(
			ev("output", "TestA", "a.go:1:1: looks like compile\n"),
			ev("pass", "TestA", ""),
		), ""},
		{"unrelated loose lines", join("hello", "world"), ""},
		{"capped at 240 plus ellipsis", "panic: " + long, ("panic: " + long)[:240] + "..."},
		{"exactly 240 not capped", "panic: " + long[:233], "panic: " + long[:233]},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := symptomFailureExcerpt(tc.out); got != tc.want {
				t.Fatalf("excerpt = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestSymptomFailureExcerptDefaultEmpty(t *testing.T) {
	if got := NewWithRunner(nil, t.TempDir()).SymptomFailureExcerpt(); got != "" {
		t.Fatalf("default excerpt = %q, want empty", got)
	}
}

func jsonQuote(s string) string {
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`, "\t", `\t`)
	return `"` + r.Replace(s) + `"`
}
