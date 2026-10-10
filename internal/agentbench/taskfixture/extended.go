package taskfixture

import (
	"fmt"
	"strings"
)

const (
	SuiteDefault  = "default"
	SuiteExtended = "extended"
)

// Suite selects a named fixture set. The default suite is exactly Cases; the
// extended suite appends ExtendedCases so model A/B runs reach at least 30 tasks.
func Suite(name string, includeHeldout bool) ([]Fixture, error) {
	switch strings.TrimSpace(name) {
	case "", SuiteDefault:
		return Cases(includeHeldout), nil
	case SuiteExtended:
		return append(Cases(includeHeldout), ExtendedCases()...), nil
	default:
		return nil, fmt.Errorf("taskfixture: unknown suite %q (want %s or %s)", name, SuiteDefault, SuiteExtended)
	}
}

// ByID finds a fixture in any suite, heldout included.
func ByID(id string) (Fixture, bool) {
	for _, fixture := range append(Cases(true), ExtendedCases()...) {
		if fixture.ID == id {
			return fixture, true
		}
	}
	return Fixture{}, false
}

type extendedFamily struct {
	family, target, spec, broken, fixed string
	tasks                               [3]extendedTask
}

type extendedTask struct{ id, visible, oracle string }

// ExtendedCases returns single-function repair tasks across nine small Go
// families. Within a family, task i uses workflow behavior_fix, tdd, then
// stale_reread, so every family exercises all three evidenced workflows.
func ExtendedCases() []Fixture {
	families := extendedFamilies()
	out := make([]Fixture, 0, 3*len(families))
	for _, family := range families {
		for index, task := range family.tasks {
			workflow := extendedWorkflow(index, family.family)
			out = append(out, Fixture{
				ID: task.id, Family: family.family, TargetFile: family.target, Workflow: workflow,
				Prompt:       "Specification: " + family.spec + "\n\n" + workflowPrompt(workflow),
				BrokenSource: family.broken, FixedSource: family.fixed,
				VisibleTest: task.visible, OracleTest: task.oracle,
			})
		}
	}
	return out
}

func extendedWorkflow(index int, area string) Workflow {
	switch index {
	case 1:
		return Workflow{Kind: "tdd", Area: area, RequiredSteps: []string{"inspect", "test_fail", "edit", "test_pass"}}
	case 2:
		return Workflow{Kind: "stale_reread", Area: area, RequiredSteps: []string{"inspect", "edit", "reread", "test_pass"}}
	default:
		return Workflow{Kind: "behavior_fix", Area: area, RequiredSteps: []string{"inspect", "edit", "test_pass"}}
	}
}

func goTest(imports, name, body string) string {
	header := "package fixture\n\nimport \"testing\"\n\n"
	if imports != "" {
		header = "package fixture\n\nimport (\n" + imports + "\t\"testing\"\n)\n\n"
	}
	return header + "func " + name + "(t *testing.T) {\n" + body + "}\n"
}

func extendedFamilies() []extendedFamily {
	return []extendedFamily{
		clampFamily(), dedupeFamily(), movingSumFamily(), mergeFamily(), wrapFamily(),
		clockFamily(), pathFamily(), ringFamily(), portFamily(),
	}
}

func clampFamily() extendedFamily {
	oracle := func(cases string) string {
		return goTest("", "TestOracle", "\tfor _, tc := range []struct{ v, lo, hi, want int }{"+cases+"} {\n\t\tif got := Clamp(tc.v, tc.lo, tc.hi); got != tc.want { t.Fatalf(\"Clamp(%d, %d, %d) = %d, want %d\", tc.v, tc.lo, tc.hi, got, tc.want) }\n\t}\n")
	}
	visible := func(call string, want int) string {
		return goTest("", "TestClamp", "\tif got := "+call+"; got != "+fmt.Sprint(want)+" { t.Fatalf(\"got %d\", got) }\n")
	}
	return extendedFamily{
		family: "clamp", target: "clamp.go",
		spec:   "Clamp(v, lo, hi) returns v limited to the inclusive range [lo, hi]: lo when v is below it, hi when v is above it, and v otherwise.",
		broken: "package fixture\n\nfunc Clamp(v, lo, hi int) int {\n\tif v < lo { return lo }\n\tif v > hi { return lo }\n\treturn v\n}\n",
		fixed:  "package fixture\n\nfunc Clamp(v, lo, hi int) int {\n\tif v < lo { return lo }\n\tif v > hi { return hi }\n\treturn v\n}\n",
		tasks: [3]extendedTask{
			{"clamp-above-range", visible("Clamp(15, 0, 10)", 10), oracle("{15, 0, 10, 10}, {-3, 0, 10, 0}, {5, 0, 10, 5}, {10, 0, 10, 10}, {0, 0, 10, 0}, {11, -5, -1, -1}")},
			{"clamp-negative-range", visible("Clamp(0, -10, -2)", -2), oracle("{0, -10, -2, -2}, {-20, -10, -2, -10}, {-5, -10, -2, -5}, {1000, -10, -2, -2}")},
			{"clamp-wide-range", visible("Clamp(250, 1, 100)", 100), oracle("{250, 1, 100, 100}, {101, 1, 100, 100}, {100, 1, 100, 100}, {1, 1, 100, 1}, {-7, 1, 100, 1}, {9, 4, 4, 4}")},
		},
	}
}

func dedupeFamily() extendedFamily {
	oracle := func(cases string) string {
		return goTest("\t\"fmt\"\n", "TestOracle", "\tfor _, tc := range []struct {\n\t\tin   []string\n\t\twant string\n\t}{"+cases+"} {\n\t\tif got := fmt.Sprint(Dedupe(tc.in)); got != tc.want { t.Fatalf(\"Dedupe(%q) = %s, want %s\", tc.in, got, tc.want) }\n\t}\n\tif got := Dedupe(nil); got == nil || len(got) != 0 { t.Fatalf(\"Dedupe(nil) = %#v, want empty non-nil\", got) }\n")
	}
	visible := func(in, want string) string {
		return goTest("\t\"fmt\"\n", "TestDedupe", "\tif got := fmt.Sprint(Dedupe([]string{"+in+"})); got != \""+want+"\" { t.Fatalf(\"got %s\", got) }\n")
	}
	return extendedFamily{
		family: "dedupe", target: "dedupe.go",
		spec:   "Dedupe(xs) returns the distinct strings of xs in first-occurrence order and returns an empty, non-nil slice for empty input.",
		broken: "package fixture\n\nfunc Dedupe(xs []string) []string {\n\tseen := map[string]bool{}\n\tout := []string{}\n\tfor _, x := range xs {\n\t\tif seen[x] { continue }\n\t\tout = append(out, x)\n\t}\n\treturn out\n}\n",
		fixed:  "package fixture\n\nfunc Dedupe(xs []string) []string {\n\tseen := map[string]bool{}\n\tout := []string{}\n\tfor _, x := range xs {\n\t\tif seen[x] { continue }\n\t\tseen[x] = true\n\t\tout = append(out, x)\n\t}\n\treturn out\n}\n",
		tasks: [3]extendedTask{
			{"dedupe-adjacent", visible(`"a", "a", "b"`, "[a b]"), oracle(`{[]string{"a", "a", "b"}, "[a b]"}, {[]string{"solo"}, "[solo]"}, {[]string{}, "[]"}, {[]string{"x", "y", "x", "z", "y"}, "[x y z]"}`)},
			{"dedupe-scattered", visible(`"x", "y", "x", "z", "y"`, "[x y z]"), oracle(`{[]string{"x", "y", "x", "z", "y"}, "[x y z]"}, {[]string{"", "", "a"}, "[ a]"}, {[]string{"q", "r"}, "[q r]"}`)},
			{"dedupe-case-sensitive", visible(`"Go", "go", "Go"`, "[Go go]"), oracle(`{[]string{"Go", "go", "Go", "GO"}, "[Go go GO]"}, {[]string{"b", "a", "b", "a"}, "[b a]"}, {[]string{"z"}, "[z]"}`)},
		},
	}
}

func movingSumFamily() extendedFamily {
	oracle := func(cases string) string {
		return goTest("\t\"fmt\"\n", "TestOracle", "\tfor _, tc := range []struct {\n\t\tin   []int\n\t\tk    int\n\t\twant string\n\t}{"+cases+"} {\n\t\tif got := fmt.Sprint(MovingSum(tc.in, tc.k)); got != tc.want { t.Fatalf(\"MovingSum(%v, %d) = %s, want %s\", tc.in, tc.k, got, tc.want) }\n\t}\n\tif got := MovingSum([]int{1}, 2); got != nil { t.Fatalf(\"k > len(xs) = %v, want nil\", got) }\n\tif got := MovingSum([]int{1, 2}, 0); got != nil { t.Fatalf(\"k == 0 = %v, want nil\", got) }\n")
	}
	visible := func(in string, k int, want string) string {
		return goTest("\t\"fmt\"\n", "TestMovingSum", "\tif got := fmt.Sprint(MovingSum([]int{"+in+"}, "+fmt.Sprint(k)+")); got != \""+want+"\" { t.Fatalf(\"got %s\", got) }\n")
	}
	return extendedFamily{
		family: "moving-sum", target: "window.go",
		spec:   "MovingSum(xs, k) returns the sum of every contiguous window of length k in order, which is len(xs)-k+1 sums; it returns nil when k <= 0 or k > len(xs).",
		broken: "package fixture\n\nfunc MovingSum(xs []int, k int) []int {\n\tif k <= 0 || k > len(xs) { return nil }\n\tout := make([]int, 0, len(xs)-k+1)\n\tfor i := 0; i+k < len(xs); i++ {\n\t\tsum := 0\n\t\tfor _, x := range xs[i : i+k] { sum += x }\n\t\tout = append(out, sum)\n\t}\n\treturn out\n}\n",
		fixed:  "package fixture\n\nfunc MovingSum(xs []int, k int) []int {\n\tif k <= 0 || k > len(xs) { return nil }\n\tout := make([]int, 0, len(xs)-k+1)\n\tfor i := 0; i+k <= len(xs); i++ {\n\t\tsum := 0\n\t\tfor _, x := range xs[i : i+k] { sum += x }\n\t\tout = append(out, sum)\n\t}\n\treturn out\n}\n",
		tasks: [3]extendedTask{
			{"window-last", visible("1, 2, 3, 4", 2, "[3 5 7]"), oracle(`{[]int{1, 2, 3, 4}, 2, "[3 5 7]"}, {[]int{1, 2, 3, 4}, 4, "[10]"}, {[]int{2, 2}, 3, "[]"}`)},
			{"window-full", visible("5, 6, 7", 3, "[18]"), oracle(`{[]int{5, 6, 7}, 3, "[18]"}, {[]int{5, 6, 7}, 2, "[11 13]"}, {[]int{-1, 1, -1, 1}, 2, "[0 0 0]"}`)},
			{"window-unit", visible("4, -1, 9", 1, "[4 -1 9]"), oracle(`{[]int{4, -1, 9}, 1, "[4 -1 9]"}, {[]int{3, 1, 4, 1, 5}, 3, "[8 6 10]"}, {[]int{7}, 1, "[7]"}`)},
		},
	}
}

func mergeFamily() extendedFamily {
	oracle := func(cases string) string {
		return goTest("\t\"fmt\"\n", "TestOracle", "\tfor _, tc := range []struct {\n\t\tbase, override map[string]int\n\t\twant           string\n\t}{"+cases+"} {\n\t\tbefore := fmt.Sprint(tc.base)\n\t\tif got := fmt.Sprint(Merge(tc.base, tc.override)); got != tc.want { t.Fatalf(\"Merge(%v, %v) = %s, want %s\", tc.base, tc.override, got, tc.want) }\n\t\tif fmt.Sprint(tc.base) != before { t.Fatalf(\"base mutated to %v\", tc.base) }\n\t}\n\tif got := Merge(nil, nil); got == nil || len(got) != 0 { t.Fatalf(\"Merge(nil, nil) = %#v, want empty map\", got) }\n")
	}
	visible := func(base, override, want string) string {
		return goTest("\t\"fmt\"\n", "TestMerge", "\tif got := fmt.Sprint(Merge(map[string]int{"+base+"}, map[string]int{"+override+"})); got != \""+want+"\" { t.Fatalf(\"got %s\", got) }\n")
	}
	return extendedFamily{
		family: "map-merge", target: "merge.go",
		spec:   "Merge(base, override) returns a new map holding every key of base and override; on a shared key the override value wins. Neither input map is modified, and nil inputs are treated as empty.",
		broken: "package fixture\n\nfunc Merge(base, override map[string]int) map[string]int {\n\tout := make(map[string]int, len(base)+len(override))\n\tfor k, v := range override { out[k] = v }\n\tfor k, v := range base { out[k] = v }\n\treturn out\n}\n",
		fixed:  "package fixture\n\nfunc Merge(base, override map[string]int) map[string]int {\n\tout := make(map[string]int, len(base)+len(override))\n\tfor k, v := range base { out[k] = v }\n\tfor k, v := range override { out[k] = v }\n\treturn out\n}\n",
		tasks: [3]extendedTask{
			{"merge-override-wins", visible(`"a": 1, "b": 2`, `"b": 20`, "map[a:1 b:20]"), oracle(`{map[string]int{"a": 1, "b": 2}, map[string]int{"b": 20}, "map[a:1 b:20]"}, {nil, map[string]int{"k": 1}, "map[k:1]"}, {map[string]int{"k": 1}, nil, "map[k:1]"}`)},
			{"merge-shared-and-new", visible(`"x": 1`, `"x": 5, "y": 6`, "map[x:5 y:6]"), oracle(`{map[string]int{"x": 1}, map[string]int{"x": 5, "y": 6}, "map[x:5 y:6]"}, {map[string]int{"p": 1, "q": 2}, map[string]int{"q": 3, "r": 4}, "map[p:1 q:3 r:4]"}`)},
			{"merge-zero-override", visible(`"n": 7`, `"n": 0`, "map[n:0]"), oracle(`{map[string]int{"n": 7}, map[string]int{"n": 0}, "map[n:0]"}, {map[string]int{"n": 7, "m": -1}, map[string]int{"m": 0}, "map[m:0 n:7]"}`)},
		},
	}
}

func wrapFamily() extendedFamily {
	oracle := func(op string) string {
		return "package fixture\n\nimport (\n\t\"errors\"\n\t\"testing\"\n)\n\nfunc TestOracle(t *testing.T) {\n" + "\tif Wrap(\"" + op + "\", nil) != nil { t.Fatal(\"nil error must stay nil\") }\n\terr := Wrap(\"" + op + "\", ErrNotFound)\n\tif err == nil || err.Error() != \"" + op + ": not found\" { t.Fatalf(\"message = %v\", err) }\n\tif !errors.Is(err, ErrNotFound) { t.Fatal(\"errors.Is lost the cause\") }\n\tif !errors.Is(Wrap(\"outer\", err), ErrNotFound) { t.Fatal(\"nested wrap lost the cause\") }\n\tcause := &oracleCodeError{code: 7}\n\tvar target *oracleCodeError\n\tif !errors.As(Wrap(\"" + op + "\", cause), &target) || target.code != 7 { t.Fatal(\"errors.As lost the typed cause\") }\n}\n\ntype oracleCodeError struct{ code int }\n\nfunc (e *oracleCodeError) Error() string { return \"code error\" }\n"
	}
	return extendedFamily{
		family: "error-wrap", target: "wrap.go",
		spec:   "Wrap(op, err) returns nil for a nil err; otherwise it returns an error whose message is \"<op>: <err>\" and which still matches err under errors.Is, errors.As and errors.Unwrap.",
		broken: "package fixture\n\nimport (\n\t\"errors\"\n\t\"fmt\"\n)\n\nvar ErrNotFound = errors.New(\"not found\")\n\nfunc Wrap(op string, err error) error {\n\tif err == nil { return nil }\n\treturn fmt.Errorf(\"%s: %v\", op, err)\n}\n",
		fixed:  "package fixture\n\nimport (\n\t\"errors\"\n\t\"fmt\"\n)\n\nvar ErrNotFound = errors.New(\"not found\")\n\nfunc Wrap(op string, err error) error {\n\tif err == nil { return nil }\n\treturn fmt.Errorf(\"%s: %w\", op, err)\n}\n",
		tasks: [3]extendedTask{
			{"wrap-is", goTest("\t\"errors\"\n", "TestWrap", "\tif !errors.Is(Wrap(\"read\", ErrNotFound), ErrNotFound) { t.Fatal(\"cause lost\") }\n"), oracle("read")},
			{"wrap-nested", goTest("\t\"errors\"\n", "TestWrap", "\tif !errors.Is(Wrap(\"load\", Wrap(\"read\", ErrNotFound)), ErrNotFound) { t.Fatal(\"cause lost\") }\n"), oracle("load")},
			{"wrap-unwrap", goTest("\t\"errors\"\n", "TestWrap", "\tif got := errors.Unwrap(Wrap(\"open\", ErrNotFound)); got != ErrNotFound { t.Fatalf(\"unwrap = %v\", got) }\n"), oracle("open")},
		},
	}
}

func clockFamily() extendedFamily {
	oracle := func(cases string) string {
		return goTest("", "TestOracle", "\tfor _, tc := range []struct {\n\t\tin   int\n\t\twant string\n\t}{"+cases+"} {\n\t\tif got := Clock(tc.in); got != tc.want { t.Fatalf(\"Clock(%d) = %q, want %q\", tc.in, got, tc.want) }\n\t}\n")
	}
	visible := func(in int, want string) string {
		return goTest("", "TestClock", "\tif got := Clock("+fmt.Sprint(in)+"); got != \""+want+"\" { t.Fatalf(\"got %q\", got) }\n")
	}
	return extendedFamily{
		family: "clock-format", target: "clock.go",
		spec:   "Clock(seconds) formats a duration as M:SS below one hour and H:MM:SS from one hour up; every field after the leading one is zero-padded to two digits, and negative input is treated as 0.",
		broken: "package fixture\n\nimport \"fmt\"\n\nfunc Clock(seconds int) string {\n\tif seconds < 0 { seconds = 0 }\n\th, m, s := seconds/3600, seconds/60%60, seconds%60\n\tif h > 0 { return fmt.Sprintf(\"%d:%02d:%d\", h, m, s) }\n\treturn fmt.Sprintf(\"%d:%d\", m, s)\n}\n",
		fixed:  "package fixture\n\nimport \"fmt\"\n\nfunc Clock(seconds int) string {\n\tif seconds < 0 { seconds = 0 }\n\th, m, s := seconds/3600, seconds/60%60, seconds%60\n\tif h > 0 { return fmt.Sprintf(\"%d:%02d:%02d\", h, m, s) }\n\treturn fmt.Sprintf(\"%d:%02d\", m, s)\n}\n",
		tasks: [3]extendedTask{
			{"clock-pad-seconds", visible(65, "1:05"), oracle(`{65, "1:05"}, {0, "0:00"}, {59, "0:59"}, {600, "10:00"}, {-5, "0:00"}`)},
			{"clock-hours", visible(3661, "1:01:01"), oracle(`{3661, "1:01:01"}, {3600, "1:00:00"}, {3605, "1:00:05"}, {36000, "10:00:00"}, {3599, "59:59"}`)},
			{"clock-under-ten", visible(7, "0:07"), oracle(`{7, "0:07"}, {70, "1:10"}, {7209, "2:00:09"}, {-1, "0:00"}`)},
		},
	}
}

func pathFamily() extendedFamily {
	oracle := func(cases string) string {
		return goTest("", "TestOracle", "\tfor _, tc := range []struct{ in, want string }{"+cases+"} {\n\t\tif got := NormalizePath(tc.in); got != tc.want { t.Fatalf(\"NormalizePath(%q) = %q, want %q\", tc.in, got, tc.want) }\n\t}\n")
	}
	visible := func(in, want string) string {
		return goTest("", "TestNormalizePath", "\tif got := NormalizePath(\""+in+"\"); got != \""+want+"\" { t.Fatalf(\"got %q\", got) }\n")
	}
	return extendedFamily{
		family: "path-normalize", target: "paths.go",
		spec:   "NormalizePath(p) returns the cleaned absolute slash path for p: it always starts with \"/\", collapses duplicate slashes and dot segments, never climbs above the root, and has no trailing slash except for the root itself.",
		broken: "package fixture\n\nimport \"path\"\n\nfunc NormalizePath(p string) string {\n\treturn path.Clean(p)\n}\n",
		fixed:  "package fixture\n\nimport \"path\"\n\nfunc NormalizePath(p string) string {\n\treturn path.Clean(\"/\" + p)\n}\n",
		tasks: [3]extendedTask{
			{"path-relative", visible("a/b/../c/", "/a/c"), oracle(`{"a/b/../c/", "/a/c"}, {"/x//y/", "/x/y"}, {"a", "/a"}, {"/", "/"}`)},
			{"path-empty", visible("", "/"), oracle(`{"", "/"}, {"./", "/"}, {".", "/"}, {"/srv/", "/srv"}`)},
			{"path-climb", visible("../etc/./hosts", "/etc/hosts"), oracle(`{"../etc/./hosts", "/etc/hosts"}, {"../../z", "/z"}, {"a/./b/..", "/a"}, {"/../", "/"}`)},
		},
	}
}

func ringFamily() extendedFamily {
	oracle := func(cases string) string {
		return goTest("\t\"fmt\"\n", "TestOracle", "\tfor _, tc := range []struct {\n\t\tbuf      []int\n\t\tv, limit int\n\t\twant     string\n\t}{"+cases+"} {\n\t\tif got := fmt.Sprint(Push(tc.buf, tc.v, tc.limit)); got != tc.want { t.Fatalf(\"Push(%v, %d, %d) = %s, want %s\", tc.buf, tc.v, tc.limit, got, tc.want) }\n\t}\n\tif got := Push([]int{1}, 2, 0); got != nil { t.Fatalf(\"limit 0 = %v, want nil\", got) }\n")
	}
	visible := func(buf string, v, limit int, want string) string {
		return goTest("\t\"fmt\"\n", "TestPush", "\tif got := fmt.Sprint(Push([]int{"+buf+"}, "+fmt.Sprint(v)+", "+fmt.Sprint(limit)+")); got != \""+want+"\" { t.Fatalf(\"got %s\", got) }\n")
	}
	return extendedFamily{
		family: "bounded-buffer", target: "ring.go",
		spec:   "Push(buf, v, limit) appends v to buf and, when the result is longer than limit, keeps only the newest limit items by dropping the oldest; it returns nil when limit <= 0.",
		broken: "package fixture\n\nfunc Push(buf []int, v, limit int) []int {\n\tif limit <= 0 { return nil }\n\tbuf = append(buf, v)\n\tif len(buf) > limit { buf = buf[:limit] }\n\treturn buf\n}\n",
		fixed:  "package fixture\n\nfunc Push(buf []int, v, limit int) []int {\n\tif limit <= 0 { return nil }\n\tbuf = append(buf, v)\n\tif len(buf) > limit { buf = buf[len(buf)-limit:] }\n\treturn buf\n}\n",
		tasks: [3]extendedTask{
			{"ring-drop-oldest", visible("1, 2, 3", 4, 3, "[2 3 4]"), oracle(`{[]int{1, 2, 3}, 4, 3, "[2 3 4]"}, {[]int{1, 2}, 3, 5, "[1 2 3]"}, {nil, 9, 2, "[9]"}`)},
			{"ring-limit-one", visible("8", 9, 1, "[9]"), oracle(`{[]int{8}, 9, 1, "[9]"}, {[]int{}, 4, 1, "[4]"}, {[]int{5, 6, 7}, 8, 1, "[8]"}`)},
			{"ring-overfull", visible("1, 2, 3, 4", 5, 2, "[4 5]"), oracle(`{[]int{1, 2, 3, 4}, 5, 2, "[4 5]"}, {[]int{1, 2, 3}, 4, 4, "[1 2 3 4]"}, {[]int{0, 0, 1}, 2, 3, "[0 1 2]"}`)},
		},
	}
}

func portFamily() extendedFamily {
	oracle := func(extra string) string {
		return goTest("", "TestOracle", "\tfor _, tc := range []struct {\n\t\tin   string\n\t\twant int\n\t\tok   bool\n\t}{{\"0\", 0, false}, {\"1\", 1, true}, {\"65535\", 65535, true}, {\"65536\", 0, false}, {\"-1\", 0, false}, {\"http\", 0, false}, {\"\", 0, false}, {\"8080\", 8080, true}, "+extra+"} {\n\t\tif got, ok := ParsePort(tc.in); got != tc.want || ok != tc.ok { t.Fatalf(\"ParsePort(%q) = %d, %t; want %d, %t\", tc.in, got, ok, tc.want, tc.ok) }\n\t}\n")
	}
	visible := func(in string) string {
		return goTest("", "TestParsePort", "\tif got, ok := ParsePort(\""+in+"\"); ok || got != 0 { t.Fatalf(\"ParsePort(%q) = %d, %t; want rejection\", \""+in+"\", got, ok) }\n")
	}
	return extendedFamily{
		family: "port-parse", target: "port.go",
		spec:   "ParsePort(s) accepts a decimal TCP port in the range 1..65535 and returns (port, true); anything else, including 0, negative values, values above 65535 and non-numeric text, returns (0, false).",
		broken: "package fixture\n\nimport \"strconv\"\n\nfunc ParsePort(s string) (int, bool) {\n\tn, err := strconv.Atoi(s)\n\tif err != nil || n < 0 || n > 65535 { return 0, false }\n\treturn n, true\n}\n",
		fixed:  "package fixture\n\nimport \"strconv\"\n\nfunc ParsePort(s string) (int, bool) {\n\tn, err := strconv.Atoi(s)\n\tif err != nil || n < 1 || n > 65535 { return 0, false }\n\treturn n, true\n}\n",
		tasks: [3]extendedTask{
			{"port-zero", visible("0"), oracle(`{"443", 443, true}`)},
			{"port-zero-padded", visible("00"), oracle(`{"00", 0, false}, {"080", 80, true}`)},
			{"port-negative-zero", visible("-0"), oracle(`{"-0", 0, false}, {"22", 22, true}`)},
		},
	}
}
