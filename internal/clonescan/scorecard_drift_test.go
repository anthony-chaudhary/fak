package clonescan

// The clone definition lives twice by construction: here (the authoring-time query)
// and in tools/code_slop_scorecard.py kpi_duplication (the batch CI grade). The whole
// value of the early query is that driving its warnings to zero drives the scorecard's
// dup_extractable to zero BY CONSTRUCTION — an invariant that holds only while the two
// engines agree on what a clone is. This file is the drift guard (#2527): it reads the
// Python source and fails if any shared constant or token table diverges, and it runs
// a fixture through BOTH live engines and fails if their verdicts or token streams
// disagree. A change to either side must land with the matching change to the other.

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

// scorecardRelPath is where the batch half of the definition lives, relative to this
// package directory (the go test working directory).
var scorecardRelPath = filepath.Join("..", "..", "tools", "code_slop_scorecard.py")

func TestSharedCloneDefinitionMatchesScorecard(t *testing.T) {
	src, err := os.ReadFile(scorecardRelPath)
	if err != nil {
		// Fail, not skip: if the scorecard moved or was renamed, this guard must be
		// re-pointed in the same change, or the pairing is silently unenforced.
		t.Fatalf("cannot read the scorecard half of the clone definition: %v", err)
	}
	py := string(src)

	t.Run("window constants", func(t *testing.T) {
		checkInt(t, py, "CLONE_WINDOW_TOKENS", WindowTokens)
		checkInt(t, py, "CLONE_MIN_LOGIC_TOKENS", MinLogicTokens)
		checkInt(t, py, "CLONE_MIN_OCCURRENCES", MinOccurrences)
	})

	t.Run("token tables", func(t *testing.T) {
		checkSet(t, py, "_GO_KEYWORDS", goKeywords)
		checkSet(t, py, "_LOGIC_KEYWORDS", logicKeywords)
		checkSet(t, py, "_LOGIC_OPS", logicOps)
		checkSet(t, py, "_ASSIGN_OPS", assignOps)

		// _GO_OPS is compared as an ORDERED sequence, not a set: both lexers match
		// greedily longest-first, so reordering alone changes tokenization.
		pyOps := pyStrings(t, py, "_GO_OPS")
		if len(pyOps) != len(goOps) {
			t.Fatalf("op table length drift: Go goOps has %d entries, Python _GO_OPS has %d — these must move together (#2527)",
				len(goOps), len(pyOps))
		}
		for i := range goOps {
			if goOps[i] != pyOps[i] {
				t.Errorf("op table drift at index %d: Go goOps=%q, Python _GO_OPS=%q — greedy longest-first order is semantic; these must move together (#2527)",
					i, goOps[i], pyOps[i])
			}
		}
	})

	t.Run("cross-engine fixture", func(t *testing.T) {
		crossEngineFixture(t)
	})
}

// checkInt asserts a `NAME = <int>` module-level assignment in the Python source
// equals the Go constant of the same meaning.
func checkInt(t *testing.T, py, name string, goVal int) {
	t.Helper()
	re := regexp.MustCompile(`(?m)^` + regexp.QuoteMeta(name) + ` = (\d+)\b`)
	m := re.FindStringSubmatch(py)
	if m == nil {
		t.Fatalf("constant %s not found in %s — if it was renamed, re-point this guard in the same change (#2527)",
			name, scorecardRelPath)
	}
	pyVal, err := strconv.Atoi(m[1])
	if err != nil {
		t.Fatalf("constant %s: %v", name, err)
	}
	if pyVal != goVal {
		t.Errorf("constant drift: Go %d vs Python %s = %d — \"warned at write time\" and \"counted at CI\" no longer agree; these must move together (#2527)",
			goVal, name, pyVal)
	}
}

// checkSet asserts a Python string-collection assignment holds exactly the keys of
// the Go table of the same meaning.
func checkSet(t *testing.T, py, name string, goTable map[string]bool) {
	t.Helper()
	pyList := pyStrings(t, py, name)
	pySet := make(map[string]bool, len(pyList))
	for _, s := range pyList {
		if pySet[s] {
			t.Errorf("Python %s holds %q twice — malformed table edit", name, s)
		}
		pySet[s] = true
	}
	var missing, extra []string
	for s := range goTable {
		if !pySet[s] {
			missing = append(missing, s)
		}
	}
	for s := range pySet {
		if !goTable[s] {
			extra = append(extra, s)
		}
	}
	sort.Strings(missing)
	sort.Strings(extra)
	if len(missing) > 0 || len(extra) > 0 {
		t.Errorf("table drift in %s: in Go but not Python %q; in Python but not Go %q — these must move together (#2527)",
			name, missing, extra)
	}
}

// pyStrings extracts, in order, every string literal inside the bracketed value of
// the first module-level `NAME = ` assignment. It is a narrow scanner for the
// literal frozenset/tuple tables the scorecard declares: it tracks bracket depth
// only OUTSIDE string literals (the op table itself contains "(" and "{"), skips
// `#` comments, and fails closed on anything it cannot read.
func pyStrings(t *testing.T, py, name string) []string {
	t.Helper()
	re := regexp.MustCompile(`(?m)^` + regexp.QuoteMeta(name) + ` = `)
	loc := re.FindStringIndex(py)
	if loc == nil {
		t.Fatalf("table %s not found in %s — if it was renamed, re-point this guard in the same change (#2527)",
			name, scorecardRelPath)
	}
	var out []string
	depth := 0
	i := loc[1]
	for i < len(py) {
		c := py[i]
		switch {
		case c == '#':
			j := strings.IndexByte(py[i:], '\n')
			if j < 0 {
				i = len(py)
			} else {
				i += j
			}
		case c == '"' || c == '\'':
			lit, next, ok := pyStringLit(py, i)
			if !ok {
				t.Fatalf("table %s: unreadable string literal at byte %d of %s", name, i, scorecardRelPath)
			}
			out = append(out, lit)
			i = next
		case c == '(' || c == '{' || c == '[':
			depth++
			i++
		case c == ')' || c == '}' || c == ']':
			depth--
			i++
			if depth == 0 {
				if len(out) == 0 {
					t.Fatalf("table %s parsed empty — the scanner no longer reads the scorecard's shape; fix the guard, do not delete it (#2527)", name)
				}
				return out
			}
		case c == '\n' && depth == 0:
			t.Fatalf("table %s: value is not a bracketed literal — the scorecard's shape changed; re-teach the guard (#2527)", name)
		default:
			i++
		}
	}
	t.Fatalf("table %s: brackets never closed — truncated read of %s", name, scorecardRelPath)
	return nil
}

// pyStringLit reads one single- or double-quoted Python string literal starting at
// py[at], returning its decoded value and the index just past the closing quote.
// The tables hold plain ASCII operator/keyword text, so only backslash pass-through
// is handled; a literal this scanner cannot finish reports !ok.
func pyStringLit(py string, at int) (lit string, next int, ok bool) {
	q := py[at]
	var b strings.Builder
	i := at + 1
	for i < len(py) {
		c := py[i]
		if c == '\\' && i+1 < len(py) {
			b.WriteByte(py[i+1])
			i += 2
			continue
		}
		if c == q {
			return b.String(), i + 1, true
		}
		if c == '\n' {
			return "", 0, false
		}
		b.WriteByte(c)
		i++
	}
	return "", 0, false
}

// fixtureCase is one shared clone/not-clone verdict both engines must produce. Cases
// stay inside the region where the engines agree BY DESIGN: the batch scorecard
// layers extra precision gates (sort-scaffold, flag-plumbing, dispatch-arm, sub-6-line
// span) the query deliberately lacks, so positives span >=6 real logic lines and avoid
// those shapes, and negatives are negative at the window level itself.
type fixtureCase struct {
	name  string
	a, b  string // two tracked files' source text
	clone bool
}

// fixtureBody is a real multi-line logic block: >34 normalized tokens, >6 source
// lines, control flow + computation, no sort/flag/case-arm shapes.
const fixtureBody = `
func %s(items []int) int {
	total := 0
	for i := 0; i < len(items); i++ {
		if items[i] > 0 {
			total += items[i] * 2
		} else {
			total -= items[i]
		}
	}
	return total
}
`

func fixtureCases() []fixtureCase {
	alpha := "package a\n" + strings.Replace(fixtureBody, "%s", "alpha", 1)
	beta := "package b\n" + strings.Replace(fixtureBody, "%s", "beta", 1)
	// Same token stream as fixtureBody, different line breaks and comments: the
	// definition is a normalized token window, so formatting must not matter.
	reflowed := `package b

// reformatted copy — comments and line breaks differ, tokens do not
func beta(items []int) int {
	total := 0
	for i := 0; i < len(items); i++ { // walk
		if items[i] > 0 { total += items[i] * 2 } else { total -= items[i] }
	}
	return total
}
`
	// Every identifier renamed: identifiers are kept verbatim (normalize_idents=False
	// on both sides), so a rename changes the window identity by design.
	renamed := `package b
func gamma(vals []int) int {
	acc := 0
	for k := 0; k < len(vals); k++ {
		if vals[k] > 0 {
			acc += vals[k] * 2
		} else {
			acc -= vals[k]
		}
	}
	return acc
}
`
	// Token-identical in both files and well over one window long, but its only logic
	// tokens are bare assignments — the context gate zeroes them, so no window
	// qualifies and an identical declaration run is data, not a clone.
	decls := `package a
var (
	name00 = "v00"
	name01 = "v01"
	name02 = "v02"
	name03 = "v03"
	name04 = "v04"
	name05 = "v05"
	name06 = "v06"
	name07 = "v07"
	name08 = "v08"
	name09 = "v09"
	name10 = "v10"
	name11 = "v11"
)
`
	// Identical real logic, but the whole file is under one 34-token window.
	tiny := `package a
func tiny(x int) int {
	if x > 0 {
		return x * 2
	}
	return -x
}
`
	return []fixtureCase{
		{name: "copied logic block", a: alpha, b: beta, clone: true},
		{name: "reformatted copy still matches", a: alpha, b: reflowed, clone: true},
		{name: "renamed identifiers do not match", a: alpha, b: renamed, clone: false},
		{name: "identical declaration run is data not logic", a: decls, b: strings.Replace(decls, "package a", "package b", 1), clone: false},
		{name: "identical fragment under one window", a: tiny, b: strings.Replace(tiny, "package a", "package b", 1), clone: false},
	}
}

// pyDriver runs the fixture through the REAL Python engine: kpi_duplication for the
// clone verdict and go_tokens for the normalized stream, so tokenizer-code drift is
// caught behaviorally even when the tables still match. sum(subcategories) is the
// surviving HARD group count (score and detail are derived from it).
const pyDriver = `
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("scorecard", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
req = json.load(sys.stdin)
out = {}
for c in req["cases"]:
    files = {"a.go": c["a"], "b.go": c["b"]}
    r = mod.kpi_duplication(files)
    out[c["name"]] = {
        "clone": sum(r["subcategories"].values()) > 0,
        "tokens": {rel: [[s, l, bool(g)] for (s, l, g) in
                         mod.go_tokens(files[rel], normalize_idents=False)]
                   for rel in files},
    }
print(json.dumps(out))
`

type pyCaseResult struct {
	Clone  bool                `json:"clone"`
	Tokens map[string][][3]any `json:"tokens"`
}

func crossEngineFixture(t *testing.T) {
	t.Helper()
	interp := findPython(t)
	if interp == "" {
		// The source-level guards above still enforce the tables everywhere; CI has
		// python3 (the Makefile's python gates prove it), so the behavioral half is
		// enforced where the scorecard actually runs.
		t.Skip("no python3/python on PATH; cross-engine fixture runs in CI")
	}
	scorecardAbs, err := filepath.Abs(scorecardRelPath)
	if err != nil {
		t.Fatalf("abs: %v", err)
	}

	cases := fixtureCases()
	req := struct {
		Cases []map[string]string `json:"cases"`
	}{}
	for _, c := range cases {
		req.Cases = append(req.Cases, map[string]string{"name": c.name, "a": c.a, "b": c.b})
	}
	reqJSON, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, interp, "-c", pyDriver, scorecardAbs)
	cmd.Stdin = strings.NewReader(string(reqJSON))
	outBytes, err := cmd.Output()
	if err != nil {
		stderr := ""
		if ee, ok := err.(*exec.ExitError); ok {
			stderr = string(ee.Stderr)
		}
		t.Fatalf("python engine failed: %v\n%s", err, stderr)
	}
	var pyRes map[string]pyCaseResult
	if err := json.Unmarshal(outBytes, &pyRes); err != nil {
		t.Fatalf("unmarshal python result: %v\nraw: %s", err, outBytes)
	}

	for _, c := range cases {
		res, found := pyRes[c.name]
		if !found {
			t.Fatalf("case %q: no python result", c.name)
		}
		goClone := len(Query(c.a, map[string]string{"b.go": c.b}, "", 0)) > 0
		if goClone != c.clone {
			t.Errorf("case %q: Go engine says clone=%v, fixture expects %v", c.name, goClone, c.clone)
		}
		if res.Clone != c.clone {
			t.Errorf("case %q: Python engine says clone=%v, fixture expects %v — the engines have drifted (#2527)",
				c.name, res.Clone, c.clone)
		}
		for rel, text := range map[string]string{"a.go": c.a, "b.go": c.b} {
			compareStreams(t, c.name, rel, goTokens(text, false), res.Tokens[rel])
		}
	}
}

// compareStreams asserts the Go and Python tokenizers emitted the identical
// (sym, line, is_logic) stream for one fixture file.
func compareStreams(t *testing.T, caseName, rel string, goToks []token, pyToks [][3]any) {
	t.Helper()
	if len(goToks) != len(pyToks) {
		t.Errorf("case %q %s: token stream length drift: Go %d vs Python %d (#2527)",
			caseName, rel, len(goToks), len(pyToks))
		return
	}
	for i, gt := range goToks {
		sym, _ := pyToks[i][0].(string)
		lineF, _ := pyToks[i][1].(float64)
		logic, _ := pyToks[i][2].(bool)
		if gt.sym != sym || gt.line != int(lineF) || gt.isLogic != logic {
			t.Errorf("case %q %s: token %d drift: Go (%q line=%d logic=%v) vs Python (%q line=%d logic=%v) (#2527)",
				caseName, rel, i, gt.sym, gt.line, gt.isLogic, sym, int(lineF), logic)
			return
		}
	}
}

// findPython returns a runnable Python 3 interpreter, or "" when none is on PATH.
// python3 is probed before python, and each candidate must actually run --version
// (a Windows Store alias resolves on PATH but cannot execute).
func findPython(t *testing.T) string {
	t.Helper()
	for _, name := range []string{"python3", "python"} {
		path, err := exec.LookPath(name)
		if err != nil {
			continue
		}
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		err = exec.CommandContext(ctx, path, "--version").Run()
		cancel()
		if err == nil {
			return path
		}
	}
	return ""
}
