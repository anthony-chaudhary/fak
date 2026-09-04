package hooks

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/token"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// gate_pythongate.go — the whole-tree gate that catches a NEW tools/*.py before it reaches
// the shared trunk. internal/pythongate's TestNoNewPythonTools already reds CI when a tracked
// tools/*.py is not in the grandfathered de-Python baseline, but that test only fires AFTER
// the offending commit is on the trunk and `go test ./...` runs — so on a hot, many-session
// trunk the ratchet reds the release-gating fast subset (ci-fast.yml) minutes after the push,
// and holds the release cadence on CI_BASE_RED until a human notices. This gate surfaces the
// same NEW_PYTHON_TOOL gap one boundary earlier, in `fak hygiene` (the pre-push --audit-tree
// backstop), so a contributor sees it BEFORE the trunk goes red (epic #2653).
//
// It is the pythongate twin of gate_tierdeclared.go: like that gate, it does NOT import its
// tier-2 scorecard package (an upward import a tier-1 hooks package may not make). It parses
// the baseline plus exact reviewed test-companion declarations from the tracked
// tree and compares them to the tracked tools/*.py set, so it can never become a
// rival authority to the ratchet it fronts.

// pythonBaselineFile is the single source of truth for the de-Python grandfathered baseline.
const pythonBaselineFile = "internal/pythongate/baseline.go"

const pythonTestCompanionFile = "internal/pythongate/testcompanions.go"

// reasonNewPythonTool mirrors pythongate.ReasonNewPythonTool without importing the tier-2
// package (kept as a literal so the tier-1 gate stays import-clean; the test asserts they agree).
const reasonNewPythonTool = "NEW_PYTHON_TOOL"

// pyBaselineEntryRE matches a `"tools/....py"` string literal — one grandfathered path per
// match. Anchored to the tools/ prefix + .py suffix so a stray quoted string in a comment or
// the regenerate recipe in the file's doc block is not miscounted as a baseline entry.
var pyBaselineEntryRE = regexp.MustCompile(`"(tools/[^"]+\.py)"`)

// declaredPyBaseline parses the grandfathered tools/*.py paths out of baseline.go read from
// the tracked tree. ok is false when the file cannot be read or holds no entries (the gate
// then fails open via ErrCouldNotRun, never a false NEW_PYTHON_TOOL on an unreadable source).
func declaredPyBaseline(t *TrackedTree) (map[string]bool, bool) {
	body, exists := t.FileBytes(pythonBaselineFile)
	if !exists {
		return nil, false
	}
	declared := map[string]bool{}
	inBlock := false
	for _, line := range strings.Split(string(body), "\n") {
		if !inBlock {
			if strings.Contains(line, "var grandfathered = []string{") {
				inBlock = true
			}
			continue
		}
		// The slice literal closes at the first line that is just `}`.
		if strings.TrimSpace(line) == "}" {
			break
		}
		for _, m := range pyBaselineEntryRE.FindAllStringSubmatch(line, -1) {
			declared[m[1]] = true
		}
	}
	if len(declared) == 0 {
		return nil, false // the marker moved or the file shape changed — fail open
	}
	return declared, true
}

func declaredPyTestCompanions(t *TrackedTree, baseline map[string]bool) map[string]bool {
	body, exists := t.FileBytes(pythonTestCompanionFile)
	if !exists {
		return nil
	}
	file, err := parser.ParseFile(token.NewFileSet(), pythonTestCompanionFile, body, 0)
	if err != nil {
		return nil
	}
	allowed := map[string]bool{}
	ast.Inspect(file, func(node ast.Node) bool {
		decl, ok := node.(*ast.ValueSpec)
		if !ok || len(decl.Names) != 1 || decl.Names[0].Name != "testCompanions" || len(decl.Values) != 1 {
			return true
		}
		list, ok := decl.Values[0].(*ast.CompositeLit)
		if !ok {
			return false
		}
		for _, element := range list.Elts {
			row, ok := element.(*ast.CompositeLit)
			if !ok || len(row.Elts) < 2 {
				continue
			}
			pathLit, pathOK := row.Elts[0].(*ast.BasicLit)
			moduleLit, moduleOK := row.Elts[1].(*ast.BasicLit)
			if !pathOK || !moduleOK || pathLit.Kind != token.STRING || moduleLit.Kind != token.STRING {
				continue
			}
			companion, pathErr := strconv.Unquote(pathLit.Value)
			module, moduleErr := strconv.Unquote(moduleLit.Value)
			if pathErr != nil || moduleErr != nil || companion != strings.TrimSuffix(module, ".py")+"_test.py" || !baseline[module] {
				continue
			}
			if !containsTrackedPath(t.Paths, companion) || !containsTrackedPath(t.Paths, module) {
				continue
			}
			source, ok := t.FileBytes(companion)
			if ok && pythonSyntaxImports(source, strings.TrimSuffix(path.Base(module), ".py")) {
				allowed[companion] = true
			}
		}
		return false
	})
	return allowed
}

func containsTrackedPath(paths []string, want string) bool {
	for _, candidate := range paths {
		if candidate == want {
			return true
		}
	}
	return false
}

func pythonSyntaxImports(source []byte, module string) bool {
	const inspectImports = `import ast, sys
target = sys.argv[1]
try:
    tree = ast.parse(sys.stdin.read())
except SyntaxError:
    raise SystemExit(2)
for node in ast.walk(tree):
    if isinstance(node, ast.Import) and any(alias.name == target for alias in node.names):
        raise SystemExit(0)
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == target:
        raise SystemExit(0)
raise SystemExit(1)
`
	cmd := windowgate.Command("python3", "-I", "-S", "-c", inspectImports, module)
	cmd.Stdin = bytes.NewReader(source)
	return cmd.Run() == nil
}

// gatePythonToolTree emits a NEW_PYTHON_TOOL finding for every tracked tools/*.py that is not
// in the grandfathered baseline — the same verdict pythongate.ScanTree computes, one boundary
// earlier. It reads the tracked path set (so an untracked scratch .py in tools/ is correctly
// ignored, exactly like git ls-files). Returns ErrCouldNotRun when the baseline cannot be
// parsed (fail open, exit 2 → the pythongate TEST still catches it in CI as the backstop).
func gatePythonToolTree(t *TrackedTree) ([]Finding, error) {
	baseline, ok := declaredPyBaseline(t)
	if !ok {
		return nil, ErrCouldNotRun
	}
	testCompanions := declaredPyTestCompanions(t, baseline)
	var findings []Finding
	for _, p := range t.Paths {
		// prefix+suffix at ANY depth is the ratchet's real scope: git pathspec `*` is not
		// shell-glob `*` — it spans `/`, so `git ls-files tools/*.py` returns nested
		// tools/**/x.py too (the tracked tree has one today). Restricting this gate to a
		// single level would let a NEW nested python tool pass pre-push and then red CI when
		// pythongate.ScanTree, which uses that very pathspec, sees it.
		if !strings.HasPrefix(p, "tools/") || !strings.HasSuffix(p, ".py") {
			continue
		}
		if baseline[p] || testCompanions[p] {
			continue
		}
		findings = append(findings, Finding{
			Gate: reasonNewPythonTool,
			File: p,
			Detail: p + " is a NEW python tool; port it to Go instead (" + reasonNewPythonTool +
				"). If it is a genuinely new tool that must stay Python, add its path to the " +
				"grandfathered baseline in " + pythonBaselineFile + " (the single source of truth " +
				"pythongate.TestNoNewPythonTools also reads).",
		})
	}
	sort.Slice(findings, func(i, j int) bool { return findings[i].File < findings[j].File })
	return findings, nil
}
