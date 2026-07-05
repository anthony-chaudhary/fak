package hooks

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"sort"
	"strings"
)

// gate_godfile.go — the god-file / god-function GROWTH ratchet (#2868, Track I of #2834).
//
// Hermes' core is concentrated in god-files (`gateway/run.py` is 20,320 lines; `cli.py`
// 16,128) and its own rubric begs for refactors — the monoliths accrete faster than they
// are paid down because nothing REFUSES the next line of growth. fak already names the
// ceilings (tools/code_quality_scorecard.py: FILE_HARD_MAX=1500 / FUNC_HARD_MAX=200 are
// "hard debt") and already owns the paydown loop (/modularize, tools/godsplit_plan.py);
// this gate turns the doctrine preventive. Full write-up:
// docs/explainers/god-file-growth-gate.md.
//
// The ratchet, pythongate-style: every tracked non-test .go file already over a ceiling
// when the gate shipped is GRANDFATHERED at its then-size (godfile_baseline.go). The gate
// refuses (a) a NEW file/function crossing its ceiling and (b) any GROWTH of a
// grandfathered offender past its frozen size. Shrinking is always clean, and the
// baseline may only ever shrink or lose entries — never grow — so the god-code surface
// is monotonically non-increasing. Enforced twice: here in `fak hygiene` (pre-push,
// before the trunk goes red) and by TestGodfileGate_LiveTreeClean under `make ci`.

// reasonGodFileGrowth is the closed-vocabulary refusal code for god-file / god-function
// growth: a new file or function over the hard ceiling, or a grandfathered one that grew
// past its frozen size. One token covers both classes; the Detail names which.
const reasonGodFileGrowth = "GOD_FILE_GROWTH"

// The hard ceilings mirror tools/code_quality_scorecard.py (FILE_HARD_MAX / FUNC_HARD_MAX):
// the sizes the code-quality scorecard already counts as hard architecture debt. The gate
// refuses growth past them; the scorecard keeps scoring the grandfathered stock.
const (
	godFileMaxLines = 1500 // a tracked non-test .go file over this is a god-file
	godFuncMaxLines = 200  // a declared function/method spanning more lines is a god-function
)

// godScan is one scanned file: its line count plus the functions found over the ceiling.
type godScan struct {
	path  string
	lines int
	funcs []godFunc
}

// godFunc is one over-ceiling function: its baseline key ("path:Name" or
// "path:(*T).Name"), the line its declaration starts on, and its line span.
type godFunc struct {
	key  string
	line int
	span int
}

// gateGodFileGrowthTree emits a GOD_FILE_GROWTH finding for every tracked non-test .go
// file over godFileMaxLines lines and every function over godFuncMaxLines lines that is
// not covered by the frozen grandfathered baseline — where "covered" means the entry
// exists AND the current size does not exceed the frozen one. Never errors: the baseline
// is compiled into this package (godfile_baseline.go), so there is no unreadable source
// of truth to fail open on; an unparseable file simply skips the function scan.
func gateGodFileGrowthTree(t *TrackedTree) ([]Finding, error) {
	return godGrowthOffenses(godfileScanTree(t), godfileGrandfathered, godfuncGrandfathered), nil
}

// godfileScanTree folds the tracked tree into the per-file scans the ratchet core
// judges. Scope mirrors the scorecard's architecture corpus: tracked *.go, excluding
// _test.go (tests are graded by the tests KPI, not architecture) and fixture/vendored
// trees. Files at or under godFuncMaxLines lines are skipped whole — they can hold
// neither a god-file nor a god-function.
func godfileScanTree(t *TrackedTree) []godScan {
	var scans []godScan
	for _, p := range t.Paths {
		if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") || godfileExcludedPath(p) {
			continue
		}
		body, ok := t.FileBytes(p)
		if !ok {
			continue
		}
		lines := countLines(body)
		if lines <= godFuncMaxLines {
			continue
		}
		scans = append(scans, godScan{path: p, lines: lines, funcs: longFuncs(p, body)})
	}
	return scans
}

// godfileExcludedPath reports whether a path sits in a tree the scorecard's architecture
// corpus also skips: fixtures are intentionally odd, vendored/generated trees are not
// ours to grade (GO_EXCLUDE_DIRS in tools/code_quality_scorecard.py).
func godfileExcludedPath(p string) bool {
	for _, seg := range strings.Split(p, "/") {
		switch seg {
		case "testdata", "vendor", "node_modules", "__pycache__":
			return true
		}
		if strings.HasPrefix(seg, ".") {
			return true
		}
	}
	return false
}

// countLines counts the file's lines: newline count, plus one for a trailing
// unterminated line. The frozen baseline records THIS scanner's numbers, so freeze and
// check can never disagree with each other (other tools may count ±1 differently).
func countLines(body []byte) int {
	n := bytes.Count(body, []byte("\n"))
	if len(body) > 0 && body[len(body)-1] != '\n' {
		n++
	}
	return n
}

// longFuncs parses the file and returns the declared functions/methods spanning more
// than godFuncMaxLines lines (declaration line through closing brace, doc comment
// excluded). go/parser makes the span un-gameable by literals or comments — the same
// reason the scorecard blanks literals before its brace scan. A parse failure returns
// nil: the file-line rule still applies, and the file will not compile anyway.
func longFuncs(path string, body []byte) []godFunc {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, body, parser.SkipObjectResolution)
	if err != nil {
		return nil
	}
	var funcs []godFunc
	for _, d := range f.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Body == nil {
			continue
		}
		start := fset.Position(fd.Pos()).Line
		span := fset.Position(fd.End()).Line - start + 1
		if span > godFuncMaxLines {
			funcs = append(funcs, godFunc{key: godFuncKey(path, fd), line: start, span: span})
		}
	}
	return funcs
}

// godFuncKey renders a function's stable baseline identity: "path:Name" for a function,
// "path:(T).Name" / "path:(*T).Name" for a method — receiver included so two same-named
// methods in one file cannot shadow each other's ceiling.
func godFuncKey(path string, fd *ast.FuncDecl) string {
	name := fd.Name.Name
	if fd.Recv != nil && len(fd.Recv.List) == 1 {
		name = "(" + recvTypeString(fd.Recv.List[0].Type) + ")." + name
	}
	return path + ":" + name
}

// recvTypeString renders a receiver type expression ("T", "*T", "T[P]") without
// importing go/types — generics' type parameters are rendered by their identifiers.
func recvTypeString(e ast.Expr) string {
	switch t := e.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return "*" + recvTypeString(t.X)
	case *ast.IndexExpr:
		return recvTypeString(t.X) + "[" + recvTypeString(t.Index) + "]"
	case *ast.IndexListExpr:
		parts := make([]string, len(t.Indices))
		for i, ix := range t.Indices {
			parts[i] = recvTypeString(ix)
		}
		return recvTypeString(t.X) + "[" + strings.Join(parts, ", ") + "]"
	default:
		return "?"
	}
}

// godGrowthOffenses is the pure ratchet core, split out so it can be unit-tested on a
// synthetic corpus + baselines: one Finding per file over the file ceiling and per
// function over the function ceiling, unless the frozen baseline grandfathers it at or
// above its current size. Sorted by (file, line) for stable output.
func godGrowthOffenses(scans []godScan, fileBase, funcBase map[string]int) []Finding {
	var findings []Finding
	for _, s := range scans {
		if s.lines > godFileMaxLines {
			if frozen, ok := fileBase[s.path]; !ok {
				findings = append(findings, Finding{
					Gate: reasonGodFileGrowth,
					File: s.path,
					Detail: fmt.Sprintf("NEW god-file: %s is %d lines (> %d). %s",
						s.path, s.lines, godFileMaxLines, godfileFixHint),
				})
			} else if s.lines > frozen {
				findings = append(findings, Finding{
					Gate: reasonGodFileGrowth,
					File: s.path,
					Detail: fmt.Sprintf("grandfathered god-file GREW: %s is %d lines, frozen ceiling %d. %s",
						s.path, s.lines, frozen, godfileFixHint),
				})
			}
		}
		for _, fn := range s.funcs {
			if frozen, ok := funcBase[fn.key]; !ok {
				findings = append(findings, Finding{
					Gate: reasonGodFileGrowth,
					File: s.path,
					Line: fn.line,
					Detail: fmt.Sprintf("NEW god-function: %s spans %d lines (> %d). %s",
						fn.key, fn.span, godFuncMaxLines, godfileFixHint),
				})
			} else if fn.span > frozen {
				findings = append(findings, Finding{
					Gate: reasonGodFileGrowth,
					File: s.path,
					Line: fn.line,
					Detail: fmt.Sprintf("grandfathered god-function GREW: %s spans %d lines, frozen ceiling %d. %s",
						fn.key, fn.span, frozen, godfileFixHint),
				})
			}
		}
	}
	sort.Slice(findings, func(i, j int) bool {
		if findings[i].File != findings[j].File {
			return findings[i].File < findings[j].File
		}
		return findings[i].Line < findings[j].Line
	})
	return findings
}

// godfileFixHint is the recover-by line every GOD_FILE_GROWTH finding carries: split,
// don't grow — and tighten the baseline only after a genuine split.
const godfileFixHint = "Split along concern seams instead of growing the monolith — " +
	"/modularize owns the recipe (tools/godsplit_plan.py plans the boundaries); see " +
	"docs/explainers/god-file-growth-gate.md (Hermes' 20,320-line gateway/run.py is what " +
	"accretes without this gate). After a genuine split lands, regenerate " +
	"internal/hooks/godfile_baseline.go (FAK_GODFILE_BASELINE_REGEN=1) — entries may only " +
	"shrink or disappear, never grow."
