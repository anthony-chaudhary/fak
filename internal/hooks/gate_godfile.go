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
// are paid down because nothing REFUSES the next line of growth. fak owns the paydown loop
// (/modularize, tools/godsplit_plan.py); this gate turns the doctrine preventive. Full
// write-up: docs/explainers/god-file-growth-gate.md.
//
// TWO SIBLING GATES, ONE FILE LINE. This gate is the FUNCTION-and-file GROWTH ratchet; its
// file-only sibling internal/godfileceiling (#2898) is the hard "NO NEW GOD-FILE" ceiling at a
// non-tunable const 1500. On the FILE dimension they deliberately agree at 1500 — a brand-new
// file monolith is the thing you most want to refuse, so the file line is held firm, not
// loosened. This gate's UNIQUE contribution is the FUNCTION dimension (godfileceiling has none)
// plus a growth ratchet over grandfathered offenders. Raising godFileMaxLines alone would be
// dead code — godfileceiling shadows it at 1500 — so it stays at 1500 and only godFuncMaxLines
// is set loose.
//
// MEASURE STRICT, BLOCK LOOSE (functions). The scorecard (tools/code_quality_scorecard.py:
// FUNC_HARD_MAX=200) MEASURES function debt at the tight line so the dashboard stays honest.
// The FUNCTION blocking ceiling here sits DELIBERATELY higher (godFuncMaxLines default 400) so
// the gate only reds on an *egregious* new function, never on a legitimately large-but-bounded
// one — a dashboard that flags a 250-line function is useful; a build that REFUSES it is a false
// block. Both ceilings are operator-tunable (FAK_GODFILE_MAX_LINES / FAK_GODFUNC_MAX_LINES) and
// the whole gate has an escape hatch (ALLOW_GOD_FILE) for the genuine one-off.
//
// The ratchet, pythongate-style: every tracked non-test .go file already over a ceiling
// when the gate shipped is GRANDFATHERED at its then-size (godfile_baseline.go). The gate
// refuses (a) a NEW file/function crossing its ceiling and (b) GROWTH of a grandfathered
// offender past its frozen size PLUS a bounded slack band (godGrowthSlackPct, default 20%;
// FAK_GODFILE_GROWTH_SLACK_PCT=0 restores the strict ratchet). Shrinking is always clean, and
// the frozen anchor may only ever shrink or lose entries on regen — never grow. So the surface
// is BOUNDED rather than strictly monotonic: a grandfathered offender may drift within its slack
// band, but a runaway still reds and the anchor never ratchets upward. That slack is what keeps
// an ordinary edit to an already-large function from false-blocking a busy shared trunk (the
// concrete case: cmd/fak/guard.go:cmdGuard, frozen at 1181, which a one-line peer edit used to
// red). Enforced twice: here in `fak hygiene` (pre-push, before the trunk goes red) and by
// TestGodfileGate_LiveTreeClean under `make ci`.

// reasonGodFileGrowth is the closed-vocabulary refusal code for god-file / god-function
// growth: a new file or function over the hard ceiling, or a grandfathered one that grew
// past its frozen size. One token covers both classes; the Detail names which.
const reasonGodFileGrowth = "GOD_FILE_GROWTH"

// The BLOCKING ceilings — the sizes the gate reds a build on. The FILE ceiling stays at the
// scorecard debt line (1500), aligned with the sibling godfileceiling gate; the FUNCTION
// ceiling sits above the scorecard's 200 so the gate refuses only an egregious new function,
// not every large one (see the "TWO SIBLING GATES" note above). Operator-tunable: a fleet that
// wants a tighter or looser floor moves it with an env var, no code change. A zero/negative/
// garbage value falls back to the default (that is what ALLOW_GOD_FILE is for, not a 0 ceiling).
var (
	// File ceiling: held at the scorecard's debt line (1500), NOT loosened, because the
	// sibling gate internal/godfileceiling (#2898) is the hard "NO NEW GOD-FILE" ratchet at a
	// non-tunable const 1500 — raising this alone would be dead (godfileceiling shadows it) and
	// desync two gates the tree deliberately keeps aligned on files. The FUNCTION dimension,
	// which godfileceiling has no equivalent for, is where this gate loosens (see below).
	godFileMaxLines = gateEnvInt("FAK_GODFILE_MAX_LINES", 1500) // a tracked non-test .go file over this is a god-file
	godFuncMaxLines = gateEnvInt("FAK_GODFUNC_MAX_LINES", 400)  // a declared function/method spanning more lines is a god-function
)

// godGrowthSlackPct is the BOUNDED-DRIFT allowance for grandfathered offenders. Raising the
// ceilings only helps NEW code; a function already frozen far above the ceiling (cmd/fak/guard.go
// :cmdGuard was frozen at 1181) still reds the build the moment a peer adds a line to it — the
// commonest false block on a busy shared trunk, and the one that forces a mid-flight split of a
// function nobody set out to refactor. So a grandfathered offender may drift up to this percentage
// above its frozen size before the gate re-engages. The surface is therefore BOUNDED (a runaway
// function still reds past the band) rather than strictly monotonic — a deliberate trade of the
// last increment of strictness for far fewer false blocks. Set FAK_GODFILE_GROWTH_SLACK_PCT=0 to
// restore the strict ratchet. The frozen anchor itself only ever tightens (regen refuses to raise
// it), so slack gives runway above the anchor without ever ratcheting the anchor upward.
var godGrowthSlackPct = gateEnvNonNegInt("FAK_GODFILE_GROWTH_SLACK_PCT", 20)

// godFileGateEscapeEnv opts a single run out of GOD_FILE_GROWTH entirely — the same escape-hatch
// idiom the pre-commit gate set carries (hooks.go EscapeEnv). For the legitimate large file the
// ceilings still refuse, an operator sets this rather than being forced off-trunk.
const godFileGateEscapeEnv = "ALLOW_GOD_FILE"

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
	// Escape hatch: an operator can admit a run that the ceilings would otherwise refuse,
	// so a legitimate large file never forces the author off-trunk (#goal: gates that
	// over-refuse must have a witnessed override, not a hard wall).
	if gateEnvTruthy(godFileGateEscapeEnv) {
		return nil, nil
	}
	return godGrowthOffenses(godfileScanTree(t), godfileGrandfathered, godfuncGrandfathered, godGrowthSlackPct), nil
}

// godSlackCeiling is the effective ceiling a grandfathered offender may reach before it reds:
// its frozen anchor plus slackPct percent of that anchor (floored to whole lines). slackPct==0
// reproduces the strict ratchet (effective == frozen).
func godSlackCeiling(frozen, slackPct int) int {
	return frozen + frozen*slackPct/100
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
		if lines <= godFuncMaxLines && lines <= godFileMaxLines {
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
// within slackPct percent of its frozen size. slackPct==0 is the strict ratchet (no drift
// above the frozen anchor). Sorted by (file, line) for stable output.
func godGrowthOffenses(scans []godScan, fileBase, funcBase map[string]int, slackPct int) []Finding {
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
			} else if ceiling := godSlackCeiling(frozen, slackPct); s.lines > ceiling {
				findings = append(findings, Finding{
					Gate: reasonGodFileGrowth,
					File: s.path,
					Detail: fmt.Sprintf("grandfathered god-file GREW: %s is %d lines, past its frozen ceiling %d (+%d%% growth slack = %d). %s",
						s.path, s.lines, frozen, slackPct, ceiling, godfileFixHint),
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
			} else if ceiling := godSlackCeiling(frozen, slackPct); fn.span > ceiling {
				findings = append(findings, Finding{
					Gate: reasonGodFileGrowth,
					File: s.path,
					Line: fn.line,
					Detail: fmt.Sprintf("grandfathered god-function GREW: %s spans %d lines, past its frozen ceiling %d (+%d%% growth slack = %d). %s",
						fn.key, fn.span, frozen, slackPct, ceiling, godfileFixHint),
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
// don't grow — and tighten the baseline only after a genuine split. It also names the two
// operator overrides, so an author who hits a false block on a legitimately large file has
// the witnessed way out in front of them instead of reaching for a feature branch.
const godfileFixHint = "Split along concern seams instead of growing the monolith — " +
	"/modularize owns the recipe (tools/godsplit_plan.py plans the boundaries); see " +
	"docs/explainers/god-file-growth-gate.md (Hermes' 20,320-line gateway/run.py is what " +
	"accretes without this gate). After a genuine split lands, regenerate " +
	"internal/hooks/godfile_baseline.go (FAK_GODFILE_BASELINE_REGEN=1) — entries may only " +
	"shrink or disappear, never grow. If the size is legitimate, widen the growth slack or raise " +
	"the ceiling for the run (FAK_GODFILE_GROWTH_SLACK_PCT / FAK_GODFILE_MAX_LINES / " +
	"FAK_GODFUNC_MAX_LINES) or admit it with ALLOW_GOD_FILE=1."
