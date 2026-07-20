package boundarylint

import (
	"errors"
	"go/scanner"
	"path/filepath"
	"strings"
)

// CodeUnparseableSource is the closed-vocabulary code for a source file the scanner
// could not parse, and therefore could not lint.
const CodeUnparseableSource = "UNPARSEABLE_SOURCE"

// ScanUnparseable walks each root and reports every .go file that parser.ParseFile
// could not read — the scanner's own blind spot, surfaced as a recorded skip.
//
// The tell it closes is a fail-OPEN default. Every other walk here silently drops an
// unparseable file ("the compiler owns that error"), so such a file contributes zero
// findings and is indistinguishable from a genuinely clean one: the linter reports
// success over source it never actually read. That is the inverse of the fail-CLOSED
// discipline the kernel applies to undecidable input elsewhere (internal/adjudicator's
// argcanon returns MALFORMED rather than a default-allow when an arg's quotes never
// close). A detector that cannot decide should say so, not say "nothing found".
//
// UNPARSEABLE_SOURCE is a SOFT signal, the same class as SKIP_DEBT: it is reported by
// `fak boundary` and never gates. That is deliberate rather than a weaker compromise —
// this is a shared, permanently peer-dirty trunk where half-written .go files from other
// live sessions are normal, so failing the build on one would red the trunk for everyone
// on a peer's in-flight edit while saying nothing about the committed tree. The compiler
// (and `fak buildcheck`) remain the authority on whether source is valid; this witness
// only ensures the LINTER's silence is never mistaken for a clean bill of health.
//
// It walks both production and test sources, since Scan (non-test) and ScanTests
// (_test.go) each skip unparseable files in their own half of the tree.
//
// Findings carry no //boundarylint:ignore suppression: the directive is collected from
// the parsed AST, which by definition does not exist here. A soft, never-gating tell
// needs no escape hatch — the fix is to make the file parse, or to let it leave the tree.
func ScanUnparseable(roots []string) ([]Finding, error) {
	return scanFiltered(roots, nil, func(path string) bool {
		return strings.HasSuffix(path, ".go")
	}, true)
}

// unparseableFinding builds the UNPARSEABLE_SOURCE finding for path, anchored at the
// first syntax error's line when the parser reported a position (go/parser returns a
// scanner.ErrorList) and at line 0 when it did not — an IO error carries no position.
func unparseableFinding(path string, perr error) Finding {
	line, reason := 0, perr.Error()
	var list scanner.ErrorList
	if errors.As(perr, &list) && len(list) > 0 {
		line = list[0].Pos.Line
		reason = list[0].Msg
	}
	return Finding{
		Code: CodeUnparseableSource,
		File: filepath.ToSlash(path),
		Line: line,
		Detail: "source could not be parsed (" + reason + "), so no boundary rule ran over it — " +
			"this file is a recorded SKIP, not a clean result; fix the syntax so the linter can read it",
	}
}
