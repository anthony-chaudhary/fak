package adjudicator

// lintwrites.go is the in-process realization of the opt-in write-scoped lint
// verdict (#536): a whole-file write whose content does not parse is refused
// with MALFORMED before it lands. It is the decide-path dual of codelint's
// advisory write-lint (internal/swebench lintWritten), with two load-bearing
// differences from calling codelint directly:
//
//   - No os/exec. codelint's Python/CUDA packs shell out, and this package sits
//     on the live decide path (internal/registrations blank-imports the
//     adjudicator), so importing codelint would put a non-literal
//     exec.CommandContext on the request-path closure and break architest's
//     TestRequestPathInterpreterFree. The Go and JSON grammars parse in-process
//     via the stdlib, so those two — the common agent languages — are checked
//     here; everything else DEFERs (fail open).
//   - Bounded disclosure. The deny carries one finding (file:line:col), never
//     the file body: the deny channel is not a content oracle.
//
// The parse is the same stable stdlib primitive codelint's in-process packs use
// (go/parser, encoding/json); the small duplication is the price of keeping the
// decide path subprocess-free.

import (
	"encoding/json"
	"errors"
	"fmt"
	"go/parser"
	"go/scanner"
	"go/token"
	"path/filepath"
	"strings"
)

// wholeFileWrite reports whether tool carries a FULL file body (write/create
// semantics), as opposed to a partial edit/patch/replace whose fragment would
// never parse standalone. The lint rung is scoped to whole-file writes so a
// partial edit is never false-denied as MALFORMED — it DEFERs (fail open).
func wholeFileWrite(tool string) bool {
	low := strings.ToLower(tool)
	for _, w := range []string{"write", "create"} {
		if strings.Contains(low, w) {
			return true
		}
	}
	return false
}

// writeContent extracts the full-file content string a write tool carries, trying
// the common content arg keys across agent tool dialects (write_file's "content",
// the Aider/Claude "file_text"). Returns ("", false) when no string content arg
// is present — a write with no content DEFERs.
func writeContent(args map[string]any) (string, bool) {
	for _, k := range []string{"content", "file_text", "text"} {
		if v, ok := args[k]; ok {
			if s, ok := v.(string); ok {
				return s, true
			}
		}
	}
	return "", false
}

// lintWriteMalformed parses the content of a whole-file write with the in-process
// grammar its extension implies and returns a bounded file:line:col witness for
// the first hard parse error, or "" when the write is clean OR in a language the
// decide path has no in-process checker for (.py/.cu/unlinted) — those DEFER
// (fail open), since lint is a quality signal, never a security gate.
func lintWriteMalformed(path string, args map[string]any) string {
	content, ok := writeContent(args)
	if !ok {
		return "" // no content to lint: defer
	}
	switch strings.ToLower(filepath.Ext(path)) {
	case ".go":
		return goParseWitness(path, []byte(content))
	case ".json":
		return jsonParseWitness(path, []byte(content))
	default:
		return "" // no in-process checker owns this language: defer (fail open)
	}
}

// goParseWitness parses Go source and returns a bounded "<path>:<line>:<col>:
// <msg>" witness for the first syntax error, or "" when the source parses. It
// mirrors codelint's goCheck (parser.AllErrors) but reports only the FIRST error
// (bounded disclosure). Semantic/type errors are intentionally not surfaced — a
// single file in isolation has no package context, so flagging them would
// false-deny perfectly good code.
func goParseWitness(name string, src []byte) string {
	fset := token.NewFileSet()
	if _, err := parser.ParseFile(fset, name, src, parser.AllErrors); err != nil {
		var el scanner.ErrorList
		if errors.As(err, &el) && len(el) > 0 {
			e := el[0] // first error only — bounded disclosure
			return findingLine(name, e.Pos.Line, e.Pos.Column, e.Msg)
		}
		return findingLine(name, 0, 0, err.Error())
	}
	return ""
}

// jsonParseWitness parses JSON and returns a bounded "<path>:<line>:<col>: <msg>"
// witness for the first syntax error, or "" when the JSON is well-formed. Mirrors
// codelint's jsonCheck (the stdlib decoder surfaces a byte offset, converted to
// line:col).
func jsonParseWitness(name string, src []byte) string {
	var v any
	if err := json.Unmarshal(src, &v); err != nil {
		line, col := 0, 0
		var se *json.SyntaxError
		if errors.As(err, &se) {
			line, col = offsetToLineCol(src, int(se.Offset))
		}
		return findingLine(name, line, col, err.Error())
	}
	return ""
}

// findingLine renders the bounded file:line:col witness the MALFORMED deny
// carries. The detail is the checker's message (never the file content); a
// zero line/col (the checker did not locate it) collapses to "<path>: <msg>".
func findingLine(name string, line, col int, detail string) string {
	if line <= 0 {
		return name + ": " + detail
	}
	if col <= 0 {
		return fmt.Sprintf("%s:%d: %s", name, line, detail)
	}
	return fmt.Sprintf("%s:%d:%d: %s", name, line, col, detail)
}

// offsetToLineCol maps a 0-based byte offset into 1-based line/column. Mirrors
// codelint's helper (used by the JSON pack to convert the decoder's byte offset
// into a citable location).
func offsetToLineCol(src []byte, off int) (line, col int) {
	if off < 0 {
		return 0, 0
	}
	if off > len(src) {
		off = len(src)
	}
	line, col = 1, 1
	for i := 0; i < off; i++ {
		if src[i] == '\n' {
			line++
			col = 1
		} else {
			col++
		}
	}
	return line, col
}
