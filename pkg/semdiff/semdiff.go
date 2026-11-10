// Package semdiff classifies a change to one Go source file as comment-only or
// logic, so a gate can skip package witnesses that a comment edit cannot move.
// It fails closed: any doubt is ClassLogic.
//
// Known residual risk: adding or removing comment lines shifts the line numbers
// seen by runtime.Caller, panic traces, and other position-reporting code.
package semdiff

import (
	"bytes"
	"go/parser"
	"go/scanner"
	"go/token"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

// Class is the semantic class of a change.
type Class string

const (
	// ClassCommentOnly means the token stream is identical ignoring comments
	// and no significant comment changed.
	ClassCommentOnly Class = "comment_only"
	ClassLogic       Class = "logic"
)

// Reason is the closed token explaining a ClassLogic verdict; it is empty for
// ClassCommentOnly.
type Reason string

const (
	ReasonNone             Reason = ""
	ReasonScanError        Reason = "scan_error"
	ReasonParseError       Reason = "parse_error"
	ReasonCodeChanged      Reason = "code_changed"
	ReasonDirectiveChanged Reason = "directive_changed"
	ReasonCgo              Reason = "cgo"
	ReasonExampleOutput    Reason = "example_output"
)

// ClassifyGo compares two versions of one Go source file. name is used only to
// detect a _test.go file.
func ClassifyGo(name string, before, after []byte) (Class, Reason) {
	if bytes.Equal(before, after) {
		return ClassCommentOnly, ReasonNone
	}
	beforeToks, beforeComments, ok := scan(before)
	if !ok {
		return ClassLogic, ReasonScanError
	}
	afterToks, afterComments, ok := scan(after)
	if !ok {
		return ClassLogic, ReasonScanError
	}
	if !slices.Equal(beforeToks, afterToks) {
		return ClassLogic, ReasonCodeChanged
	}
	if !slices.Equal(significant(beforeComments), significant(afterComments)) {
		return ClassLogic, ReasonDirectiveChanged
	}
	isTest := strings.HasSuffix(name, "_test.go")
	beforeInfo, ok := inspect(before, isTest)
	if !ok {
		return ClassLogic, ReasonParseError
	}
	afterInfo, ok := inspect(after, isTest)
	if !ok {
		return ClassLogic, ReasonParseError
	}
	if beforeInfo.cgo || afterInfo.cgo {
		return ClassLogic, ReasonCgo
	}
	if !slices.Equal(beforeInfo.outputs, afterInfo.outputs) {
		return ClassLogic, ReasonExampleOutput
	}
	return ClassCommentOnly, ReasonNone
}

type tok struct {
	tok token.Token
	lit string
}

// scan returns the code token stream and the ordered comment texts. A real ";"
// and an auto-inserted newline semicolon are the same token to the parser, so
// both normalize to a bare SEMICOLON; a comment that changes semicolon insertion
// still changes the stream. Comments come from a second pass because
// ScanComments mode alters where semicolons are reported.
func scan(src []byte) ([]tok, []string, bool) {
	var toks []tok
	ok := scanEach(src, 0, func(t token.Token, lit string) {
		if t == token.SEMICOLON {
			lit = ""
		}
		toks = append(toks, tok{t, lit})
	})
	if !ok {
		return nil, nil, false
	}
	var comments []string
	ok = scanEach(src, scanner.ScanComments, func(t token.Token, lit string) {
		if t == token.COMMENT {
			comments = append(comments, lit)
		}
	})
	return toks, comments, ok
}

func scanEach(src []byte, mode scanner.Mode, emit func(token.Token, string)) bool {
	fset := token.NewFileSet()
	file := fset.AddFile("", fset.Base(), len(src))
	var s scanner.Scanner
	errs := 0
	s.Init(file, src, func(token.Position, string) { errs++ }, mode)
	for {
		_, t, lit := s.Scan()
		if t == token.EOF {
			break
		}
		emit(t, lit)
	}
	return errs == 0
}

var directivePattern = regexp.MustCompile(`^//[a-z0-9]+:[a-z]`)

var significantPrefixes = []string{
	"//go:", "//line ", "/*line ", "// +build", "//+build", "//export ", "//extern ", "//sys",
}

func significant(comments []string) []string {
	var out []string
	for _, c := range comments {
		if directivePattern.MatchString(c) || hasAnyPrefix(c, significantPrefixes) {
			out = append(out, c)
		}
	}
	return out
}

func hasAnyPrefix(s string, prefixes []string) bool {
	for _, p := range prefixes {
		if strings.HasPrefix(s, p) {
			return true
		}
	}
	return false
}

var outputLine = regexp.MustCompile(`(?im)^\s*(unordered )?output:`)

type fileInfo struct {
	cgo     bool
	outputs []string
}

func inspect(src []byte, isTest bool) (fileInfo, bool) {
	fset := token.NewFileSet()
	mode := parser.ImportsOnly
	if isTest {
		mode = parser.ParseComments
	}
	f, err := parser.ParseFile(fset, "", src, mode)
	if err != nil {
		return fileInfo{}, false
	}
	var info fileInfo
	for _, imp := range f.Imports {
		if path, err := strconv.Unquote(imp.Path.Value); err == nil && path == "C" {
			info.cgo = true
		}
	}
	if isTest {
		for _, group := range f.Comments {
			if !outputLine.MatchString(group.Text()) {
				continue
			}
			raw := make([]string, 0, len(group.List))
			for _, c := range group.List {
				raw = append(raw, c.Text)
			}
			info.outputs = append(info.outputs, strings.Join(raw, "\n"))
		}
	}
	return info, true
}
