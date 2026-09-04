// Package scratchmark detects source files whose leading comments declare the
// file disposable. It does not decide what a caller should do with a match.
//
// Invariant: scratchmark detection is fail-closed and deterministic. Unreadable
// files fail closed with an error, and non-comment or trailing prose is never
// treated as disposable.
// Guard: inspection is strictly bounded by MaxHeaderBytes to isolate classification
// strictly to the file's leading comment block.
package scratchmark

import (
	"fmt"
	"io"
	"os"
	"strings"
	"unicode"
)

const (
	// KeepDirective suppresses a scratch marker in the same leading comment
	// block. Callers can surface this spelling without duplicating policy.
	KeepDirective = "scratchmark:keep"

	// MaxHeaderBytes bounds detection work and prevents a marker deep in a file
	// from being mistaken for a declaration about the source artifact itself.
	MaxHeaderBytes = 16 * 1024
)

// Result describes the first disposable-source marker in the leading comment
// block. A keep directive wins over every marker in that block.
type Result struct {
	Marked bool
	Kept   bool
	Marker string
	Line   int
}

// Scan reads only the bounded header of path and classifies it.
func Scan(path string) (Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return Result{}, fmt.Errorf("open %q: %w", path, err)
	}
	defer f.Close()

	source, err := io.ReadAll(io.LimitReader(f, MaxHeaderBytes))
	if err != nil {
		return Result{}, fmt.Errorf("read %q: %w", path, err)
	}
	return Detect(source), nil
}

// Detect classifies at most MaxHeaderBytes from source. Only the leading
// comment block is inspected; code, prose after code, and string literals are
// outside the detector's scope.
func Detect(source []byte) Result {
	if len(source) > MaxHeaderBytes {
		source = source[:MaxHeaderBytes]
	}

	comments := leadingComments(string(source))
	var marked Result
	for _, comment := range comments {
		if hasKeepDirective(comment.text) {
			return Result{Kept: true, Line: comment.line}
		}
		if marked.Marked {
			continue
		}
		if marker := disposableMarker(comment.text); marker != "" {
			marked = Result{Marked: true, Marker: marker, Line: comment.line}
		}
	}
	return marked
}

type headerComment struct {
	text string
	line int
}

type blockComment byte

const (
	noBlock blockComment = iota
	cBlock
	htmlBlock
)

func leadingComments(source string) []headerComment {
	lines := strings.Split(source, "\n")
	comments := make([]headerComment, 0, len(lines))
	block := noBlock

	for i, raw := range lines {
		line := strings.TrimSpace(strings.TrimSuffix(raw, "\r"))
		if i == 0 {
			line = strings.TrimPrefix(line, "\ufeff")
		}

		if block != noBlock {
			text, closed := blockLine(line, block)
			comments = append(comments, headerComment{text: text, line: i + 1})
			if closed {
				block = noBlock
			}
			continue
		}
		if line == "" {
			continue
		}

		text, next, ok := commentLine(line)
		if !ok {
			break
		}
		comments = append(comments, headerComment{text: text, line: i + 1})
		block = next
	}
	return comments
}

func commentLine(line string) (string, blockComment, bool) {
	for _, prefix := range []string{"//", "#", ";", "--"} {
		if strings.HasPrefix(line, prefix) {
			return strings.TrimSpace(strings.TrimPrefix(line, prefix)), noBlock, true
		}
	}
	if strings.HasPrefix(line, "/*") {
		text := strings.TrimSpace(strings.TrimPrefix(line, "/*"))
		if before, _, closed := strings.Cut(text, "*/"); closed {
			return strings.TrimSpace(before), noBlock, true
		}
		return text, cBlock, true
	}
	if strings.HasPrefix(line, "<!--") {
		text := strings.TrimSpace(strings.TrimPrefix(line, "<!--"))
		if before, _, closed := strings.Cut(text, "-->"); closed {
			return strings.TrimSpace(before), noBlock, true
		}
		return text, htmlBlock, true
	}
	return "", noBlock, false
}

func blockLine(line string, block blockComment) (string, bool) {
	end := "*/"
	if block == htmlBlock {
		end = "-->"
	}
	before, _, closed := strings.Cut(line, end)
	before = strings.TrimSpace(before)
	if block == cBlock {
		before = strings.TrimSpace(strings.TrimPrefix(before, "*"))
	}
	return before, closed
}

func hasKeepDirective(comment string) bool {
	return strings.Contains(strings.ToLower(comment), KeepDirective)
}

var markerWords = map[string]bool{
	"disposable": true,
	"scratch":    true,
	"temporary":  true,
	"throwaway":  true,
}

var artifactWords = map[string]bool{
	"artifact":       true,
	"code":           true,
	"command":        true,
	"executable":     true,
	"experiment":     true,
	"file":           true,
	"helper":         true,
	"implementation": true,
	"probe":          true,
	"program":        true,
	"prototype":      true,
	"script":         true,
	"shim":           true,
	"source":         true,
	"tool":           true,
	"utility":        true,
	"workaround":     true,
}

func disposableMarker(comment string) string {
	words := commentWords(comment)
	if len(words) == 0 {
		return ""
	}
	for i, word := range words {
		if !markerWords[word] {
			continue
		}
		if directDeclaration(words, i) || selfDeclaration(words, i) {
			return word
		}
	}
	return ""
}

func directDeclaration(words []string, marker int) bool {
	if marker > 1 || marker == 1 && !isLabel(words[0]) {
		return false
	}
	if marker == len(words)-1 {
		return true
	}
	limit := marker + 5
	if limit > len(words) {
		limit = len(words)
	}
	for _, word := range words[marker+1 : limit] {
		if artifactWords[word] {
			return true
		}
	}
	return false
}

func selfDeclaration(words []string, marker int) bool {
	start := marker - 5
	if start < 0 {
		start = 0
	}
	self, artifact, copula := false, false, false
	for _, word := range words[start:marker] {
		switch {
		case word == "this" || word == "it":
			self = true
		case artifactWords[word]:
			artifact = true
		case word == "is" || word == "was" || word == "remains":
			copula = true
		}
	}
	if !copula || !(self || artifact) {
		return false
	}
	if self && !artifact {
		for _, word := range words[marker+1:] {
			if artifactWords[word] {
				return true
			}
		}
	}
	return artifact || marker == len(words)-1
}

func isLabel(word string) bool {
	switch word {
	case "hack", "note", "todo", "warning":
		return true
	default:
		return false
	}
}

func commentWords(comment string) []string {
	return strings.FieldsFunc(strings.ToLower(comment), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}
