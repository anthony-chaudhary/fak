package hooks

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"unicode"
)

const commentQualityGate = "COMMENT_QUALITY"

// gateCommentQuality gives changed implementation comments a deliberately narrow advisory pass.
// It ignores documentation and special comments, and only reports conspicuously long blocks or
// stock narration. Comment style is contextual, so this gate must remain warn-by-default.
func gateCommentQuality(d *StagedDiff) ([]Finding, error) {
	var findings []Finding
	candidates := 0
	for _, path := range d.StagedPaths {
		if filepath.Ext(path) != ".go" || strings.HasSuffix(path, ".pb.go") || strings.HasSuffix(path, "_generated.go") {
			continue
		}
		added := d.AddedByFile[path]
		if len(added) == 0 {
			continue
		}
		content, ok := d.FileBytes(path)
		if !ok {
			return nil, ErrCouldNotRun
		}
		if generatedGo(content) {
			continue
		}
		fset := token.NewFileSet()
		file, err := parser.ParseFile(fset, path, content, parser.ParseComments)
		if err != nil {
			continue // syntax belongs to codelint/build gates; do not duplicate it here
		}
		docs := documentationGroups(file)
		addedLines := make(map[int]bool, len(added))
		for _, line := range added {
			addedLines[line.New] = true
		}
		for _, group := range file.Comments {
			if docs[group] || !implementationComment(file, group) || !commentGroupTouches(group, fset, addedLines) || specialCommentGroup(group) {
				continue
			}
			candidates++
			text := strings.TrimSpace(group.Text())
			lineCount := fset.Position(group.End()).Line - fset.Position(group.Pos()).Line + 1
			if narrationComment(text) {
				findings = append(findings, Finding{Gate: commentQualityGate, File: path, Line: fset.Position(group.Pos()).Line, Detail: "comment narrates the code; remove it or state the non-obvious reason"})
				continue
			}
			if lineCount >= 6 && len(text) >= 420 {
				findings = append(findings, Finding{Gate: commentQualityGate, File: path, Line: fset.Position(group.Pos()).Line, Detail: "long implementation comment; tighten it or move durable explanation to documentation"})
			}
		}
		for _, decl := range file.Decls {
			checkDocTautology(decl, fset, addedLines, path, &findings)
		}
	}
	d.NoteCandidates(commentQualityGate, candidates, "changed implementation comment blocks")
	return findings, nil
}

func documentationGroups(file *ast.File) map[*ast.CommentGroup]bool {
	out := make(map[*ast.CommentGroup]bool)
	if file.Doc != nil {
		out[file.Doc] = true
	}
	for _, decl := range file.Decls {
		switch n := decl.(type) {
		case *ast.FuncDecl:
			if n.Doc != nil {
				out[n.Doc] = true
			}
		case *ast.GenDecl:
			if n.Doc != nil {
				out[n.Doc] = true
			}
			for _, spec := range n.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					if s.Doc != nil {
						out[s.Doc] = true
					}
				case *ast.ValueSpec:
					if s.Doc != nil {
						out[s.Doc] = true
					}
				case *ast.ImportSpec:
					if s.Doc != nil {
						out[s.Doc] = true
					}
				}
			}
		}
	}
	return out
}

func implementationComment(file *ast.File, group *ast.CommentGroup) bool {
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if ok && strings.HasPrefix(fn.Name.Name, "Example") {
			continue
		}
		if ok && fn.Body != nil && group.Pos() >= fn.Body.Pos() && group.End() <= fn.Body.End() {
			return true
		}
	}
	return false
}
func commentGroupTouches(group *ast.CommentGroup, fset *token.FileSet, added map[int]bool) bool {
	for line := fset.Position(group.Pos()).Line; line <= fset.Position(group.End()).Line; line++ {
		if added[line] {
			return true
		}
	}
	return false
}

func generatedGo(content []byte) bool {
	head := string(content)
	if len(head) > 2048 {
		head = head[:2048]
	}
	return strings.Contains(head, "Code generated ") && strings.Contains(head, " DO NOT EDIT.")
}

func specialCommentGroup(group *ast.CommentGroup) bool {
	text := strings.TrimSpace(group.Text())
	lower := strings.ToLower(text)
	return strings.HasPrefix(text, "go:") || strings.HasPrefix(text, "+build") ||
		strings.Contains(lower, "copyright") || strings.Contains(lower, "license") ||
		strings.HasPrefix(lower, "nolint") || strings.HasPrefix(lower, "lint:")
}

func narrationComment(text string) bool {
	lower := strings.ToLower(strings.TrimSpace(text))
	for _, prefix := range []string{"this function loops ", "this function iterates ", "this method loops ", "this method iterates ", "the following code loops ", "the following code iterates ", "here we loop ", "here we iterate ", "now we loop ", "now we iterate "} {
		if strings.HasPrefix(lower, prefix) {
			return true
		}
	}
	return false
}

func checkDocTautology(decl ast.Decl, fset *token.FileSet, addedLines map[int]bool, path string, findings *[]Finding) {
	switch d := decl.(type) {
	case *ast.FuncDecl:
		if ast.IsExported(d.Name.Name) && d.Doc != nil && commentGroupTouches(d.Doc, fset, addedLines) {
			if isTautologicalDoc(d.Name.Name, d.Doc.Text()) {
				*findings = append(*findings, Finding{
					Gate:   commentQualityGate,
					File:   path,
					Line:   fset.Position(d.Doc.Pos()).Line,
					Detail: fmt.Sprintf("doc comment on %s is tautological; explain semantics or non-obvious invariants rather than restating the identifier", d.Name.Name),
				})
			}
		}
	case *ast.GenDecl:
		for _, spec := range d.Specs {
			switch s := spec.(type) {
			case *ast.TypeSpec:
				if ast.IsExported(s.Name.Name) {
					doc := s.Doc
					if doc == nil {
						doc = d.Doc
					}
					if doc != nil && commentGroupTouches(doc, fset, addedLines) {
						if isTautologicalDoc(s.Name.Name, doc.Text()) {
							*findings = append(*findings, Finding{
								Gate:   commentQualityGate,
								File:   path,
								Line:   fset.Position(doc.Pos()).Line,
								Detail: fmt.Sprintf("doc comment on %s is tautological; explain semantics or non-obvious invariants rather than restating the identifier", s.Name.Name),
							})
						}
					}
				}
			case *ast.ValueSpec:
				for _, name := range s.Names {
					if ast.IsExported(name.Name) {
						doc := s.Doc
						if doc == nil {
							doc = d.Doc
						}
						if doc != nil && commentGroupTouches(doc, fset, addedLines) {
							if isTautologicalDoc(name.Name, doc.Text()) {
								*findings = append(*findings, Finding{
									Gate:   commentQualityGate,
									File:   path,
									Line:   fset.Position(doc.Pos()).Line,
									Detail: fmt.Sprintf("doc comment on %s is tautological; explain semantics or non-obvious invariants rather than restating the identifier", name.Name),
								})
							}
						}
					}
				}
			}
		}
	}
}

func splitIdentifierWords(name string) map[string]bool {
	set := make(map[string]bool)
	set[strings.ToLower(name)] = true
	var curr strings.Builder
	for i, r := range name {
		if r == '_' || r == '-' {
			if curr.Len() > 0 {
				set[strings.ToLower(curr.String())] = true
				curr.Reset()
			}
			continue
		}
		if unicode.IsUpper(r) && i > 0 && curr.Len() > 0 {
			set[strings.ToLower(curr.String())] = true
			curr.Reset()
		}
		curr.WriteRune(r)
	}
	if curr.Len() > 0 {
		set[strings.ToLower(curr.String())] = true
	}
	return set
}

func isTautologicalDoc(name string, text string) bool {
	nameLower := strings.ToLower(name)
	text = strings.TrimSpace(text)
	if text == "" {
		return true
	}
	fields := strings.Fields(text)
	if len(fields) == 0 {
		return true
	}
	firstWord := strings.Trim(strings.ToLower(fields[0]), ":,.-()")
	if firstWord != nameLower && !strings.HasPrefix(strings.ToLower(text), nameLower) {
		return false
	}
	remainder := strings.TrimSpace(text[len(firstWord):])
	words := strings.FieldsFunc(remainder, func(r rune) bool {
		return unicode.IsSpace(r) || unicode.IsPunct(r)
	})

	fillers := map[string]bool{
		"is": true, "are": true, "does": true, "do": true, "returns": true, "return": true,
		"represents": true, "represent": true, "holds": true, "hold": true, "the": true,
		"a": true, "an": true, "of": true, "for": true, "to": true, "that": true, "which": true,
		"will": true, "can": true, "provides": true, "provide": true, "specifies": true,
		"specify": true, "defines": true, "define": true, "indicates": true, "indicate": true,
		"details": true, "detail": true, "records": true, "record": true, "encapsulates": true,
		"encapsulate": true, "captures": true, "capture": true, "contains": true, "contain": true,
	}

	nameParts := splitIdentifierWords(name)
	meaningfulWords := 0
	for _, w := range words {
		wl := strings.ToLower(w)
		if fillers[wl] || nameParts[wl] {
			continue
		}
		meaningfulWords++
	}
	return meaningfulWords < 2
}
