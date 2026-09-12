package taskrun

import (
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
)

// validateCandidateSource permits a candidate to change bytes only inside one
// original top-level function body. Package text, imports, signatures, comments,
// directives, globals, and every other declaration remain byte-identical.
func validateCandidateSource(original, candidate string) error {
	if strings.Contains(candidate, "//go:") || strings.Contains(candidate, "/*go:") {
		return errors.New("candidate contains a Go compiler directive")
	}
	originalSet := token.NewFileSet()
	originalFile, err := parser.ParseFile(originalSet, "original.go", original, parser.ParseComments|parser.AllErrors)
	if err != nil {
		return fmt.Errorf("parse original fixture: %w", err)
	}
	candidateSet := token.NewFileSet()
	candidateFile, err := parser.ParseFile(candidateSet, "candidate.go", candidate, parser.ParseComments|parser.AllErrors)
	if err != nil {
		return fmt.Errorf("parse candidate source: %w", err)
	}

	originalBodies := functionBodies(originalSet, originalFile)
	candidateBodies := functionBodies(candidateSet, candidateFile)
	if len(originalBodies) == 0 || len(originalBodies) != len(candidateBodies) {
		return errors.New("candidate changed top-level function declarations")
	}
	for index, originalBody := range originalBodies {
		candidateBody := candidateBodies[index]
		if originalBody.name != candidateBody.name {
			continue
		}
		if bytes.Equal([]byte(original[:originalBody.open+1]), []byte(candidate[:candidateBody.open+1])) &&
			bytes.Equal([]byte(original[originalBody.close:]), []byte(candidate[candidateBody.close:])) {
			return nil
		}
	}
	return errors.New("candidate changed source outside one target function body")
}

type sourceFunctionBody struct {
	name        string
	open, close int
}

func functionBodies(files *token.FileSet, file *ast.File) []sourceFunctionBody {
	bodies := make([]sourceFunctionBody, 0)
	for _, declaration := range file.Decls {
		function, ok := declaration.(*ast.FuncDecl)
		if !ok || function.Body == nil {
			continue
		}
		positionFile := files.File(function.Pos())
		if positionFile == nil {
			continue
		}
		bodies = append(bodies, sourceFunctionBody{
			name:  function.Name.Name,
			open:  positionFile.Offset(function.Body.Lbrace),
			close: positionFile.Offset(function.Body.Rbrace),
		})
	}
	return bodies
}
