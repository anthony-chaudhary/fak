package ctxmmu

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
)

// SkeletonizeGo parses Go source code and produces a lightweight structural
// skeleton: function and method bodies are elided into empty blocks while
// package declarations, imports, type definitions, interface methods,
// signatures, parameter types, return types, global constants/variables,
// and docstrings are preserved.
func SkeletonizeGo(src []byte) ([]byte, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", src, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("skeletonize parse: %w", err)
	}

	type bodySpan struct {
		start token.Pos
		end   token.Pos
	}
	var elidedSpans []bodySpan

	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Body != nil {
			elidedSpans = append(elidedSpans, bodySpan{
				start: fn.Body.Lbrace,
				end:   fn.Body.Rbrace,
			})
			fn.Body = &ast.BlockStmt{
				List: nil,
			}
		}
	}

	// Filter out comments originating inside elided function bodies,
	// while preserving package, type, field, global, and function doc comments.
	if len(elidedSpans) > 0 {
		var kept []*ast.CommentGroup
		for _, cg := range file.Comments {
			inside := false
			for _, sp := range elidedSpans {
				if cg.Pos() >= sp.start && cg.End() <= sp.end {
					inside = true
					break
				}
			}
			if !inside {
				kept = append(kept, cg)
			}
		}
		file.Comments = kept
	}

	var buf bytes.Buffer
	if err := format.Node(&buf, fset, file); err != nil {
		return nil, fmt.Errorf("skeletonize format: %w", err)
	}

	formatted, err := format.Source(buf.Bytes())
	if err == nil {
		return formatted, nil
	}
	return buf.Bytes(), nil
}
