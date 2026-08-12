package devindex

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const vsLexiconFloor = 20

type PreState uint8

const (
	PreUnverified PreState = iota
	PreNone
	PreStructural
	PreRuntime
	PreNotApplicable
)

func (p PreState) String() string {
	switch p {
	case PreNone:
		return "NONE"
	case PreStructural:
		return "STRUCTURAL"
	case PreRuntime:
		return "RUNTIME"
	case PreNotApplicable:
		return "N/A"
	default:
		return "UNVERIFIED"
	}
}

type SurfacePre struct {
	State PreState
	Codes []string
	Notes []string
}

type ReasonLexicon struct {
	Codes map[string]struct{}
}

// BuildReasonLexicon derives fak's refusal vocabulary from Go source. It deliberately
// does not use runtime help or a hand-maintained table: reason constants and typed
// string literals are the source of truth.
func BuildReasonLexicon(root string) (*ReasonLexicon, error) {
	dirs := []string{"cmd/fak", "internal"}
	codes := map[string]struct{}{}
	fset := token.NewFileSet()
	for _, rel := range dirs {
		base := filepath.Join(root, filepath.FromSlash(rel))
		err := filepath.WalkDir(base, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				if path != base && (d.Name() == "testdata" || strings.HasPrefix(d.Name(), ".")) {
					return filepath.SkipDir
				}
				return nil
			}
			if filepath.Ext(path) != ".go" || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			src, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			file, err := parser.ParseFile(fset, path, src, 0)
			if err != nil {
				// Shared-trunk peers can be mid-write. A malformed untracked/modified file
				// cannot define a committed refusal vocabulary, so omit it rather than
				// making this source index nondeterministically un-runnable.
				if rel, relErr := filepath.Rel(root, path); relErr == nil && !vsGitTracked(root, filepath.ToSlash(rel)) {
					return nil
				}
				return fmt.Errorf("parse %s: %w", path, err)
			}
			ast.Inspect(file, func(n ast.Node) bool {
				lit, ok := n.(*ast.BasicLit)
				if !ok || lit.Kind != token.STRING {
					return true
				}
				value, err := strconv.Unquote(lit.Value)
				if err != nil {
					return true
				}
				for _, code := range refusalCodesInString(value) {
					codes[code] = struct{}{}
				}
				return true
			})
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return &ReasonLexicon{Codes: codes}, nil
}

func refusalCodesInString(s string) []string {
	fields := strings.FieldsFunc(s, func(r rune) bool {
		return !(r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_')
	})
	seen := map[string]bool{}
	var out []string
	for _, f := range fields {
		if len(f) < 4 || !strings.Contains(f, "_") || strings.Trim(f, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != "" || !looksLikeRefusalCode(f) || seen[f] {
			continue
		}
		seen[f] = true
		out = append(out, f)
	}
	sort.Strings(out)
	return out
}

func looksLikeRefusalCode(code string) bool {
	parts := strings.Split(code, "_")
	if len(parts) < 2 {
		return false
	}
	// Reject environment/configuration identifiers (CODEX_HOME, API_KEY, ...).
	// Refusal codes use verdict nouns or guard/action words in at least one segment.
	words := map[string]bool{
		"BLOCK": true, "BLOCKED": true, "BUSY": true, "DENY": true, "DENIED": true,
		"ERROR": true, "FAIL": true, "FAILED": true, "FORBIDDEN": true, "INVALID": true,
		"LOCK": true, "MISSING": true, "NOT": true, "OFF": true, "OVERLAP": true,
		"REFUSE": true, "REFUSED": true, "REQUIRED": true, "STALE": true, "TIMEOUT": true,
		"UNAVAILABLE": true, "UNVERIFIED": true, "UNWIRED": true, "VIOLATION": true,
	}
	for _, part := range parts {
		if words[part] {
			return true
		}
	}
	return false
}
func vsPreconditions(p *vsPkg, leaf SurfaceLeaf, reach []string, lex *ReasonLexicon) SurfacePre {
	if leaf.Fn == "" {
		return SurfacePre{State: PreUnverified}
	}
	found := map[string]bool{}
	runtime := false
	for _, fn := range reach {
		decl := p.funcs[fn]
		if decl == nil {
			continue
		}
		ast.Inspect(decl.Body, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.BasicLit:
				if n.Kind != token.STRING {
					return true
				}
				value, err := strconv.Unquote(n.Value)
				if err != nil {
					return true
				}
				for _, code := range refusalCodesInString(value) {
					if _, ok := lex.Codes[code]; ok {
						found[code] = true
					}
				}
			case *ast.SelectorExpr:
				name := n.Sel.Name
				if strings.HasPrefix(name, "Reason") {
					code := camelReasonCode(strings.TrimPrefix(name, "Reason"))
					if _, ok := lex.Codes[code]; ok {
						found[code] = true
					}
				}
			case *ast.CallExpr:
				if sel, ok := n.Fun.(*ast.SelectorExpr); ok {
					if id, ok := sel.X.(*ast.Ident); ok && (id.Name == "exec" || id.Name == "http" || id.Name == "net") {
						runtime = true
					}
				}
			}
			return true
		})
	}
	codes := make([]string, 0, len(found))
	for code := range found {
		codes = append(codes, code)
	}
	sort.Strings(codes)
	if len(codes) > 0 {
		return SurfacePre{State: PreStructural, Codes: codes}
	}
	if runtime {
		return SurfacePre{State: PreRuntime}
	}
	return SurfacePre{State: PreNone}
}

func camelReasonCode(s string) string {
	var b strings.Builder
	for i, r := range s {
		if i > 0 && r >= 'A' && r <= 'Z' {
			b.WriteByte('_')
		}
		b.WriteRune(r)
	}
	return strings.ToUpper(b.String())
}

func vsGitTracked(root, rel string) bool {
	cmd := windowgate.Command("git", "-C", root, "ls-files", "--error-unmatch", "--", rel)
	return cmd.Run() == nil
}
