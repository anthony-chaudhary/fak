// Package pathlint is a static witness for one external-boundary claim: that every
// user-supplied filesystem-path flag is normalized before it reaches the OS.
//
// A CLI that declares `-gguf`/`-hf`/`-tok`/`-tokenizer`/`-dir` is implicitly claiming
// "give me a path and I'll open it." But Go's flag parsing never expands a leading ~,
// and PowerShell/most quoting pass ~ through literally, so `-gguf ~/m.gguf` is opened
// as a literal "~" directory and fails ("the system cannot find the path specified").
// The claim is only true if the value passes through pathutil.ExpandTilde. This package
// checks that claim from the source itself rather than trusting that each author
// remembered — the offenses it returns carry the closed-vocabulary reason
// UNEXPANDED_USER_PATH.
//
// Analysis is FUNCTION-scoped: each flag variable must be expanded within the same
// function that declares it. That matters for multi-subcommand binaries like cmd/fak,
// where several functions each declare their own `-dir`; a file-scoped check would be
// fooled into passing once any single one was fixed.
package pathlint

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
)

// ReasonUnexpandedUserPath is the closed-vocabulary refusal code for a path flag that
// never reaches ExpandTilde — the structured, machine-checkable form a DOS-style
// witness refuses with instead of free text.
const ReasonUnexpandedUserPath = "UNEXPANDED_USER_PATH"

// PathFlags is the vocabulary of flag names that name a user-supplied filesystem path
// which the command then opens (a model checkpoint, a HuggingFace dir, a tokenizer dir,
// or a model export dir). These are the names a user is most likely to hand as ~/...
var PathFlags = map[string]bool{
	"gguf":      true,
	"hf":        true,
	"tok":       true,
	"tokenizer": true,
	"dir":       true,
}

// Offense is one unexpanded path flag at one declaration site.
type Offense struct {
	Cmd  string // command dir name, e.g. "diagtok"
	Func string // enclosing function, e.g. "cmdRecall" (or "<package>" for a global)
	Flag string // flag name, e.g. "gguf"
}

func (o Offense) String() string {
	return fmt.Sprintf("cmd/%s %s(): -%s is a user path but never reaches pathutil.ExpandTilde (%s)",
		o.Cmd, o.Func, o.Flag, ReasonUnexpandedUserPath)
}

// ScanCmdTree walks <repoRoot>/cmd/* and returns every path-flag declaration that is
// not routed through ExpandTilde, sorted for stable output.
func ScanCmdTree(repoRoot string) ([]Offense, error) {
	cmdDir := filepath.Join(repoRoot, "cmd")
	entries, err := os.ReadDir(cmdDir)
	if err != nil {
		return nil, fmt.Errorf("read cmd dir: %w", err)
	}
	var offenses []Offense
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		off, err := scanCmd(filepath.Join(cmdDir, e.Name()), e.Name())
		if err != nil {
			return nil, err
		}
		offenses = append(offenses, off...)
	}
	sort.Slice(offenses, func(i, j int) bool {
		a, b := offenses[i], offenses[j]
		if a.Cmd != b.Cmd {
			return a.Cmd < b.Cmd
		}
		if a.Func != b.Func {
			return a.Func < b.Func
		}
		return a.Flag < b.Flag
	})
	return offenses, nil
}

// scanCmd analyzes one command directory's non-test Go sources.
func scanCmd(dir, name string) ([]Offense, error) {
	fset := token.NewFileSet()
	files, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", dir, err)
	}
	var parsed []*ast.File
	for _, f := range files {
		if f.IsDir() || !strings.HasSuffix(f.Name(), ".go") || strings.HasSuffix(f.Name(), "_test.go") {
			continue
		}
		af, perr := parser.ParseFile(fset, filepath.Join(dir, f.Name()), nil, 0)
		if perr != nil {
			continue // not parseable as standalone Go (e.g. a //go:build cuda cgo file); skip
		}
		parsed = append(parsed, af)
	}
	return analyzeFiles(name, parsed), nil
}

// analyzeFiles is the AST core, split out so it can be unit-tested on synthetic source.
// Each function is analyzed independently (its declared path flags must be expanded
// within it); package-level flag vars are checked against file-wide expansion.
func analyzeFiles(cmd string, files []*ast.File) []Offense {
	var offenses []Offense
	for _, f := range files {
		// Package-level flag vars (e.g. `var gguf = flag.String(...)`): scope is the file.
		fileExpanded := collectExpanded(f)
		for v, name := range packageFlagVars(f) {
			if !fileExpanded[v] {
				offenses = append(offenses, Offense{Cmd: cmd, Func: "<package>", Flag: name})
			}
		}
		// Function-local flag vars: scope is the declaring function.
		for _, d := range f.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			localExpanded := collectExpanded(fn.Body)
			for v, name := range localFlagVars(fn.Body) {
				if !localExpanded[v] {
					offenses = append(offenses, Offense{Cmd: cmd, Func: fn.Name.Name, Flag: name})
				}
			}
		}
	}
	return offenses
}

// collectExpanded returns the set of variable names passed to an ExpandTilde call
// anywhere under root (accepting both `ExpandTilde(*v)` and `ExpandTilde(v)`).
func collectExpanded(root ast.Node) map[string]bool {
	out := map[string]bool{}
	ast.Inspect(root, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok && isExpandTilde(call.Fun) && len(call.Args) == 1 {
			if v := argVarName(call.Args[0]); v != "" {
				out[v] = true
			}
		}
		return true
	})
	return out
}

// localFlagVars maps each `<var> := <recv>.String("<pathflag>", ...)` in body to its
// flag name.
func localFlagVars(body *ast.BlockStmt) map[string]string {
	out := map[string]string{}
	ast.Inspect(body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
			return true
		}
		lhs, ok := as.Lhs[0].(*ast.Ident)
		if !ok {
			return true
		}
		if name, ok := stringFlagName(as.Rhs[0]); ok && PathFlags[name] {
			out[lhs.Name] = name
		}
		return true
	})
	return out
}

// packageFlagVars maps each package-level `var <name> = <recv>.String("<pathflag>", ...)`
// to its flag name.
func packageFlagVars(f *ast.File) map[string]string {
	out := map[string]string{}
	for _, d := range f.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok || gd.Tok != token.VAR {
			continue
		}
		for _, spec := range gd.Specs {
			vs, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}
			for i, val := range vs.Values {
				if i >= len(vs.Names) {
					break
				}
				if name, ok := stringFlagName(val); ok && PathFlags[name] {
					out[vs.Names[i].Name] = name
				}
			}
		}
	}
	return out
}

// stringFlagName returns the flag name if expr is a `<x>.String("name", ...)` call.
func stringFlagName(expr ast.Expr) (string, bool) {
	call, ok := expr.(*ast.CallExpr)
	if !ok {
		return "", false
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || sel.Sel.Name != "String" || len(call.Args) == 0 {
		return "", false
	}
	lit, ok := call.Args[0].(*ast.BasicLit)
	if !ok || lit.Kind != token.STRING {
		return "", false
	}
	name, err := strconv.Unquote(lit.Value)
	if err != nil {
		return "", false
	}
	return name, true
}

// isExpandTilde reports whether fun names ExpandTilde (bare or pkg-qualified).
func isExpandTilde(fun ast.Expr) bool {
	switch f := fun.(type) {
	case *ast.Ident:
		return f.Name == "ExpandTilde"
	case *ast.SelectorExpr:
		return f.Sel.Name == "ExpandTilde"
	}
	return false
}

// argVarName extracts the variable name from an ExpandTilde argument, accepting both
// `ExpandTilde(*gguf)` (the *string flag) and `ExpandTilde(gguf)`.
func argVarName(arg ast.Expr) string {
	switch a := arg.(type) {
	case *ast.StarExpr:
		if id, ok := a.X.(*ast.Ident); ok {
			return id.Name
		}
	case *ast.Ident:
		return a.Name
	}
	return ""
}
