//go:build windows

package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestWindowsBackgroundSpawnsSuppressConsoleWindows guards diagnostic and
// recovery probes that run beneath an existing UI. A missed configuration here
// flashes a new console for every poll or recovery attempt.
func TestWindowsBackgroundSpawnsSuppressConsoleWindows(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, "_windows.go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		source, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(source), "exec.Command") {
			continue
		}
		file, err := parser.ParseFile(token.NewFileSet(), name, source, 0)
		if err != nil {
			t.Fatal(err)
		}
		ast.Inspect(file, func(node ast.Node) bool {
			if node == nil {
				return false
			}
			assign, ok := node.(*ast.AssignStmt)
			if !ok || len(assign.Lhs) != 1 || len(assign.Rhs) != 1 {
				return true
			}
			ident, ok := assign.Lhs[0].(*ast.Ident)
			call, isCall := assign.Rhs[0].(*ast.CallExpr)
			if !ok || !isCall || !isExecCommand(call) {
				return true
			}
			if !backgroundConfigured(file, ident.Name) {
				t.Errorf("%s: exec command %q lacks windowgate.ConfigureBackgroundCommand", filepath.Base(name), ident.Name)
			}
			return true
		})
	}
}

func isExecCommand(call *ast.CallExpr) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	pkg, isIdent := sel.X.(*ast.Ident)
	return isIdent && pkg.Name == "exec" && (sel.Sel.Name == "Command" || sel.Sel.Name == "CommandContext")
}

func backgroundConfigured(file *ast.File, commandName string) bool {
	configured := false
	ast.Inspect(file, func(node ast.Node) bool {
		if node == nil {
			return false
		}
		call, ok := node.(*ast.CallExpr)
		if !ok || len(call.Args) != 1 {
			return true
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		pkg, isIdent := sel.X.(*ast.Ident)
		arg, isArgIdent := call.Args[0].(*ast.Ident)
		if isIdent && isArgIdent && pkg.Name == "windowgate" && sel.Sel.Name == "ConfigureBackgroundCommand" && arg.Name == commandName {
			configured = true
		}
		return true
	})
	return configured
}
