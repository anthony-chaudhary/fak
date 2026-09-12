//go:build windows

package main

import (
	"bytes"
	"go/ast"
	"go/format"
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
		for _, command := range unconfiguredBackgroundCommands(name, file) {
			t.Errorf("%s: exec command %q lacks a windowless configuration or the console-inheritance witness", filepath.Base(name), command)
		}
	}
}

func unconfiguredBackgroundCommands(name string, file *ast.File) []string {
	var missing []string
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		ast.Inspect(fn.Body, func(node ast.Node) bool {
			assign, ok := node.(*ast.AssignStmt)
			if !ok || len(assign.Lhs) != 1 || len(assign.Rhs) != 1 {
				return true
			}
			ident, ok := assign.Lhs[0].(*ast.Ident)
			call, isCall := assign.Rhs[0].(*ast.CallExpr)
			if ok && isCall && isExecCommand(call) &&
				!backgroundConfigured(fn.Body, ident) &&
				!consoleInheritanceWitness(name, fn, assign, ident) {
				missing = append(missing, ident.Name)
			}
			return true
		})
	}
	return missing
}

func isExecCommand(call *ast.CallExpr) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	pkg, isIdent := sel.X.(*ast.Ident)
	return isIdent && pkg.Name == "exec" && (sel.Sel.Name == "Command" || sel.Sel.Name == "CommandContext")
}

func backgroundConfigured(body *ast.BlockStmt, command *ast.Ident) bool {
	configured := false
	ast.Inspect(body, func(node ast.Node) bool {
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
		if isIdent && isArgIdent && pkg.Name == "windowgate" && (sel.Sel.Name == "ConfigureBackgroundCommand" || sel.Sel.Name == "ConfigureDetachedCommand") && sameCommand(arg, command) {
			configured = true
		}
		return true
	})
	return configured
}

// Parser object identity prevents a same-named command in another function or
// nested scope from satisfying this command's configuration requirement.
func sameCommand(a, b *ast.Ident) bool {
	return a.Obj != nil && a.Obj == b.Obj
}

func spawnSyntax(node ast.Node) string {
	var b bytes.Buffer
	_ = format.Node(&b, token.NewFileSet(), node)
	return b.String()
}

// This one ordinary descendant intentionally inherits the attended root console.
// Accept it only while its positive source contract holds; changing its spawn
// mode would erase the console-inheritance diagnostic.
func consoleInheritanceWitness(name string, fn *ast.FuncDecl, command *ast.AssignStmt, ident *ast.Ident) bool {
	if filepath.Base(name) != "windowgate_selfcheck_windows.go" || fn.Name.Name != "runDesktopConsoleSelfcheckRoot" ||
		spawnSyntax(command) != `cmd := exec.Command(path, "windowgate", "--selfcheck")` {
		return false
	}
	required := map[string]bool{
		`childEnv := envMap(os.Environ())`:                   false,
		`childEnv[desktopConsoleSelfcheckRoleEnv] = "child"`: false,
		`childEnv[desktopConsoleSelfcheckLabelEnv] = label`:  false,
		`childEnv[desktopConsoleSelfcheckDirEnv] = dir`:      false,
		`cmd.Env = envSliceFromMap(childEnv)`:                false,
		`cmd.Stdout, cmd.Stderr = stdout, stderr`:            false,
	}
	valid := true
	ast.Inspect(fn.Body, func(node ast.Node) bool {
		if assign, ok := node.(*ast.AssignStmt); ok {
			syntax := spawnSyntax(assign)
			if _, ok := required[syntax]; ok {
				required[syntax] = true
			}
			for _, lhs := range assign.Lhs {
				if sel, ok := lhs.(*ast.SelectorExpr); ok {
					if id, ok := sel.X.(*ast.Ident); ok && sameCommand(id, ident) {
						if _, allowed := required[syntax]; !allowed {
							valid = false
						}
					}
				}
			}
		}
		if call, ok := node.(*ast.CallExpr); ok {
			// Passing the command to any helper could override inheritance.
			for _, arg := range call.Args {
				if id, ok := arg.(*ast.Ident); ok && sameCommand(id, ident) {
					valid = false
				}
			}
		}
		return true
	})
	for _, present := range required {
		valid = valid && present
	}
	return valid
}

func TestBackgroundConfiguredFixtures(t *testing.T) {
	for _, tc := range []struct {
		name, source string
		missing      int
	}{
		{"ordinary", `func probe() { cmd := exec.Command("helper"); cmd.Run() }`, 1},
		{"background", `func probe() { cmd := exec.Command("helper"); windowgate.ConfigureBackgroundCommand(cmd); cmd.Run() }`, 0},
		{"detached", `func probe() { cmd := exec.CommandContext(ctx, "helper"); windowgate.ConfigureDetachedCommand(cmd); cmd.Run() }`, 0},
		{"other function", `func probe() { cmd := exec.Command("helper"); cmd.Run() }; func other() { cmd := exec.Command("helper"); windowgate.ConfigureBackgroundCommand(cmd) }`, 1},
		{"shadowed command", `func probe() { cmd := exec.Command("helper"); { cmd := exec.Command("other"); windowgate.ConfigureBackgroundCommand(cmd) }; cmd.Run() }`, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			file, err := parser.ParseFile(token.NewFileSet(), "fixture_windows.go", "package main\n"+tc.source, 0)
			if err != nil {
				t.Fatal(err)
			}
			if got := len(unconfiguredBackgroundCommands("fixture_windows.go", file)); got != tc.missing {
				t.Fatalf("unconfigured commands = %d, want %d", got, tc.missing)
			}
		})
	}
}

func TestBackgroundConfiguredConsoleInheritanceWitness(t *testing.T) {
	file, err := parser.ParseFile(token.NewFileSet(), "windowgate_selfcheck_windows.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	witnesses := 0
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		ast.Inspect(fn.Body, func(node ast.Node) bool {
			assign, ok := node.(*ast.AssignStmt)
			if !ok || len(assign.Lhs) != 1 || len(assign.Rhs) != 1 {
				return true
			}
			ident, ok := assign.Lhs[0].(*ast.Ident)
			call, isCall := assign.Rhs[0].(*ast.CallExpr)
			if ok && isCall && isExecCommand(call) && consoleInheritanceWitness("windowgate_selfcheck_windows.go", fn, assign, ident) {
				witnesses++
			}
			return true
		})
	}
	if witnesses != 1 {
		t.Fatalf("production console-inheritance witnesses = %d, want 1", witnesses)
	}
	const source = `package main
func runDesktopConsoleSelfcheckRoot() {
	childEnv := envMap(os.Environ())
	childEnv[desktopConsoleSelfcheckRoleEnv] = "child"
	childEnv[desktopConsoleSelfcheckLabelEnv] = label
	childEnv[desktopConsoleSelfcheckDirEnv] = dir
	cmd := exec.Command(path, "windowgate", "--selfcheck")
	cmd.Env = envSliceFromMap(childEnv)
	cmd.Stdout, cmd.Stderr = stdout, stderr
	cmd.Start()
}
`
	for _, tc := range []struct {
		name, old, replacement string
		want                   bool
	}{
		{"ordinary inheritance", "", "", true},
		{"wrong role", `"child"`, `"root"`, false},
		{"missing environment", "cmd.Env = envSliceFromMap(childEnv)", "", false},
		{"redirected streams", "cmd.Stdout, cmd.Stderr = stdout, stderr", "cmd.Stdout, cmd.Stderr = nil, nil", false},
		{"process override", "cmd.Start()", "cmd.SysProcAttr = attr; cmd.Start()", false},
		{"helper override", "cmd.Start()", "configure(cmd); cmd.Start()", false},
		{"background override", "cmd.Start()", "windowgate.ConfigureBackgroundCommand(cmd); cmd.Start()", false},
		{"wrong function", "runDesktopConsoleSelfcheckRoot", "ordinaryHelper", false},
		{"wrong command", `"--selfcheck"`, `"--other"`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := source
			if tc.old != "" {
				src = strings.Replace(src, tc.old, tc.replacement, 1)
			}
			file, err := parser.ParseFile(token.NewFileSet(), "windowgate_selfcheck_windows.go", src, 0)
			if err != nil {
				t.Fatal(err)
			}
			fn := file.Decls[0].(*ast.FuncDecl)
			var got bool
			ast.Inspect(fn.Body, func(node ast.Node) bool {
				assign, ok := node.(*ast.AssignStmt)
				if !ok || len(assign.Lhs) != 1 || len(assign.Rhs) != 1 {
					return true
				}
				call, ok := assign.Rhs[0].(*ast.CallExpr)
				if ok && isExecCommand(call) {
					got = consoleInheritanceWitness("windowgate_selfcheck_windows.go", fn, assign, assign.Lhs[0].(*ast.Ident))
				}
				return true
			})
			if got != tc.want {
				t.Fatalf("inheritance witness = %v, want %v", got, tc.want)
			}
		})
	}
}
