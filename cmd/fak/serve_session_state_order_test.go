package main

import (
	"context"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// Estimate only; this test has not been timed.
// fak-test:runtime fast est=100ms lane=default
func TestServeSessionStateRestorePrecedesExpensiveBoot(t *testing.T) {
	root := repoRootFromTest(t)
	fset := token.NewFileSet()
	parse := func(name string) *ast.File {
		f, err := parser.ParseFile(fset, filepath.Join(root, "cmd", "fak", name), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		return f
	}
	callName := func(c *ast.CallExpr) string {
		switch x := c.Fun.(type) {
		case *ast.Ident:
			return x.Name
		case *ast.SelectorExpr:
			return x.Sel.Name
		}
		return ""
	}
	var boot *ast.FuncDecl
	for _, decl := range parse("serve.go").Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "cmdServe" {
			boot = fn
		}
	}
	if boot == nil {
		t.Fatal("actual cmdServe owner missing")
	}
	calls := map[string][]token.Pos{}
	ast.Inspect(boot.Body, func(n ast.Node) bool {
		if c, ok := n.(*ast.CallExpr); ok {
			name := callName(c)
			calls[name] = append(calls[name], c.Pos())
		}
		return true
	})
	restores := calls["restoreServeSessions"]
	if len(restores) != 1 {
		t.Fatalf("cmdServe actual restore count = %d, want 1", len(restores))
	}
	restore := restores[0]
	// Require a direct top-level error-check initializer, not a nested/dead branch.
	direct := false
	for _, stmt := range boot.Body.List {
		guard, ok := stmt.(*ast.IfStmt)
		if !ok {
			continue
		}
		assign, ok := guard.Init.(*ast.AssignStmt)
		if !ok || len(assign.Rhs) != 1 {
			continue
		}
		c, ok := assign.Rhs[0].(*ast.CallExpr)
		if !ok || c.Pos() != restore {
			continue
		}
		if len(c.Args) != 2 {
			t.Fatal("restore lost table/path arguments")
		}
		tbl, ok := c.Args[0].(*ast.Ident)
		if !ok || tbl.Name != "serveSessions" {
			t.Fatal("restore no longer targets live session table")
		}
		path, ok := c.Args[1].(*ast.StarExpr)
		if !ok {
			t.Fatal("restore no longer consumes resolved path")
		}
		sel, ok := path.X.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "sessionStatePath" {
			t.Fatal("restore path changed")
		}
		direct = true
	}
	if !direct {
		t.Fatal("session restore is not a direct cmdServe startup gate")
	}
	for _, name := range []string{"runServePolicyCheck", "runServeSizingJSON", "runServeOpenCodeConfig", "runServePiConfig", "runServeCodexConfig", "configureServeToolEngines"} {
		if len(calls[name]) == 0 {
			t.Fatalf("startup boundary %s missing", name)
		}
		for _, at := range calls[name] {
			if at >= restore {
				t.Fatalf("restore precedes dry-run/validation boundary %s", name)
			}
		}
	}
	for _, name := range []string{"maybeStartQwen38Delegation", "resolveCompute", "loadServeModelWithVulkanLease", "loadModel", "resolveSessionPlane", "buildGateway"} {
		if len(calls[name]) == 0 {
			t.Fatalf("expensive/later boundary %s missing", name)
		}
		for _, at := range calls[name] {
			if at <= restore {
				t.Fatalf("restore follows %s", name)
			}
		}
	}
	ast.Inspect(parse("serve_stages.go"), func(n ast.Node) bool {
		if c, ok := n.(*ast.CallExpr); ok && callName(c) == "restoreServeSessions" {
			t.Fatal("late stage rereads/restores the snapshot")
		}
		return true
	})
}

// Estimate only; this test has not been timed. Reuses the existing cmdServe child harness.
// fak-test:runtime medium est=5s lane=default
func TestServeSessionStateCorruptionRefusesBeforeCompute(t *testing.T) {
	const backendSentinel = "session-state-order-unregistered-backend"
	for _, mode := range []string{"default", "explicit", "off"} {
		t.Run(mode, func(t *testing.T) {
			root := t.TempDir()
			home, config := filepath.Join(root, "home"), filepath.Join(root, "config")
			defaultConfig := config
			if runtime.GOOS == "darwin" {
				defaultConfig = filepath.Join(home, "Library", "Application Support")
			}
			snapshotPath := filepath.Join(defaultConfig, "fak", serveSessionStateFile)
			if err := os.MkdirAll(filepath.Dir(snapshotPath), 0o700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(snapshotPath, []byte("{not a valid snapshot envelope"), 0o600); err != nil {
				t.Fatal(err)
			}
			args := []string{"-test.run=^TestControlIngressLiveServeHelper$", "--", "--addr", "127.0.0.1:0", "--mock", "--keep-awake", "off", "--backend", backendSentinel}
			if mode == "explicit" {
				args = append(args, "--session-state", snapshotPath)
			}
			if mode == "off" {
				args = append(args, "--session-state", "off")
			}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], args...)
			cmd.Dir = root
			for _, entry := range os.Environ() {
				key, _, _ := strings.Cut(entry, "=")
				if strings.HasPrefix(key, "FAK_") || key == "HOME" || key == "USERPROFILE" || key == "XDG_CONFIG_HOME" || key == "APPDATA" {
					continue
				}
				cmd.Env = append(cmd.Env, entry)
			}
			cmd.Env = append(cmd.Env, "HOME="+home, "USERPROFILE="+home, "XDG_CONFIG_HOME="+config, "APPDATA="+config, controlIngressLiveHelperEnv+"=1")
			output, err := cmd.CombinedOutput()
			if ctx.Err() != nil {
				t.Fatalf("startup refusal timed out: %s", output)
			}
			var exit *exec.ExitError
			if !errors.As(err, &exit) {
				t.Fatalf("expected refusal exit: %v, output=%s", err, output)
			}
			if mode == "off" {
				if !strings.Contains(string(output), backendSentinel) || strings.Contains(string(output), "--session-state "+snapshotPath) {
					t.Fatalf("off did not preserve compute-stage control: %s", output)
				}
			} else if exit.ExitCode() != 1 || !strings.Contains(string(output), "--session-state "+snapshotPath) || !strings.Contains(string(output), "snapshot:") || strings.Contains(string(output), backendSentinel) {
				t.Fatalf("snapshot did not refuse before compute: exit=%d output=%s", exit.ExitCode(), output)
			}
		})
	}
}
