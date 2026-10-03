package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// fak-test:runtime fast est=1s
func TestServeWorkspaceAdmissionPermissiveFlagAndEnvironment(t *testing.T) {
	for _, tc := range []struct {
		name, env string
		args      []string
		want      bool
	}{
		{name: "default"},
		{name: "flag", args: []string{"--workspace-admission-permissive"}, want: true},
		{name: "environment", env: "true", want: true},
		{name: "explicit false overrides environment", env: "true", args: []string{"--workspace-admission-permissive=false"}},
		{name: "explicit true overrides environment", env: "false", args: []string{"--workspace-admission-permissive=true"}, want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("FAK_WORKSPACE_ADMISSION_PERMISSIVE", tc.env)
			fs, sf := newServeFlagSet()
			if err := fs.Parse(tc.args); err != nil {
				t.Fatal(err)
			}
			if sf.workspaceAdmissionPermissive == nil || *sf.workspaceAdmissionPermissive != tc.want {
				t.Fatalf("workspace admission flag=%v, want %v", sf.workspaceAdmissionPermissive, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestServeWorkspaceAdmissionPermissiveBuildGatewaySeam(t *testing.T) {
	file, err := parser.ParseFile(token.NewFileSet(), "serve.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var wired bool
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "buildGateway" {
			continue
		}
		ast.Inspect(fn.Body, func(node ast.Node) bool {
			field, ok := node.(*ast.KeyValueExpr)
			if !ok {
				return true
			}
			key, ok := field.Key.(*ast.Ident)
			if !ok || key.Name != "WorkspaceAdmissionPermissive" {
				return true
			}
			deref, ok := field.Value.(*ast.StarExpr)
			if !ok {
				return true
			}
			selector, ok := deref.X.(*ast.SelectorExpr)
			if !ok || selector.Sel.Name != "workspaceAdmissionPermissive" {
				return true
			}
			owner, ok := selector.X.(*ast.Ident)
			wired = ok && owner.Name == "sf"
			return true
		})
	}
	if !wired {
		t.Fatal("buildGateway does not pass the parsed workspace admission flag into gateway configuration")
	}
}
