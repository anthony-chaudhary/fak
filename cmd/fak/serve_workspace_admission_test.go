package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
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
func TestServeWorkspaceAdmissionModeFlagAndEnvironment(t *testing.T) {
	for _, tc := range []struct {
		name string
		env  map[string]string
		args []string
		want string // the resolved gateway.Config.WorkspaceAdmission value
	}{
		{name: "default open", want: "open"},
		{name: "flag strict", args: []string{"--workspace-admission=strict"}, want: "strict"},
		{name: "flag permissive", args: []string{"--workspace-admission=permissive"}, want: "permissive"},
		{name: "environment strict", env: map[string]string{"FAK_WORKSPACE_ADMISSION": "strict"}, want: "strict"},
		{name: "explicit flag beats environment", env: map[string]string{"FAK_WORKSPACE_ADMISSION": "open"}, args: []string{"--workspace-admission=strict"}, want: "strict"},
		{name: "legacy permissive bool alone resolves permissive", env: map[string]string{"FAK_WORKSPACE_ADMISSION_PERMISSIVE": "true"}, want: "permissive"},
		{name: "explicit mode beats legacy permissive bool", env: map[string]string{"FAK_WORKSPACE_ADMISSION_PERMISSIVE": "true"}, args: []string{"--workspace-admission=strict"}, want: "strict"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for key, value := range tc.env {
				t.Setenv(key, value)
			}
			fs, sf := newServeFlagSet()
			if err := parseServeArgs(fs, sf, tc.args); err != nil {
				t.Fatal(err)
			}
			got, err := resolveWorkspaceAdmission(*sf.workspaceAdmission, *sf.workspaceAdmissionPermissive)
			if err != nil {
				t.Fatalf("resolve: %v", err)
			}
			if got != tc.want {
				t.Fatalf("resolved mode=%q, want %q (flag=%q permissive=%v)", got, tc.want, *sf.workspaceAdmission, *sf.workspaceAdmissionPermissive)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestServeWorkspaceAdmissionUnknownTokenFailsLoud(t *testing.T) {
	fs, sf := newServeFlagSet()
	err := parseServeArgs(fs, sf, []string{"--workspace-admission=bogus"})
	if err == nil {
		t.Fatal("unknown --workspace-admission token must fail loud")
	}
	if !strings.Contains(err.Error(), "want open|permissive|strict") {
		t.Fatalf("error does not name the valid set: %v", err)
	}
}

// fak-test:runtime fast est=1s
func TestServeWorkspaceAdmissionPermissiveBuildGatewaySeam(t *testing.T) {
	file, err := parser.ParseFile(token.NewFileSet(), "serve.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var wired, modeWired bool
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
			if !ok {
				return true
			}
			switch key.Name {
			case "WorkspaceAdmissionPermissive":
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
			case "WorkspaceAdmission":
				modeWired = true
			}
			return true
		})
	}
	if !wired {
		t.Fatal("buildGateway does not pass the parsed workspace admission flag into gateway configuration")
	}
	if !modeWired {
		t.Fatal("buildGateway does not pass the resolved workspace admission mode into gateway configuration")
	}
}
