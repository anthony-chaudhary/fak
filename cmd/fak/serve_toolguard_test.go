package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1s
func TestServeToolGuardResolution(t *testing.T) {
	for _, tc := range []struct {
		name string
		env  map[string]string
		args []string
		want string // the resolved gateway.Config.ToolGuard value
		err  bool
	}{
		{name: "default is raw pass-through", want: "off"},
		{name: "explicit --guard arms the floor", args: []string{"--guard"}, want: "on"},
		{name: "explicit --guard=true arms the floor", args: []string{"--guard=true"}, want: "on"},
		{name: "explicit --guard=false stays raw", args: []string{"--guard=false"}, want: "off"},
		{name: "FAK_GUARD=on arms the floor", env: map[string]string{"FAK_GUARD": "on"}, want: "on"},
		{name: "FAK_GUARD=off stays raw", env: map[string]string{"FAK_GUARD": "off"}, want: "off"},
		{name: "FAK_GUARD empty resolves the default", env: map[string]string{"FAK_GUARD": ""}, want: "off"},
		{name: "explicit flag beats FAK_GUARD=off", env: map[string]string{"FAK_GUARD": "off"}, args: []string{"--guard"}, want: "on"},
		{name: "explicit --guard=false beats FAK_GUARD=on", env: map[string]string{"FAK_GUARD": "on"}, args: []string{"--guard=false"}, want: "off"},
		{name: "unparsable FAK_GUARD fails loud", env: map[string]string{"FAK_GUARD": "maybe"}, err: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for key, value := range tc.env {
				t.Setenv(key, value)
			}
			fs, sf := newServeFlagSet()
			if err := parseServeArgs(fs, sf, tc.args); err != nil {
				t.Fatal(err)
			}
			got, err := resolveServeToolGuard(*sf.guard, explicitFlagNames(fs), os.Getenv)
			if tc.err {
				if err == nil {
					t.Fatalf("resolved %q, want error", got)
				}
				if !strings.Contains(err.Error(), "FAK_GUARD") {
					t.Fatalf("error does not name the offending env var: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("resolve: %v", err)
			}
			if got != tc.want {
				t.Fatalf("resolved posture=%q, want %q (flag=%v)", got, tc.want, *sf.guard)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestServeToolGuardEnvSpellings(t *testing.T) {
	for _, tc := range []struct {
		raw  string
		want string
	}{
		{raw: "ON", want: "on"},
		{raw: " on ", want: "on"},
		{raw: "1", want: "on"},
		{raw: "true", want: "on"},
		{raw: "yes", want: "on"},
		{raw: "enabled", want: "on"},
		{raw: "OFF", want: "off"},
		{raw: "0", want: "off"},
		{raw: "false", want: "off"},
		{raw: "no", want: "off"},
		{raw: "disabled", want: "off"},
		{raw: "raw", want: "off"},
	} {
		t.Run(fmt.Sprintf("%q", tc.raw), func(t *testing.T) {
			t.Setenv("FAK_GUARD", tc.raw)
			got, err := resolveServeToolGuard(false, nil, os.Getenv)
			if err != nil {
				t.Fatalf("resolve: %v", err)
			}
			if got != tc.want {
				t.Fatalf("FAK_GUARD=%q resolved %q, want %q", tc.raw, got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestServeToolGuardBuildGatewaySeam(t *testing.T) {
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
			if ok && key.Name == "ToolGuard" {
				wired = true
			}
			return true
		})
	}
	if !wired {
		t.Fatal("buildGateway does not pass the resolved tool guard posture into gateway configuration")
	}
}
