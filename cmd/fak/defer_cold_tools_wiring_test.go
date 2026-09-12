package main

// defer_cold_tools_wiring_test.go — the regression lock for the 10x floor lever's front-door
// wiring (#3530 → flipped default-ON by #3537, under epic #3229). The gateway lever itself
// (internal/gateway.maybeDeferColdTools, gated on Config.DeferColdTools) shipped with #3232; the
// flag threading shipped with #3530 default-off; #3537 flipped the default to ON once the A/B
// (token-delta × held-accuracy × poison) gates reported PASS. This test pins the flipped posture:
// BOTH front doors (fak guard, fak serve) declare --defer-cold-tools defaulting to
// gateway.DefaultDeferColdTools (which is true), wire the parsed value into the gateway Config,
// and the explicit opt-out (--defer-cold-tools=false) still parses to OFF — so a peer who silently
// unwires the lever, or regresses the default in either direction, reds here, not in production.
// Each assertion reads the REAL entrypoint declaration, never a copy that drifts.

import (
	"flag"
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// TestDeferColdToolsFlagDeclaredOnBothDoors pins the flag declaration on both front doors so the
// lever is reachable without editing env. Since #3537 it is DEFAULT ON via the single named
// gateway.DefaultDeferColdTools const — a literal true/false in a door's declaration is a drift
// from the one authoritative default and reds here.
func TestDeferColdToolsFlagDeclaredOnBothDoors(t *testing.T) {
	if !gateway.DefaultDeferColdTools {
		t.Errorf("gateway.DefaultDeferColdTools must be true (the #3537 default-on flip)")
	}
	for _, f := range []string{"guard.go", "serve.go"} {
		if !strings.Contains(readEntrypoint(t, f), `fs.Bool("defer-cold-tools", gateway.DefaultDeferColdTools`) {
			t.Errorf("%s must declare --defer-cold-tools as fs.Bool(..., gateway.DefaultDeferColdTools, ...) (the #3537 default-on flip of the 10x floor lever)", f)
		}
	}
}

// TestDeferColdToolsWiredIntoGatewayConfig pins the flag through to the gateway Config on both
// front doors: the parsed value must actually reach gateway.Config.DeferColdTools, or the flag is
// inert (dead-lever). Guard sets it from the local *deferColdTools; serve from *sf.deferColdTools.
func TestDeferColdToolsWiredIntoGatewayConfig(t *testing.T) {
	for _, door := range []struct {
		file     string
		receiver string
	}{
		{file: "guard.go"},
		{file: "serve.go", receiver: "sf"},
	} {
		if !hasDeferColdToolsGatewayWiring(t, readEntrypoint(t, door.file), door.receiver) {
			t.Errorf("%s must wire the parsed deferColdTools flag into gateway.Config.DeferColdTools", door.file)
		}
	}
}

// hasDeferColdToolsGatewayWiring checks a real gateway.Config key/value pair.
// An empty receiver selects *deferColdTools; otherwise require *receiver.deferColdTools.
func hasDeferColdToolsGatewayWiring(t *testing.T, source, receiver string) bool {
	t.Helper()
	file, err := parser.ParseFile(token.NewFileSet(), "entrypoint.go", source, 0)
	if err != nil {
		t.Fatalf("parse entrypoint: %v", err)
	}
	found := false
	ast.Inspect(file, func(node ast.Node) bool {
		literal, ok := node.(*ast.CompositeLit)
		if !ok {
			return true
		}
		config, ok := literal.Type.(*ast.SelectorExpr)
		if !ok || config.Sel.Name != "Config" {
			return true
		}
		pkg, ok := config.X.(*ast.Ident)
		if !ok || pkg.Name != "gateway" {
			return true
		}
		for _, element := range literal.Elts {
			field, ok := element.(*ast.KeyValueExpr)
			if !ok {
				continue
			}
			key, ok := field.Key.(*ast.Ident)
			if !ok || key.Name != "DeferColdTools" {
				continue
			}
			value, ok := field.Value.(*ast.StarExpr)
			if !ok {
				continue
			}
			if receiver == "" {
				name, ok := value.X.(*ast.Ident)
				found = found || (ok && name.Name == "deferColdTools")
				continue
			}
			selector, ok := value.X.(*ast.SelectorExpr)
			if !ok || selector.Sel.Name != "deferColdTools" {
				continue
			}
			name, ok := selector.X.(*ast.Ident)
			found = found || (ok && name.Name == receiver)
		}
		return true
	})
	return found
}

func TestDeferColdToolsGatewayWiringMatcher(t *testing.T) {
	for _, fixture := range []struct {
		name     string
		source   string
		receiver string
		want     bool
	}{
		{"guard compact", "var _ = gateway.Config{DeferColdTools:*deferColdTools}", "", true},
		{"serve aligned", "var _ = gateway.Config{\n\tDeferColdTools:  *sf.deferColdTools,\n}", "sf", true},
		{"guard formatted", "var _ = &gateway.Config{\nDeferColdTools: /* parsed flag */ *deferColdTools,\n}", "", true},
		{"serve compact", "var _ = gateway.Config{DeferColdTools:*sf.deferColdTools}", "sf", true},
		{"false", "var _ = gateway.Config{DeferColdTools:false}", "", false},
		{"true", "var _ = gateway.Config{DeferColdTools:true}", "sf", false},
		{"wrong variable", "var _ = gateway.Config{DeferColdTools:*other}", "", false},
		{"wrong receiver", "var _ = gateway.Config{DeferColdTools:*other.deferColdTools}", "sf", false},
		{"wrong selector", "var _ = gateway.Config{DeferColdTools:*sf.other}", "sf", false},
		{"guard rejects serve", "var _ = gateway.Config{DeferColdTools:*sf.deferColdTools}", "", false},
		{"serve rejects guard", "var _ = gateway.Config{DeferColdTools:*deferColdTools}", "sf", false},
		{"missing dereference", "var _ = gateway.Config{DeferColdTools:deferColdTools}", "", false},
		{"wrong field", "var _ = gateway.Config{Other:*deferColdTools}", "", false},
		{"wrong package", "var _ = other.Config{DeferColdTools:*deferColdTools}", "", false},
		{"wrong type", "var _ = gateway.Other{DeferColdTools:*deferColdTools}", "", false},
		{"comment only", "// gateway.Config{DeferColdTools: *deferColdTools,}\nvar _ = gateway.Config{}", "", false},
		{"string only", "var _ = \"gateway.Config{DeferColdTools: *sf.deferColdTools,}\"", "sf", false},
	} {
		t.Run(fixture.name, func(t *testing.T) {
			got := hasDeferColdToolsGatewayWiring(t, "package main\n"+fixture.source, fixture.receiver)
			if got != fixture.want {
				t.Errorf("wiring matched = %v, want %v", got, fixture.want)
			}
		})
	}
}

// TestDeferColdToolsFlagParsesDefaultOnWithOptOut is the behavioral half: a FlagSet declared
// exactly as the front doors declare it defaults to ON with no argument (the #3537 flip), and the
// explicit --defer-cold-tools=false opt-out still turns it off. This proves the default-ON posture
// AND the preserved opt-out at parse time, not just by source-string match.
func TestDeferColdToolsFlagParsesDefaultOnWithOptOut(t *testing.T) {
	fs := flag.NewFlagSet("t", flag.ContinueOnError)
	got := fs.Bool("defer-cold-tools", gateway.DefaultDeferColdTools, "")
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("parse empty args: %v", err)
	}
	if !*got {
		t.Fatalf("--defer-cold-tools must default ON (no args, the #3537 flip), got false")
	}

	fs2 := flag.NewFlagSet("t", flag.ContinueOnError)
	off := fs2.Bool("defer-cold-tools", gateway.DefaultDeferColdTools, "")
	if err := fs2.Parse([]string{"-defer-cold-tools=false"}); err != nil {
		t.Fatalf("parse opt-out: %v", err)
	}
	if *off {
		t.Fatalf("--defer-cold-tools=false must keep the explicit opt-out, got true")
	}
}
