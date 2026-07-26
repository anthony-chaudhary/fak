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
	if !strings.Contains(readEntrypoint(t, "guard.go"), "DeferColdTools: *deferColdTools,") {
		t.Errorf("guard.go must set gateway Config DeferColdTools: *deferColdTools (wire the parsed flag through)")
	}
	if !strings.Contains(readEntrypoint(t, "serve.go"), "DeferColdTools: *sf.deferColdTools,") {
		t.Errorf("serve.go must set gateway Config DeferColdTools: *sf.deferColdTools (wire the parsed flag through)")
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
