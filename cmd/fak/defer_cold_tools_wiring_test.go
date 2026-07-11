package main

// defer_cold_tools_wiring_test.go — the regression lock for the 10x floor lever's front-door
// wiring (#3530, under epic #3229). The gateway lever itself (internal/gateway.maybeDeferColdTools,
// gated on Config.DeferColdTools) shipped with #3232, but a lever nothing sets is dead: without a
// flag threading the parsed value into gateway.Config.DeferColdTools an operator could never turn
// it on from the front door (only the FAK_DEFER_COLD_TOOLS env fallback baked into gateway.New).
// This test pins the flag: BOTH front doors (fak guard, fak serve) declare --defer-cold-tools and
// wire the parsed value into the gateway Config, so a peer who silently unwires it reds here, not
// in production. Each assertion reads the REAL entrypoint declaration, never a copy that drifts.

import (
	"flag"
	"strings"
	"testing"
)

// TestDeferColdToolsFlagDeclaredOnBothDoors pins the flag declaration on both front doors so the
// lever is reachable without editing env. It is DEFAULT OFF (the epic's highest-risk lever), so the
// declaration is a plain fs.Bool default-false — the on/off default flips to on in #3537, not here.
func TestDeferColdToolsFlagDeclaredOnBothDoors(t *testing.T) {
	for _, f := range []string{"guard.go", "serve.go"} {
		if !strings.Contains(readEntrypoint(t, f), `fs.Bool("defer-cold-tools", false`) {
			t.Errorf("%s must declare --defer-cold-tools as fs.Bool(..., false, ...) (the 10x floor lever, #3530)", f)
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

// TestDeferColdToolsFlagParsesDefaultOff is the behavioral half: a FlagSet declared exactly as the
// front doors declare it defaults to false with no argument, and an explicit --defer-cold-tools
// turns it on. This proves the default-OFF posture at parse time, not just by source-string match.
func TestDeferColdToolsFlagParsesDefaultOff(t *testing.T) {
	fs := flag.NewFlagSet("t", flag.ContinueOnError)
	got := fs.Bool("defer-cold-tools", false, "")
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("parse empty args: %v", err)
	}
	if *got {
		t.Fatalf("--defer-cold-tools must default OFF (no args), got true")
	}

	fs2 := flag.NewFlagSet("t", flag.ContinueOnError)
	on := fs2.Bool("defer-cold-tools", false, "")
	if err := fs2.Parse([]string{"-defer-cold-tools"}); err != nil {
		t.Fatalf("parse opt-in: %v", err)
	}
	if !*on {
		t.Fatalf("--defer-cold-tools must turn the lever ON, got false")
	}
}
