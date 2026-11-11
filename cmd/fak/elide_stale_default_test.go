package main

// elide_stale_default_test.go — the regression lock for the read-lifecycle STALE-elision
// posture split. maybeElideStaleReads (internal/gateway) and its fail-safe transform
// (internal/agent.ElideStaleReadsWithOutcome) were implemented and witnessed —
// cache-prefix byte-identity (TestReadLifecycleElidesStaleKeepsFreshAndPrefix) and the
// fire+restore round-trip (TestMaybeElideStaleReadsRoundTrip) — then flipped default-ON
// on BOTH front doors (the original default-on flip), and re-split on 2026-11-10 by the
// hosting-simplification flip: the GUARD door keeps the armed library default
// (gateway.DefaultElideStaleReads, true), while the SERVE door resolves its own raw
// serving-router posture (gateway.DefaultServeElideStaleReads, false) — the serving
// router forwards the Anthropic body byte-for-byte unless the operator re-arms the
// lever by name. Each assertion reads the REAL entrypoint declaration or the gateway
// const, never a copy that drifts. Mirrors vcache_anchor_wiring_test.go, the sibling
// posture-split lock.

import (
	"flag"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// TestElideStaleReadsPostureSplit pins the split posture: the gateway library const
// stays default-on (the harness products keep the armed floor), guard.go wires the
// flag to it, and serve.go wires its own raw serving-router const. A peer who
// silently unwires the lever on either door (or flips either const) reds here, not
// in production.
func TestElideStaleReadsPostureSplit(t *testing.T) {
	if !gateway.DefaultElideStaleReads {
		t.Fatalf("gateway.DefaultElideStaleReads must stay default-on (true): the guard door's armed posture")
	}
	if gateway.DefaultServeElideStaleReads {
		t.Fatalf("gateway.DefaultServeElideStaleReads must be the raw serving-router posture (false): the 2026-11-10 hosting-simplification flip")
	}
	if !strings.Contains(readEntrypoint(t, "guard.go"), `fs.Bool("elide-stale-reads", gateway.DefaultElideStaleReads`) {
		t.Errorf("guard.go must wire --elide-stale-reads to gateway.DefaultElideStaleReads (the armed harness posture)")
	}
	if !strings.Contains(readEntrypoint(t, "serve.go"), `fs.Bool("elide-stale-reads", gateway.DefaultServeElideStaleReads`) {
		t.Errorf("serve.go must wire --elide-stale-reads to gateway.DefaultServeElideStaleReads (the raw serving-router posture)")
	}
}

// TestElideStaleReadsWiredIntoGatewayConfig pins the flag through to the gateway Config on both
// front doors: the parsed value must actually reach gateway.Config.ElideStaleReads, or the
// posture flag is inert dead code (the exact failure vcache_anchor_wiring_test.go guards for its
// sibling lever). Guard sets it from the local *elideStaleReads; serve from *sf.elideStaleReads.
func TestElideStaleReadsWiredIntoGatewayConfig(t *testing.T) {
	// Collapse gofmt's struct-field alignment padding so the assertion pins the wiring, not the
	// column width (which shifts as sibling field names grow).
	collapse := func(s string) string { return strings.Join(strings.Fields(s), " ") }
	if !strings.Contains(collapse(readEntrypoint(t, "guard.go")), "ElideStaleReads: *elideStaleReads,") {
		t.Errorf("guard.go must set gateway Config ElideStaleReads from the parsed *elideStaleReads flag")
	}
	if !strings.Contains(collapse(readEntrypoint(t, "serve.go")), "ElideStaleReads: *sf.elideStaleReads,") {
		t.Errorf("serve.go must set gateway Config ElideStaleReads from the parsed *sf.elideStaleReads flag")
	}
}

// TestElideStaleReadsFlagParsesBothPostures is the behavioral half: a FlagSet declared
// exactly as each door declares it parses its own posture — the guard declaration
// (gateway.DefaultElideStaleReads) defaults ON with no argument and an explicit
// --elide-stale-reads=false is the byte-for-byte opt-out; the serve declaration
// (gateway.DefaultServeElideStaleReads) defaults OFF with no argument and an explicit
// --elide-stale-reads re-arms the lever. This proves both postures at parse time, not
// just by source-string match.
func TestElideStaleReadsFlagParsesBothPostures(t *testing.T) {
	fs := flag.NewFlagSet("t", flag.ContinueOnError)
	got := fs.Bool("elide-stale-reads", gateway.DefaultElideStaleReads, "")
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("parse empty args: %v", err)
	}
	if !*got {
		t.Fatalf("guard declaration: --elide-stale-reads must default ON (no args), got false")
	}

	fs2 := flag.NewFlagSet("t", flag.ContinueOnError)
	off := fs2.Bool("elide-stale-reads", gateway.DefaultElideStaleReads, "")
	if err := fs2.Parse([]string{"-elide-stale-reads=false"}); err != nil {
		t.Fatalf("parse opt-out: %v", err)
	}
	if *off {
		t.Fatalf("guard declaration: --elide-stale-reads=false must turn the transform OFF, got true")
	}

	fs3 := flag.NewFlagSet("t", flag.ContinueOnError)
	serveOff := fs3.Bool("elide-stale-reads", gateway.DefaultServeElideStaleReads, "")
	if err := fs3.Parse(nil); err != nil {
		t.Fatalf("parse empty args (serve): %v", err)
	}
	if *serveOff {
		t.Fatalf("serve declaration: --elide-stale-reads must default OFF (no args), got true")
	}

	fs4 := flag.NewFlagSet("t", flag.ContinueOnError)
	serveOn := fs4.Bool("elide-stale-reads", gateway.DefaultServeElideStaleReads, "")
	if err := fs4.Parse([]string{"-elide-stale-reads"}); err != nil {
		t.Fatalf("parse re-arm: %v", err)
	}
	if !*serveOn {
		t.Fatalf("serve declaration: bare --elide-stale-reads must re-arm the transform, got false")
	}
}
