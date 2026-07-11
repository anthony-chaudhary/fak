package main

// elide_stale_default_test.go — the regression lock for the read-lifecycle STALE-elision default-on
// flip. maybeElideStaleReads (internal/gateway) and its fail-safe transform (internal/agent.
// ElideStaleReadsWithOutcome) were implemented and witnessed — cache-prefix byte-identity
// (TestReadLifecycleElidesStaleKeepsFreshAndPrefix) and the fire+restore round-trip
// (TestMaybeElideStaleReadsRoundTrip) — but the lever shipped OFF: both front doors defaulted the
// flag false, so a plain `fak guard -- claude` / `fak serve` run never fired it and the "safer,
// restorable sibling of --elide-result-bytes" delivered nothing out of the box. This test pins the
// flip: the gateway default const is true, and BOTH front doors wire --elide-stale-reads to it — so
// the transform runs on the Anthropic passthrough with no operator configuration, and a peer who
// silently unwires the lever (or flips the const) reds here, not in production. Each assertion reads
// the REAL entrypoint declaration or the gateway const, never a copy that drifts. Mirrors
// vcache_anchor_wiring_test.go, the sibling default-on bool lock.

import (
	"flag"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// TestElideStaleReadsDefaultsOn pins read-lifecycle STALE elision ON by default: the gateway const
// is true, and both front doors wire the flag to it, so a superseded pre-edit Read snapshot is
// replaced with a restorable marker on the Anthropic passthrough with no flags. Wiring the flag to
// the const keeps the on/off decision a single edit to gateway.DefaultElideStaleReads.
func TestElideStaleReadsDefaultsOn(t *testing.T) {
	if !gateway.DefaultElideStaleReads {
		t.Fatalf("gateway.DefaultElideStaleReads must be default-on (true): the restorable, strictly-more-conservative sibling of --elide-result-bytes")
	}
	for _, f := range []string{"serve.go", "guard.go"} {
		if !strings.Contains(readEntrypoint(t, f), `fs.Bool("elide-stale-reads", gateway.DefaultElideStaleReads`) {
			t.Errorf("%s must wire --elide-stale-reads to gateway.DefaultElideStaleReads (default-on read-lifecycle STALE elision)", f)
		}
	}
}

// TestElideStaleReadsWiredIntoGatewayConfig pins the flag through to the gateway Config on both
// front doors: the parsed value must actually reach gateway.Config.ElideStaleReads, or the
// default-on flag is inert dead code (the exact failure vcache_anchor_wiring_test.go guards for its
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

// TestElideStaleReadsFlagParsesDefaultOn is the behavioral half: a FlagSet declared exactly as the
// front doors declare it defaults to true with no argument, and an explicit --elide-stale-reads=false
// is the byte-for-byte opt-out. This proves the default-on posture at parse time, not just by
// source-string match.
func TestElideStaleReadsFlagParsesDefaultOn(t *testing.T) {
	fs := flag.NewFlagSet("t", flag.ContinueOnError)
	got := fs.Bool("elide-stale-reads", gateway.DefaultElideStaleReads, "")
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("parse empty args: %v", err)
	}
	if !*got {
		t.Fatalf("--elide-stale-reads must default ON (no args), got false")
	}

	fs2 := flag.NewFlagSet("t", flag.ContinueOnError)
	off := fs2.Bool("elide-stale-reads", gateway.DefaultElideStaleReads, "")
	if err := fs2.Parse([]string{"-elide-stale-reads=false"}); err != nil {
		t.Fatalf("parse opt-out: %v", err)
	}
	if *off {
		t.Fatalf("--elide-stale-reads=false must turn the transform OFF, got true")
	}
}
