package main

// vcache_anchor_wiring_test.go — the regression lock for the M2 star-anchor pre-flight gate
// (#1493). The gate itself (internal/gateway.maybeAnchorAnthropicRaw → cachemeta.RecommendLayout)
// was implemented and its byte-stability + honesty witnesses landed, but it was DEAD: nothing set
// gateway.Config.VCacheAnchor, so it defaulted to the Go zero value (false) on every real session
// and the "DEFAULT-ON" claim was untrue. This test pins the fix: the gateway default const is
// true, and BOTH front doors (fak guard, fak serve) wire --vcache-anchor to it — so a plain
// `fak guard -- claude` / `fak serve` run applies the anchor with no operator configuration, and a
// peer who silently unwires the lever (or flips the const) reds here, not in production. Each
// assertion reads the REAL entrypoint declaration or the gateway const, never a copy that drifts.

import (
	"flag"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// TestVCacheAnchorDefaultsOn pins the M2 star-anchor pre-flight gate's 2026-11-10
// hosting-simplification split posture: the GUARD door keeps the armed library default
// (gateway.DefaultVCacheAnchor, true — the #1493 fix), while the SERVE door resolves its own raw
// serving-router posture (gateway.DefaultServeVCacheAnchor, false — the outbound body is never
// re-anchored). Each door must wire ITS OWN posture const; a door silently inheriting the
// other's posture is the drift this lock exists for.
func TestVCacheAnchorDefaultsOn(t *testing.T) {
	if !gateway.DefaultVCacheAnchor {
		t.Fatalf("gateway.DefaultVCacheAnchor must be default-on (true) for the guard door's M2 star-anchor pre-flight gate (#1493)")
	}
	if gateway.DefaultServeVCacheAnchor {
		t.Fatalf("gateway.DefaultServeVCacheAnchor must be false (the raw serving-router posture, 2026-11-10 hosting-simplification flip)")
	}
	if !strings.Contains(readEntrypoint(t, "guard.go"), `fs.Bool("vcache-anchor", gateway.DefaultVCacheAnchor`) {
		t.Errorf("guard.go must wire --vcache-anchor to gateway.DefaultVCacheAnchor (the armed M2 star-anchor gate, #1493)")
	}
	if !strings.Contains(readEntrypoint(t, "serve.go"), `fs.Bool("vcache-anchor", gateway.DefaultServeVCacheAnchor`) {
		t.Errorf("serve.go must wire --vcache-anchor to gateway.DefaultServeVCacheAnchor (the raw serving-router posture, 2026-11-10 hosting-simplification flip)")
	}
}

// TestVCacheAnchorWiredIntoGatewayConfig pins the flag through to the gateway Config on both front
// doors: the parsed *vcacheAnchor value must actually reach gateway.Config.VCacheAnchor, or the
// default-on flag is inert (the exact dead-code failure #1493 fixes). Guard sets it from the local
// *vcacheAnchor; serve from the serveFlags field *sf.vcacheAnchor.
func TestVCacheAnchorWiredIntoGatewayConfig(t *testing.T) {
	if !strings.Contains(readEntrypoint(t, "guard.go"), "VCacheAnchor:") {
		t.Errorf("guard.go must set gateway Config VCacheAnchor: *vcacheAnchor (wire the parsed flag through)")
	}
	if !strings.Contains(readEntrypoint(t, "serve.go"), "VCacheAnchor:") {
		t.Errorf("serve.go must set gateway Config VCacheAnchor: *sf.vcacheAnchor (wire the parsed flag through)")
	}
}

// TestVCacheAnchorFlagParsesDefaultOn is the behavioral half: a FlagSet declared exactly as the
// front doors declare it defaults to true with no argument, and an explicit --vcache-anchor=false
// is the byte-for-byte opt-out. This proves the default-on posture at parse time, not just by
// source-string match.
func TestVCacheAnchorFlagParsesDefaultOn(t *testing.T) {
	fs := flag.NewFlagSet("t", flag.ContinueOnError)
	got := fs.Bool("vcache-anchor", gateway.DefaultVCacheAnchor, "")
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("parse empty args: %v", err)
	}
	if !*got {
		t.Fatalf("--vcache-anchor must default ON (no args), got false")
	}

	fs2 := flag.NewFlagSet("t", flag.ContinueOnError)
	off := fs2.Bool("vcache-anchor", gateway.DefaultVCacheAnchor, "")
	if err := fs2.Parse([]string{"-vcache-anchor=false"}); err != nil {
		t.Fatalf("parse opt-out: %v", err)
	}
	if *off {
		t.Fatalf("--vcache-anchor=false must turn the gate OFF, got true")
	}
}
