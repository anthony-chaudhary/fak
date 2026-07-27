package engine

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// TestEngineRouteAgreesWithResidencyFloor is the cross-package witness for the
// account-switcher's load-bearing security contract: the route string
// modelroute.Target.EngineRoute() stamps into abi.ToolCall.Engine must be classified
// by THIS package's residency floor (remoteRoute) exactly as the account's DECLARED
// locality says. If a future provider kind emitted a route the floor could not place,
// it would fail OPEN (a remote payload read as on-box) — this test makes that a
// build-time failure, not a runtime bypass.
//
// It lives in internal/engine (not modelroute) so it can call the unexported
// remoteRoute directly without duplicating the floor logic or adding an export; the
// import is one-way (modelroute is stdlib-only and never imports engine, so there is
// no cycle).
func TestEngineRouteAgreesWithResidencyFloor(t *testing.T) {
	kinds := []modelroute.ProviderKind{
		modelroute.KindOpenAI,
		modelroute.KindOpenAIResponses,
		modelroute.KindAnthropic,
		modelroute.KindGemini,
		modelroute.KindXAI,
		modelroute.KindDeepSeek,
		modelroute.KindLocal,
		modelroute.KindFleet,
	}
	for _, k := range kinds {
		tg := modelroute.Target{Kind: k, Account: "acct", UpstreamModel: "the-model"}
		route := tg.EngineRoute()
		if got, want := remoteRoute(route), tg.Remote(); got != want {
			t.Fatalf("kind %q: remoteRoute(%q)=%v but Target.Remote()=%v — the floor and the switch disagree (fail-OPEN risk)",
				k, route, got, want)
		}
	}
}

// TestLocalRoutePrefixWinsOverDeceptiveName pins both directions of the
// account-id-vs-locality collision the security review flagged. The route's LEADING
// prefix is the sole locality signal, and EngineRoute always leads with "local:" (for a
// local kind) or the validated "<kind>:" keyword (for a remote kind) — never the account
// id. So the account id's spelling can never flip the floor's decision.
func TestLocalRoutePrefixWinsOverDeceptiveName(t *testing.T) {
	// A LOCAL account whose id contains a remote keyword still routes LOCAL.
	tg := modelroute.Target{Kind: modelroute.KindLocal, Account: "openai-mirror", UpstreamModel: "gpt-clone"}
	route := tg.EngineRoute() // "local:openai-mirror/gpt-clone"
	if remoteRoute(route) {
		t.Fatalf("a local-prefixed route must read as on-box even when it contains 'openai': %q", route)
	}
	// A REMOTE account named "local" still routes REMOTE — the openai: kind prefix leads.
	remote := modelroute.Target{Kind: modelroute.KindOpenAI, Account: "local", UpstreamModel: "gpt-5.5"}
	rroute := remote.EngineRoute() // "openai:local/gpt-5.5"
	if !remoteRoute(rroute) {
		t.Fatalf("a remote account named 'local' must still route remote (kind leads): %q", rroute)
	}
}

// TestTierOneRouteMirrorAgreesWithTheFloor pins the DUPLICATION that
// internal/modelroute.IsRemoteRoute necessarily is. modelroute sits at architest
// tier 1 (stdlib-only) and cannot import this package, so it keeps its own copy of
// the on-box engine-family list — the copy the comment on localRoute names as the
// MIRROR. A family added on one side and forgotten on the other would let a
// tier-1 caller label a route on-box while THIS floor denies it (or worse, the
// reverse), so both classifiers run over one corpus here and must agree
// byte-for-byte on every shape the floor accepts.
func TestTierOneRouteMirrorAgreesWithTheFloor(t *testing.T) {
	routes := []string{
		"", "inkernel", "mock", "cassette", "kernel",
		"local", "local:box/llama3.2", "local-gpu", "local/llama",
		"on-device:0", "ondevice-1", "on-device/gpu",
		"fleet:gpu07/glm-5.2", "fleet-a/kimi", "fleet",
		"openai:acct/gpt-5.5", "anthropic:work/claude-opus-5", "gemini:g/pro",
		"xai:x/grok", "deepseek:d/v4", "openai-responses:acct/o",
		"notlocal:acct/m", "fleetwood:acct/mac", "localish", "something-unknown",
		"  local:box/m  ", "LOCAL:BOX/M", "FLEET:gpu07/x",
	}
	for _, r := range routes {
		if got, want := modelroute.IsRemoteRoute(r), remoteRoute(r); got != want {
			t.Fatalf("route %q: modelroute.IsRemoteRoute=%v but engine.remoteRoute=%v — "+
				"the tier-1 mirror and the enforcing floor disagree (fail-OPEN risk)", r, got, want)
		}
	}
}

// TestFleetZoneStaysRemoteAtTheFloor is the executable statement of the
// placement-zone slice's fail-closed boundary, asserted at the floor that
// actually enforces it. A ZoneFleet route is SELF-HOSTED (the org owns the
// silicon, so a usage ledger may count its tokens as saved) yet still OFF-BOX, so
// this floor must keep denying a tenant-scoped payload routed there. Widening
// that is an operator-declared trust-boundary change; it must never arrive as a
// side effect of naming the zone.
func TestFleetZoneStaysRemoteAtTheFloor(t *testing.T) {
	tg := modelroute.Target{Kind: modelroute.KindFleet, Account: "gpu07", UpstreamModel: "glm-5.2"}
	route := tg.EngineRoute()
	if !remoteRoute(route) {
		t.Fatalf("a fleet route %q must read REMOTE at the residency floor — org-owned hardware is still off-box", route)
	}
	if zone := modelroute.ZoneOfRoute(route); !zone.SelfHosted() {
		t.Fatalf("a fleet route %q must still be attributable as self-hosted (zone %q)", route, zone)
	}
}
