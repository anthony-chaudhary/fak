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
		modelroute.KindLocal,
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
