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

// TestLocalRoutePrefixWinsOverDeceptiveName pins the correct-by-design collision case
// the security review flagged: a LOCAL account whose id/upstream happens to contain a
// remote keyword ("openai") still routes local, because remoteRoute checks the local
// prefix FIRST and short-circuits. The inverse — a remote account named "local-ish" —
// is impossible because Validate rejects a reserved-token account id and a remote
// route always leads with its <kind>: keyword, never the account id.
func TestLocalRoutePrefixWinsOverDeceptiveName(t *testing.T) {
	tg := modelroute.Target{Kind: modelroute.KindLocal, Account: "openai-mirror", UpstreamModel: "gpt-clone"}
	route := tg.EngineRoute() // "local:openai-mirror/gpt-clone"
	if remoteRoute(route) {
		t.Fatalf("a local-prefixed route must read as on-box even when it contains 'openai': %q", route)
	}
}
