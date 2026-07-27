package engine

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

// TestPlacementZoneStringsMatchModelroute pins the string coupling between the
// placement vocabulary (internal/modelroute.PlacementZone) and the audit classifier
// that consumes it (internal/sessionaudit.BucketForPlacement).
//
// sessionaudit is an architest pureRoot leaf — it may import nothing internal — so it
// takes the zone as a plain string and hard-codes "device"/"fleet"/"vendor" in a
// switch. That is a MIRROR, with the usual mirror hazard: renaming a zone constant
// would leave the switch matching nothing, and BucketForPlacement's unrecognized-zone
// arm falls back to "no placement claim". The failure would therefore be SILENT — a
// fleet's tokens would quietly stop counting as self-hosted and the headline fraction
// in epic #5416 would sag with no error anywhere. This test lives here because
// internal/engine (tier 2) may import both tier-1 leaves.
//
// Every zone on the ladder must round-trip: its String() must be a zone
// BucketForPlacement recognizes, and the resulting bucket's self-hosted verdict must
// match the zone's own SelfHosted().
func TestPlacementZoneStringsMatchModelroute(t *testing.T) {
	const model = "glm-5.2" // an open-weights id — servable in ANY of the three zones
	for _, zone := range modelroute.Zones() {
		bucket := sessionaudit.BucketForPlacement(model, zone.String())

		selfHosted, known := sessionaudit.BucketIsSelfHosted(bucket)
		if !known {
			t.Fatalf("zone %q produced bucket %q, which makes no placement claim — "+
				"the zone string is no longer recognized by sessionaudit.BucketForPlacement "+
				"(a rename on one side of the mirror)", zone, bucket)
		}
		if selfHosted != zone.SelfHosted() {
			t.Fatalf("zone %q: sessionaudit bucket %q reports selfHosted=%v but "+
				"modelroute.PlacementZone.SelfHosted()=%v — the two halves disagree",
				zone, bucket, selfHosted, zone.SelfHosted())
		}
	}
}

// TestEngineRouteClassifiesIntoAnAuditBucket closes the loop end to end, over the
// exact string that travels through the kernel. A Target stamps abi.ToolCall.Engine
// via EngineRoute(); modelroute.ZoneOfRoute parses that string back to a zone; and
// sessionaudit buckets it. So a usage record carrying nothing but the engine route
// can still be attributed to the right rung — which is what makes the self-hosted
// token fraction computable from data the serving path already has.
//
// It also re-asserts the fail-closed boundary from the zone slice at this seam: a
// fleet route is attributable as self-hosted while remaining REMOTE to the residency
// floor. Attribution and permission are different questions.
func TestEngineRouteClassifiesIntoAnAuditBucket(t *testing.T) {
	cases := []struct {
		kind           modelroute.ProviderKind
		model          string
		wantSelfHosted bool
		wantRemote     bool
	}{
		{modelroute.KindLocal, "qwen3.6-4b", true, false},
		{modelroute.KindFleet, "glm-5.2", true, true},
		{modelroute.KindFleet, "kimi-k3", true, true},
		{modelroute.KindDeepSeek, "deepseek-v4-pro", false, true},
		{modelroute.KindAnthropic, "claude-opus-4-6", false, true},
		{modelroute.KindOpenAI, "gpt-5.5", false, true},
	}
	for _, c := range cases {
		tg := modelroute.Target{Kind: c.kind, Account: "acct", UpstreamModel: c.model}
		route := tg.EngineRoute()

		zone := modelroute.ZoneOfRoute(route)
		bucket := sessionaudit.BucketForPlacement(c.model, zone.String())
		selfHosted, known := sessionaudit.BucketIsSelfHosted(bucket)
		if !known {
			t.Fatalf("route %q (kind %q) did not reach an attributable bucket, got %q", route, c.kind, bucket)
		}
		if selfHosted != c.wantSelfHosted {
			t.Fatalf("route %q: bucket %q selfHosted=%v, want %v", route, bucket, selfHosted, c.wantSelfHosted)
		}
		// The floor is unmoved by any of this.
		if got := remoteRoute(route); got != c.wantRemote {
			t.Fatalf("route %q: residency floor remoteRoute=%v, want %v — attribution must not change enforcement",
				route, got, c.wantRemote)
		}
	}
}
