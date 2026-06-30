package storedrv

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// placement.go is the thin adapter (MLCACHE5, #1471) that lets the byte-MOVING Router
// honor the byte-FREE cachemeta placement policy. Before this, the storage router and
// the KV-residency policy were orthogonal: the Router picked a tier purely by payload
// SIZE (plus a Hint's plane/scope/taint/durability) and consumed no PlacementDecision
// and no TierPressure. So under real hot-tier PRESSURE a payload the size heuristic
// would keep "hot" stayed hot even when the placement policy — which weighs live tier
// fullness against recompute cost — would have routed it one tier colder, and the two
// planes never met.
//
// PutPlaced closes that gap WITHOUT touching the frozen abi.Resolver default path. It
// is additive and opt-in, the same posture PutHinted already takes: it is reachable
// only through *Router (never the abi.Resolver interface), so an unaware caller's Put is
// byte-for-byte unchanged. The CALLER derives a cachemeta.PlacementDecision from the
// live per-tier pressure (cachemeta.PlanPlacement) — keeping cachemeta the policy that
// owns no bytes — and hands it in; the router maps that verdict onto its hot-vs-durable
// tiers. The router stays the byte-mover; cachemeta stays the policy.

// PutPlaced stores b honoring an optional cachemeta placement verdict, then falls back
// to the size/durability route for everything the verdict does not override.
//
// When the policy is content to keep the span hot — ActionKeep or ActionPromote (toward
// a hotter tier), and the zero verdict (no opinion) — the router defers to PutHinted:
// the byte heuristic is the right call when the hot tier still has room. When the policy
// says the span belongs COLDER than the hot tier (demote / spill / compress-demote /
// evict — every verdict PlanPlacement emits once the hot tier is under pressure), the
// router routes the bytes to the first durable tier even though their byte size alone
// would have kept them hot. The router is a byte-mover, so a policy "evict" (drop and
// recompute later) still lands the bytes durably rather than dropping them — it was
// handed bytes, not a recomputable KV span.
//
// This is genuinely pressure-aware, not a rename of PutHinted: the SAME small,
// non-durable payload routes hot under a no-pressure (Keep) verdict and durable under a
// high-hot-pressure (Demote) verdict — a flip neither the size heuristic nor PutHinted's
// durability routing can produce on its own.
func (r *Router) PutPlaced(ctx context.Context, b []byte, h Hint, d cachemeta.PlacementDecision) (abi.Ref, error) {
	if len(b) <= InlineMax {
		// Tiny payloads ride inline regardless of policy (no tier is touched).
		return r.Put(ctx, b)
	}
	if placementKeepsHot(d.Action) {
		// The policy leaves the span hot (or has no opinion): the byte/durability route
		// stands, so an opted-in caller with a no-pressure verdict is identical to PutHinted.
		return r.PutHinted(ctx, b, h)
	}
	if i := r.firstDurable(); i >= 0 {
		atomic.AddInt64(&r.puts, 1)
		ref, err := r.tiers[i].Driver.Put(ctx, b)
		if err != nil {
			return abi.Ref{}, fmt.Errorf("storedrv: put-placed -> tier %s: %w", r.tiers[i].Driver.ID(), err)
		}
		return ref, nil
	}
	// No durable tier exists to route colder into — fall back to the size/durability route
	// rather than fail (a single-tier router has nowhere colder to honor the verdict).
	return r.PutHinted(ctx, b, h)
}

// placementKeepsHot reports whether a placement verdict leaves the span in (or moves it
// toward) the hot tier, so the router defers to its byte-size route. ActionKeep and
// ActionPromote are hot-or-hotter, and the zero action ("" — a caller that passed no
// real decision) fails open to the existing route so PutPlaced is a safe superset of
// PutHinted. Every other verdict — ActionDemote, ActionSpill, ActionCompressDemote,
// ActionEvict — is the policy asking for a COLDER seat than the hot tier, which the
// byte-mover honors by routing to the durable tier.
func placementKeepsHot(a cachemeta.PlacementAction) bool {
	switch a {
	case cachemeta.ActionKeep, cachemeta.ActionPromote, "":
		return true
	default:
		return false
	}
}
