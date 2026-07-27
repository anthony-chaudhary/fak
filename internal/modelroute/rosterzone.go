package modelroute

import "strings"

// WHICH RUNG SERVES THIS MODEL — ANSWERED ONLY FOR A MODEL THE ROSTER ACTUALLY BINDS.
//
// A usage fold that wants to report "what share of our tokens did we self-host?" has one
// obvious-looking implementation:
//
//	t, err := roster.Resolve(model)
//	if err == nil { count(t.Zone()) }
//
// and that implementation is wrong in the one direction that matters. Resolve is a
// DISPATCH primitive: its job is to always produce somewhere to send the call, so an id
// with no explicit Binding falls back to r.Default (see Resolve's doc comment — the
// fallback is deliberate and correct for dispatch). Read backwards by a fold, that
// fallback answers a question nobody asked: it reports the DEFAULT account's rung for a
// model the roster never placed. Point it at a fleet-default roster and every unrecognized
// worker model — every typo, every model pinned by a peer who never added a binding, every
// id that arrived from an env var — is counted as self-hosted.
//
// That error is not symmetric. Over-reporting the self-hosted share is the failure that
// gets believed: it says the token-economics goal is met when the tokens actually went to
// a vendor. Under-reporting merely says "we cannot attribute this yet", which is a work
// item, not a false conclusion. So attribution asks a strictly narrower question than
// dispatch, and BoundZone is that question.
//
// The rule this encodes, the same one internal/dispatchtick.AttributeZone enforces one
// layer up: A RUNG IS NEVER ASSUMED. An unbound id is unattributed, not local.

// BoundZone reports the placement zone serving a model the roster EXPLICITLY binds.
//
// ok is false — with no zone — when the roster does not bind this id, even when Resolve
// would happily dispatch it to the default account, and when the binding points at an
// account the roster does not define. It is the attribution counterpart to Resolve: same
// roster, same account lookup, same Kind-derived zone, minus the dispatch fallback that
// would let an unplaced model inherit a rung nobody declared for it.
//
// Its shape is exactly dispatchtick.ZoneResolver's, so a caller passes the method value
// (roster.BoundZone) with no adapter and no chance to drop the boolean on the way.
func (r Roster) BoundZone(modelID string) (PlacementZone, bool) {
	id := strings.TrimSpace(modelID)
	if id == "" {
		return "", false
	}
	acctID := ""
	for _, b := range r.Bindings {
		if b.Model == id {
			acctID = b.Account
			break
		}
	}
	if acctID == "" {
		return "", false // unbound: NOT r.Default's zone
	}
	a, ok := r.account(acctID)
	if !ok {
		return "", false // a dangling binding declares no rung
	}
	return a.Zone(), true
}
