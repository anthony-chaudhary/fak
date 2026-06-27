package modelroute

// Co-residency bridge from ensemble roles to polymodel spec-decode (#604, epic #595).
//
// specbridge.go answers "which member drafts and which verifies?" PURELY from
// Member.Role, keeping that leaf stdlib-only. It does NOT consult residency: a
// drafter and a verifier can be named in a Plan while sitting in different
// families (different tokenizers), where spec-decode is INVALID — the drafter's
// token ids are not a meaningful draft for the verifier. This file closes that
// gap. It reads the live residency view (internal/polymodel.Pool) and only
// bridges a roles-tagged Plan to a drafter/verifier pairing when the two members
// are genuinely CO-RESIDENT — the same non-empty Family AND mutually shareable
// (polymodel.CanShare) — so a cross-family pair is refused, never silently run as
// spec-decode against an incompatible target.
//
// THE TIER SEAM. internal/polymodel is a tier-1 foundation leaf (stdlib-only,
// below even the root ABI); internal/modelroute is also tier-1. A same-tier
// import is permitted by the layered-DAG rule (internal/architest
// TestNoUpwardImports refuses only a STRICTLY-upward edge, to > from), so this
// file may read polymodel's residency types directly. It is the ONE file in this
// leaf that imports polymodel; the routing spine (modelroute.go) and the
// closure-seam bridge (specbridge.go) stay stdlib-only. The drafter/verifier role
// extraction is reused from specbridge.go (Plan.SpecRoles), not re-derived.
//
// SCOPE OF "DETERMINISTIC" (the package voice): the MAPPING here — role
// extraction, the co-residency verdict, the drafter pick — is pure and
// deterministic over a fixed Plan + Pool. The model OUTPUTS that spec-decode then
// produces are not (they come from non-bit-exact engines). Determinism is pinned
// to the bridge decision, never to end-to-end answer reproducibility. The LIVE
// verify pass (running the verifier to PRODUCE the argmax tokens, and the
// model.KVCache.Evict rollback of rejected drafts) is DEFERRED engine wiring above
// this bridge — see specbridge.go and docs/serving/polymodel-prefill-share-plan.md.

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// PolyBridge is the resolved spec-decode pairing a co-resident roles-tagged Plan
// maps onto: the drafter member proposes tokens, the verifier (target) member
// accepts the longest correct prefix in one pass. Both ids are polymodel.ModelID
// so the result is handed straight to polymodel.PickDrafter / Schedule /
// AcceptGreedy without a re-lookup. Drafter is the member tagged role=drafter;
// Verifier is the member tagged role=verifier and is the spec-decode TARGET.
type PolyBridge struct {
	Drafter  polymodel.ModelID `json:"drafter"`
	Verifier polymodel.ModelID `json:"verifier"`
}

// ErrNotBridgeable is returned by BridgeToPolymodel when a roles-tagged Plan
// CANNOT be run as co-resident spec-decode: the members are not both resident, or
// they are resident but not co-resident (different Family / not CanShare). It is a
// clear "not bridgeable" signal — never a panic and never a fail-open pairing — so
// the host falls back to the answer-level reduce (or plain decode) for that Plan.
var ErrNotBridgeable = fmt.Errorf("modelroute: ensemble is not a co-resident drafter/verifier spec pair")

// BridgeToPolymodel maps a roles-tagged ensemble Plan onto a polymodel
// drafter/verifier pairing, gated on CO-RESIDENCY. It (1) extracts the
// drafter/verifier roles via Plan.SpecRoles (exactly one of each, else not a spec
// ensemble), then (2) requires BOTH members resident in pool AND co-resident —
// the same non-empty Family and mutually shareable (polymodel.CanShare), i.e. a
// valid speculation target for the drafter. Only then does it return the
// PolyBridge a host hands to polymodel.Schedule / AcceptGreedy. If the members are
// not co-resident it returns ErrNotBridgeable wrapping the specific reason (no
// panic, no fail-open). The mapping is pure over a fixed (Plan, Pool): same inputs
// always yield the same verdict, and member order is preserved (SpecRoles reads
// Members in declared order).
//
// pool is the residency view (nil pool => nothing resident => not bridgeable).
// This is the pure residency MAPPING; it does not consult Enabled(). A caller
// putting the resulting pairing on a LIVE request path MUST gate that on
// polymodel.Enabled() (FAK_POLYMODEL) — see BridgeToPolymodelEnabled.
func (p Plan) BridgeToPolymodel(pool *polymodel.Pool) (PolyBridge, error) {
	roles, err := p.SpecRoles()
	if err != nil {
		// Not a 1-drafter/1-verifier ensemble at all — surface the role error.
		return PolyBridge{}, err
	}
	if pool == nil {
		return PolyBridge{}, fmt.Errorf("%w: no residency view (nil pool)", ErrNotBridgeable)
	}
	draft := polymodel.ModelID(roles.Drafter)
	verify := polymodel.ModelID(roles.Verifier)

	dm, dok := pool.Get(draft)
	if !dok {
		return PolyBridge{}, fmt.Errorf("%w: drafter %q is not resident", ErrNotBridgeable, draft)
	}
	vm, vok := pool.Get(verify)
	if !vok {
		return PolyBridge{}, fmt.Errorf("%w: verifier %q is not resident", ErrNotBridgeable, verify)
	}
	// Co-residency: the drafter must be a VALID speculator for the verifier — the
	// same non-empty tokenizer family, mutually prefill-shareable. CanShare is the
	// family+digest verdict; a distinct Family (or an empty one) makes the draft
	// tokens meaningless for the target, so the pair is not bridgeable.
	if dm.Family == "" || dm.Family != vm.Family {
		return PolyBridge{}, fmt.Errorf("%w: drafter family %q != verifier family %q", ErrNotBridgeable, dm.Family, vm.Family)
	}
	if !polymodel.CanShare(vm, dm) {
		return PolyBridge{}, fmt.Errorf("%w: drafter %q and verifier %q are not co-resident (CanShare false)", ErrNotBridgeable, draft, verify)
	}
	return PolyBridge{Drafter: draft, Verifier: verify}, nil
}

// BridgeToPolymodelEnabled is BridgeToPolymodel gated on the live FAK_POLYMODEL
// flag — the form a request-path integration calls. When the poly-model lane is
// OFF (the default), it refuses with ErrNotBridgeable regardless of residency, so
// the spec-decode lane never activates implicitly; the host falls back to the
// answer-level reduce. When ON, it is exactly BridgeToPolymodel. The pure mapping
// (BridgeToPolymodel) stays testable without the env gate; only this live form
// consults Enabled().
func (p Plan) BridgeToPolymodelEnabled(pool *polymodel.Pool) (PolyBridge, error) {
	if !polymodel.Enabled() {
		return PolyBridge{}, fmt.Errorf("%w: poly-model lane disabled (set %s=on)", ErrNotBridgeable, polymodel.FlagEnv)
	}
	return p.BridgeToPolymodel(pool)
}

// PickPolyDrafter resolves the drafter polymodel would itself select for the Plan's
// verifier among the OTHER resident co-residents, via polymodel.PickDrafter, and
// confirms it agrees with the Plan's declared drafter role. It is the cross-check
// that the role tag and the residency-derived pick name the SAME model: a Plan may
// declare draft-7b as drafter, but if a cheaper co-resident family member is warm,
// PickDrafter would prefer it — this surfaces that divergence instead of hiding it.
// Returns the bridged pairing (declared roles) plus the residency-preferred drafter
// id; agree reports whether they match. Not-bridgeable Plans propagate the error.
func (p Plan) PickPolyDrafter(pool *polymodel.Pool) (bridge PolyBridge, preferred polymodel.ModelID, agree bool, err error) {
	b, err := p.BridgeToPolymodel(pool)
	if err != nil {
		return PolyBridge{}, "", false, err
	}
	preferred = polymodel.PickDrafter(b.Verifier, pool)
	return b, preferred, preferred == b.Drafter, nil
}
