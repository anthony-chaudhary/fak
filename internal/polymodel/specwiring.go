package polymodel

// Ensemble-roles -> spec-decode wiring bridge (#604, epic #595).
//
// THE SEAM, AND WHY THIS HALF LIVES HERE. internal/modelroute owns the routing
// POLICY: a Plan is a set of Members, and a Member may carry Role=="drafter" /
// Role=="verifier". modelroute already extracts that pair (modelroute.Plan.SpecRoles
// -> {Drafter, Verifier} as plain strings) and even returns a co-residency verdict
// (modelroute.Plan.BridgeToPolymodel) by reaching DOWN into this leaf. What it does
// NOT — and by the tier rule MUST not — do is assemble the polymodel SPEC-DECODE
// WIRING: the residency-validated drafter pick (PickDrafter) bound to the verify
// pass (AcceptGreedy / AcceptTree). Those primitives live here, in the tier-1
// foundation leaf, so the resolved wiring object that drives them belongs here too.
//
// THE IMPORT DIRECTION IS FORCED. internal/modelroute imports internal/polymodel
// (its polybridge.go reads the residency Pool), so polymodel importing modelroute
// would be an import CYCLE — and would break this leaf's stdlib-only invariant
// (architest: "polymodel ... imports nothing internal"). So the role pair crosses
// the seam exactly the way modelroute.SpecAccept already passes accept results: as
// plain values (the two model ids), never as a modelroute import. A caller wires
// the two halves with one line:
//
//	roles, _ := plan.SpecRoles()                 // modelroute: policy -> {Drafter, Verifier}
//	w, err := polymodel.BridgeRoles(             // polymodel: ids -> resolved spec wiring
//	    ModelID(roles.Drafter), ModelID(roles.Verifier), pool)
//	if err == nil {
//	    res := w.Accept(draft, targetArgmax)     // the verifier's accept pass
//	}
//
// SCOPE. This is the pure decision + accounting layer: BridgeRoles resolves the
// pairing and the accept rule deterministically over a fixed (ids, Pool). It runs no
// engine — producing targetArgmax (the live verify pass) and applying the KVCache.Evict
// rollback of EvictKV positions are the DEFERRED engine wiring above this leaf (see
// doc.go and docs/serving/polymodel-prefill-share-plan.md). A request-path caller
// must additionally gate on Enabled() (FAK_POLYMODEL); BridgeRoles itself is a pure
// library call and does not consult the flag, the same contract Schedule/AcceptGreedy
// already follow.

import "errors"

// ErrNotCoResident is returned by BridgeRoles when a declared drafter/verifier pair
// CANNOT be run as co-resident spec-decode: a member is not resident in the pool, or
// both are resident but not a valid speculation pair (different/empty Family, or not
// CanShare — the prefill-share band differs, so the draft tokens are not a meaningful
// proposal for the target). It is a clear "fall back to plain decode / answer-level
// reduce" signal — never a panic and never a fail-open pairing.
var ErrNotCoResident = errors.New("polymodel: drafter/verifier pair is not co-resident spec-decodable")

// ErrEmptyRole is returned by BridgeRoles when either role id is empty — a single-model
// Plan (no ensemble) carries no drafter/verifier split, so there is nothing to bridge
// to spec-decode. The caller treats this as "no speculative pass" (plain self-decode),
// distinct from ErrNotCoResident (a real pair that is simply not co-resident).
var ErrEmptyRole = errors.New("polymodel: empty drafter or verifier role (no spec ensemble)")

// SpecWiring is the resolved spec-decode pairing a co-resident drafter/verifier
// ensemble maps onto: the Drafter proposes tokens, the Verifier (the spec-decode
// TARGET) accepts the longest correct prefix in one verify pass. Both ids are
// resident and co-resident (validated at construction). Drafter is the role-declared
// drafter; PreferredDrafter is the drafter polymodel's own residency policy would
// pick (PickDrafter — the cheapest co-resident family member), and DrafterIsPreferred
// reports whether the two agree, so a declared drafter that a cheaper warm peer would
// beat is SURFACED, not hidden. Tree selects which accept rule Accept runs.
type SpecWiring struct {
	Drafter            ModelID
	Verifier           ModelID
	PreferredDrafter   ModelID
	DrafterIsPreferred bool
	// Tree, when true, makes Accept run the TREE accept rule (AcceptTree) instead of
	// the linear greedy rule (AcceptGreedy). It is set by BridgeRolesTree; BridgeRoles
	// leaves it false (the linear chain, which AcceptTree reduces to exactly).
	Tree bool
}

// BridgeRoles maps a role-declared drafter/verifier pair onto the polymodel
// spec-decode wiring, gated on CO-RESIDENCY. It is the polymodel-side half of the
// ensemble->spec-decode bridge (#604): the caller extracts the role ids from a
// modelroute.Plan (Plan.SpecRoles), and this resolves them against the live residency
// Pool into a SpecWiring whose Accept pass the host then drives. The preconditions,
// in order:
//
//  1. both ids non-empty (else ErrEmptyRole — a single-model Plan has no spec pair);
//  2. both resident in pool (else ErrNotCoResident);
//  3. co-resident: a DIFFERENT model in the same non-empty Family AND mutually
//     prefill-shareable (CanShare(verifier, drafter)) — i.e. the drafter's token ids
//     are a valid draft for the verifier's vocabulary (else ErrNotCoResident).
//
// drafter == verifier is refused: a model cannot speculatively draft for itself (no
// throughput, and PickDrafter excludes the active model). On success it also resolves
// PickDrafter(verifier) — the residency-preferred drafter — and records whether the
// declared drafter agrees. pool == nil => nothing resident => ErrNotCoResident.
//
// The mapping is PURE over a fixed (drafter, verifier, Pool): same inputs always yield
// the same SpecWiring. It does NOT consult Enabled(); a live-path caller gates that
// separately (see doc.go).
func BridgeRoles(drafter, verifier ModelID, pool *Pool) (SpecWiring, error) {
	if drafter == "" || verifier == "" {
		return SpecWiring{}, ErrEmptyRole
	}
	if drafter == verifier {
		return SpecWiring{}, ErrNotCoResident
	}
	if pool == nil {
		return SpecWiring{}, ErrNotCoResident
	}
	dm, dok := pool.Get(drafter)
	if !dok {
		return SpecWiring{}, ErrNotCoResident
	}
	vm, vok := pool.Get(verifier)
	if !vok {
		return SpecWiring{}, ErrNotCoResident
	}
	// Co-residency: same non-empty Family (shared tokenizer, so the draft tokens are
	// meaningful for the target) AND CanShare (byte-identical prefill band, so the pair
	// is a real speculation target). A distinct/empty family or a digest mismatch makes
	// the draft tokens invalid for the verifier — refuse, never fail open.
	if dm.Family == "" || dm.Family != vm.Family {
		return SpecWiring{}, ErrNotCoResident
	}
	if !CanShare(vm, dm) {
		return SpecWiring{}, ErrNotCoResident
	}
	preferred := PickDrafter(verifier, pool)
	return SpecWiring{
		Drafter:            drafter,
		Verifier:           verifier,
		PreferredDrafter:   preferred,
		DrafterIsPreferred: preferred == drafter,
	}, nil
}

// BridgeRolesTree is BridgeRoles for a TREE-shaped speculation (Medusa / EAGLE-2 /
// SpecInfer): the resolved SpecWiring's AcceptTree pass verifies a whole token tree in
// one pass. The co-residency preconditions are identical to BridgeRoles; only the
// accept rule the wiring carries differs (Tree=true). A linear chain verified through
// AcceptTree reduces exactly to AcceptGreedy, so this is a strict generalization.
func BridgeRolesTree(drafter, verifier ModelID, pool *Pool) (SpecWiring, error) {
	w, err := BridgeRoles(drafter, verifier, pool)
	if err != nil {
		return SpecWiring{}, err
	}
	w.Tree = true
	return w, nil
}

// Accept runs the verifier's speculative-accept pass over the drafter's proposed
// tokens: given the drafter's draft token ids and the verifier's argmax at each of
// those positions (from the single verify pass the engine layer produces), it returns
// the SpecResult accounting (Accepted / Advance / KeepKV / EvictKV). It is the linear
// greedy rule (AcceptGreedy); the KEEP/EVICT split maps 1:1 onto the model leaf's
// bit-exact KV primitives (KeepKV stays, EvictKV positions are rolled back with
// KVCache.Evict — the deferred engine wiring). For a Tree wiring (BridgeRolesTree), use
// AcceptTree directly with the resolved roles; Accept is the chain form.
func (w SpecWiring) Accept(draft, targetArgmax []int) SpecResult {
	return AcceptGreedy(draft, targetArgmax)
}

// AcceptTree runs the verifier's TREE accept pass over a speculation tree the drafter
// proposed (the Tree-wiring form). It is a method shim onto the package AcceptTree so a
// caller drives the resolved wiring uniformly: the accounting (Path / Advance / KeepKV /
// EvictKV) is identical to the package function.
func (w SpecWiring) AcceptTree(t SpecTree) TreeResult {
	return AcceptTree(t)
}
