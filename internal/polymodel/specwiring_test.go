package polymodel

import (
	"errors"
	"testing"
)

// #604 — the polymodel-side half of the ensemble-roles -> spec-decode bridge. These
// witnesses cover the issue's bar from this leaf: (a) a co-resident drafter+verifier
// pair resolves to a SpecWiring whose drafter selection + verify/accept pass are the
// right polymodel primitives, (b) a non-co-resident pair is refused (no fail-open),
// and (c) a single-model "ensemble" (an empty role) bridges to NO speculative pass.

// coResidentSpecPool admits a cheap drafter and an expensive verifier in the SAME
// family with the SAME prefix digest, so CanShare(verifier, drafter) holds — the
// co-residency BridgeRoles requires.
func coResidentSpecPool(t *testing.T) *Pool {
	t.Helper()
	p := NewPool(1 << 20)
	mustAdmit(t, p, Model{ID: "draft-7b", Family: "glm", PrefixDigest: "sha256:band", WeightBytes: 7000})
	mustAdmit(t, p, Model{ID: "verify-70b", Family: "glm", PrefixDigest: "sha256:band", WeightBytes: 70000})
	return p
}

// TestBridgeRolesCoResidentResolvesWiring proves a co-resident drafter/verifier pair
// resolves to a SpecWiring that names the right drafter + verifier, agrees with the
// residency-preferred drafter (PickDrafter), and whose Accept pass IS the polymodel
// greedy accept rule (it commits the longest matching prefix + 1 correction, with the
// KeepKV/EvictKV split AcceptGreedy computes).
func TestBridgeRolesCoResidentResolvesWiring(t *testing.T) {
	pool := coResidentSpecPool(t)
	w, err := BridgeRoles("draft-7b", "verify-70b", pool)
	if err != nil {
		t.Fatalf("co-resident pair should bridge: %v", err)
	}
	if w.Drafter != "draft-7b" || w.Verifier != "verify-70b" {
		t.Fatalf("wiring = %+v, want drafter=draft-7b verifier=verify-70b", w)
	}
	// The declared drafter is the only other glm member, so PickDrafter agrees.
	if w.PreferredDrafter != "draft-7b" || !w.DrafterIsPreferred {
		t.Fatalf("preferred drafter = %q agree=%v, want draft-7b/true", w.PreferredDrafter, w.DrafterIsPreferred)
	}
	if w.Tree {
		t.Fatal("BridgeRoles should not set Tree (linear greedy rule)")
	}
	// The verify pass IS AcceptGreedy: a draft of [5,6,7] with target argmax [5,6,9,2]
	// accepts the leading [5,6] (2), corrects at the divergence (advance 3), keeps 2
	// speculative KV positions and evicts 1. This must equal the package rule exactly.
	draft := []int{5, 6, 7}
	target := []int{5, 6, 9, 2}
	got := w.Accept(draft, target)
	want := AcceptGreedy(draft, target)
	if got != want {
		t.Fatalf("wiring Accept %+v != AcceptGreedy %+v", got, want)
	}
	if want.Accepted != 2 || want.Advance != 3 || want.KeepKV != 2 || want.EvictKV != 1 {
		t.Fatalf("accept accounting unexpected: %+v", want)
	}
}

// TestBridgeRolesPicksCheaperCoResidentDrafter proves the wiring SURFACES a divergence
// between the declared drafter and the residency-preferred one: when a CHEAPER
// co-resident family member is warm, PickDrafter prefers it, so a Plan that declared
// the costlier drafter no longer agrees — the bridge does not silently override the
// role, it records the disagreement for the caller.
func TestBridgeRolesPicksCheaperCoResidentDrafter(t *testing.T) {
	pool := coResidentSpecPool(t)
	mustAdmit(t, pool, Model{ID: "draft-1b", Family: "glm", PrefixDigest: "sha256:band", WeightBytes: 1000})
	w, err := BridgeRoles("draft-7b", "verify-70b", pool)
	if err != nil {
		t.Fatalf("bridge: %v", err)
	}
	if w.Drafter != "draft-7b" {
		t.Fatalf("declared drafter must be preserved, got %q", w.Drafter)
	}
	if w.PreferredDrafter != "draft-1b" || w.DrafterIsPreferred {
		t.Fatalf("a cheaper co-resident should be preferred: preferred=%q agree=%v", w.PreferredDrafter, w.DrafterIsPreferred)
	}
}

// TestBridgeRolesDifferentFamilyRefused proves a drafter/verifier pair in DIFFERENT
// families (incompatible tokenizers) is refused with ErrNotCoResident — never silently
// run as spec-decode against a target whose vocabulary the draft tokens do not fit.
func TestBridgeRolesDifferentFamilyRefused(t *testing.T) {
	p := NewPool(1 << 20)
	mustAdmit(t, p, Model{ID: "draft-a", Family: "glm", PrefixDigest: "sha256:x", WeightBytes: 7000})
	mustAdmit(t, p, Model{ID: "verify-b", Family: "qwen", PrefixDigest: "sha256:y", WeightBytes: 70000})
	_, err := BridgeRoles("draft-a", "verify-b", p)
	if !errors.Is(err, ErrNotCoResident) {
		t.Fatalf("different-family pair must be ErrNotCoResident, got %v", err)
	}
}

// TestBridgeRolesSameFamilyDigestMismatchRefused proves same-Family is necessary but
// not sufficient: distinct prefix digests fail CanShare (the reused prefill band would
// not be bit-identical), so the pair is still refused.
func TestBridgeRolesSameFamilyDigestMismatchRefused(t *testing.T) {
	p := NewPool(1 << 20)
	mustAdmit(t, p, Model{ID: "draft-7b", Family: "glm", PrefixDigest: "sha256:a", WeightBytes: 7000})
	mustAdmit(t, p, Model{ID: "verify-70b", Family: "glm", PrefixDigest: "sha256:b", WeightBytes: 70000})
	if _, err := BridgeRoles("draft-7b", "verify-70b", p); !errors.Is(err, ErrNotCoResident) {
		t.Fatalf("same family, mismatched digest must be ErrNotCoResident, got %v", err)
	}
}

// TestBridgeRolesNotResidentRefused proves a pair whose members are not both resident
// is not bridgeable: a nil pool, and a half-resident pool (only the drafter warm),
// both refuse with ErrNotCoResident.
func TestBridgeRolesNotResidentRefused(t *testing.T) {
	if _, err := BridgeRoles("d", "v", nil); !errors.Is(err, ErrNotCoResident) {
		t.Fatalf("nil pool should be ErrNotCoResident, got %v", err)
	}
	p := NewPool(1 << 20)
	mustAdmit(t, p, Model{ID: "draft-7b", Family: "glm", PrefixDigest: "sha256:band", WeightBytes: 7000})
	if _, err := BridgeRoles("draft-7b", "verify-70b", p); !errors.Is(err, ErrNotCoResident) {
		t.Fatalf("absent verifier should be ErrNotCoResident, got %v", err)
	}
}

// TestBridgeRolesSingleModelNoSpecPass proves a single-model Plan (no ensemble) bridges
// to NO speculative pass: an empty drafter (or empty verifier) returns ErrEmptyRole —
// distinct from ErrNotCoResident — so the host runs plain self-decode, not a spec pass.
func TestBridgeRolesSingleModelNoSpecPass(t *testing.T) {
	pool := coResidentSpecPool(t)
	// A single-model Plan's role extraction yields an empty drafter (no role=drafter
	// member). The bridge must say "no spec ensemble", not "not co-resident".
	if _, err := BridgeRoles("", "verify-70b", pool); !errors.Is(err, ErrEmptyRole) {
		t.Fatalf("empty drafter should be ErrEmptyRole, got %v", err)
	}
	if _, err := BridgeRoles("draft-7b", "", pool); !errors.Is(err, ErrEmptyRole) {
		t.Fatalf("empty verifier should be ErrEmptyRole, got %v", err)
	}
	// And ErrEmptyRole is genuinely distinct from the not-co-resident path.
	if errors.Is(ErrEmptyRole, ErrNotCoResident) {
		t.Fatal("ErrEmptyRole must be distinct from ErrNotCoResident")
	}
}

// TestBridgeRolesSelfDraftRefused proves a model cannot speculatively draft for itself
// (drafter == verifier): there is no throughput in a model verifying its own draft,
// and PickDrafter excludes the active model, so the pair is refused.
func TestBridgeRolesSelfDraftRefused(t *testing.T) {
	pool := coResidentSpecPool(t)
	if _, err := BridgeRoles("verify-70b", "verify-70b", pool); !errors.Is(err, ErrNotCoResident) {
		t.Fatalf("self-draft must be refused, got %v", err)
	}
}

// TestBridgeRolesDeterministic proves the mapping is deterministic: the same
// (drafter, verifier, Pool) always yields the same SpecWiring.
func TestBridgeRolesDeterministic(t *testing.T) {
	pool := coResidentSpecPool(t)
	first, err := BridgeRoles("draft-7b", "verify-70b", pool)
	if err != nil {
		t.Fatalf("bridge: %v", err)
	}
	for i := 0; i < 50; i++ {
		got, err := BridgeRoles("draft-7b", "verify-70b", pool)
		if err != nil {
			t.Fatalf("bridge at %d: %v", i, err)
		}
		if got != first {
			t.Fatalf("non-deterministic bridge at %d: %+v != %+v", i, got, first)
		}
	}
}

// TestBridgeRolesTreeAcceptPass proves the Tree form resolves the same co-resident
// pairing but carries the TREE accept rule: SpecWiring.AcceptTree on a linear chain
// reduces exactly to the greedy rule (the strict-generalization invariant), and the
// wiring is flagged Tree.
func TestBridgeRolesTreeAcceptPass(t *testing.T) {
	pool := coResidentSpecPool(t)
	w, err := BridgeRolesTree("draft-7b", "verify-70b", pool)
	if err != nil {
		t.Fatalf("tree bridge: %v", err)
	}
	if !w.Tree {
		t.Fatal("BridgeRolesTree must set Tree=true")
	}
	if w.Drafter != "draft-7b" || w.Verifier != "verify-70b" {
		t.Fatalf("tree wiring = %+v", w)
	}
	// A LINEAR chain tree: root predicts token 5; node 1 (token 5) predicts 6; node 2
	// (token 6) predicts 9. The drafter proposed [5,6], the target's argmax matches both,
	// so the accepted path is [1,2], advance 3 — exactly what the greedy chain gives.
	tree := SpecTree{Nodes: []TreeNode{
		{TargetArgmax: 5, Children: []int{1}},
		{Token: 5, TargetArgmax: 6, Children: []int{2}},
		{Token: 6, TargetArgmax: 9, Children: nil},
	}}
	got := w.AcceptTree(tree)
	want := AcceptTree(tree)
	if got.Advance != want.Advance || got.KeepKV != want.KeepKV || got.EvictKV != want.EvictKV {
		t.Fatalf("wiring AcceptTree %+v != package AcceptTree %+v", got, want)
	}
	if want.Advance != 3 || want.KeepKV != 2 || want.EvictKV != 0 {
		t.Fatalf("linear-chain tree accounting unexpected: %+v", want)
	}
}
