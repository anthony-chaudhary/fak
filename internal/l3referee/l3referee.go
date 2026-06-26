// Package l3referee stands fak up as an admission + attestation REFEREE in front
// of an external L3 KV store's get/set seam — the "ship-tomorrow" rung of the
// disaggregated-cache study (#78, epic #504; L3-DISAGGREGATED-CACHE-REIMAGINED.md
// §4 Option A + §7). It is the SAME shape as the gateway in front of the Claude
// API: the page bytes pass through BYTE-EXACT so RDMA zero-copy + prefix-caching
// are preserved, and fak adjudicates only on the CONTROL PATH.
//
// CONTROL PATH ONLY — the load-bearing constraint. Neither referee method ever
// takes the page payload ([]byte): AdmitSet sees a Ref + a durability hint;
// VerifyGet sees a Ref + a digest the caller already computed. The referee never
// sits on the 1–5 µs RDMA data path; verified bytes still flow client-direct.
// TestNoDataPathHook proves this structurally (no []byte in either signature).
//
// Two gaps, the issue's deliberate first slice (highest value-per-hour pair):
//
//   - G6 durability-gated admission (on set): admit to the shared L3 tier only a
//     page that EARNS it. The S7 classifier (#496 rung-1 / #498) labels a page
//     turn|session|bounded|durable; the referee admits at-or-above a configured
//     FLOOR (default Bounded) and refuses the rest — turn/session pages stay
//     local. An unknown class DEFAULT-EXPIRES (refuse): fail closed.
//   - G1 return-digest verification (on get): refuse a returned page whose digest
//     does not match the Ref.Digest fak recorded at write, so a hash collision or
//     a mis-computed prefix is caught instead of silently returning wrong KV.
//
// The seam is STORE-AGNOSTIC: CAMA is the reference target but the referee names
// no store. Out of scope (study §6.3 over-reach): middle-eviction (G2) and L3
// deletion certs (G3) — a pure sidecar does not own the KV math; those need the
// L3RegionBackend child.
//
// Tier 1 (foundation): a pure control-path decision leaf; imports only abi (and
// stdlib in the fake backend), registers nothing into the kernel, off the hot
// path. The live wiring (classifier from #498, the store seam) is the caller's.
package l3referee

import "github.com/anthony-chaudhary/fak/internal/abi"

// refereeBy stamps Verdict.By for forensics (which adjudicator decided).
const refereeBy = "l3referee"

// metaReason is the Verdict.Meta key under which the precise L3 sub-reason token
// rides. Verdict.Reason carries a core abi.ReasonCode (the closed, trainable
// vocabulary every leaf shares — house style; no leaf registers out-of-tree
// reasons); this Meta token is the finer L3 classification a consumer switches on.
const metaReason = "reason"

// Durability is the S7 classifier's output vocabulary (#498): how long a page's
// truth lasts. The referee CONSUMES this label; it does not classify (the
// classifier stays the caller's, decoupled so #498 can be stubbed then wired).
// The string values are byte-identical to internal/ctxplan's so a
// Meta["durability"] hint flows straight through, but they are redeclared here to
// keep this leaf store-agnostic and abi-only (no ctxplan import).
const (
	DurabilityTurn    = "turn"    // this turn only — never leaves the local tier
	DurabilitySession = "session" // this session — stays local
	DurabilityBounded = "bounded" // bounded TTL — earns the shared tier
	DurabilityDurable = "durable" // long-lived — earns the shared tier
)

// Typed L3 refusal tokens, carried in Verdict.Meta["reason"]. Exported so a
// caller (or a test) can switch on the precise reason without parsing prose.
const (
	ReasonDigestMismatch    = "L3_DIGEST_MISMATCH"     // G1: returned page ≠ recorded digest
	ReasonDurabilityFloor   = "L3_DURABILITY_FLOOR"    // G6/G8: class below the admission floor
	ReasonDurabilityUnknown = "L3_DURABILITY_UNKNOWN"  // G6: unclassified — default-expire
	ReasonNoRecordedDigest  = "L3_NO_RECORDED_DIGEST"  // G1: nothing to verify against — quarantine
)

// durabilityRank orders the classes; a higher rank earns more of the shared
// tier. -1 = an unknown class (default-expire).
func durabilityRank(class string) int {
	switch class {
	case DurabilityTurn:
		return 0
	case DurabilitySession:
		return 1
	case DurabilityBounded:
		return 2
	case DurabilityDurable:
		return 3
	default:
		return -1
	}
}

// L3Referee adjudicates an external L3 KV store's two control-path events. Both
// methods return a typed abi.Verdict — Allow, Quarantine (hold out: unprovable),
// or a Deny carrying a reason. NEITHER takes the page payload: control path only.
type L3Referee interface {
	// AdmitSet decides whether a page being WRITTEN earns the shared L3 tier,
	// from its durability class (G6). Below the floor or unclassified => refuse.
	AdmitSet(ref abi.Ref, durabilityHint string) abi.Verdict
	// VerifyGet decides whether a page just READ back is the page fak recorded,
	// by comparing the caller-computed pageDigest to ref.Digest (G1). Mismatch
	// => refuse; no recorded digest => quarantine; match => allow.
	VerifyGet(ref abi.Ref, pageDigest string) abi.Verdict
}

// Referee is the default, store-agnostic L3Referee. The zero value is usable and
// fail-closed: an empty Floor resolves to the Bounded default, so turn/session
// pages are refused unless the operator deliberately lowers the floor.
type Referee struct {
	// Floor is the lowest durability class admitted to the shared tier; a page
	// classified below it stays local. Empty => DurabilityBounded.
	Floor string
}

// compile-time proof the default impl satisfies the interface.
var _ L3Referee = (*Referee)(nil)

// New returns a Referee with the given durability floor (empty => Bounded).
func New(floor string) *Referee { return &Referee{Floor: floor} }

func (r *Referee) floor() string {
	if r.Floor == "" {
		return DurabilityBounded
	}
	return r.Floor
}

// AdmitSet implements G6: durability-gated admission on set.
func (r *Referee) AdmitSet(ref abi.Ref, durabilityHint string) abi.Verdict {
	rank := durabilityRank(durabilityHint)
	if rank < 0 {
		// Unknown class — default-expire (the issue's "Default = expire"): a page
		// fak cannot place in the truth-duration lattice never reaches the shared
		// tier. Fail closed.
		return refuse(abi.ReasonPolicyBlock, ReasonDurabilityUnknown, durabilityHint)
	}
	if rank < durabilityRank(r.floor()) {
		// Below the configured floor (G6/G8): turn/session pages stay local.
		return refuse(abi.ReasonPolicyBlock, ReasonDurabilityFloor, durabilityHint)
	}
	return abi.Verdict{
		Kind: abi.VerdictAllow,
		By:   refereeBy,
		Meta: map[string]string{"durability": durabilityHint},
	}
}

// VerifyGet implements G1: return-digest verification on get.
func (r *Referee) VerifyGet(ref abi.Ref, pageDigest string) abi.Verdict {
	if ref.Digest == "" {
		// No recorded digest to check against — the referee cannot PROVE the
		// returned page is correct, so it holds it out rather than trust it.
		return abi.Verdict{
			Kind:    abi.VerdictQuarantine,
			Payload: abi.QuarantinePayload{PageOut: true},
			Reason:  abi.ReasonUnwitnessed,
			By:      refereeBy,
			Meta:    map[string]string{metaReason: ReasonNoRecordedDigest},
		}
	}
	if pageDigest != ref.Digest {
		// G1 bite: the page that came back is not the page fak recorded — a hash
		// collision or a mis-computed prefix. Refuse; never return wrong KV.
		v := refuse(abi.ReasonTrustViolation, ReasonDigestMismatch, "")
		v.Meta["recorded"] = ref.Digest
		v.Meta["returned"] = pageDigest
		return v
	}
	return abi.Verdict{Kind: abi.VerdictAllow, By: refereeBy}
}

// refuse builds a Deny verdict citing a core reason code and the precise L3 token.
func refuse(code abi.ReasonCode, token, durability string) abi.Verdict {
	meta := map[string]string{metaReason: token}
	if durability != "" {
		meta["durability"] = durability
	}
	return abi.Verdict{Kind: abi.VerdictDeny, Reason: code, By: refereeBy, Meta: meta}
}
