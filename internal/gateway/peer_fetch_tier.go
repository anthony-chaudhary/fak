package gateway

// peer_fetch_tier.go — model another node's held KV as a READ ONLY fetch tier, and
// the deterministic choice of whether to FETCH a needed prefix from that peer tier
// or RECOMPUTE it locally (issue #5269, epic #4296 — distributed KV mobility).
//
// residency_router.go already routes a REQUEST to the worker that holds the shared
// prefix (the control plane half): a "control plane signal only, no KV bytes move
// here" comment marks that it deliberately defers moving the bytes. This file is the
// data plane half. When co-locating the request on the warm peer is not possible, a
// node may instead treat the peer's held KV as an extra, slower, READ ONLY residency
// tier it can FETCH from rather than recompute the prefix cold.
//
// READ ONLY means the local node never writes or evicts into the peer tier — every
// method here takes a VALUE receiver, so a choice can never mutate the peer's held
// state. It composes with the tier weighted overlap credit (tier_overlap_credit.go):
// that term scores WHERE a request should land; this term decides, once landed, HOW
// to obtain a prefix the local node lacks. Deterministic and wall clock free — no
// network, no GPU, no time source. Every degenerate input fails closed to RECOMPUTE
// (the local, always safe move).

import "math"

// PrefixNeed is the run of leading prefix blocks a request needs KV for but the local
// node does not already hold. Blocks is the count the node must obtain (by fetch or by
// recompute) before it can serve the request.
type PrefixNeed struct {
	Blocks int
}

// PeerKVTier models another node's held KV as a READ ONLY residency tier the local
// node may fetch from. HeldBlocks is the leading run of the needed prefix the peer
// holds KV for; BytesPerBlock is the transfer size of one block over the link. Value
// semantics keep it read only: a choice reads these fields and never writes them, so
// the local node can never evict or overwrite the peer's state through this type.
type PeerKVTier struct {
	HeldBlocks    int
	BytesPerBlock int64
}

// FetchLink is the read only transfer path from the peer tier to the local node.
// BytesPerUnit is the deterministic transfer rate: bytes moved per unit of cost (the
// bytes-over-link term). A larger rate makes the same transfer cheaper.
type FetchLink struct {
	BytesPerUnit float64
}

// RecomputeLocal is the local cost of recomputing the needed prefix from scratch.
// PerBlock is the cost of recomputing one prefix block on the local node, in the same
// unit as a fetch cost so the two are directly comparable.
type RecomputeLocal struct {
	PerBlock float64
}

// FetchVerdict is the deterministic outcome of the fetch-or-recompute choice. Fetch is
// true only when pulling from the peer tier strictly beats recomputing locally; the
// two cost fields expose the compared numbers so a caller (or a witness) can see why.
// Reason is a short, closed-vocabulary token for the branch taken.
type FetchVerdict struct {
	Fetch         bool
	FetchCost     float64
	RecomputeCost float64
	Reason        string
}

const (
	// reasonFetch: the peer holds the full needed prefix and the fetch is strictly
	// cheaper than a local recompute, so pull the bytes.
	reasonFetch = "fetch"
	// reasonRecomputeCheaper: a valid fetch is possible but costs at least as much as
	// recomputing, so keep the local move (fetch is never chosen when it is slower).
	reasonRecomputeCheaper = "recompute_cheaper"
	// reasonPeerShort: the peer does not hold the full needed prefix; a read only tier
	// cannot be written to fill the gap, so recompute.
	reasonPeerShort = "peer_short"
	// reasonNoNeed: the request needs nothing, so there is nothing to fetch.
	reasonNoNeed = "no_need"
	// reasonBadInput: a degenerate cost input (non-positive rate/size, NaN/Inf, negative
	// recompute) fails closed to the safe local recompute.
	reasonBadInput = "bad_input"
)

// Choose is the pure, deterministic fetch-or-recompute core. Given what the peer holds
// (the read only receiver), the request's need, the link rate, and the local recompute
// cost, it returns whether to FETCH from the peer tier or RECOMPUTE locally. Properties
// the witness pins:
//   - Peer holds the full prefix AND fetch is strictly cheaper → Fetch.
//   - Peer holds nothing (or less than needed) → recompute (read only: no gap fill).
//   - Fetch cost at or above recompute cost → recompute (never chosen when slower).
//   - Read only: a VALUE receiver, so the choice cannot mutate the peer's held state.
//   - Every degenerate input (no need, non-positive rate/size, NaN/Inf, negative
//     recompute) fails closed to recompute, the local and always safe move.
func (t PeerKVTier) Choose(need PrefixNeed, link FetchLink, local RecomputeLocal) FetchVerdict {
	// Nothing needed: no fetch, recompute is a no-op of zero cost. Fail closed.
	if need.Blocks <= 0 {
		return FetchVerdict{Fetch: false, Reason: reasonNoNeed}
	}

	// Read only tier: it can only serve a prefix it fully holds. A partial hold cannot
	// be topped up by writing into the peer, so recompute the whole run locally.
	if t.HeldBlocks < need.Blocks {
		return FetchVerdict{Fetch: false, Reason: reasonPeerShort}
	}

	// Degenerate cost inputs fail closed to the local recompute.
	if t.BytesPerBlock <= 0 ||
		link.BytesPerUnit <= 0 || math.IsNaN(link.BytesPerUnit) || math.IsInf(link.BytesPerUnit, 0) ||
		local.PerBlock < 0 || math.IsNaN(local.PerBlock) || math.IsInf(local.PerBlock, 0) {
		return FetchVerdict{Fetch: false, Reason: reasonBadInput}
	}

	// Only the NEEDED blocks move over the link, even if the peer holds more.
	bytes := float64(need.Blocks) * float64(t.BytesPerBlock)
	fetchCost := bytes / link.BytesPerUnit
	recomputeCost := float64(need.Blocks) * local.PerBlock

	if math.IsNaN(fetchCost) || math.IsInf(fetchCost, 0) {
		return FetchVerdict{Fetch: false, Reason: reasonBadInput}
	}

	v := FetchVerdict{FetchCost: fetchCost, RecomputeCost: recomputeCost}
	// Strictly cheaper wins; a tie keeps the local recompute so fetch is never chosen
	// when it is not an improvement.
	if fetchCost < recomputeCost {
		v.Fetch = true
		v.Reason = reasonFetch
	} else {
		v.Reason = reasonRecomputeCheaper
	}
	return v
}
