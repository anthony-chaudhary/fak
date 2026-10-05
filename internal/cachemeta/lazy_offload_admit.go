package cachemeta

import "strings"

// Lazy-offload store refusal reasons, declared here so the admission gate stays a
// bounded, self-contained leaf. They extend the shared LookupReason vocabulary with
// the closed causes this decision can produce.
const (
	// ReasonUnidentifiedSpan: a store event named no stable span/block identity.
	ReasonUnidentifiedSpan LookupReason = "unidentified_span"
	// ReasonPrefixUnknown: the caller did not assert prefix continuity.
	ReasonPrefixUnknown LookupReason = "prefix_unknown"
	// ReasonPrefixBroken: the caller asserted a broken prefix chain.
	ReasonPrefixBroken LookupReason = "prefix_discontinuity"
	// ReasonOrphanNoReceipt: an orphaned batch reached no terminal receipt.
	ReasonOrphanNoReceipt LookupReason = "orphaned_without_receipt"
	// ReasonReceiptPending: the batch's only receipt is still pending.
	ReasonReceiptPending LookupReason = "receipt_pending"
	// ReasonReceiptFailed: the batch's terminal receipt is a failure.
	ReasonReceiptFailed LookupReason = "receipt_failed"
)

// This file adapts the lazy-offload store-safety rules from LMCache's
// eviction-aware integration design into one PURE admission decision. LMCache
// buffers produced K/V spans and appends them lazily; a span with no stable
// identity, a broken prefix chain, or an orphaned batch that never produced a
// terminal receipt can be stored but can never be reliably evicted or retrieved
// again. Refusing such a store event BEFORE it happens keeps the cache from
// acquiring unreachable or unverifiable residency.
//
// Source: LMCache @ 1b7dff2cd83fc634326b5989eb71aa5f05b4f426,
// lmcache/docs/design/integration/vllm/lazy_offload_policy/eviction_aware.md:30-70
// (Apache-2.0). This is an attributed Go adaptation of the protocol rules; no
// Python bytes are copied.
//
// The decision is deterministic and wall-clock free: it is a function of the
// explicit facts the caller supplies and nothing else. The caller must assert
// the facts — this gate does not infer prefix continuity or receipt state from
// bytes, and it never inspects payload.

// PrefixContinuity is the caller's explicit claim about whether a store event's
// span continues the prefix chain it extends. Unknown (the zero value) is
// refused when admission facts are supplied: an unasserted prefix is not a
// continuous one.
type PrefixContinuity string

const (
	PrefixContinuityUnknown PrefixContinuity = ""
	PrefixContinuityKnown   PrefixContinuity = "known"
	PrefixContinuityBroken  PrefixContinuity = "broken"
)

// BatchState is the caller's explicit claim about the batch that produced a store
// event. An orphaned batch (one whose owner is gone before it reached a terminal
// receipt) is only safe to drain when a successful terminal receipt proves it.
type BatchState string

const (
	BatchUnknown  BatchState = ""
	BatchActive   BatchState = "active"
	BatchOrphaned BatchState = "orphaned"
)

// ReceiptState is the caller's explicit claim about a batch's store receipt.
type ReceiptState string

const (
	ReceiptUnknown ReceiptState = ""
	ReceiptPending ReceiptState = "pending"
	ReceiptSuccess ReceiptState = "success"
	ReceiptFailed  ReceiptState = "failed"
)

// StoreAdmission is the explicit, payload-free fact set a caller supplies for a
// lazy-offload store event. It is optional on LMCTransferEvent: a nil
// StoreAdmission preserves the legacy mapping exactly and is NEVER evidence that
// store safety was checked.
type StoreAdmission struct {
	// CoveredBlocks names the block identities the store event covers. An empty
	// list means the event does not identify what it stores (a hash-less span),
	// which is refused.
	CoveredBlocks []string
	// PrefixContinuity is the explicit prefix-chain claim. Unknown is refused.
	PrefixContinuity PrefixContinuity
	// Batch is the explicit producer-batch state.
	Batch BatchState
	// Aborted records that the producing batch was aborted. Abort ALONE is not a
	// refusal reason: an otherwise valid buffered append still drains.
	Aborted bool
	// Receipt is the explicit store receipt state for the batch.
	Receipt ReceiptState
}

// StoreRefusal is one closed reason a lazy-offload store event was refused, or the
// empty StoreAllowed value when it is admissible.
type StoreRefusal string

const (
	// StoreAllowed is the zero value: the store event is admissible.
	StoreAllowed StoreRefusal = ""
	// StoreRefusedUnidentified: the span names no stable identity (missing span
	// digest or no covered block identities).
	StoreRefusedUnidentified StoreRefusal = "unidentified_span"
	// StoreRefusedPrefixUnknown: the caller did not assert prefix continuity.
	StoreRefusedPrefixUnknown StoreRefusal = "prefix_unknown"
	// StoreRefusedPrefixBroken: the caller asserted an explicitly broken prefix
	// chain.
	StoreRefusedPrefixBroken StoreRefusal = "prefix_discontinuity"
	// StoreRefusedOrphanNoReceipt: an orphaned batch reached no terminal receipt.
	StoreRefusedOrphanNoReceipt StoreRefusal = "orphaned_without_receipt"
	// StoreRefusedReceiptPending: the only receipt is still pending, so a later
	// eviction/retrieval cannot be tracked.
	StoreRefusedReceiptPending StoreRefusal = "receipt_pending"
	// StoreRefusedReceiptFailed: the terminal receipt is a failure.
	StoreRefusedReceiptFailed StoreRefusal = "receipt_failed"
)

// Allowed reports whether the refusal is the allowed zero value.
func (r StoreRefusal) Allowed() bool { return r == StoreAllowed }

// Reason maps a refusal onto the LookupReason the shared verdict vocabulary
// carries, so a refused append surfaces as a typed MISS rather than an ad-hoc
// string. StoreAllowed maps to ReasonNone.
func (r StoreRefusal) Reason() LookupReason {
	switch r {
	case StoreRefusedUnidentified:
		return ReasonUnidentifiedSpan
	case StoreRefusedPrefixUnknown:
		return ReasonPrefixUnknown
	case StoreRefusedPrefixBroken:
		return ReasonPrefixBroken
	case StoreRefusedOrphanNoReceipt:
		return ReasonOrphanNoReceipt
	case StoreRefusedReceiptPending:
		return ReasonReceiptPending
	case StoreRefusedReceiptFailed:
		return ReasonReceiptFailed
	default:
		return ReasonNone
	}
}

// StoreAdmissible is the pure lazy-offload store-admission decision. It reads
// only the event's identity and its explicit StoreAdmission facts; it performs
// no I/O and consults no clock, so replaying the same event yields the same
// decision. A nil StoreAdmission returns StoreAllowed (legacy mapping): the gate
// is opt-in and never silently claims a store was checked.
func StoreAdmissible(ev LMCTransferEvent) StoreRefusal {
	s := ev.StoreAdmission
	if s == nil {
		return StoreAllowed
	}
	// Identity: a store event that names no stable span or covers no block cannot
	// be tracked for later eviction/retrieval.
	if strings.TrimSpace(ev.SpanDigest) == "" || len(s.CoveredBlocks) == 0 {
		return StoreRefusedUnidentified
	}
	// Prefix continuity must be explicitly asserted known.
	switch s.PrefixContinuity {
	case PrefixContinuityKnown:
	case PrefixContinuityBroken:
		return StoreRefusedPrefixBroken
	default:
		return StoreRefusedPrefixUnknown
	}
	// An orphaned batch is only safe to drain on a successful terminal receipt.
	// Abort alone is deliberately not consulted: an aborted-but-valid buffered
	// append still drains.
	if s.Batch == BatchOrphaned {
		switch s.Receipt {
		case ReceiptSuccess:
			// may drain
		case ReceiptPending:
			return StoreRefusedReceiptPending
		case ReceiptFailed:
			return StoreRefusedReceiptFailed
		default:
			return StoreRefusedOrphanNoReceipt
		}
	}
	return StoreAllowed
}
