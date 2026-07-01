package cachemeta

import "strconv"

// nixl_lease.go witnesses an EXTERNAL vLLM NIXL KV lease as a first-class cachemeta
// record, so a remote/disaggregated K/V span stops being an opaque "warm bit" and
// becomes a typed residency claim fak can fold, expire, and demote.
//
// vLLM's disaggregated-prefill design (docs/design/nixl_kv_cache_lease.md,
// nixl_kv_push_connector.md) works like this: a PREFILL node pins the K/V blocks for
// a request under a LEASE; a DECODE node heartbeats that lease from scheduler
// admission; and the pinned blocks FREE on transfer completion, finish/abort, or
// lease expiry. Push mode writes the K/V straight into decode memory over a NIXL
// WRITE and signals completion. From fak's control-plane view the lease is the only
// honest evidence that the remote span is resident: while the lease is held the span
// is warm and a cache-aware route to that engine is valid; the moment it frees
// (delivered, aborted, or expired) the warm claim is dead and routing on it would be
// a false hit.
//
// This file keeps cachemeta's field-only, wall-clock-FREE posture (the caller injects
// nowMillis, exactly like lifecycle.go's Advance): a lease's residency verdict is a
// deterministic function of its last event and the clock, so a disaggregated workload
// replays reproducibly. It reuses FromKVTransfer to place the lease on the same
// kv_transfer plane as every other residency event, and the LookupVerdict trichotomy
// to express "warm/routable" (Hit) vs "demote, do not route" (Miss/Fault) — never a
// second parallel state model.

// NIXLTransferRole names which side of a disaggregated NIXL transfer a lease belongs
// to. The prefill node pins the blocks; the decode node pulls them (or, in push mode,
// receives a direct NIXL WRITE into its memory).
type NIXLTransferRole string

const (
	// NIXLRolePrefillPin — the prefill node holding the K/V blocks pinned under the
	// lease so a decode node can still transfer them.
	NIXLRolePrefillPin NIXLTransferRole = "prefill_pin"
	// NIXLRoleDecodePull — the decode node pulling the pinned blocks over NIXL READ.
	NIXLRoleDecodePull NIXLTransferRole = "decode_pull"
	// NIXLRolePushWrite — push mode: the prefill node writing K/V directly into decode
	// memory over NIXL WRITE, with a completion notification.
	NIXLRolePushWrite NIXLTransferRole = "push_write"
)

// NIXLLeaseEvent names an observed transition in a remote lease's life. These are the
// events a NixlConnector/scheduler surface emits; fak witnesses them rather than
// inferring residency from a bare warm flag.
type NIXLLeaseEvent string

const (
	// NIXLLeaseCreate — the prefill node pinned the blocks and opened the lease.
	NIXLLeaseCreate NIXLLeaseEvent = "create"
	// NIXLLeaseHeartbeat — the decode node re-asserted the lease from scheduler
	// admission; it refreshes the expiry window.
	NIXLLeaseHeartbeat NIXLLeaseEvent = "heartbeat"
	// NIXLLeaseTransferComplete — the K/V reached the decode node; the pin is released
	// and the blocks free. A SUCCESS that nonetheless ends the remote residency claim.
	NIXLLeaseTransferComplete NIXLLeaseEvent = "transfer_complete"
	// NIXLLeaseExpiry — the lease's TTL elapsed with no heartbeat; the blocks free.
	NIXLLeaseExpiry NIXLLeaseEvent = "expiry"
	// NIXLLeaseAbort — the request finished/aborted or the transfer failed; the blocks
	// free without a successful delivery.
	NIXLLeaseAbort NIXLLeaseEvent = "abort"
)

// NIXLLease is the field-only witness of one external vLLM NIXL KV lease. It is keyed
// by the request/trace it serves and the remote engine that holds it, names the block
// identity (the content-addressed span digest + its length), the transfer role, the
// last observed event, and the wall-clock instants that bound the lease. cachemeta
// never calls a vLLM API; a NixlConnector adapter populates this shape.
type NIXLLease struct {
	// LeaseID is the remote engine's own handle for the lease (its residency lease).
	LeaseID string
	// TraceID / RequestID scope the lease to the agent request it serves.
	TraceID   string
	RequestID string
	// RemoteEngineID identifies the vLLM engine instance that holds the pinned blocks.
	RemoteEngineID string
	// SpanDigest + Tokens are the block identity: the content-addressed K/V span and
	// its length in positions. Empty digest = an unidentified lease (fail-closed: it
	// can never be a warm hit, see NIXLLeaseVerdict).
	SpanDigest string
	Tokens     int64
	// ModelID / TokenizerID bind the span to the model that produced it.
	ModelID     string
	TokenizerID string
	// Role is which side of the transfer this lease belongs to.
	Role NIXLTransferRole
	// Event is the most recent observed transition.
	Event NIXLLeaseEvent
	// GrantedAtMillis is when the lease was created/last refreshed; ExpiresAtMillis is
	// the instant it lapses absent a heartbeat. A non-positive ExpiresAtMillis means
	// "no expiry clock" (residency is governed only by explicit complete/abort events).
	GrantedAtMillis int64
	ExpiresAtMillis int64
	// BytesMoved records transfer volume when known (0 until a completion is observed).
	BytesMoved int64
}

// NIXLLeaseState is the residency state of a lease at a given clock: whether the
// remote span is still a warm, routable claim, or the claim is dead (delivered,
// expired, or failed) and must be demoted from the warm set.
type NIXLLeaseState string

const (
	// NIXLStateActive — lease held and unexpired; the remote span is warm and a
	// cache-aware route to RemoteEngineID is valid.
	NIXLStateActive NIXLLeaseState = "active"
	// NIXLStateCompleted — the transfer completed; the pin freed. Not warm anymore.
	NIXLStateCompleted NIXLLeaseState = "completed"
	// NIXLStateExpired — the lease TTL elapsed; the pin freed. Not warm anymore.
	NIXLStateExpired NIXLLeaseState = "expired"
	// NIXLStateFailed — the request aborted or the transfer failed; the pin freed
	// without delivery. Not warm; a fault a router must see.
	NIXLStateFailed NIXLLeaseState = "failed"
	// NIXLStateUnidentified — the lease names no span; it can never back a hit.
	NIXLStateUnidentified NIXLLeaseState = "unidentified"
)

// ReasonLeaseReleased is the miss cause for a remote span whose lease FREED after a
// successful transfer completion — a distinct, non-fault reason from an expiry so a
// router/metric can tell "delivered, pin gone" apart from "TTL lapsed".
const ReasonLeaseReleased LookupReason = "lease_released"

// StateAt folds the lease's last event and the injected clock into its residency
// state. A terminal event (complete/abort) decides regardless of the clock; otherwise
// an unexpired create/heartbeat is Active, and a create/heartbeat whose expiry
// instant has passed is Expired — so a router that never saw an explicit expiry event
// still demotes a lapsed lease purely from the clock. An empty span digest is always
// Unidentified (fail-closed).
func (l NIXLLease) StateAt(nowMillis int64) NIXLLeaseState {
	if l.SpanDigest == "" {
		return NIXLStateUnidentified
	}
	switch l.Event {
	case NIXLLeaseTransferComplete:
		return NIXLStateCompleted
	case NIXLLeaseAbort:
		return NIXLStateFailed
	case NIXLLeaseExpiry:
		return NIXLStateExpired
	case NIXLLeaseCreate, NIXLLeaseHeartbeat:
		if l.ExpiresAtMillis > 0 && nowMillis >= l.ExpiresAtMillis {
			return NIXLStateExpired
		}
		return NIXLStateActive
	default:
		// Unknown/zero event: never assume warm.
		return NIXLStateExpired
	}
}

// FromNIXLLease lowers the lease witness onto the kv_transfer plane, reusing
// FromKVTransfer so a remote lease flows through the SAME residency stream as every
// other KV movement. Residency is TierRemote owned by the remote engine and held
// under LeaseID; the lease's role, event, request/trace, and expiry are recorded as
// labels so an observing sink separates them without re-deriving. The transfer
// direction reflects the role (a prefill pin/push is an offload of residency to the
// remote holder; a decode pull is a restore toward the consumer). Outcome tracks the
// event: an abort lowers to a typed FAULT, everything else to OK residency — the
// warm/demote decision itself is NIXLLeaseVerdict's job, off this record.
func FromNIXLLease(l NIXLLease, nowMillis int64, opts ...Option) Entry {
	dir := KVOffload
	if l.Role == NIXLRoleDecodePull {
		dir = KVRestore
	}
	outcome := KVTransferOK
	faultReason := ""
	if l.StateAt(nowMillis) == NIXLStateFailed {
		outcome = KVTransferFault
		faultReason = "nixl_lease_abort"
	}
	e := FromKVTransfer(KVTransfer{
		Direction:   dir,
		SpanDigest:  l.SpanDigest,
		Tokens:      l.Tokens,
		ModelID:     l.ModelID,
		TokenizerID: l.TokenizerID,
		ToTier:      TierRemote,
		Owner:       l.RemoteEngineID,
		Lease:       l.LeaseID,
		Outcome:     outcome,
		FaultReason: faultReason,
		BytesMoved:  l.BytesMoved,
	}, opts...)
	if e.Labels == nil {
		e.Labels = map[string]string{}
	}
	e.Labels["nixl_lease"] = l.LeaseID
	e.Labels["nixl_role"] = string(l.Role)
	e.Labels["nixl_event"] = string(l.Event)
	e.Labels["nixl_state"] = string(l.StateAt(nowMillis))
	if l.RemoteEngineID != "" {
		e.Labels["remote_engine"] = l.RemoteEngineID
	}
	if l.TraceID != "" {
		e.Labels["trace_id"] = l.TraceID
	}
	if l.RequestID != "" {
		e.Labels["request_id"] = l.RequestID
	}
	if l.ExpiresAtMillis > 0 {
		e.Labels["lease_expires_at"] = strconv.FormatInt(l.ExpiresAtMillis, 10)
	}
	// An external lease is refuted the moment it frees; mark it so the coherence plane
	// does not treat it as a policy-governed local borrow.
	e.Coherence.InvalidationMode = InvalidationExternalRefutation
	return e
}

// NIXLLeaseVerdict folds a lease's residency state at nowMillis into the typed lookup
// verdict a router MUST consult before taking a cache-aware route to the remote span:
//
//   - Active     -> HIT: the span is warm; a cache-aware route is valid.
//   - Completed  -> MISS(lease_released): delivered, pin freed; demote from warm set.
//   - Expired    -> MISS(expired_ttl): TTL lapsed, pin freed; demote from warm set.
//   - Failed     -> FAULT(residency_fault): aborted/failed transfer; demote + fault.
//   - Unidentified -> MISS(absent): the lease names no span; never a hit.
//
// Only the Active verdict returns CanServe() == true, so a caller that gates routing
// on CanServe never routes on a freed/expired/failed lease — that is the acceptance
// guarantee "no cache-aware route is taken on an expired entry", expressed in the
// package's own verdict vocabulary.
func NIXLLeaseVerdict(l NIXLLease, nowMillis int64) LookupVerdict {
	e := FromNIXLLease(l, nowMillis)
	switch l.StateAt(nowMillis) {
	case NIXLStateActive:
		return Hit(e)
	case NIXLStateCompleted:
		return Miss(ReasonLeaseReleased)
	case NIXLStateExpired:
		return Miss(ReasonExpiredTTL)
	case NIXLStateFailed:
		return Fault(e, ReasonResidencyFault)
	default:
		return Miss(ReasonAbsent)
	}
}

// Warm reports whether the lease backs a warm, routable remote span at nowMillis. It
// is the boolean projection of NIXLLeaseVerdict for a warm-set membership check: a
// warm-set holding this entry must DEMOTE it as soon as Warm returns false.
func (l NIXLLease) Warm(nowMillis int64) bool {
	return l.StateAt(nowMillis) == NIXLStateActive
}

// NIXLClearProof classifies how well fak can prove a remote deletion/clear actually
// removed the K/V it targeted — the acceptance requirement that a deletion event
// record exact-span proof, a whole-prefix reset, or no proof at all. It maps onto the
// existing KVEvictionScope vocabulary (exact_span / whole_prefix_cache) plus an
// explicit no-proof floor, so a caller never mistakes "we reset something coarse" or
// "we have no receipt" for a precise eviction.
type NIXLClearProof string

const (
	// NIXLClearExactSpan — the engine confirmed it evicted the exact named block; fak
	// holds precise proof.
	NIXLClearExactSpan NIXLClearProof = "exact_span"
	// NIXLClearWholePrefix — the engine has no exact-span API, so fak reset the whole
	// prefix cache: coarse but attested over-invalidation.
	NIXLClearWholePrefix NIXLClearProof = "whole_prefix"
	// NIXLClearNoProof — no eviction receipt and/or no span identity; fak cannot prove
	// anything was cleared. Fail-closed: this is the default for an unwitnessed clear.
	NIXLClearNoProof NIXLClearProof = "none"
)

// NIXLClear is the field-only witness of a remote deletion/clear event: the block it
// targeted (SpanDigest), whether the engine confirmed an exact-span eviction, and
// whether a whole-prefix reset was performed as the coarse fallback.
type NIXLClear struct {
	SpanDigest         string
	ExactSpanConfirmed bool
	WholePrefixReset   bool
}

// ClassifyNIXLClear records which proof of removal fak holds for a clear event, in
// strict precedence: a confirmed exact-span eviction of a NAMED span is the strongest
// (exact_span); absent that, an attested whole-prefix reset is coarse proof
// (whole_prefix); absent both — or a clear that names no span — fak has no proof
// (none). The named-span requirement on exact_span is the fail-closed seam: an
// "exact" claim with no block identity can never be precise, so it degrades to none.
func ClassifyNIXLClear(c NIXLClear) NIXLClearProof {
	if c.ExactSpanConfirmed && c.SpanDigest != "" {
		return NIXLClearExactSpan
	}
	if c.WholePrefixReset {
		return NIXLClearWholePrefix
	}
	return NIXLClearNoProof
}

// EvictionScope projects a clear proof onto the shared KVEvictionScope vocabulary for
// callers that already speak it. Exact-span proof maps to the exact-span scope; a
// whole-prefix reset to the whole-prefix-cache scope. A no-proof clear has no valid
// scope to report (false), so a caller requiring an attested scope fails closed
// rather than reporting a phantom eviction.
func (p NIXLClearProof) EvictionScope() (KVEvictionScope, bool) {
	switch p {
	case NIXLClearExactSpan:
		return KVEvictionScopeExactSpan, true
	case NIXLClearWholePrefix:
		return KVEvictionScopeWholePrefixCache, true
	default:
		return "", false
	}
}
