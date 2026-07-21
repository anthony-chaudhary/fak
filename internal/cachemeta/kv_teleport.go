package cachemeta

// kv_teleport.go is the deterministic, no-hardware half of the KV "teleport" witness
// (#4301). When a load balancer re-routes a live conversation from node A to node B
// (A got busy, or a rolling deploy drained it), B can either recompute the whole prefix
// from scratch — throwing the warm KV on A away — or TELEPORT the serialized KV span
// A->B over the transport plane so the warm prefix survives the hop. This file quantifies
// WHICH of those two the re-route should take: it is the "session-migration policy on top
// of that transport" the issue names, expressed as a pure value.
//
// The transport itself (the RDMA/NVLink/TCP byte push, rehydrate, and resume) is the
// LIVE cross-machine half — it needs two nodes and a fabric, so it is DEFERRED to its
// hardware follow-on (the same split remote_dram.go takes for #4306 / #5066: the modeled
// cost is the CPU-witnessable half, the 2-node measurement is the private-lab half). This
// file moves no bytes and opens no socket; it reuses the SAME stageNanos / recomputeNanos
// cost model the placement policy (placement.go) weighs demote-vs-evict with, so the
// teleport-vs-recompute verdict is exactly the cost the rest of the plane already speaks.
//
// The core comparison mirrors RetainCheaperThanRecompute: teleporting a warm span wins
// precisely when the byte-copy over the link is cheaper than the full re-prefill it
// avoids AND it can complete inside the re-route's deadline. Fail-closed: if the link is
// too slow to finish the transfer within the deadline (or is unprofiled), the verdict is
// recompute-from-scratch — never a teleport that blows the budget and strands the request.
// It is deterministic and wall-clock-free: every quantity is a pure function of the
// injected request (the deadline budget is an input, not a clock read), so a replay is
// reproducible from the request alone.

// KVTeleportOutcome is the verdict of the re-route migration policy: move the warm KV, or
// fall back to recomputing the prefix from scratch on the target.
type KVTeleportOutcome string

const (
	// TeleportKV — push the serialized KV span A->B over the transport plane and resume
	// on the warm prefix. Chosen when the modeled transfer beats recompute and fits the
	// re-route deadline.
	TeleportKV KVTeleportOutcome = "teleport"
	// RecomputePrefix — drop the warm KV and re-prefill the whole prefix on the target.
	// The fail-closed fallback: taken when there is nothing warm worth moving, when
	// recompute is cheaper than the transfer, or when the transfer cannot finish inside
	// the deadline (the issue's "recompute-from-scratch fallback if teleport can't
	// complete within a deadline").
	RecomputePrefix KVTeleportOutcome = "recompute"
)

// KVTeleportRequest is the field-only input to the re-route migration decision. It
// describes the warm span being considered for a hop (its serialized size and its prefix
// length in positions), the transport LINK it would cross (a TierProfile, so the same
// stageNanos model quantifies first-byte latency + bytes/bandwidth the way it does for
// any other tier move), the target node's per-token prefill cost (what a recompute would
// pay), and the re-route DEADLINE budget the teleport must finish inside. It carries no
// bytes and no clock — the deadline is an injected budget, not a wall-clock read.
type KVTeleportRequest struct {
	// SpanBytes is the size of the serialized KV span (the SerializeSpan blob) the
	// teleport would push over the link. Non-positive means there is nothing warm to
	// move, so the verdict is recompute.
	SpanBytes int64
	// Tokens is the prefix length in positions — exactly what a recompute-from-scratch
	// would have to re-prefill on the target. Non-positive means recompute is free, so a
	// teleport can never pay.
	Tokens int64
	// Link is the physical profile of the transport hop A->B (its bandwidth and first-byte
	// latency). Reusing TierProfile lets the shared stageNanos model the transfer time; a
	// zero-bandwidth/unprofiled link yields a large sentinel, so an unknown link never
	// looks cheap and fails closed to recompute.
	Link TierProfile
	// PerTokenPrefillNanos is the target node's cost to re-prefill one token — the quantity
	// the teleport is weighed against (tokens x per-token prefill == recompute cost).
	PerTokenPrefillNanos int64
	// DeadlineNanos is the re-route budget: the teleport MUST finish within it or the
	// verdict fails closed to recompute. Non-positive means the re-route imposed no
	// deadline (an explicit migrate with no time bound), so the transfer time is not
	// gated on a budget — only the cost comparison decides.
	DeadlineNanos int64
}

// KVTeleportVerdict is the verdict plus the two modeled costs it compared and whether the
// teleport fit the deadline, so an observing sink can see WHY the re-route chose what it
// did without re-deriving the model.
type KVTeleportVerdict struct {
	Outcome KVTeleportOutcome
	// TeleportNanos is the modeled serialize+transfer time of pushing SpanBytes over the
	// Link (stageNanos: first-byte latency + bytes/bandwidth). A large sentinel when the
	// link is unprofiled.
	TeleportNanos int64
	// RecomputeNanos is the modeled cost of re-prefilling the whole prefix on the target
	// (tokens x per-token prefill).
	RecomputeNanos int64
	// WithinDeadline reports whether the modeled teleport finishes inside DeadlineNanos.
	// Always true when no deadline was imposed (DeadlineNanos <= 0).
	WithinDeadline bool
	// Reason is a stable, metric-readable tag for the verdict.
	Reason string
}

// Teleported reports whether the verdict chose to move the warm KV rather than recompute.
func (v KVTeleportVerdict) Teleported() bool { return v.Outcome == TeleportKV }

// ResolveKVTeleport decides whether a re-routed session should teleport its warm KV span
// to the target node or recompute the prefix from scratch. It is pure, deterministic, and
// hardware-free: every quantity is a function of the injected request. The order:
//
//  1. Nothing warm to move (SpanBytes <= 0) or nothing to save (Tokens <= 0) -> recompute:
//     a teleport that carries no warm prefix can never beat re-prefilling.
//  2. The transfer cannot finish inside the re-route deadline -> recompute (fail-closed):
//     the issue's explicit "recompute-from-scratch fallback if teleport can't complete
//     within a deadline". An unprofiled link (sentinel transfer time) lands here too.
//  3. The transfer is not cheaper than the recompute it avoids -> recompute: a tiny warm
//     prefix is not worth a byte-copy over the wire (the same retain-vs-recompute bar
//     PlanPlacement uses for demote-vs-evict).
//  4. Otherwise -> teleport: the byte-copy survives the warm prefix for less than the
//     re-prefill it saves, inside budget.
func ResolveKVTeleport(req KVTeleportRequest) KVTeleportVerdict {
	teleportNanos := stageNanos(req.SpanBytes, req.Link)
	recompute := recomputeNanos(req.Tokens, req.PerTokenPrefillNanos)
	within := req.DeadlineNanos <= 0 || teleportNanos <= req.DeadlineNanos

	v := KVTeleportVerdict{
		TeleportNanos:  teleportNanos,
		RecomputeNanos: recompute,
		WithinDeadline: within,
	}

	if req.SpanBytes <= 0 || req.Tokens <= 0 {
		v.Outcome = RecomputePrefix
		v.Reason = "no_warm_span"
		return v
	}
	if !within {
		// Fail-closed: the transfer would blow the re-route budget (or the link is
		// unprofiled), so re-prefill from scratch rather than strand the request.
		v.Outcome = RecomputePrefix
		v.Reason = "teleport_exceeds_deadline"
		return v
	}
	if teleportNanos >= recompute {
		// The warm prefix is too cheap to rebuild to be worth moving over the link.
		v.Outcome = RecomputePrefix
		v.Reason = "recompute_cheaper_than_teleport"
		return v
	}
	v.Outcome = TeleportKV
	v.Reason = "teleport_beats_recompute"
	return v
}
