package toolproc

// repeatreason.go — the REGISTERED verdict vocabulary of the reuse decision
// (#5119, the last DoD rung of #4764). repeatreuse.go already owns the leaf's
// own CLOSED token set: ReuseReason is what a Receipt cites, and it stays a
// plain string enum so the fold remains pure, init-free, and free of any
// registry dependency at decision time. This file is the BRIDGE between that
// leaf token and the kernel's registered refusal vocabulary.
//
// WHY A BRIDGE AND NOT A REWRITE. A live seam that SERVES a repeat has to be
// able to say why, in the same vocabulary every other kernel verdict uses —
// `abi.ReasonName(code)` must round-trip to a stable token, never free text.
// But the classifier must keep working with no registry at all (the offline
// `fak toolproc repeats` path folds rollout logs in a process where nothing is
// registered). Binding the two as DATA satisfies both: the leaf keeps citing
// ReuseReason, and a consumer that needs kernel-legible verdicts maps them.
//
// WHO REGISTERS. Not this leaf. ReuseReasonPairs() is data; the CONSUMER hands
// it to abi.RegisterReason, exactly as ReasonPairs() is registered by its
// consumers and never by this package's init. That is what lets
// internal/toolproc stay an init-free fold with no defconfig entry.
//
// AND THE CONSUMER IS internal/toolprocgate/reusearm.go (#5407). This file
// shipped as a FOUNDATION with nothing in the tree calling either exported
// function; that posture is now retired, and the promotion evidence it named is
// exactly what landed. ReuseArm.Serve answers a repeat from the armed store and
// cites the code the receipt maps to, rendering the stable token on
// abi.Result.Meta["reuse_reason"] — beside kill_reason, the surface an operator
// already reads for a quarantined completion, so a served repeat is
// distinguishable from a fresh fetch without a second reporting channel. The
// pairs register in that leaf's own init: one site, run once, which is what
// makes a second registration under a different name unreachable rather than
// merely discouraged. The fail-closed half is honoured at the call site and not
// only documented here: a reason outside the closed set names NOTHING and also
// serves nothing, so bytes no verdict blessed never answer a call.
//
// Registration still does not belong to THIS leaf's init. The offline
// `fak toolproc repeats` fold runs in a process where nothing is registered and
// stays init-free, exactly as before — arming a serving seam changed the
// consumer, not the classifier.
//
// THE BLOCK. The process-table family holds 1040–1044 (toolproc.go's four
// supervision verdicts plus monitor.go's coverage verdict). The reuse verdicts
// are a DISTINCT family with a distinct consumer and a distinct lifetime, so
// they take their own contiguous block at 1090–1095 instead of consuming the
// process-table block's headroom. Every code sits above abi.ReasonCoreMax
// (1023) — the sanctioned out-of-tree range, not a core edit.

import "github.com/anthony-chaudhary/fak/internal/abi"

// The out-of-tree refusal codes a reuse decision cites. One per ReuseReason:
// the map below is total, so a receipt can always be rendered as a registered
// verdict and a new ReuseReason cannot silently fall back to free text.
const (
	ReasonReuseKeyedHit      abi.ReasonCode = 1090
	ReasonReuseFreshnessHit  abi.ReasonCode = 1091
	ReasonReuseFirstFetch    abi.ReasonCode = 1092
	ReasonReuseDigestChanged abi.ReasonCode = 1093
	ReasonReuseWindowExpired abi.ReasonCode = 1094
	ReasonReuseNeverReused   abi.ReasonCode = 1095
)

// The stable names for the codes above — the closed verdict vocabulary a reuse
// hit (or a reuse refusal) cites on the wire.
const (
	ReasonReuseKeyedHitName      = "REUSE_KEYED_HIT"
	ReasonReuseFreshnessHitName  = "REUSE_FRESHNESS_HIT"
	ReasonReuseFirstFetchName    = "REUSE_FIRST_FETCH"
	ReasonReuseDigestChangedName = "REUSE_DIGEST_CHANGED"
	ReasonReuseWindowExpiredName = "REUSE_WINDOW_EXPIRED"
	ReasonReuseNeverReusedName   = "REUSE_NEVER_REUSED"
)

// reuseReasonCodes binds each leaf token to its registered code. Kept as the
// single source of truth so ReuseReasonPairs and ReuseReasonCode cannot drift.
var reuseReasonCodes = map[ReuseReason]abi.ReasonCode{
	ReasonKeyedHit:      ReasonReuseKeyedHit,
	ReasonFreshnessHit:  ReasonReuseFreshnessHit,
	ReasonFirstFetch:    ReasonReuseFirstFetch,
	ReasonDigestChanged: ReasonReuseDigestChanged,
	ReasonWindowExpired: ReasonReuseWindowExpired,
	ReasonNeverReused:   ReasonReuseNeverReused,
}

// ReuseReasonPairs lists the reuse vocabulary for a consumer to register with
// abi.RegisterReason. It is deliberately SEPARATE from ReasonPairs(): the
// process-table verdicts are registered by every toolproc consumer (including
// the offline CLI), while these are only meaningful where a repeat is actually
// served — the armed live seam. A consumer that serves nothing registers
// nothing. The order is the decision order (hits, then the three refusals).
func ReuseReasonPairs() []ReasonPair {
	return []ReasonPair{
		{ReasonReuseKeyedHit, ReasonReuseKeyedHitName},
		{ReasonReuseFreshnessHit, ReasonReuseFreshnessHitName},
		{ReasonReuseFirstFetch, ReasonReuseFirstFetchName},
		{ReasonReuseDigestChanged, ReasonReuseDigestChangedName},
		{ReasonReuseWindowExpired, ReasonReuseWindowExpiredName},
		{ReasonReuseNeverReused, ReasonReuseNeverReusedName},
	}
}

// ReuseReasonCode maps one Receipt's ReuseReason to its registered code. ok is
// false only for a token outside the closed set (an empty Receipt.Reason, or a
// value a caller invented) — a consumer treats that as "no registered verdict"
// and MUST NOT fabricate one.
func ReuseReasonCode(r ReuseReason) (abi.ReasonCode, bool) {
	c, ok := reuseReasonCodes[r]
	return c, ok
}
