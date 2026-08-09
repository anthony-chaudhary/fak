package agent

import "sort"

// compact_bail_vocab.go — the REGISTERED CompactReason* bail vocabulary, owned by the
// package that EMITS the reasons, plus the one thing a consumer cannot derive from the
// token itself: WHERE in the compactor the identity-return was decided.
//
// Two consumers were drifting from the const block above it, in opposite directions:
//
//   - internal/gateway's Prometheus HELP for fak_gateway_compaction_bail_reason_total
//     spelled the vocabulary out by hand and claimed it CLOSED. It listed 9 members while
//     this package emitted 13 (#5441). That is not a stale comment: a reader who trusts the
//     claim builds an alert over 9 label values and silently drops the rest — including
//     decode_failed, which #5387 split out of too_few_msgs precisely BECAUSE a structural
//     decode failure is a bug signal and a short conversation is not.
//   - The bail RATE both internal/gateway and internal/gatewayusageledger derive counted
//     every non-fire as a decline (#5388, #5443). The compactor is attempted on every
//     Anthropic passthrough once a budget is set, so every sub-three-message auxiliary ping
//     and every non-JSON body lands in that denominator; on mixed traffic they dominate and
//     the rate sits near 1.0 whether the compactor is perfectly healthy or completely
//     broken. A rate that always reads ~1.0 cannot be alerted on.
//
// Both cures need the same object: a vocabulary the emitter owns, that a consumer reads
// instead of re-typing, and that a test can hold the const block to. Registering a reason
// here is what makes adding one to the const block safe; TestCompactBailReasonsRegistered
// scans this package's own source and fails on any CompactReason* constant that never
// reached this map, so the drift that produced #5441 cannot recur silently.
//
// internal/gatewayusageledger keeps its own hand-spelled copy of the pre-eligibility half:
// it is a stdlib-shaped tier-1 leaf that cannot import this package, so its partition stays
// coupled by convention. internal/gateway CAN import here, and does.

// compactBailReasonPreEligible maps every registered CompactReason* identity-return token
// to whether the compactor decided it BEFORE any compactible span existed.
//
// The partition is by WHERE the decision happened, not by how benign the outcome was.
//
// true (the pre-eligibility half) is the group anchorCompactablePrefixMode returns before it
// has resolved an anchor — non_json, no_messages_key, decode_failed, too_few_msgs all return
// within 14 lines of each other, before a protected prefix or a compactible suffix exists.
// These requests were never in the running, so counting them as declines says nothing about
// compaction health.
//
// decode_failed is deliberately on that side even though it is a STRUCTURAL fault rather
// than a benign idle: it returns from the same check, so no candidate ever existed to
// decline. Nothing is hidden by the placement — it stays individually visible in
// fak_gateway_compaction_bail_reason_total, where a nonzero count is assertable as
// fak-fault on its own, exactly as prefix_mismatch is.
//
// false (the eligible half) is everything decided after that check, INCLUDING the late
// structural aborts (splice_failed, redecode_failed, prefix_mismatch, malformed_body):
// each one aborts a real candidate after the drop was already computed — a compaction that
// should have fired and did not — which is precisely what an alertable rate must keep
// measuring.
//
// CompactReasonNone is not a member: it is the FIRED outcome, not a bail.
var compactBailReasonPreEligible = map[string]bool{
	// Decided at the eligibility check — never a compaction candidate.
	CompactReasonNonJSON:      true,
	CompactReasonNoMsgsKey:    true,
	CompactReasonDecodeFailed: true,
	CompactReasonTooFewMsgs:   true,

	// Decided after it — a compactible span existed and was declined or aborted.
	CompactReasonUnderBudget:       false,
	CompactReasonNoBreakpoint:      false,
	CompactReasonCachedSpan:        false,
	CompactReasonWindowNoDrop:      false,
	CompactReasonBurstUnprofitable: false,
	CompactReasonSpliceFailed:      false,
	CompactReasonRedecodeFail:      false,
	CompactReasonPrefixMismatch:    false,
	CompactReasonMalformedBody:     false,
	// The survival-class refusal (#2421) sits on the eligible side for the same reason the late
	// structural aborts do: a real compactible candidate existed and was DECLINED. It is in fact
	// the most alertable member of that half — a sustained nonzero rate means the configured
	// budget no longer clears the session's pinned floor, so the lever is refusing every turn
	// while the resident window keeps growing.
	CompactReasonPinEvictRefused: false,
}

// CompactBailReasons returns every REGISTERED CompactReason* bail token, sorted, so a
// consumer can enumerate the vocabulary instead of re-typing it. CompactReasonNone (the
// fired outcome) is not included — this is the identity-return set.
//
// It returns a fresh slice on every call: the registry is process-global and a caller that
// sorted or appended to a shared backing array would corrupt every later reader.
func CompactBailReasons() []string {
	out := make([]string, 0, len(compactBailReasonPreEligible))
	for r := range compactBailReasonPreEligible {
		out = append(out, r)
	}
	sort.Strings(out)
	return out
}

// CompactBailPreEligible reports whether reason names an identity-return the compactor
// decided BEFORE any compactible span existed — a request that was never a compaction
// candidate, and so must not sit in the denominator of a compaction-health rate.
//
// An UNREGISTERED reason reports false, i.e. it counts as a real candidate. That direction
// is deliberate and is the only safe one: a vocabulary member added upstream and not
// registered here leaves the derived rate conservatively HIGH — it can over-report a
// problem, never silently understate one. (The compiler cannot catch the omission because
// the tokens are plain string constants; TestCompactBailReasonsRegistered is what does.)
func CompactBailPreEligible(reason string) bool {
	return compactBailReasonPreEligible[reason]
}
