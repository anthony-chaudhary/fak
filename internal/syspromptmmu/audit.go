package syspromptmmu

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/promptmmu"
)

// audit.go — Rung 6 of the system-prompt MMU (#1264, epic #1258): the observability
// witness. It re-derives the realized wire prefix from a request body and PROVES it
// equals the planned spine — divergence is an ALARM (an accidental head mutation caught
// before it costs a cache miss), not a blended metric.
//
// This is the read-only observability counterpart to SpliceSystemOverlay's internal
// bytes.Equal guard: the splice refuses to SHIP a body whose spine drifted; this audits
// ANY body after the fact and says, loudly, whether its realized spine still equals the
// plan. The check is at the content-WITNESS level (re-derive WitnessFor(block.text) and
// compare to the plan's witness), so it is immune to JSON framing/whitespace — a content
// witness is a sha256 over the block bytes, so witness-equal ⟺ spine-content byte-equal.
//
// It consumes the context-safety doctrine (#1217): a self-cross-checking roll-up where
// divergence is an error. It does NOT mint parallel numbers — it re-derives and compares.
//
// Tier: mechanism (2). Imports cachemeta(1) + promptmmu(1) + stdlib.

// Closed set of audit statuses.
const (
	// AuditOK: the body carries a fak-shaped base context and every resident block's
	// re-derived witness matches the plan — the realized prefix equals the planned spine.
	AuditOK = "ok"
	// AuditDiverged: the body is fak-shaped (breakpoint on the last resident block, at
	// least len(plan) blocks) but a resident block's content changed from the plan. THE
	// ALARM — an accidental head mutation.
	AuditDiverged = "spine-diverged"
	// AuditAbsent: the body carries no fak base context to audit (no system[] array, a
	// non-array system value, an empty array, no breakpoint, too few blocks, or the
	// breakpoint is misplaced — e.g. a harness-authored passthrough body). Neutral, NOT an
	// alarm: there is simply no fak spine present, and on today's passthrough traffic this
	// is the expected-large bucket.
	AuditAbsent = "no-fak-base-context"
	// AuditUnreadable: the body could not be READ — it is not a JSON object, or its
	// system[] value is array-shaped and still failed to decode (spans or text blocks).
	// Split out of AuditAbsent (#5442) because folding a structural failure into the
	// expected-large neutral bucket is precisely where a rise raises no suspicion: a
	// decoder regression would have shown up as fewer AuditOK, never as an error. Not the
	// spine alarm (nothing is known to have mutated) but not neutral either — a nonzero
	// count here is a malformed-input or fak-fault signal that is assertable on its own.
	AuditUnreadable = "unreadable-body"
)

// SegmentAudit is the per-resident-segment witness comparison.
type SegmentAudit struct {
	ExpectWitness string // the plan segment's content witness
	GotWitness    string // re-derived from the realized block's content
	Match         bool
}

// PrefixAudit is the Rung-6 re-derivation verdict for one wire body.
type PrefixAudit struct {
	// Present reports whether the body carries a fak-shaped base-context prefix at all.
	Present bool
	// Diverged is the ALARM: Present and at least one resident block's content changed.
	Diverged bool
	// Status is the closed-set verdict (AuditOK / AuditDiverged / AuditAbsent /
	// AuditUnreadable).
	Status string
	// ExpectDigest is the witness-chain digest of the plan (the expected spine).
	ExpectDigest string
	// GotDigest is the witness-chain digest re-derived from the realized prefix (empty
	// when Absent). GotDigest == ExpectDigest iff the realized spine equals the plan.
	GotDigest string
	// Segments is the per-resident-segment comparison (empty when Absent).
	Segments []SegmentAudit
	// BreakIdx is the realized system-block index the cache breakpoint sits on (-1 when
	// Absent).
	BreakIdx int
}

// witnessChainDigest hashes an ordered witness list, NUL-separated so no concatenation
// aliases another. It is the spine-unchanged proof at the chain level.
func witnessChainDigest(witnesses []string) string {
	h := sha256.New()
	for _, w := range witnesses {
		h.Write([]byte(w))
		h.Write([]byte{0})
	}
	return witnessPrefix + hex.EncodeToString(h.Sum(nil))
}

func planWitnesses(plan []cachemeta.PromptSegment) []string {
	out := make([]string, len(plan))
	for i, s := range plan {
		out[i] = s.Witness
	}
	return out
}

// AuditRealizedPrefix re-derives the realized system prefix from a wire body and proves
// it equals the planned spine. It decodes the system blocks, confirms the body is
// fak-shaped (the cache breakpoint sits on the last resident block and there are at least
// len(plan) blocks), then re-derives each resident block's content witness and compares
// to the plan. A content mismatch is a loud divergence (the alarm); a body with no
// fak-shaped base context is AuditAbsent (neutral); a body that could not be READ at all is
// AuditUnreadable, held apart from the neutral bucket so a decode failure is not counted as
// an ordinary passthrough. The overlay (blocks after the breakpoint) is intentionally NOT
// audited — it is the per-turn layer that is meant to change.
func AuditRealizedPrefix(raw []byte, plan []cachemeta.PromptSegment) PrefixAudit {
	a := PrefixAudit{BreakIdx: -1, ExpectDigest: witnessChainDigest(planWitnesses(plan))}
	if len(raw) == 0 || len(plan) == 0 {
		a.Status = AuditAbsent
		return a
	}
	// One decode, one reason. ArraySplicePointsWithReason names WHY no offsets came back,
	// and ArrayReasonIsStructural is the single registered partition of that closed set —
	// so "the body is malformed / its system[] would not decode" and "this body simply has
	// no fak base context" stop sharing one verdict here.
	breakIdx, _, _, reason := promptmmu.ArraySplicePointsWithReason(raw, "system")
	if reason != promptmmu.ArrayOffsetsResolved {
		a.Status = AuditAbsent
		if promptmmu.ArrayReasonIsStructural(reason) {
			a.Status = AuditUnreadable
		}
		return a
	}
	// Offsets resolved ⇒ raw IS a JSON object carrying a decodable, anchored system[].
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil {
		a.Status = AuditUnreadable
		return a
	}
	var blocks []textBlock
	if json.Unmarshal(obj["system"], &blocks) != nil {
		// STRUCTURAL: system[] yielded element spans but not text blocks. Same class as
		// promptmmu.ArrayUndecodable, so it carries the same verdict.
		a.Status = AuditUnreadable
		return a
	}
	// A fak-shaped base context has its breakpoint on the LAST resident block and at
	// least as many blocks as the plan. Anything else is not a body fak authored.
	if breakIdx != len(plan)-1 || len(blocks) < len(plan) {
		a.Status = AuditAbsent
		return a
	}

	a.Present = true
	a.BreakIdx = breakIdx
	got := make([]string, len(plan))
	for i := range plan {
		w := WitnessFor([]byte(blocks[i].Text))
		got[i] = w
		match := w == plan[i].Witness
		if !match {
			a.Diverged = true
		}
		a.Segments = append(a.Segments, SegmentAudit{
			ExpectWitness: plan[i].Witness,
			GotWitness:    w,
			Match:         match,
		})
	}
	a.GotDigest = witnessChainDigest(got)
	if a.Diverged {
		a.Status = AuditDiverged
	} else {
		a.Status = AuditOK
	}
	return a
}

// AuditBaseContext is the common case: audit a wire body against fak's own authored base
// context (BaseContextPlan). A live observability surface (#1264 / #1217) calls this per
// turn; AuditDiverged is the head-mutation alarm.
func AuditBaseContext(raw []byte) PrefixAudit {
	return AuditRealizedPrefix(raw, BaseContextPlan())
}
