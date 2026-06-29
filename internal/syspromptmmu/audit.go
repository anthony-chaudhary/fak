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
	// AuditAbsent: the body carries no fak base context to audit (no system[] array, no
	// breakpoint, too few blocks, or the breakpoint is misplaced — e.g. a harness-authored
	// passthrough body). Neutral, NOT an alarm: there is simply no fak spine present.
	AuditAbsent = "no-fak-base-context"
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
	// Status is the closed-set verdict (AuditOK / AuditDiverged / AuditAbsent).
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
// fak-shaped base context is AuditAbsent (neutral). The overlay (blocks after the
// breakpoint) is intentionally NOT audited — it is the per-turn layer that is meant to
// change.
func AuditRealizedPrefix(raw []byte, plan []cachemeta.PromptSegment) PrefixAudit {
	a := PrefixAudit{BreakIdx: -1, ExpectDigest: witnessChainDigest(planWitnesses(plan))}
	if len(raw) == 0 || len(plan) == 0 {
		a.Status = AuditAbsent
		return a
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil {
		a.Status = AuditAbsent
		return a
	}
	sysRaw, ok := obj["system"]
	if !ok {
		a.Status = AuditAbsent
		return a
	}
	breakIdx, _, _, anchored := promptmmu.ArraySplicePoints(raw, "system")
	if !anchored {
		a.Status = AuditAbsent
		return a
	}
	var blocks []textBlock
	if json.Unmarshal(sysRaw, &blocks) != nil {
		a.Status = AuditAbsent
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
