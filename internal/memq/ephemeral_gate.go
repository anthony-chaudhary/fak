package memq

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// ErrEphemeralRefused is returned when a durable-targeted Add/AddPromoted is refused
// because the body is a situational (timestamp/current-step/mood-bound) observation
// with no explicit reclassification — issue #1592's write-time gate. memq core
// otherwise avoids importing a mechanism package (memq.go's own comment: "it does not
// import the mechanism packages"), but codexbackend.go already imports ctxmmu for
// ScreenBytes at read time; this is the same precedent applied at the write/promotion
// boundary, and internal/architest's tier ladder allows it (ctxmmu is tier 2, memq is
// tier 3 — a downward import).
var ErrEphemeralRefused = errors.New("memq: situational observation refused for durable memory; expire by default, promote only with explicit reclassification")

// EphemeralReclassification is memq's plain-string mirror of
// ctxmmu.Reclassification — memq.go's existing pattern for the durability/consent
// vocabularies (plain strings, normalized fail-closed) rather than importing the
// mechanism package's enum into the public API. NormReclassification maps between
// the two.
const (
	ReclassifyNone               = "none"
	ReclassifyExplicitConsent    = "explicit_consent"
	ReclassifyUserConfirmed      = "user_confirmed"
	ReclassifyEstablishedPattern = "established_pattern"
)

// NormReclassification maps any reclassification string to the canonical vocabulary,
// failing closed to ReclassifyNone for a missing/unrecognized value — same posture as
// NormConsent/NormDurability: an unrecognized override is treated as no override.
func NormReclassification(s string) string {
	switch s {
	case ReclassifyNone, ReclassifyExplicitConsent, ReclassifyUserConfirmed, ReclassifyEstablishedPattern:
		return s
	default:
		return ReclassifyNone
	}
}

// toCtxmmuReclassification converts memq's normalized string vocabulary to
// ctxmmu.Reclassification for the actual gate call.
func toCtxmmuReclassification(s string) ctxmmu.Reclassification {
	switch NormReclassification(s) {
	case ReclassifyExplicitConsent:
		return ctxmmu.ReclassifyExplicitConsent
	case ReclassifyUserConfirmed:
		return ctxmmu.ReclassifyUserConfirmed
	case ReclassifyEstablishedPattern:
		return ctxmmu.ReclassifyEstablishedPattern
	default:
		return ctxmmu.ReclassifyNone
	}
}

// GateEphemeralPromotion is the pre-check callers are expected to run before promoting
// raw observation text into durable memory (#1592, done condition item 3): it wraps
// ctxmmu.GateEphemeral so a memq caller need not import ctxmmu directly for the common
// case. It returns the underlying ctxmmu outcome unchanged (Allowed/Situational/Reason)
// so a caller can audit WHY a promotion was refused or allowed.
func GateEphemeralPromotion(text string, reclass string) ctxmmu.EphemeralGateOutcome {
	return ctxmmu.GateEphemeral(text, toCtxmmuReclassification(reclass))
}

// AddIfDurable is Add's ephemeral-gated counterpart: it runs GateEphemeralPromotion
// over body's text BEFORE calling Add, and refuses (ErrEphemeralRefused, zero Cell)
// a durability-targeting write whose body is a situational observation with no
// explicit reclassification. A caller that already knows its durability target is
// turn/session-class (pure context, never eligible for promotion per
// CONTEXT-IS-NOT-MEMORY.md) should keep calling Add/AddPromoted directly — this gate
// only needs to run on the path that targets DurabilityBounded/DurabilityDurable,
// mirroring PromotionLedger.Record's own "only non-turn crosses the boundary" posture.
func (m *MemStore) AddIfDurable(role, kind, durability string, body []byte, sealed bool, reclass string) (Cell, error) {
	return m.AddPromotedIfDurable(role, kind, durability, body, sealed, PromotionMeta{}, reclass)
}

// AddPromotedIfDurable is AddPromoted's ephemeral-gated counterpart (#1592): for a
// write whose NORMALIZED durability target is DurabilityBounded or DurabilityDurable,
// it runs the write-time ephemeral gate over body's text first. A situational
// observation (timestamp/current-step/mood-bound, or any other non-durable-shaped
// text) is refused with ErrEphemeralRefused UNLESS reclass names an explicit
// reclassification (ReclassifyExplicitConsent/ReclassifyUserConfirmed/
// ReclassifyEstablishedPattern) — "unless explicitly reclassified" per the issue's
// done condition. A turn/session-targeted write is not gated at all: it was never
// eligible for promotion in the first place (CONTEXT-IS-NOT-MEMORY.md's own decision
// tree), so gating it would refuse ordinary context writes that were never headed for
// durable memory.
func (m *MemStore) AddPromotedIfDurable(role, kind, durability string, body []byte, sealed bool, meta PromotionMeta, reclass string) (Cell, error) {
	target := NormDurability(durability)
	if (target == DurabilityBounded || target == DurabilityDurable) && !sealed {
		outcome := GateEphemeralPromotion(string(body), reclass)
		if !outcome.Allowed {
			return Cell{}, fmt.Errorf("%w: %s (situational=%s)", ErrEphemeralRefused, outcome.Reason, outcome.Situational)
		}
	}
	return m.AddPromoted(role, kind, durability, body, sealed, meta), nil
}
