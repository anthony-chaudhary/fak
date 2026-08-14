package syspromptmmu

import (
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// steering.go — #3308 (epic #1258): cache-safe verbosity/terseness steering as one more
// after-breakpoint overlay segment. Borrowed (clean-room, Python→Go) from
// headroomlabs-ai/headroom's apply_verbosity_steering (Apache-2.0, pinned 38074888),
// which appends a leveled, sentinel-wrapped steering block AFTER the last system block so
// any earlier cache_control breakpoint — and with it the whole cached prefix — is
// untouched.
//
// This file is the PRODUCER only: steeringSegment maps a terseness level (1..4, mild →
// maximal; 0 = no-op) to a fixed canonical, sentinel-wrapped text emitted as the same
// segment shape SelectOverlay produces (SegMessage, no cache_control, content-derived
// Witness). It rides the existing Rung-2/Rung-3 path — BuildSystemValue for a fresh base,
// SpliceSystemOverlay for a per-turn swap — so it lands strictly past the breakpoint and
// never disturbs the cached resident spine+policy prefix (invariants 1+2).
//
// Byte-stable by construction: every level's text is a fixed literal and the wrapping is
// deterministic, so the same level always yields identical bytes — a cache-safe segment
// whose Witness (WitnessFor over the content) is stable per level. The sentinel names the
// block so a later turn can replace it in place through the overlay swap when the level
// changes; the same level re-applied is byte-identical (idempotent).
//
// Scope fence: advisory output-token steering only — opt-in, appended after the client's
// own system blocks, never an override of an explicit client instruction, and NOT wired
// to the live gateway here (the overlay path has no request-path caller today; the wiring
// is the follow-on rung).
//
// Tier: mechanism (2). Imports cachemeta(1) + stdlib only.

// SteeringEnvVar is the opt-in request-path knob (#5047): the env var the owned-loop
// builder reads to select a terseness level. Unset/empty/invalid ⇒ SteeringOff (steering
// stays off — the forward path does not move unless a level is deliberately configured).
const SteeringEnvVar = "FAK_STEERING_LEVEL"

// SteeringOff is the level-0 no-op sentinel: steering is opt-in, so the absence of a
// configured level (and any out-of-range level, which SteeringSegment refuses) reports as
// SteeringOff — no steering block is appended and the value is byte-identical to before.
const SteeringOff = 0

// SteeringSegment is the exported request-path entry to the steering producer: the
// agent-side owned-loop builder (#5047 wiring) calls it to append the sentinel-wrapped
// terseness segment for `level` strictly AFTER the cache breakpoint. It is steeringSegment
// (the #3308 internal producer) promoted to the package API — same bytes, same CLOSED
// 1..4 domain, same fail-safe no-op: SteeringOff and any out-of-range level return ok
// false and the zero segment, so an unknown level can never fabricate a steering block.
func SteeringSegment(level int) (cachemeta.PromptSegment, bool) {
	return steeringSegment(level)
}

// steeringSentinelOpen / steeringSentinelClose wrap every steering block so it is
// identifiable in a realized overlay: a later turn finds the sentinel and swaps the block
// in place (through the overlay splice, never an edit of the resident prefix). The
// sentinel is version-stamped like the spine/policy tiers (SpineVersion idiom): a text
// change is a deliberate version bump, and the content Witness detects drift regardless.
const (
	steeringSentinelOpen  = "<fak:steering v1>"
	steeringSentinelClose = "</fak:steering>"
)

// The leveled canonical texts, mild (1) → maximal (4) terseness. Fixed literals, never
// templated per turn, so each level's emitted segment is byte-identical every call.
const (
	steeringLevel1 = "Signal-first level 1 (light): lead with the answer. Use compact sentences and retain " +
		"full explanation where it helps the reader act. Include every safety caveat, constraint, citation, " +
		"command, code block, and explicitly requested section."
	steeringLevel2 = "Signal-first level 2 (focused): state the result first, then add only the context needed " +
		"to understand or act on it. Prefer concrete verbs and compact bullets. Include every safety caveat, " +
		"constraint, citation, command, code block, and explicitly requested section."
	steeringLevel3 = "Signal-first level 3 (compressed): deliver the essential result in short subject-first " +
		"lines. Use dense bullets when they improve scanning. Include every safety caveat, constraint, citation, " +
		"command, code block, and explicitly requested section."
	steeringLevel4 = "Signal-first level 4 (minimal): return the requested artifact or answer in its shortest " +
		"complete form. Keep exact syntax and all correctness-critical qualifications. Include every safety " +
		"caveat, constraint, citation, command, code block, and explicitly requested section."
)

// steeringText maps a level to its canonical text. The set is CLOSED: only 1..4 exist;
// 0 is the documented no-op and anything else is refused the same way (fail-safe — an
// unknown level must never fabricate a steering block).
func steeringText(level int) (string, bool) {
	switch level {
	case 1:
		return steeringLevel1, true
	case 2:
		return steeringLevel2, true
	case 3:
		return steeringLevel3, true
	case 4:
		return steeringLevel4, true
	default:
		return "", false
	}
}

// steeringSegment returns the sentinel-wrapped, byte-stable terseness segment for
// `level` (1..4), shaped exactly like a Rung-3 overlay segment (SegMessage tail content,
// no cache_control, Witness = content blob hash) so the caller hands it to
// BuildSystemValue / SpliceSystemOverlay alongside the queried capability overlay and it
// lands strictly AFTER the cache breakpoint. Level 0 is the no-op (steering off — the
// opt-in default): ok is false and the zero segment must not be appended. Deterministic:
// the same level always produces identical bytes, so re-applying a level is idempotent
// and can never bust the cached prefix.
func steeringSegment(level int) (cachemeta.PromptSegment, bool) {
	text, ok := steeringText(level)
	if !ok {
		return cachemeta.PromptSegment{}, false
	}
	content := []byte(steeringSentinelOpen + "\n" + text + "\n" + steeringSentinelClose)
	return cachemeta.PromptSegment{
		Kind:    cachemeta.SegMessage, // tail content, appended after the cached prefix
		Tokens:  estTokens(content),
		Content: content,
		Witness: WitnessFor(content), // content blob hash — drift-detectable per level
	}, true
}
