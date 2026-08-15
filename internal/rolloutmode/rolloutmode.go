// Package rolloutmode owns fak's ONE closed staged-rollout vocabulary: the
// off -> shadow -> canary -> on ladder a leaf climbs to land work in progress
// without changing default behavior (#6090).
//
// WHY A SHARED NOUN. Default-off / opt-in gating is the repo's dominant way of
// landing an unfinished capability — well over a hundred non-test Go files carry
// the prose convention. Before this package three leaves had each invented the
// ladder privately and spelled the APPLIED rung three different ways
// (dispatchtick `default`, promptmmu `on`, memorycotravel `live`), with three
// different defaults and two opposite unknown-value behaviors. Every new gated
// feature was free to pick a fourth spelling. The semantics worth keeping were
// already written out at length in internal/dispatchtick/rollout.go; this is
// that type, generalized.
//
// THE RUNGS, AND WHY EACH SPELLING WON:
//
//   - OFF — the live path is untouched: nothing computed, nothing applied. Spelled
//     `off` because all three leaves already agreed on it, and because it is the
//     rung that makes "default behavior is unchanged" PROVABLE rather than argued.
//   - SHADOW — compute the decision BESIDE the live one and report the delta, but
//     never let it take effect. All three leaves already agreed on `shadow`.
//   - CANARY — apply, but only inside a narrow declared scope, and roll back on a
//     regression. Only dispatchtick has this rung today; `canary` is the industry
//     spelling and no leaf spelled it otherwise. A leaf without a narrow scope
//     simply omits the rung (see ParseIn).
//   - ON — the applied rung: the decision changes live behavior for everything in
//     scope. `on` wins over `default` and `live` because it is the only one of the
//     three that names the RUNG rather than a claim about it. `default` describes
//     a leaf's chosen posture (and misleads, since a leaf's default may be `off`);
//     `live` describes an environment. `on` is also the rung already paired with
//     `off` and `shadow` in the two leaves that spelled it that way.
//
// FAIL-CLOSED IS PER-CALLER, NOT BAKED IN. Parse takes the caller's fallback
// rather than hardcoding a direction, because the correct direction genuinely
// differs by leaf: a routing guard must fail toward `off` (never route on a
// typo), while a curate that is deliberately on out of the box must fail toward
// `on` (never silently disable itself on a typo). Both are honest; what was NOT
// honest was each leaf re-deriving the vocabulary in order to express it. Parse
// always reports ok=false on an unrecognized value, so a caller can warn no
// matter which way it chose to fall.
//
// The parser does NOT normalize case or whitespace. The vocabulary is closed, so
// "ON" is not a rung; a caller that wants lenient input normalizes before parsing
// and keeps that leniency visible at its own seam.
package rolloutmode

// Mode is the closed vocabulary of rollout rungs. The zero value is the empty
// string, which is NOT a rung — it is Valid()==false, so a struct field left
// unset never silently reads as a rollout stage.
type Mode string

const (
	// Off is the untouched-live-path rung: nothing computed, nothing applied.
	Off Mode = "off"
	// Shadow computes the decision beside the live one and reports the delta. It
	// never changes live behavior — that is the rung's whole content.
	Shadow Mode = "shadow"
	// Canary applies the decision, but only within a leaf's narrow declared scope
	// and only until a regression rolls it back. A leaf with no narrow scope omits
	// this rung from its allowed set rather than redefining it.
	Canary Mode = "canary"
	// On is the applied rung: the decision changes live behavior for everything in
	// scope. Being spelled here does NOT make it reachable in any given leaf — a
	// leaf may name On and still refuse to operate at it (dispatchtick refuses
	// promotion past canary without a separate witness).
	On Mode = "on"
)

// ladder is the canonical, stable order: increasing exposure. A status surface
// enumerates in this order so two leaves never render the same ladder differently.
var ladder = [...]Mode{Off, Shadow, Canary, On}

// Modes returns the full ladder in canonical order. It returns a fresh slice, so
// a caller cannot mutate the shared vocabulary by writing into the result.
func Modes() []Mode {
	out := make([]Mode, len(ladder))
	copy(out, ladder[:])
	return out
}

// Valid reports whether m is one of the four rungs. Anything else — including the
// zero value and the retired `default` / `live` spellings — is not.
func (m Mode) Valid() bool { return In(m, ladder[:]) }

// String renders the rung's wire spelling.
func (m Mode) String() string { return string(m) }

// Computes reports whether the rung evaluates the gated decision at all. Off
// short-circuits; shadow, canary and on all compute. An unrecognized mode does
// not compute, so a bad value can never make a leaf do work it was not asked for.
func (m Mode) Computes() bool { return m == Shadow || m == Canary || m == On }

// Applies reports whether the rung's semantics are to CHANGE live behavior
// (canary within its scope, on everywhere in scope). This describes the RUNG, not
// a leaf's permission to be at it: a leaf may refuse to apply a rung Applies
// reports true for. An unrecognized mode never applies.
func (m Mode) Applies() bool { return m == Canary || m == On }

// In reports whether m is a member of allowed. Exported because a leaf that
// admits a subset of the ladder needs the same membership test its parse uses.
func In(m Mode, allowed []Mode) bool {
	for _, a := range allowed {
		if m == a {
			return true
		}
	}
	return false
}

// Parse resolves a raw string against the FULL ladder. An empty string means
// "unset" and resolves to fallback with ok=true; a recognized rung resolves to
// itself; anything else resolves to fallback with ok=false so the caller can warn.
// The caller supplies fallback because the safe direction is a leaf's decision,
// not the vocabulary's (see the package doc).
func Parse(s string, fallback Mode) (m Mode, ok bool) {
	return ParseIn(s, ladder[:], fallback)
}

// ParseIn is Parse restricted to the subset of rungs a leaf actually implements.
// A rung that is spelled correctly but outside allowed resolves to fallback with
// ok=false — the same answer as gibberish, because to that leaf it IS gibberish
// (a curate over a prompt body has no canary scope to canary into). fallback is
// returned verbatim and is not itself checked against allowed: it is the caller's
// declared posture, not user input.
func ParseIn(s string, allowed []Mode, fallback Mode) (m Mode, ok bool) {
	if s == "" {
		return fallback, true
	}
	cand := Mode(s)
	if In(cand, allowed) {
		return cand, true
	}
	return fallback, false
}
