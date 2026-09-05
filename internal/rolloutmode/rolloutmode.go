// Package rolloutmode provides the closed staged-rollout ladder:
// off -> shadow -> canary -> on (#6090).
package rolloutmode

// Mode is the closed vocabulary of rollout rungs; the zero value is invalid.
type Mode string

const (
	// Off leaves the live path untouched with nothing computed or applied.
	Off Mode = "off"
	// Shadow computes decisions alongside live behavior without taking effect.
	Shadow Mode = "shadow"
	// Canary applies the decision within a narrow declared scope.
	Canary Mode = "canary"
	// On changes live behavior for all operations in scope.
	On Mode = "on"
)

var ladder = [...]Mode{Off, Shadow, Canary, On}

// Modes returns the canonical rollout ladder as a fresh slice copy.
func Modes() []Mode {
	out := make([]Mode, len(ladder))
	copy(out, ladder[:])
	return out
}

// Valid reports whether m is a member of the canonical rollout ladder.
func (m Mode) Valid() bool { return In(m, ladder[:]) }

// String returns the wire spelling of the rollout mode.
func (m Mode) String() string { return string(m) }

// Computes reports whether the mode evaluates the gated decision.
func (m Mode) Computes() bool { return m == Shadow || m == Canary || m == On }

// Applies reports whether the mode changes live behavior in its scope.
func (m Mode) Applies() bool { return m == Canary || m == On }

// In reports whether m is present in the allowed slice.
func In(m Mode, allowed []Mode) bool {
	for _, a := range allowed {
		if m == a {
			return true
		}
	}
	return false
}

// Parse resolves a string against the full ladder, defaulting to fallback.
func Parse(s string, fallback Mode) (m Mode, ok bool) {
	return ParseIn(s, ladder[:], fallback)
}

// ParseIn resolves a string against an allowed subset, defaulting to fallback.
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
