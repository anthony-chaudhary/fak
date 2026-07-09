package hooks

import (
	"os"
	"strconv"
	"strings"
)

// gate_tuning.go — the shared operator knobs for the SIZE gates (GOD_FILE_GROWTH,
// FILE_ADMISSION). These gates guard against *egregious* growth, not against every large
// file; the whole point is to red the build on a monolith forming, never on a legitimately
// large-but-bounded change. The ceilings that make that call are therefore operator-tunable
// (a fleet with a different comfort line moves it without a code change) and each gate keeps
// an escape hatch, exactly like the pre-commit gate set (hooks.go EscapeEnv) — a gate that
// occasionally over-refuses must have a witnessed way to say "admit this one" instead of
// forcing the author off-trunk.

// gateEnvInt reads a positive integer tuning knob from the environment. It returns def when
// the variable is unset, unparseable, or non-positive — a zero or negative ceiling would
// disable the gate silently, which is what the explicit escape hatch is for, so a garbage
// value falls back to the (permissive) default rather than to "off".
func gateEnvInt(name string, def int) int {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return def
	}
	return n
}

// gateEnvNonNegInt is gateEnvInt's sibling for knobs where zero is a MEANINGFUL setting
// (e.g. a growth slack of 0% restores the strict ratchet). It accepts any value >= 0 and
// falls back to def only on unset / unparseable / negative input.
func gateEnvNonNegInt(name string, def int) int {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 {
		return def
	}
	return n
}

// gateEnvTruthy reports whether an escape-hatch variable is set to a truthy value
// (1/true/yes/on, case-insensitive). Anything else — including unset — is false, so a gate
// stays enforcing unless an operator deliberately opts a run out.
func gateEnvTruthy(name string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}
