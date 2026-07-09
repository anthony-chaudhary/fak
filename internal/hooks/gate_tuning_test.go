package hooks

import "testing"

// TestGateEnvInt pins the size-knob resolver: unset / garbage / non-positive all fall back to
// the (permissive) default, and only a clean positive integer overrides it. The floor-to-default
// contract is load-bearing — a 0 or negative ceiling must NOT silently disable a gate (that is
// what the explicit ALLOW_* escape hatch is for), so a fat-fingered value stays safe-permissive.
func TestGateEnvInt(t *testing.T) {
	const name = "FAK_TEST_TUNING_KNOB"
	cases := []struct {
		set  bool
		val  string
		want int
	}{
		{false, "", 2500},    // unset -> default
		{true, "", 2500},     // empty -> default
		{true, "   ", 2500},  // blank -> default
		{true, "3200", 3200}, // valid override
		{true, " 800 ", 800}, // trimmed
		{true, "0", 2500},    // zero -> default (never "off")
		{true, "-5", 2500},   // negative -> default
		{true, "abc", 2500},  // garbage -> default
		{true, "12x", 2500},  // partial parse fails -> default
	}
	for _, c := range cases {
		if c.set {
			t.Setenv(name, c.val)
		} else {
			t.Setenv(name, "") // ensure a clean baseline, then treat "" as unset
		}
		if got := gateEnvInt(name, 2500); got != c.want {
			t.Errorf("gateEnvInt(%q=%q) = %d, want %d", name, c.val, got, c.want)
		}
	}
}

// TestGateEnvNonNegInt pins the zero-allowing knob resolver used by the growth-slack setting:
// 0 is a meaningful value (strict ratchet) and must pass through, while negative / garbage /
// unset still fall back to the default.
func TestGateEnvNonNegInt(t *testing.T) {
	const name = "FAK_TEST_NONNEG_KNOB"
	cases := []struct {
		val  string
		want int
	}{
		{"", 20},     // unset/empty -> default
		{"0", 0},     // zero is meaningful -> passes through
		{"35", 35},   // valid override
		{" 5 ", 5},   // trimmed
		{"-1", 20},   // negative -> default
		{"nope", 20}, // garbage -> default
	}
	for _, c := range cases {
		t.Setenv(name, c.val)
		if got := gateEnvNonNegInt(name, 20); got != c.want {
			t.Errorf("gateEnvNonNegInt(%q=%q) = %d, want %d", name, c.val, got, c.want)
		}
	}
}

// TestGateEnvTruthy pins the escape-hatch parser: only the canonical truthy spellings enable an
// override; everything else (including unset and "0") leaves the gate enforcing.
func TestGateEnvTruthy(t *testing.T) {
	const name = "FAK_TEST_ESCAPE_HATCH"
	truthy := []string{"1", "true", "TRUE", "Yes", "on", " on "}
	for _, v := range truthy {
		t.Setenv(name, v)
		if !gateEnvTruthy(name) {
			t.Errorf("gateEnvTruthy(%q) = false, want true", v)
		}
	}
	falsy := []string{"", "0", "false", "no", "off", "maybe", "2"}
	for _, v := range falsy {
		t.Setenv(name, v)
		if gateEnvTruthy(name) {
			t.Errorf("gateEnvTruthy(%q) = true, want false", v)
		}
	}
}
