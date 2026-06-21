package gateway

import (
	"testing"
	"time"
)

// TestDurEnv pins the env-override timeout seam the dogfood launchers rely on: a
// slow local model needs a WriteTimeout far above the network-safe default, and an
// explicit 0 must select Go's "no timeout" semantics — while garbage and negative
// values fall back to the conservative default rather than silently disabling it.
func TestDurEnv(t *testing.T) {
	const name = "FAK_HTTP_WRITE_TIMEOUT_S"
	def := 90 * time.Second
	cases := []struct {
		set  bool
		val  string
		want time.Duration
	}{
		{false, "", def},                 // unset -> default
		{true, "", def},                  // empty -> default
		{true, "600", 600 * time.Second}, // raise for a slow CPU model
		{true, "0", 0},                   // explicit opt-out (no timeout)
		{true, "-5", def},                // negative rejected -> default
		{true, "abc", def},               // unparseable -> default
	}
	for _, c := range cases {
		if c.set {
			t.Setenv(name, c.val)
		}
		if got := durEnv(name, def); got != c.want {
			t.Errorf("durEnv(%q=%q) = %v, want %v", name, c.val, got, c.want)
		}
	}
}
