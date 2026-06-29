package main

import "testing"

// TestCompressActivates pins the --compress activation rule: the flag turns the
// native compressor on ONLY when FAK_COMPRESSOR is unset, so an explicit env value
// (including `noop` to opt out) always wins.
func TestCompressActivates(t *testing.T) {
	cases := []struct {
		flag bool
		env  string
		want bool
	}{
		{true, "", true},        // flag on, no env -> activate
		{false, "", false},      // flag off -> never
		{true, "noop", false},   // explicit opt-out wins
		{true, "native", false}, // explicit env already set -> flag does not re-fill
		{true, "headroom", false},
		{true, "   ", true}, // whitespace-only env is treated as unset
		{false, "native", false},
	}
	for _, c := range cases {
		if got := compressActivates(c.flag, c.env); got != c.want {
			t.Errorf("compressActivates(%v, %q) = %v, want %v", c.flag, c.env, got, c.want)
		}
	}
}
