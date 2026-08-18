package mathx

import "testing"

func TestHexNibble(t *testing.T) {
	for _, tc := range []struct {
		in   byte
		want byte
		ok   bool
	}{{'0', 0, true}, {'9', 9, true}, {'a', 10, true}, {'F', 15, true}, {'g', 0, false}} {
		got, ok := HexNibble(tc.in)
		if got != tc.want || ok != tc.ok {
			t.Errorf("HexNibble(%q) = (%d,%v), want (%d,%v)", tc.in, got, ok, tc.want, tc.ok)
		}
	}
}
