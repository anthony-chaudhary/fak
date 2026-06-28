package numfmt

import (
	"strconv"
	"testing"
)

func TestItoa(t *testing.T) {
	cases := []uint64{0, 1, 9, 10, 99, 100, 12345, 1<<63 - 1, 1 << 63, ^uint64(0)}
	for _, n := range cases {
		if got, want := Itoa(n), strconv.FormatUint(n, 10); got != want {
			t.Errorf("Itoa(%d) = %q, want %q", n, got, want)
		}
	}
}

func TestEnvPositiveInt(t *testing.T) {
	const key = "FAK_NUMFMT_TEST_ENVPOSINT"
	for _, tc := range []struct {
		name string
		set  bool
		val  string
		def  int
		want int
	}{
		{"unset returns default", false, "", 42, 42},
		{"empty returns default", true, "", 7, 7},
		{"positive parses", true, "128", 8192, 128},
		{"zero falls to default", true, "0", 8192, 8192},
		{"negative falls to default", true, "-5", 8192, 8192},
		{"non-numeric falls to default", true, "abc", 99, 99},
		{"trailing garbage falls to default", true, "12x", 99, 99},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.set {
				t.Setenv(key, tc.val)
			} else {
				t.Setenv(key, "")
			}
			if got := EnvPositiveInt(key, tc.def); got != tc.want {
				t.Errorf("EnvPositiveInt(%q=%q, def=%d) = %d, want %d", key, tc.val, tc.def, got, tc.want)
			}
		})
	}
}
