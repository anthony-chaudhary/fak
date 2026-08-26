//go:build !arm64 || (fakaccel && darwin && cgo)

package model

import "testing"

func TestEnvEnabled(t *testing.T) {
	for _, tc := range []struct {
		value string
		want  bool
	}{{"", false}, {"0", false}, {"off", false}, {"1", true}, {"on", true}, {"true", true}} {
		t.Run(tc.value, func(t *testing.T) {
			t.Setenv("FAK_TEST_ENV_ENABLED", tc.value)
			if got := envEnabled("FAK_TEST_ENV_ENABLED"); got != tc.want {
				t.Fatalf("envEnabled(%q) = %v, want %v", tc.value, got, tc.want)
			}
		})
	}
}
