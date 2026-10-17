package compute

import (
	"os"
	"testing"
)

// fak-test:runtime fast est=5ms
// Default-on contract for both Go gates: unset and "1" enable, exactly "0"
// disables, and each family ignores the other's off switch.
func TestV41TokenPanelDefaultOnPolicy(t *testing.T) {
	gates := []struct {
		name    string
		env     string
		other   string
		enabled func() bool
	}{
		{"Q2", "FAK_V41_Q2K_TOKEN_PANEL", "FAK_V41_Q3K_TOKEN_PANEL", v41Q2KTokenPanelEnabled},
		{"Q3", "FAK_V41_Q3K_TOKEN_PANEL", "FAK_V41_Q2K_TOKEN_PANEL", v41Q3KTokenPanelEnabled},
	}
	for _, gate := range gates {
		for _, tc := range []struct {
			value string
			unset bool
			want  bool
		}{
			{unset: true, want: true},
			{value: "0", want: false},
			{value: "1", want: true},
		} {
			label := tc.value
			if tc.unset {
				label = "unset"
			}
			t.Run(gate.name+"/"+label, func(t *testing.T) {
				t.Setenv(gate.other, "0")
				t.Setenv(gate.env, tc.value)
				if tc.unset {
					if err := os.Unsetenv(gate.env); err != nil {
						t.Fatal(err)
					}
				}
				if got := gate.enabled(); got != tc.want {
					t.Fatalf("%s=%s: enabled=%v want %v", gate.env, label, got, tc.want)
				}
			})
		}
	}
}
