package dispatchtick

import "testing"

// TestIsCoreSourceLaneTree pins the predicate that biases the unattended wave toward fak's
// own guard-shippable core engineering: a lane is core only when EVERY glob is self-source
// (cmd/** or internal/**) AND none is the trust-critical referee (which a guarded worker
// could never ship), so the docs/tools buckets and the held referee set both read false.
func TestIsCoreSourceLaneTree(t *testing.T) {
	cases := []struct {
		name string
		tree []string
		want bool
	}{
		{"gateway leaf is core", []string{"internal/gateway/**"}, true},
		{"agent leaf is core", []string{"internal/agent/**"}, true},
		{"cmd shim is core", []string{"cmd/fak/**"}, true},
		{"module-prefixed core", []string{"fak/internal/compute/**"}, true},
		{"windows-authored core glob", []string{"internal\\gateway\\**"}, true},
		{"docs bucket is not core", []string{"docs/**"}, false},
		{"tools bucket is not core", []string{"tools/**"}, false},
		{"trust-critical kernel is not core", []string{"internal/kernel/**"}, false},
		{"trust-critical adjudicator is not core", []string{"internal/adjudicator/**"}, false},
		{"trust-critical file glob is not core", []string{"dos.toml"}, false},
		{"mixed core+docs is not core", []string{"internal/gateway/**", "docs/**"}, false},
		{"mixed core+trust-critical is not core", []string{"internal/gateway/**", "internal/abi/**"}, false},
		{"empty tree is not core", nil, false},
		{"blank glob is not core", []string{"   "}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsCoreSourceLaneTree(tc.tree); got != tc.want {
				t.Fatalf("IsCoreSourceLaneTree(%q) = %v, want %v", tc.tree, got, tc.want)
			}
		})
	}
}
