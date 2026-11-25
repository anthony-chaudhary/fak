package validate

import (
	"reflect"
	"testing"
)

func TestValidateSmokeUsesFakUpProductEntry(t *testing.T) {
	want := []string{"up", "--model", "3B", "--dry-run", "--json", "--memory-gib", "32"}
	if got := validateSmokeUpArgs(); !reflect.DeepEqual(got, want) {
		t.Fatalf("validateSmokeUpArgs() = %q, want %q", got, want)
	}
}

func TestValidateSmokeRejectsMockOrMissingPlan(t *testing.T) {
	for _, tc := range []struct {
		name string
		out  string
		want bool
	}{
		{name: "selected 3B", out: `{"schema":"fak-macfit-turnkey/1","tier":{"name":"3B"}}`, want: true},
		{name: "wrong tier", out: `{"schema":"fak-macfit-turnkey/1","tier":{"name":"27B"}}`, want: false},
		{name: "wrong schema", out: `{"schema":"other/1","tier":{"name":"3B"}}`, want: false},
		{name: "unrelated", out: "fak version 0.57.0", want: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := validateSmokeUpOutput(tc.out); got != tc.want {
				t.Fatalf("validateSmokeUpOutput(%q) = %v, want %v", tc.out, got, tc.want)
			}
		})
	}
}
