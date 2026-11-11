package policy

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// Omitted posture is the shipped default-open guard floor; fail_closed is the
// operator/strict lever, reachable by its own name and the default_deny /
// default-deny aliases.
func TestParsePostureOmittedAndAliases(t *testing.T) {
	cases := []struct {
		in   string
		want adjudicator.Posture
	}{
		{"", adjudicator.PostureDefaultOpen},
		{"default_open", adjudicator.PostureDefaultOpen},
		{"fail_closed", adjudicator.PostureFailClosed},
		{"default_deny", adjudicator.PostureFailClosed},
		{"default-deny", adjudicator.PostureFailClosed},
		{"admit_and_log", adjudicator.PostureAdmitAndLog},
	}
	for _, tc := range cases {
		got, err := ParsePosture(tc.in)
		if err != nil {
			t.Fatalf("ParsePosture(%q): unexpected error %v", tc.in, err)
		}
		if got != tc.want {
			t.Fatalf("ParsePosture(%q) = %v, want %v", tc.in, got, tc.want)
		}
	}

	if _, err := ParsePosture("bogus"); err == nil {
		t.Fatal("ParsePosture(\"bogus\") should fail loud")
	}
}
