package fleetaccounts

import "testing"

// TestAuthBlockKindUsageBeatsAccessWall pins the fix for the fleet-roster symptom of the overage
// bug: a blocker banner that carries BOTH the permanent org-disable wording (accessWallRE) AND a
// reset window must be classified "usage" (recoverable) — NOT "access" (a permanent wall that
// excludes the seat from dispatch and pages a human). A recovering cap must never read as a wall.
func TestAuthBlockKindUsageBeatsAccessWall(t *testing.T) {
	cases := []struct {
		name string
		text string
		want string
	}{
		{"pure usage limit", "You've reached your usage limit. Resets at 8pm.", "usage"},
		{"session limit with reset", "session limit reached, try again at 3:00", "usage"},
		{"weekly limit", "You've hit your weekly limit", "usage"},
		{"overage phrasing", "overage disabled for this organization", "usage"},
		// The load-bearing case: the SAME banner names the org-disable wording (which
		// accessWallRE matches) AND a reset — a recovering cap, not a permanent wall.
		{"org-disable text BUT carries a reset => usage, not access", "organization has disabled Claude subscription access; resets in 2 hours", "usage"},
		// A genuine permanent wall (no reset word) stays access.
		{"org-disable text, no reset => access", "organization has disabled Claude subscription access. Use an Anthropic API key instead.", "access"},
		{"credit", "Your credit balance is too low", "credit"},
		{"bare auth", "please run /login", "auth"},
		{"empty", "", "auth"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := authBlockKind(c.text); got != c.want {
				t.Fatalf("authBlockKind(%q) = %q, want %q", c.text, got, c.want)
			}
		})
	}
}

func TestAuthBlockReasonUsage(t *testing.T) {
	if got := authBlockReason("usage limit reached, resets at 5pm"); got != "usage/overage cap (recovers at reset)" {
		t.Fatalf("authBlockReason(usage) = %q", got)
	}
}
