package refid

import "testing"

func TestValid(t *testing.T) {
	for _, tc := range []struct {
		id      string
		valid   bool
		session bool
	}{
		{id: "abc-123.x_y", valid: true, session: true},
		{id: "session-abc", valid: true},
		{id: "-bad"}, {id: ".bad"}, {id: "bad/name"}, {id: ""},
	} {
		if got := Valid(tc.id); got != tc.valid {
			t.Errorf("Valid(%q) = %v, want %v", tc.id, got, tc.valid)
		}
		if got := ValidSession(tc.id); got != tc.session {
			t.Errorf("ValidSession(%q) = %v, want %v", tc.id, got, tc.session)
		}
	}
}

// Seven fixed strings, with no Git subprocess; runtime estimate is unmeasured.
// fak-test:runtime fast est=1ms lane=default
func TestValidRejectsGitForbiddenComponents(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		id    string
		valid bool
	}{
		{"worker..id", false},
		{"worker.", false},
		{"worker.lock", false},
		{"worker.id", true},
		{"worker.LOCK", true},
		{"worker.locked", true},
		{"worker-id", true},
	} {
		if got := Valid(tc.id); got != tc.valid {
			t.Errorf("Valid(%q) = %v, want %v", tc.id, got, tc.valid)
		}
		if got := ValidSession(tc.id); got != tc.valid {
			t.Errorf("ValidSession(%q) = %v, want %v", tc.id, got, tc.valid)
		}
	}
}
