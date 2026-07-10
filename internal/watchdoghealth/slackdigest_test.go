package watchdoghealth

import (
	"strings"
	"testing"
)

// TestSlackHealthDigest pins the Digest → channel-card fold across the closed status
// vocabulary: the post gate must track the digest's NeedsAttention bit exactly, the severity
// must refine that gate through the triage split (a fleet-clears NOTICE is not a human ALERT),
// and the body must lead with what a person owns.
func TestSlackHealthDigest(t *testing.T) {
	cases := []struct {
		name       string
		monitors   []Monitor
		wantSev    SlackSeverity
		wantPost   bool
		wantTitle  []string // substrings the title must contain
		wantBody   []string // substrings the body must contain
		absentBody []string // substrings the body must NOT contain
	}{
		{
			name:       "all healthy is a closed, silent OK card",
			monitors:   []Monitor{{ID: "mon-a", Installed: true, Alive: true}, {ID: "mon-b", Installed: true, Alive: true}},
			wantSev:    SlackOK,
			wantPost:   false,
			wantTitle:  []string{"OK", "HEALTHY", "2 HEALTHY"},
			wantBody:   []string{"all monitors healthy"},
			absentBody: []string{"needs you", "fleet is clearing"},
		},
		{
			name:       "empty digest folds to a no-monitors OK card",
			monitors:   nil,
			wantSev:    SlackOK,
			wantPost:   false,
			wantTitle:  []string{"OK", "no monitors"},
			wantBody:   []string{"all monitors healthy", "no monitors"},
			absentBody: []string{"needs you", "fleet is clearing"},
		},
		{
			name:       "a DOWN monitor posts but stays a fleet-clears NOTICE",
			monitors:   []Monitor{{ID: "mon-down", Installed: true}, {ID: "mon-ok", Installed: true, Alive: true}},
			wantSev:    SlackNotice,
			wantPost:   true,
			wantTitle:  []string{"NOTICE", "DOWN"},
			wantBody:   []string{"fleet is clearing", "mon-down"},
			absentBody: []string{"needs you"},
		},
		{
			name:       "a GAVE_UP monitor is a human ALERT that leads the body",
			monitors:   []Monitor{{ID: "mon-gaveup", Installed: true, Attempts: 5, MaxAttempts: 5}, {ID: "mon-ok", Installed: true, Alive: true}},
			wantSev:    SlackAlert,
			wantPost:   true,
			wantTitle:  []string{"ALERT", "GAVE_UP"},
			wantBody:   []string{"needs you", "mon-gaveup"},
			absentBody: []string{"fleet is clearing"},
		},
		{
			name:       "an auth-walled DOWN monitor escalates to a human ALERT",
			monitors:   []Monitor{{ID: "mon-auth", Installed: true, LastReason: "login required"}},
			wantSev:    SlackAlert,
			wantPost:   true,
			wantTitle:  []string{"ALERT"},
			wantBody:   []string{"needs you", "mon-auth"},
			absentBody: []string{"fleet is clearing"},
		},
		{
			name:      "mixed GAVE_UP + DOWN: ALERT with both buckets, human first",
			monitors:  []Monitor{{ID: "mon-gaveup", Installed: true, Attempts: 5, MaxAttempts: 5}, {ID: "mon-down", Installed: true}},
			wantSev:   SlackAlert,
			wantPost:  true,
			wantTitle: []string{"ALERT"},
			wantBody:  []string{"needs you", "mon-gaveup", "fleet is clearing", "mon-down"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := Fold(tc.monitors)
			got := SlackHealthDigest(d)

			if got.Severity != tc.wantSev {
				t.Errorf("severity = %q, want %q", got.Severity, tc.wantSev)
			}
			// The post gate must equal the digest's own NeedsAttention bit — the exact
			// condition `fak watchdog status --check` exits non-zero on — never drift from it.
			if got.ShouldPost != tc.wantPost {
				t.Errorf("ShouldPost = %v, want %v", got.ShouldPost, tc.wantPost)
			}
			if got.ShouldPost != d.NeedsAttention {
				t.Errorf("ShouldPost %v must track Digest.NeedsAttention %v", got.ShouldPost, d.NeedsAttention)
			}
			for _, sub := range tc.wantTitle {
				if !strings.Contains(got.Title, sub) {
					t.Errorf("title %q missing %q", got.Title, sub)
				}
			}
			for _, sub := range tc.wantBody {
				if !strings.Contains(got.Body, sub) {
					t.Errorf("body %q missing %q", got.Body, sub)
				}
			}
			for _, sub := range tc.absentBody {
				if strings.Contains(got.Body, sub) {
					t.Errorf("body %q must not contain %q", got.Body, sub)
				}
			}
			// The body is never empty — even an all-clear digest renders a sentence, so a
			// card edited from ALERT back to OK reads cleanly.
			if strings.TrimSpace(got.Body) == "" {
				t.Errorf("body must never be empty")
			}
			// When a person is owed, the "needs you" section must come before the fleet
			// section so the card leads with the human residual.
			if i, j := strings.Index(got.Body, "needs you"), strings.Index(got.Body, "fleet is clearing"); i >= 0 && j >= 0 && i > j {
				t.Errorf("needs-you section must precede fleet section in body %q", got.Body)
			}
		})
	}
}

// TestSlackHealthDigestDeterministic proves the fold is a pure function of the digest: the
// same monitors fold to a byte-identical card every time, so a coalescing outbox never sees
// spurious churn from re-rendering an unchanged health state.
func TestSlackHealthDigestDeterministic(t *testing.T) {
	monitors := []Monitor{
		{ID: "mon-gaveup", Installed: true, Attempts: 3, MaxAttempts: 3},
		{ID: "mon-down", Installed: true},
		{ID: "mon-ok", Installed: true, Alive: true},
		{ID: "mon-unknown", ProbeErr: true},
	}
	a := SlackHealthDigest(Fold(monitors))
	b := SlackHealthDigest(Fold(monitors))
	if a != b {
		t.Fatalf("fold is not deterministic:\n a=%#v\n b=%#v", a, b)
	}
}
