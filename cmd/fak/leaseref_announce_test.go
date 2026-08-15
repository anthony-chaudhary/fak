package main

import (
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// TestRunLeaserefAnnounceDryRun exercises the pure edge path (no gh): --dry-run renders
// the comment body to stdout, and that body parses back to the announced record — the
// same round-trip witness, driven through the CLI surface.
func TestRunLeaserefAnnounceDryRun(t *testing.T) {
	var stdout, stderr strings.Builder
	code := runLeaserefAnnounce(&stdout, &stderr, []string{
		"--id", "foo", "--holder", "node/sess", "--generation", "3",
		"--tree", "internal/foo/**", "--ttl", "3600", "--action", "acquire", "--dry-run",
	})
	if code != 0 {
		t.Fatalf("dry-run exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	rec, ok := leaseref.ParseAnnounce(stdout.String())
	if !ok {
		t.Fatalf("dry-run body did not parse back:\n%s", stdout.String())
	}
	want := leaseref.AnnounceRecord{LeaseID: "foo", Holder: "node/sess", Generation: 3, Tree: []string{"internal/foo/**"}, TTLSeconds: 3600, Action: "acquire"}
	if !reflect.DeepEqual(rec, want) {
		t.Fatalf("dry-run parsed record mismatch\n want: %+v\n  got: %+v", want, rec)
	}
}

// TestRunLeaserefAnnounceUsageErrors confirms caller mistakes are usage errors (exit 2),
// distinct from the never-blocks post-failure path.
func TestRunLeaserefAnnounceUsageErrors(t *testing.T) {
	cases := [][]string{
		{"--holder", "h", "--action", "acquire", "--dry-run"},            // missing --id
		{"--id", "x", "--action", "acquire", "--dry-run"},                // missing --holder
		{"--id", "x", "--holder", "h", "--action", "steal", "--dry-run"}, // out-of-vocabulary action
		{"--id", "x", "--holder", "h", "--action", "acquire"},            // no --issue and not --dry-run
	}
	for _, argv := range cases {
		var stdout, stderr strings.Builder
		if code := runLeaserefAnnounce(&stdout, &stderr, argv); code != 2 {
			t.Errorf("argv %v: exit = %d, want 2 (usage); stderr=%s", argv, code, stderr.String())
		}
	}
}

// TestRunLeaserefAnnounceViewUsage confirms the fold verb rejects a missing --issue.
func TestRunLeaserefAnnounceViewUsage(t *testing.T) {
	var stdout, stderr strings.Builder
	if code := runLeaserefAnnounceView(&stdout, &stderr, nil); code != 2 {
		t.Fatalf("announce-view with no --issue: exit = %d, want 2", code)
	}
}

func TestRunLeaserefAnnounceScrubsProtectedHolder(t *testing.T) {
	cpu := "da" + "33"
	var out, errb strings.Builder
	code := runLeaserefAnnounce(&out, &errb, []string{"--id", "foo", "--holder", cpu + "/sess", "--generation", "3", "--tree", "internal/foo/**", "--ttl", "3600", "--action", "acquire", "--dry-run"})
	if code != 0 || !strings.Contains(out.String(), "CPU server/sess") || strings.Contains(strings.ToLower(out.String()), cpu) {
		t.Fatalf("body not scrubbed: code=%d body=%q stderr=%q", code, out.String(), errb.String())
	}
}
