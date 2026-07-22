package main

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/wiprecon"
)

// TestBuildWipOrphanTicket proves a QUARANTINE orphan renders the correct idempotency
// key and a body carrying the marker, the disposition, the file set, and a next step.
func TestBuildWipOrphanTicket(t *testing.T) {
	cases := []struct {
		name     string
		session  string
		startSHA string
		files    []string
		reason   string
		wantKey  string
		wantSHA  string
	}{
		{
			name:     "full sha truncates to 12",
			session:  "sessA",
			startSHA: "0123456789abcdef0123",
			files:    []string{"cmd/fak/wip.go", "internal/wiprecon/recon.go"},
			reason:   "delta unlanded and does not apply cleanly — quarantined",
			wantKey:  "wip-orphan-sessA-0123456789ab",
			wantSHA:  "0123456789ab",
		},
		{
			name:     "short sha kept whole",
			session:  "s2",
			startSHA: "deadbeef",
			files:    []string{"a.go"},
			reason:   "",
			wantKey:  "wip-orphan-s2-deadbeef",
			wantSHA:  "deadbeef",
		},
		{
			name:     "missing sha degrades to unknown (still deterministic)",
			session:  "s3",
			startSHA: "",
			files:    nil,
			reason:   "",
			wantKey:  "wip-orphan-s3-unknown",
			wantSHA:  "unknown",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tk := buildWipOrphanTicket(tc.session, tc.startSHA, tc.files, tc.reason)
			if tk.Key != tc.wantKey {
				t.Fatalf("key = %q, want %q", tk.Key, tc.wantKey)
			}
			if tk.SHA12 != tc.wantSHA {
				t.Errorf("sha12 = %q, want %q", tk.SHA12, tc.wantSHA)
			}
			// The marker must embed the key verbatim so a later dedup can recover it.
			marker := wipOrphanTicketMarker(tc.wantKey)
			if !strings.Contains(tk.Body, marker) {
				t.Errorf("body missing key marker %q:\n%s", marker, tk.Body)
			}
			// Disposition + a concrete next step must be present.
			if !strings.Contains(tk.Body, "QUARANTINE") {
				t.Errorf("body missing QUARANTINE disposition:\n%s", tk.Body)
			}
			if !strings.Contains(tk.Body, "Next step:") {
				t.Errorf("body missing next-step line:\n%s", tk.Body)
			}
			if !strings.Contains(tk.Title, "QUARANTINE") || !strings.Contains(tk.Title, tc.session) {
				t.Errorf("title = %q, want it to name QUARANTINE + session", tk.Title)
			}
			// The attributed file set is enumerated when present.
			for _, f := range tc.files {
				if !strings.Contains(tk.Body, f) {
					t.Errorf("body missing attributed file %q:\n%s", f, tk.Body)
				}
			}
			if len(tc.files) == 0 && !strings.Contains(tk.Body, "file set unavailable") {
				t.Errorf("empty file set should note it is unavailable:\n%s", tk.Body)
			}
		})
	}
}

// fakeGH is a spy seam: it records find/create calls so a test can assert dedup.
type fakeGH struct {
	avail    bool
	existing map[string][]int // key -> issue numbers that already carry the marker
	created  []string         // titles created, in order
	nextNum  int
}

func (f *fakeGH) seam() wipTicketGH {
	return wipTicketGH{
		available: func() bool { return f.avail },
		find: func(_ context.Context, key string) ([]int, error) {
			return f.existing[key], nil
		},
		create: func(_ context.Context, title, _ string) (int, error) {
			f.created = append(f.created, title)
			f.nextNum++
			return f.nextNum, nil
		},
	}
}

// TestWipEmitDedupSuppressesSecondFile proves an already-tracked marker reuses the
// existing ticket and files no duplicate.
func TestWipEmitDedupSuppressesSecondFile(t *testing.T) {
	tk := buildWipOrphanTicket("sessDup", "cafebabecafe", []string{"x.go"}, "quarantined")
	gh := &fakeGH{avail: true, existing: map[string][]int{tk.Key: {77}}}

	var out, errb bytes.Buffer
	wipEmitOrphanTickets(context.Background(), &out, &errb, []wipOrphanTicket{tk}, false, gh.seam())

	if got := out.String(); !strings.Contains(got, "already tracked: #77") {
		t.Fatalf("want 'already tracked: #77', got:\n%s", got)
	}
	if len(gh.created) != 0 {
		t.Fatalf("dedup failed: filed %d duplicate ticket(s): %v", len(gh.created), gh.created)
	}
}

// TestWipEmitFilesWhenAbsent proves an orphan with no existing ticket is filed once.
func TestWipEmitFilesWhenAbsent(t *testing.T) {
	tk := buildWipOrphanTicket("sessNew", "0011223344556677", []string{"y.go"}, "quarantined")
	gh := &fakeGH{avail: true, existing: map[string][]int{}}

	var out, errb bytes.Buffer
	wipEmitOrphanTickets(context.Background(), &out, &errb, []wipOrphanTicket{tk}, false, gh.seam())

	if got := out.String(); !strings.Contains(got, "filed: #1") {
		t.Fatalf("want 'filed: #1', got:\n%s", got)
	}
	if len(gh.created) != 1 {
		t.Fatalf("want exactly one ticket filed, got %d: %v", len(gh.created), gh.created)
	}
}

// TestWipEmitDryRunPrintsAndFilesNothing proves --dry-run prints the exact ticket and
// makes NO gh call, and that an unavailable gh takes the same safe path.
func TestWipEmitDryRunPrintsAndFilesNothing(t *testing.T) {
	tk := buildWipOrphanTicket("sessDry", "abcdef012345", []string{"z.go"}, "quarantined")

	cases := []struct {
		name   string
		dryRun bool
		avail  bool
	}{
		{"explicit dry-run with gh available", true, true},
		{"gh unavailable falls back to print", false, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gh := &fakeGH{avail: tc.avail, existing: map[string][]int{}}
			var out, errb bytes.Buffer
			wipEmitOrphanTickets(context.Background(), &out, &errb, []wipOrphanTicket{tk}, tc.dryRun, gh.seam())

			got := out.String()
			if !strings.Contains(got, "would file ticket ["+tk.Key+"]") {
				t.Fatalf("want a would-file print for %s, got:\n%s", tk.Key, got)
			}
			if !strings.Contains(got, wipOrphanTicketMarker(tk.Key)) {
				t.Errorf("dry-run print should include the exact body/marker, got:\n%s", got)
			}
			if len(gh.created) != 0 {
				t.Fatalf("dry-run/offline must file nothing, filed: %v", gh.created)
			}
		})
	}
}

// TestWipCollectOnlyQuarantine proves the collect pass renders a ticket ONLY for a
// QUARANTINE decision (RECLAIM/DISCARD_WITNESSED/SKIP are untouched in v1) and resolves
// the real start-SHA + attributed file set from the live checkpoint.
func TestWipCollectOnlyQuarantine(t *testing.T) {
	ctx := context.Background()
	dir, file := wipTestRepo(t)

	if err := os.WriteFile(file, []byte("base line\norphan edit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := wipCheckpoint(ctx, dir, "sessQ", true, 100); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}

	decisions := []wiprecon.Decision{
		{Session: "sessQ", Action: wiprecon.ActQuarantine, Reason: "quarantined"},
		{Session: "sessR", Action: wiprecon.ActReclaim, Reason: "reclaimable"},
		{Session: "sessD", Action: wiprecon.ActDiscardWitnessed, Reason: "landed"},
		{Session: "sessL", Action: wiprecon.ActSkip, Reason: "live"},
	}
	tickets, err := wipCollectOrphanTickets(ctx, dir, decisions)
	if err != nil {
		t.Fatalf("collect: %v", err)
	}
	if len(tickets) != 1 {
		t.Fatalf("want exactly one ticket (only the QUARANTINE), got %d: %+v", len(tickets), tickets)
	}
	if tickets[0].Session != "sessQ" {
		t.Errorf("ticket session = %q, want sessQ", tickets[0].Session)
	}
	if tickets[0].SHA12 == "unknown" || tickets[0].SHA12 == "" {
		t.Errorf("collect failed to resolve the checkpoint start-SHA: %+v", tickets[0])
	}
	if !strings.Contains(tickets[0].Body, "note.txt") {
		t.Errorf("ticket body missing the attributed file note.txt:\n%s", tickets[0].Body)
	}
}

// TestWipParseIssueNumberFromURL covers the gh-output parse.
func TestWipParseIssueNumberFromURL(t *testing.T) {
	cases := []struct {
		in   string
		want int
		ok   bool
	}{
		{"https://github.com/o/r/issues/5337", 5337, true},
		{"https://github.com/o/r/issues/12/", 12, true},
		{"Creating issue\nhttps://github.com/o/r/issues/9", 9, true},
		{"not a url", 0, false},
		{"", 0, false},
	}
	for _, tc := range cases {
		got, err := wipParseIssueNumberFromURL(tc.in)
		if tc.ok && (err != nil || got != tc.want) {
			t.Errorf("parse(%q) = %d, %v; want %d, nil", tc.in, got, err, tc.want)
		}
		if !tc.ok && err == nil {
			t.Errorf("parse(%q) expected an error, got %d", tc.in, got)
		}
	}
}
