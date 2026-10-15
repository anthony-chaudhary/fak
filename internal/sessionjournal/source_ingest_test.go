package sessionjournal

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestAppendDeniedSourceLeavesOnlyAuditableRefusal is the #5910 at-rest
// witness. A session rooted in a lab-access source must be refused before its
// metadata is marshalled: the source and every other captured byte stay out of
// the journal, while one content-free, closed-reason refusal remains auditable.
func TestAppendDeniedSourceLeavesOnlyAuditableRefusal(t *testing.T) {
	journalPath := filepath.Join(t.TempDir(), "session-journal.jsonl")
	deniedSource := filepath.Join(t.TempDir(), "fak-private", "LAB-ACCESS-SOURCE-5910")
	deniedPayload := "DENIED-SESSION-PAYLOAD-5910"

	err := Append(journalPath, Event{
		Kind:  KindOpen,
		ID:    "session-5910",
		TS:    "2026-08-08T00:00:00Z",
		CWD:   deniedSource,
		Model: deniedPayload,
		Argv:  []string{deniedPayload},
	})
	if !errors.Is(err, ErrSourceDenied) {
		t.Fatalf("Append denied source error = %v, want ErrSourceDenied", err)
	}

	stored, readErr := os.ReadFile(journalPath)
	if readErr != nil {
		t.Fatalf("ReadFile: %v", readErr)
	}
	for _, forbidden := range [][]byte{[]byte(deniedSource), []byte(deniedPayload)} {
		if bytes.Contains(stored, forbidden) {
			t.Fatalf("denied bytes reached the journal at rest: %q", stored)
		}
	}

	events := ParseEvents(string(stored))
	if len(events) != 1 {
		t.Fatalf("auditable rows = %d, want exactly one refusal: %s", len(events), stored)
	}
	if events[0].Kind != KindRefuse || events[0].Reason != "SECRET_EXFIL" {
		t.Fatalf("refusal = kind %q reason %q, want %q/SECRET_EXFIL", events[0].Kind, events[0].Reason, KindRefuse)
	}
	if events[0].CWD != "" || events[0].Model != "" || len(events[0].Argv) != 0 {
		t.Fatalf("refusal retained denied event fields: %+v", events[0])
	}
	if sessions := FoldEvents(events); len(sessions) != 0 {
		t.Fatalf("a refusal became a resumable session: %+v", sessions)
	}
}

// TestSessionJournalRefusalProjectionIsBoundedAndContentFree is the #13865 witness at the
// package level: the refusal projection surfaces a policy refusal (so an operator can tell
// it from an absent producer), stays content-free (no denied source, model, or argv), honors
// exact-session-id filtering and a finite limit, and never leaks into lifecycle classification.
func TestSessionJournalRefusalProjectionIsBoundedAndContentFree(t *testing.T) {
	denied := filepath.Join(t.TempDir(), "fak-private", "LAB")
	payload := "DENIED-13865-PAYLOAD"

	// Drive three real denials (two for session A, one for B) through Append, then read the
	// journal back so the projection is exercised over exactly what containment persisted.
	journalPath := filepath.Join(t.TempDir(), "j.jsonl")
	for i, id := range []string{"A", "A", "B"} {
		ev := Event{Kind: KindOpen, ID: id, TS: "2026-08-08T00:0" + string(rune('0'+i)) + ":00Z", Boot: "boot-1", CWD: denied, Model: payload, Argv: []string{payload}}
		if err := Append(journalPath, ev); !errors.Is(err, ErrSourceDenied) {
			t.Fatalf("Append(%s) denied source error = %v, want ErrSourceDenied", id, err)
		}
	}
	// One ordinary, admissible open joins the three refusals in the journal.
	if err := Append(journalPath, Event{Kind: KindOpen, ID: "C", TS: "2026-08-08T00:03:00Z", CWD: "C:/work/fak"}); err != nil {
		t.Fatalf("Append(C) admissible source error = %v, want nil", err)
	}
	stored, readErr := os.ReadFile(journalPath)
	if readErr != nil {
		t.Fatalf("ReadFile: %v", readErr)
	}
	for _, forbidden := range [][]byte{[]byte(denied), []byte(payload)} {
		if bytes.Contains(stored, forbidden) {
			t.Fatalf("denied bytes reached the journal at rest: %q", stored)
		}
	}
	events := ParseEvents(string(stored))

	if got := RefusalCount(events); got != 3 {
		t.Fatalf("RefusalCount = %d, want 3", got)
	}

	all := Refusals(events, "", 0)
	if len(all) != 3 {
		t.Fatalf("Refusals(all) = %d rows, want 3", len(all))
	}
	// Newest-first, and every row carries only the persisted content-free fields.
	if all[0].ID != "B" || all[2].ID != "A" {
		t.Fatalf("Refusals must be newest-first, got %+v", all)
	}
	if all[0].Reason != "SECRET_EXFIL" || all[0].Boot != "boot-1" || all[0].SourceDigest == "" {
		t.Fatalf("refusal row missing persisted fields: %+v", all[0])
	}
	for _, r := range all {
		if strings.Contains(r.SourceDigest, payload) || strings.Contains(r.SourceDigest, "fak-private") {
			t.Fatalf("refusal row leaked denied source content: %+v", r)
		}
	}

	if filtered := Refusals(events, "A", 0); len(filtered) != 2 {
		t.Fatalf("Refusals(id=A) = %d rows, want 2", len(filtered))
	}
	if limited := Refusals(events, "", 2); len(limited) != 2 {
		t.Fatalf("Refusals(limit=2) = %d rows, want 2", len(limited))
	}

	// The fold invariant holds: refusals never become lifecycle state.
	if sessions := FoldEvents(events); len(sessions) != 1 || sessions[0].ID != "C" {
		t.Fatalf("refusals must stay out of the fold, got %+v", sessions)
	}
}
