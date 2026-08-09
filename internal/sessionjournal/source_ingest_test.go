package sessionjournal

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
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
