package sessiondesc

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

func TestFoldJournalUsesDurableRegistrationView(t *testing.T) {
	registration := &sessionjournal.RegistrationCarry{RegistrationID: "reg-b", Runtime: "codex", SessionID: "session-b", State: "active"}
	events := []sessionjournal.Event{{Schema: sessionjournal.Schema, Kind: sessionjournal.KindOpen, ID: "reg-b", TS: "2026-08-18T01:00:00Z", Registration: registration}}
	got := FoldJournal(events)
	if len(got) != 1 {
		t.Fatalf("descriptors=%d want 1", len(got))
	}
	if got[0].ID != "reg-b" || got[0].Harness.Presence != Bound || got[0].Harness.Agent != "codex" || got[0].Harness.Identity != "session-b" {
		t.Fatalf("descriptor=%+v", got[0])
	}
	if got[0].Ref.Presence != AbsentNotObserved {
		t.Fatalf("ref presence=%s: journal view must not infer TTL liveness", got[0].Ref.Presence)
	}
}
