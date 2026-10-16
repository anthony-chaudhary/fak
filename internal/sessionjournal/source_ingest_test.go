package sessionjournal

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
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

// TestSessionJournalRefusalLimits keeps default and explicit reports finite without
// retaining all matching events. The full-sort oracle is test-only.
// fak-test:runtime fast est=100ms lane=default
func TestSessionJournalRefusalLimits(t *testing.T) {
	const defaultLimit, maximumLimit = 100, 1000
	base := time.Date(2026, 8, 8, 0, 0, 0, 0, time.UTC)
	events := make([]Event, maximumLimit*3+1)
	for i := range events {
		// Repeated timestamps and a nonchronological journal exercise stable top-K.
		events[i] = Event{Kind: KindRefuse, ID: fmt.Sprint(i), TS: base.Add(time.Duration((i*17)%701) * time.Second).Format(time.RFC3339), Reason: "SECRET_EXFIL", CWD: "DENIED-CWD", Model: "DENIED-MODEL", Argv: []string{"DENIED-ARGV"}}
	}
	oracle := append([]Event(nil), events...)
	sort.SliceStable(oracle, func(i, j int) bool { return eventTime(oracle[i]).After(eventTime(oracle[j])) })
	for _, tc := range []struct {
		name        string
		limit, want int
	}{
		{"default", 0, defaultLimit},
		{"negative", -1, defaultLimit},
		{"one", 1, 1},
		{"finite", 7, 7},
		{"maximum", maximumLimit, maximumLimit},
		{"oversized", int(^uint(0) >> 1), maximumLimit},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := Refusals(events, "", tc.limit)
			if len(got) != tc.want {
				t.Fatalf("Refusals(limit=%d) returned %d rows, want %d", tc.limit, len(got), tc.want)
			}
			for i, row := range got {
				if row.ID != oracle[i].ID || row.TS != oracle[i].TS {
					t.Fatalf("row %d = %+v, want id=%q ts=%q", i, row, oracle[i].ID, oracle[i].TS)
				}
			}
			encoded, err := json.Marshal(got)
			if err != nil || bytes.Contains(encoded, []byte("DENIED-")) {
				t.Fatalf("unsafe projection or encoding error: %v", err)
			}
		})
	}
	if count := RefusalCount(events); count != len(events) {
		t.Fatalf("total refusal count = %d, want %d", count, len(events))
	}
	if sessions := FoldEvents(events); len(sessions) != 0 {
		t.Fatalf("refusals became lifecycle sessions: %d", len(sessions))
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSessionJournalRefusalExactIDs(t *testing.T) {
	events := []Event{
		{Kind: KindRefuse, ID: " A ", TS: "invalid", Reason: "first"},
		{Kind: KindRefuse, ID: "A", TS: "", Reason: "second"},
		{Kind: KindRefuse, ID: " A", TS: "invalid", Reason: "third"},
		{Kind: KindOpen, ID: "A", TS: "2026-08-08T00:00:00Z"},
	}
	for _, id := range []string{"A", " A ", " A"} {
		got := Refusals(events, id, 1)
		if len(got) != 1 || got[0].ID != id {
			t.Fatalf("exact filter %q returned %+v", id, got)
		}
	}
	if got := Refusals(events, "missing", 1); got == nil || len(got) != 0 {
		t.Fatalf("missing filter must return an empty array, got %+v", got)
	}
	got := Refusals(events, "", 2)
	if len(got) != 2 || got[0].Reason != "first" || got[1].Reason != "second" {
		t.Fatalf("invalid timestamps must retain journal tie order, got %+v", got)
	}
}
