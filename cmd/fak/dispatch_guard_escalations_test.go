package main

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func TestHumanBlockedGuardEscalationsLatestDispositionOnly(t *testing.T) {
	path := filepath.Join(t.TempDir(), "stops.jsonl")
	rows := []guardStopRecord{
		{Schema: guardStopRecordSchema, Session: "escalated", Disposition: string(stopDispOperatorDirectedContinue)},
		{Schema: guardStopRecordSchema, Session: "continued", Disposition: string(stopDispOperatorDirectedEscalate)},
		{Schema: guardStopRecordSchema, Session: "escalated", Disposition: string(stopDispOperatorDirectedEscalate)},
		{Schema: guardStopRecordSchema, Session: "continued", Disposition: string(stopDispOperatorDirectedWarn)},
		{Schema: guardStopRecordSchema, Session: "shadow", Disposition: string(stopDispOperatorDirectedShadow)},
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range rows {
		if _, err := fmt.Fprintf(f, `{"schema":%q,"session":%q,"disposition":%q}`+"\n", row.Schema, row.Session, row.Disposition); err != nil {
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	got := humanBlockedGuardEscalations(path)
	if len(got) != 1 || got[0].Title != "guard session escalated" || got[0].Reason != reasonBlockedByHuman {
		t.Fatalf("guard escalation fold=%+v, want only latest escalated session under %s", got, reasonBlockedByHuman)
	}
}

func TestMergeHumanBlockedSkippedDeduplicates(t *testing.T) {
	router := []dispatchtick.SkippedIssue{{Number: 42, Title: "issue row", Reason: reasonBlockedByHuman}, {Title: "guard session s1", Reason: reasonBlockedByHuman}}
	guard := []dispatchtick.SkippedIssue{{Title: "guard session s1", Reason: reasonBlockedByHuman}, {Title: "guard session s2", Reason: reasonBlockedByHuman}}
	got := mergeHumanBlockedSkipped(router, guard)
	if len(got) != 3 {
		t.Fatalf("merged rows=%+v, want 3 unique entries", got)
	}
	for _, row := range got {
		if row.Reason != reasonBlockedByHuman {
			t.Fatalf("new skip token introduced: %+v", row)
		}
	}
}
