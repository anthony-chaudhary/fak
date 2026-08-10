package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

func TestConceptEpisodesPreserveDrillDownAndOrder(t *testing.T) {
	turns := []trajectory.Turn{
		{TraceID: "b", Seq: 2, Tool: "read", Verdict: "DENY"},
		{TraceID: "a", Seq: 1, Tool: "search", Verdict: "ALLOW"},
		{TraceID: "b", Seq: 1, Tool: "search", Verdict: "ALLOW"},
		{TraceID: "a", Seq: 2, Tool: "read", Verdict: "ALLOW"},
	}
	got := conceptEpisodes(turns)
	if len(got) != 2 || got[0].ID != "b" || got[1].ID != "a" {
		t.Fatalf("episodes=%#v", got)
	}
	if got[0].Calls[0].Tool != "search" || got[0].Calls[1].Tool != "read" || !got[0].Calls[1].Error {
		t.Fatalf("ordered calls=%#v", got[0])
	}
}
