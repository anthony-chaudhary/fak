package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

func TestFoldAuditDatasetJoinsOutcomesAndOrdersBySeq(t *testing.T) {
	rows := []journal.Row{
		{Seq: 9, Kind: "QUARANTINE", CallSeq: 42, Verdict: "QUARANTINE", Reason: "TAINT", Taint: "untrusted"},
		{Seq: 7, TSUnixNano: 123, Kind: "DECIDE", CallSeq: 42, TraceID: "tr", Tool: "read", Verdict: "ALLOW", Reason: "", By: "floor", ArgsDigest: "sha256:x", ArgsLabel: "object{path}", Witness: "w"},
		{Seq: 2, TSUnixNano: 100, Kind: "DENY", CallSeq: 4, Tool: "write", Verdict: "DENY", Reason: "POLICY_BLOCK", By: "floor"},
	}
	got, problems := foldAuditDataset(rows)
	if len(problems) != 0 {
		t.Fatalf("problems=%+v", problems)
	}
	if len(got) != 2 {
		t.Fatalf("rows=%+v", got)
	}
	if got[0].Seq != 2 || got[1].Seq != 7 {
		t.Fatalf("nondeterministic order: %+v", got)
	}
	joined := got[1]
	if joined.Schema != auditDatasetSchema || joined.ResultVerdict != "QUARANTINE" || joined.ResultReason != "TAINT" || joined.Taint != "untrusted" {
		t.Fatalf("join=%+v", joined)
	}
	if joined.ArgsDigest != "sha256:x" || joined.ArgsLabel != "object{path}" || joined.Witness != "w" {
		t.Fatalf("bounded disclosure fields lost: %+v", joined)
	}
}

func TestFoldAuditDatasetSurfacesUnkeyableRows(t *testing.T) {
	rows := []journal.Row{
		{Seq: 1, Kind: "DECIDE", Tool: "missing-key", Verdict: "ALLOW"},
		{Seq: 2, Kind: "RESULT_DENY", CallSeq: 99, Verdict: "DENY"},
		{Seq: 3, Kind: "QUARANTINE"},
	}
	got, problems := foldAuditDataset(rows)
	if len(got) != 0 {
		t.Fatalf("dataset=%+v", got)
	}
	if len(problems) != 3 {
		t.Fatalf("problems=%+v", problems)
	}
}

func TestFoldAuditDatasetDoesNotEmitRawArgs(t *testing.T) {
	got, _ := foldAuditDataset([]journal.Row{{Seq: 1, Kind: "DECIDE", CallSeq: 1, Verdict: "ALLOW", ArgsDigest: "digest", ArgsLabel: "label"}})
	if len(got) != 1 || got[0].ArgsDigest != "digest" {
		t.Fatalf("dataset=%+v", got)
	}
}
