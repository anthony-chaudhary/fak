package journal

import (
	"path/filepath"
	"testing"
)

// TestCapabilityGrantAppendChainsAndVerifies pins the emit half of #5178: a
// CAPABILITY_GRANT row is a genuine chained row (it consumes a seq and verifies
// with the rest of the chain), carries the widened-knob / channel / actor
// identity on the frozen decision fields, and carries the full provenance record
// (old→new, class, source) on the non-chained Grant payload.
func TestCapabilityGrantAppendChainsAndVerifies(t *testing.T) {
	j := OpenMemory()
	j.Emit(testDenyEvent("bash", "gw-1", `{"cmd":"rm"}`)) // a real decision row ahead of the grant, so the grant chains onto history

	row := j.AppendCapabilityGrant(CapabilityGrantRow{
		Knob:    "Allow",
		Class:   "GATED_WIDEN",
		Old:     "",
		New:     "read_docs",
		Channel: GrantChannelOperatorOverlay,
		Actor:   "operator:alice",
		Source:  ".fak/guard/allow.json",
		Reason:  "docs sub-agent needs the reader",
	})

	if row.Kind != KindCapabilityGrant {
		t.Fatalf("Kind = %q, want %q", row.Kind, KindCapabilityGrant)
	}
	if row.Tool != "Allow" {
		t.Fatalf("Tool = %q, want the widened knob %q", row.Tool, "Allow")
	}
	if row.Reason != GrantChannelOperatorOverlay {
		t.Fatalf("Reason = %q, want the mirrored channel %q", row.Reason, GrantChannelOperatorOverlay)
	}
	if row.By != "operator:alice" {
		t.Fatalf("By = %q, want the granting actor %q", row.By, "operator:alice")
	}
	if row.Grant == nil {
		t.Fatal("Grant payload missing from committed row")
	}
	if row.Grant.Schema != CapabilityGrantSchema {
		t.Fatalf("payload schema = %q, want %q", row.Grant.Schema, CapabilityGrantSchema)
	}
	// Direction is stamped by the appender, not supplied: a grant is a widening
	// by construction, so no caller can record one that claims otherwise.
	if row.Grant.Direction != GrantDirectionWiden {
		t.Fatalf("payload direction = %q, want %q", row.Grant.Direction, GrantDirectionWiden)
	}
	if row.Grant.Old != "" || row.Grant.New != "read_docs" {
		t.Fatalf("payload lost the old->new values: %+v", row.Grant)
	}
	if row.Grant.Class != "GATED_WIDEN" || row.Grant.Source != ".fak/guard/allow.json" {
		t.Fatalf("payload lost class/source: %+v", row.Grant)
	}
	// The timestamp the issue asks for is the row's own anchor, not a payload field.
	if row.TSUnixNano == 0 {
		t.Fatal("committed grant carries no timestamp anchor")
	}
	if row.Seq == 0 || row.Hash == "" {
		t.Fatalf("row not committed through the chain: seq=%d hash=%q", row.Seq, row.Hash)
	}
	// The load-bearing property: the payload is NOT part of the pre-image, so the
	// chain over (decision row, grant row) verifies exactly like any other journal.
	if n, err := VerifyRows(j.Recent(0)); err != nil || n != 2 {
		t.Fatalf("VerifyRows = (%d, %v), want (2, nil)", n, err)
	}
}

// TestCapabilityGrantNilJournalNoOp pins the caller contract: every widening site
// calls this unconditionally on journal.Active(), so a run with no audit trail
// (nil journal) must be a safe no-op, not a panic.
func TestCapabilityGrantNilJournalNoOp(t *testing.T) {
	var j *Journal
	row := j.AppendCapabilityGrant(CapabilityGrantRow{Knob: "Allow", Channel: GrantChannelOperatorOverlay})
	if row.Kind != "" || row.Seq != 0 {
		t.Fatalf("nil journal must return the zero Row, got %+v", row)
	}
}

// TestCapabilityGrantUnattributedActorFallback pins the launch-time case: an
// overlay applied at boot has no named operator, and a row with an empty By
// would read as though it came from nowhere. The grant still records the
// channel, which is the attribution that actually exists.
func TestCapabilityGrantUnattributedActorFallback(t *testing.T) {
	j := OpenMemory()
	row := j.AppendCapabilityGrant(CapabilityGrantRow{Knob: "AllowPrefix", New: "read_", Channel: GrantChannelOperatorOverlay})
	if row.By != grantActorFallback {
		t.Fatalf("By = %q, want the unattributed fallback %q", row.By, grantActorFallback)
	}
	if row.Grant.Actor != "" {
		t.Fatalf("payload Actor = %q, want it left empty rather than backfilled with the fallback", row.Grant.Actor)
	}
}

// TestCapabilityGrantPersistsAndVerifies pins the durable path: a grant written
// through a file-backed journal survives close, comes back through ReadRows with
// its full provenance intact, and the file still passes journal.Verify end to
// end — the `fak audit verify` half of the issue's witness.
func TestCapabilityGrantPersistsAndVerifies(t *testing.T) {
	path := filepath.Join(t.TempDir(), "audit.jsonl")
	j, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	j.AppendCapabilityGrant(CapabilityGrantRow{
		Knob:    "Posture",
		Class:   "GATED_WIDEN",
		Old:     "fail_closed",
		New:     "admit_and_log",
		Channel: GrantChannelLiveReload,
		Actor:   "reload:gateway",
		Reason:  "FAK_POLICY_RELOAD_ALLOW_WIDEN=1",
	})
	if err := j.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	rows, err := ReadRows(path)
	if err != nil {
		t.Fatalf("ReadRows: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("len(rows) = %d, want 1", len(rows))
	}
	got := rows[0]
	if got.Kind != KindCapabilityGrant || got.Reason != GrantChannelLiveReload || got.Grant == nil {
		t.Fatalf("reloaded row lost its identity: %+v (capability_grant=%+v)", got, got.Grant)
	}
	if got.Grant.Knob != "Posture" || got.Grant.Old != "fail_closed" || got.Grant.New != "admit_and_log" {
		t.Fatalf("reloaded payload lost its old->new provenance: %+v", got.Grant)
	}
	if got.Grant.Actor != "reload:gateway" || got.Grant.Direction != GrantDirectionWiden {
		t.Fatalf("reloaded payload lost actor/direction: %+v", got.Grant)
	}
	// Acceptance: Verify still passes over a journal that now carries
	// CAPABILITY_GRANT rows.
	if n, err := Verify(path); err != nil || n != 1 {
		t.Fatalf("Verify = (%d, %v), want (1, nil)", n, err)
	}
}
