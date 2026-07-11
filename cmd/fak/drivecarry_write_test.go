package main

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// seedDriveCarryWorld wires the two disjoint stores the producer joins: the A1 identity
// ledger (uuid <-> trace) and the durable descriptor registry holding a live State under the
// trace. It returns the resolved regDir (where resume_drivecarry.jsonl lands). The hook mode
// envs are pinned off so the drivers below stay hermetic (no gateway /metrics fetch).
func seedDriveCarryWorld(t *testing.T, uuid, trace string, st session.State) string {
	t.Helper()
	dir := t.TempDir()
	regPath := filepath.Join(dir, "session-registry.json")
	t.Setenv("FLEET_REG_DIR", dir)            // resolveSweepRegDir + identity/carry ledgers
	t.Setenv("FAK_SESSION_REGISTRY", regPath) // defaultSessionRegistryPath (the descriptor store)
	t.Setenv("ANTHROPIC_BASE_URL", "")        // no gateway to scrape
	t.Setenv("FAK_GUARD_PRECOMPACT_MODE", "off")
	t.Setenv("FAK_GUARD_DENYALL_MODE", "off")

	if err := appendJSONL(resume.IdentityLedgerPath(dir), resume.IdentityRow{UUID: uuid, Trace: trace}); err != nil {
		t.Fatalf("seed identity: %v", err)
	}
	reg := session.NewRegistry(session.NewFileStore(regPath))
	if _, err := reg.Register(trace, "host", st, time.Hour, time.Now()); err != nil {
		t.Fatalf("seed descriptor: %v", err)
	}
	return dir
}

func loadDriveCarryRows(t *testing.T, regDir string) []resume.DriveCarryRow {
	t.Helper()
	raw, err := os.ReadFile(rwDriveCarryLedger(regDir))
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		t.Fatalf("read carry ledger: %v", err)
	}
	return jsonlledger.Parse[resume.DriveCarryRow](string(raw), nil)
}

// TestDriveCarryWritePreCompactBounded is the witness: driving runGuardPreCompact with the
// transcript UUID set and a bounded live descriptor appends exactly one carry row whose
// remaining axes equal the live State's.
func TestDriveCarryWritePreCompactBounded(t *testing.T) {
	const uuid, trace = "uuid-carry-1", "trace-carry-1"
	st := session.State{
		TraceID: trace, Run: session.Running,
		Budget:     session.Budget{TurnsLeft: 5, TokensLeft: 1200, ContextTokensLeft: 800, SpendMicroCentsLeft: 4500},
		Priority:   2,
		Pace:       session.Pace{MaxTokensPerTurn: 900, MinTurnGapMs: 250},
		Generation: 1,
	}
	dir := seedDriveCarryWorld(t, uuid, trace, st)
	t.Setenv("CLAUDE_CODE_SESSION_ID", uuid)

	if code := runGuardPreCompact(io.Discard, io.Discard, strings.NewReader(""), nil); code != 0 {
		t.Fatalf("runGuardPreCompact code=%d, want 0", code)
	}
	rows := loadDriveCarryRows(t, dir)
	if len(rows) != 1 {
		t.Fatalf("carry rows=%d, want exactly 1: %+v", len(rows), rows)
	}
	got := rows[0]
	if got.Session != uuid {
		t.Fatalf("carry session=%q, want %q", got.Session, uuid)
	}
	if got.TurnsLeft != 5 || got.TokensLeft != 1200 || got.ContextTokensLeft != 800 || got.SpendMicroCentsLeft != 4500 {
		t.Fatalf("carry remaining budget = %+v, want it to match the live State", got)
	}
	if got.Priority != 2 || got.PaceMaxTokensPerTurn != 900 || got.PaceMinTurnGapMs != 250 || got.Generation != 1 {
		t.Fatalf("carry priority/pace/generation = %+v", got)
	}
}

// TestDriveCarryWriteCarriesObjective is the #4121 write-side witness: a live State holding a
// standing ObjectivePin projects that pin's safe extractive triple onto the carry row, and the
// reconstructed pin reconciles as ObjectivePreserved against the original — so a relaunched
// child re-pins the SAME objective (#1583) instead of silently dropping it. This closes the
// producer half; internal/resume TestObjectivePinCarry proves the record round-trip.
func TestDriveCarryWriteCarriesObjective(t *testing.T) {
	const uuid, trace = "uuid-carry-obj", "trace-carry-obj"
	pin := ctxplan.NewObjectivePin("obj-4121", "ship the drivecarry objective triple", 3)
	st := session.State{
		TraceID: trace, Run: session.Running,
		Budget:       session.Budget{TurnsLeft: 4, TokensLeft: 500}, // bounded so a row is written
		ObjectivePin: pin,
	}
	dir := seedDriveCarryWorld(t, uuid, trace, st)
	t.Setenv("CLAUDE_CODE_SESSION_ID", uuid)

	if code := runGuardPreCompact(io.Discard, io.Discard, strings.NewReader(""), nil); code != 0 {
		t.Fatalf("runGuardPreCompact code=%d, want 0", code)
	}
	rows := loadDriveCarryRows(t, dir)
	if len(rows) != 1 {
		t.Fatalf("carry rows=%d, want exactly 1: %+v", len(rows), rows)
	}
	got := rows[0]
	if got.ObjectivePinID != pin.PinID || got.ObjectiveText != pin.Text || got.ObjectiveDigest != pin.Digest {
		t.Fatalf("carry objective triple = (%q,%q,%q), want (%q,%q,%q)",
			got.ObjectivePinID, got.ObjectiveText, got.ObjectiveDigest, pin.PinID, pin.Text, pin.Digest)
	}
	if d := ctxplan.ReconcileObjective(pin, got.ObjectivePin()); d.Outcome != ctxplan.ObjectivePreserved {
		t.Fatalf("reconcile carried objective = %q, want %q", d.Outcome, ctxplan.ObjectivePreserved)
	}
}

// TestDriveCarryWriteEnvUnset asserts the fail-open floor: with CLAUDE_CODE_SESSION_ID unset
// (a resumed child has it stripped) the hook appends no carry row at all.
func TestDriveCarryWriteEnvUnset(t *testing.T) {
	const uuid, trace = "uuid-carry-2", "trace-carry-2"
	st := session.State{TraceID: trace, Run: session.Running, Budget: session.Budget{TurnsLeft: 3, TokensLeft: 100}}
	dir := seedDriveCarryWorld(t, uuid, trace, st)
	t.Setenv("CLAUDE_CODE_SESSION_ID", "")

	if code := runGuardPreCompact(io.Discard, io.Discard, strings.NewReader(""), nil); code != 0 {
		t.Fatalf("runGuardPreCompact code=%d, want 0", code)
	}
	if rows := loadDriveCarryRows(t, dir); len(rows) != 0 {
		t.Fatalf("carry rows=%d with env unset, want 0: %+v", len(rows), rows)
	}
}

// TestDriveCarryWriteUnboundedNoRow asserts an unbounded live budget (the DefaultState shape)
// appends nothing — there is no remaining allotment worth carrying.
func TestDriveCarryWriteUnboundedNoRow(t *testing.T) {
	const uuid, trace = "uuid-carry-3", "trace-carry-3"
	st := session.State{TraceID: trace, Run: session.Running, Budget: session.Budget{TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded}}
	dir := seedDriveCarryWorld(t, uuid, trace, st)
	t.Setenv("CLAUDE_CODE_SESSION_ID", uuid)

	if code := runGuardPreCompact(io.Discard, io.Discard, strings.NewReader(""), nil); code != 0 {
		t.Fatalf("runGuardPreCompact code=%d, want 0", code)
	}
	if rows := loadDriveCarryRows(t, dir); len(rows) != 0 {
		t.Fatalf("carry rows=%d for an unbounded budget, want 0: %+v", len(rows), rows)
	}
}

// TestDriveCarryWriteStopHook covers the second producer site: the Stop hook records the same
// bounded budget on the transcript-UUID key.
func TestDriveCarryWriteStopHook(t *testing.T) {
	const uuid, trace = "uuid-carry-4", "trace-carry-4"
	st := session.State{TraceID: trace, Run: session.Running, Budget: session.Budget{TurnsLeft: 9, TokensLeft: 700}}
	dir := seedDriveCarryWorld(t, uuid, trace, st)
	t.Setenv("CLAUDE_CODE_SESSION_ID", uuid)

	if code := runGuardStopHook(io.Discard, strings.NewReader(""), nil); code != 0 {
		t.Fatalf("runGuardStopHook code=%d, want 0", code)
	}
	rows := loadDriveCarryRows(t, dir)
	if len(rows) != 1 {
		t.Fatalf("carry rows=%d after Stop, want 1: %+v", len(rows), rows)
	}
	if rows[0].TurnsLeft != 9 || rows[0].TokensLeft != 700 {
		t.Fatalf("Stop carry row = %+v, want turns_left=9 tokens_left=700", rows[0])
	}
}
