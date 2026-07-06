package journal

import (
	"testing"
	"time"
)

// AppendCrash writes a chained CHILD_CRASH row that carries the crash identity on
// the frozen decision fields (Kind/Tool/TraceID/Reason) plus the non-chained exit
// code, and the row verifies as part of the chain.
func TestAppendCrashChainsAndVerifies(t *testing.T) {
	j := OpenMemory()
	j.clock = func() time.Time { return time.Unix(5, 0) }

	// A prior decision, then a crash — the crash must chain onto the decision head.
	j.Emit(testDenyEvent("send_email", "trace-a", `{}`))
	row := j.AppendCrash("claude", "guard-xyz", CrashSignal, -1)

	if row.Kind != KindChildCrash || row.Tool != "claude" || row.TraceID != "guard-xyz" {
		t.Fatalf("crash row identity = %+v", row)
	}
	if row.Reason != CrashSignal || row.ExitCode != -1 || row.By != "guard-supervisor" {
		t.Fatalf("crash row class/exit/by = %+v", row)
	}
	if row.Seq != 2 || row.PrevHash == "" || row.Hash == "" {
		t.Fatalf("crash row not chained: %+v", row)
	}
	if n, err := VerifyRows(j.Recent(0)); err != nil || n != 2 {
		t.Fatalf("VerifyRows = n=%d err=%v, want 2 nil", n, err)
	}
}

// The crash-specific field (ExitCode) is OUTSIDE the hash-chain pre-image: two
// rows identical in every chained field hash the same regardless of ExitCode, so
// appending the field leaves every existing journal verifying byte-for-byte.
func TestExitCodeIsOutsideChainPreimage(t *testing.T) {
	base := Row{Kind: KindChildCrash, Tool: "claude", TraceID: "g", Reason: CrashOOM, By: "guard-supervisor", Seq: 1, TSUnixNano: 7}
	withCode := base
	withCode.ExitCode = 137
	if chainHash("", base) != chainHash("", withCode) {
		t.Fatal("ExitCode entered the hash pre-image — it must be a non-chained field so old journals verify unchanged")
	}
}

// A nil receiver is a safe no-op: a caller that guarded the journal on may call
// AppendCrash unconditionally.
func TestAppendCrashNilReceiverIsNoop(t *testing.T) {
	var j *Journal
	if got := j.AppendCrash("claude", "g", CrashNonzeroExit, 2); got != (Row{}) {
		t.Fatalf("nil-receiver AppendCrash = %+v, want zero Row", got)
	}
}

// A crash row round-trips through a file-backed journal and Verify (the durable
// path the guard front door actually uses).
func TestCrashRowSurvivesFileRoundTrip(t *testing.T) {
	path := t.TempDir() + "/audit.jsonl"
	j, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	j.Emit(testDenyEvent("Read", "a", `{}`))
	j.AppendCrash("claude", "guard-1", CrashOOM, 137)
	if err := j.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	n, err := Verify(path)
	if err != nil || n != 2 {
		t.Fatalf("Verify = n=%d err=%v, want 2 nil", n, err)
	}
	rows, err := ReadRows(path)
	if err != nil {
		t.Fatalf("ReadRows: %v", err)
	}
	if len(rows) != 2 || rows[1].Kind != KindChildCrash || rows[1].ExitCode != 137 {
		t.Fatalf("read-back crash row = %+v", rows)
	}
}
