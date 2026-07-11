package journal

import (
	"path/filepath"
	"testing"
)

// TestConfigSwapAppendChainsAndVerifies pins the emit half of #3959: a
// CONFIG_SWAP row is a genuine chained row (it consumes a seq and verifies with
// the rest of the chain), carries the swapped-surface/outcome identity on the
// frozen decision fields, and carries the full correlated record (source path +
// installed-bytes digest) on the non-chained ConfigSwap payload.
func TestConfigSwapAppendChainsAndVerifies(t *testing.T) {
	j := OpenMemory()
	j.Emit(testDenyEvent("bash", "gw-1", `{"cmd":"rm"}`)) // a real decision row ahead of the swap, so the swap chains onto history

	row := j.AppendConfigSwap(ConfigSwapFloor, "/etc/fak/floor.json", "sha256:abc123", ConfigSwapOK, "")

	if row.Kind != KindConfigSwap {
		t.Fatalf("Kind = %q, want %q", row.Kind, KindConfigSwap)
	}
	if row.Tool != ConfigSwapFloor {
		t.Fatalf("Tool = %q, want the swapped surface %q", row.Tool, ConfigSwapFloor)
	}
	if row.Reason != ConfigSwapOK {
		t.Fatalf("Reason = %q, want the mirrored outcome %q", row.Reason, ConfigSwapOK)
	}
	if row.ConfigSwap == nil {
		t.Fatal("ConfigSwap payload missing from committed row")
	}
	if row.ConfigSwap.Schema != ConfigSwapSchema {
		t.Fatalf("payload schema = %q, want %q", row.ConfigSwap.Schema, ConfigSwapSchema)
	}
	if row.ConfigSwap.Source != "/etc/fak/floor.json" || row.ConfigSwap.Digest != "sha256:abc123" {
		t.Fatalf("payload lost source/digest: %+v", row.ConfigSwap)
	}
	if row.Seq == 0 || row.Hash == "" {
		t.Fatalf("row not committed through the chain: seq=%d hash=%q", row.Seq, row.Hash)
	}
	// The load-bearing property: the payload is NOT part of the pre-image, so the
	// chain over (decision row, swap row) verifies exactly like any other journal.
	if n, err := VerifyRows(j.Recent(0)); err != nil || n != 2 {
		t.Fatalf("VerifyRows = (%d, %v), want (2, nil)", n, err)
	}
}

// TestConfigSwapNilJournalNoOp pins the caller contract: reloadPolicy and the
// route-watcher OnEvent call this unconditionally on journal.Active(), so a
// run with no audit trail (nil journal) must be a safe no-op, not a panic.
func TestConfigSwapNilJournalNoOp(t *testing.T) {
	var j *Journal
	row := j.AppendConfigSwap(ConfigSwapFloor, "/etc/fak/floor.json", "sha256:abc", ConfigSwapOK, "")
	if row.Kind != "" || row.Seq != 0 {
		t.Fatalf("nil journal must return the zero Row, got %+v", row)
	}
}

// TestConfigSwapRejectedPersistsAndVerifies pins the rejected-outcome half plus
// the durable path: a rejected swap (kept last-good, but recorded because a
// refused edit is what an auditor asks about) written through a file-backed
// journal survives close, comes back through ReadRows with its rejection reason
// intact, and the file still passes journal.Verify end to end.
func TestConfigSwapRejectedPersistsAndVerifies(t *testing.T) {
	path := filepath.Join(t.TempDir(), "audit.jsonl")
	j, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	j.AppendConfigSwap(ConfigSwapRoute, "/etc/fak/route.json", "", ConfigSwapRejected, "parse manifest: unexpected EOF")
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
	if got.Kind != KindConfigSwap || got.Reason != ConfigSwapRejected || got.ConfigSwap == nil {
		t.Fatalf("reloaded row lost its identity: %+v (config_swap=%+v)", got, got.ConfigSwap)
	}
	if got.ConfigSwap.Kind != ConfigSwapRoute || got.ConfigSwap.Reason != "parse manifest: unexpected EOF" {
		t.Fatalf("reloaded payload lost its rejection detail: %+v", got.ConfigSwap)
	}
	// Acceptance: Verify still passes over a journal that now carries CONFIG_SWAP rows.
	if n, err := Verify(path); err != nil || n != 1 {
		t.Fatalf("Verify = (%d, %v), want (1, nil)", n, err)
	}
}
