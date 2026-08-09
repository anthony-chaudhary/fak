package sessionledger

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestCheckpointWitnessBinds is #2425's named witness: a checkpoint binds the transcript
// head to the git tree witness in ONE record, a clean round trip verifies, and mutating
// exactly one half fails naming exactly that axis — the tree when the workspace drifts,
// the transcript when the ledger itself is rewritten.
func TestCheckpointWitnessBinds(t *testing.T) {
	dir := t.TempDir()
	l, err := Open(dir)
	if err != nil {
		t.Fatalf("open ledger: %v", err)
	}
	const trace = "trace-2425"
	if _, err := l.Append(trace, "turn", []byte(`{"n":1}`)); err != nil {
		t.Fatalf("seed turn: %v", err)
	}

	head := "3f2a1c9d4e5b6a7c8d9e0f1a2b3c4d5e6f708192"
	tree, err := NewTreeWitness(head, []DirtyEntry{
		{Path: "internal/sessionledger/checkpoint.go", Status: "??", SHA256: "aa01"},
		{Path: "cmd/fak/session_cmd.go", Status: " M", SHA256: "bb02"},
	})
	if err != nil {
		t.Fatalf("tree witness: %v", err)
	}

	cp, err := l.Checkpoint(trace, tree)
	if err != nil {
		t.Fatalf("mint checkpoint: %v", err)
	}

	t.Run("binds both hashes in one record", func(t *testing.T) {
		if cp.Hash == "" {
			t.Fatal("checkpoint carries no id hash")
		}
		if cp.LedgerHead == "" {
			t.Fatal("checkpoint carries no ledger head hash (the transcript axis)")
		}
		if cp.LedgerHead == cp.Hash {
			t.Fatalf("checkpoint id must be the record hash, not the head it was minted over (%s)", cp.Hash)
		}
		if cp.Tree != tree {
			t.Fatalf("checkpoint tree witness = %+v, want %+v", cp.Tree, tree)
		}
		if cp.LedgerHead != l.Head(trace) && cp.Hash != l.Head(trace) {
			t.Fatalf("checkpoint is not on the trace's chain: head=%s", l.Head(trace))
		}
		// The id is the digest over BOTH axes, each isolated against a fresh trace so
		// only one half varies at a time. Same head + same tree ⇒ the SAME id (a
		// checkpoint is content-addressed); vary either half and the id moves.
		other, err := NewTreeWitness(head, []DirtyEntry{{Path: "README.md", Status: " M", SHA256: "cc03"}})
		if err != nil {
			t.Fatalf("other witness: %v", err)
		}
		rootA := mintOn(t, l, "solo-a", tree)  // head "" + tree
		rootB := mintOn(t, l, "solo-b", tree)  // head "" + tree
		rootC := mintOn(t, l, "solo-c", other) // head "" + a different tree
		if rootA.Hash != rootB.Hash {
			t.Fatalf("same head + same tree gave different ids %s / %s", rootA.Hash, rootB.Hash)
		}
		if rootC.Hash == rootA.Hash {
			t.Fatal("a different tree witness produced the same checkpoint id — the tree axis is not bound")
		}
		if _, err := l.Append("solo-d", "turn", []byte(`{"n":1}`)); err != nil {
			t.Fatalf("seed solo-d: %v", err)
		}
		rootD := mintOn(t, l, "solo-d", tree) // a different head + the SAME tree
		if rootD.LedgerHead == rootA.LedgerHead {
			t.Fatalf("solo-d should have been minted over a non-empty head, got %q", rootD.LedgerHead)
		}
		if rootD.Hash == rootA.Hash {
			t.Fatal("a different ledger head produced the same checkpoint id — the transcript axis is not bound")
		}
	})

	t.Run("clean round trip verifies", func(t *testing.T) {
		if err := l.VerifyCheckpoint(cp, tree); err != nil {
			t.Fatalf("checkpoint then verify should pass, got %v", err)
		}
		back, err := l.LatestCheckpoint(trace)
		if err != nil {
			t.Fatalf("latest checkpoint: %v", err)
		}
		if back != cp {
			t.Fatalf("ledger-recovered checkpoint = %+v, want %+v", back, cp)
		}
	})

	t.Run("mutated tracked file fails naming the tree axis", func(t *testing.T) {
		mutated, err := NewTreeWitness(head, []DirtyEntry{
			{Path: "internal/sessionledger/checkpoint.go", Status: "??", SHA256: "aa01"},
			{Path: "cmd/fak/session_cmd.go", Status: " M", SHA256: "bb02-EDITED"},
		})
		if err != nil {
			t.Fatalf("mutated witness: %v", err)
		}
		assertAxis(t, l.VerifyCheckpoint(cp, mutated), AxisTree)
	})

	t.Run("moved HEAD fails naming the tree axis", func(t *testing.T) {
		moved, err := NewTreeWitness("0000000000000000000000000000000000000000", []DirtyEntry{
			{Path: "internal/sessionledger/checkpoint.go", Status: "??", SHA256: "aa01"},
			{Path: "cmd/fak/session_cmd.go", Status: " M", SHA256: "bb02"},
		})
		if err != nil {
			t.Fatalf("moved witness: %v", err)
		}
		assertAxis(t, l.VerifyCheckpoint(cp, moved), AxisTree)
	})

	t.Run("rewritten ledger entry fails naming the transcript axis", func(t *testing.T) {
		// Tamper an ANCESTOR, not the checkpoint record: the binding must catch a
		// rewrite anywhere in the history the checkpoint claims to sit on top of.
		tamper := filepath.Join(t.TempDir(), "ledger.jsonl")
		rewriteLedger(t, filepath.Join(dir, LogName), tamper, func(r map[string]json.RawMessage) bool {
			if string(r["kind"]) != `"turn"` {
				return false
			}
			r["content"] = json.RawMessage(`{"n":99}`)
			return true
		})
		tampered := openLedgerFile(t, tamper)
		assertAxis(t, tampered.VerifyCheckpoint(cp, tree), AxisTranscript)
	})

	t.Run("dropped checkpoint record fails naming the transcript axis", func(t *testing.T) {
		dropped := filepath.Join(t.TempDir(), "ledger.jsonl")
		rewriteLedger(t, filepath.Join(dir, LogName), dropped, func(r map[string]json.RawMessage) bool {
			return string(r["hash"]) == `"`+string(cp.Hash)+`"`
		}, dropMatching)
		truncated := openLedgerFile(t, dropped)
		assertAxis(t, truncated.VerifyCheckpoint(cp, tree), AxisTranscript)
	})

	t.Run("unknown trace fails naming the transcript axis", func(t *testing.T) {
		fresh, err := Open(t.TempDir())
		if err != nil {
			t.Fatalf("open fresh: %v", err)
		}
		assertAxis(t, fresh.VerifyCheckpoint(cp, tree), AxisTranscript)
	})
}

// TestTreeWitnessIsOrderIndependent pins the property two peers depend on: the same
// workspace state digests identically however git happened to order the dirty set, and a
// witness never mutates the caller's slice.
func TestTreeWitnessIsOrderIndependent(t *testing.T) {
	dirty := []DirtyEntry{
		{Path: "b.go", Status: " M", SHA256: "22"},
		{Path: "a.go", Status: "??", SHA256: "11"},
		{Path: "c.go", Status: "R ", SHA256: "33", Origin: "old.go"},
	}
	shuffled := []DirtyEntry{dirty[2], dirty[0], dirty[1]}
	first, err := NewTreeWitness("abc123", dirty)
	if err != nil {
		t.Fatalf("witness: %v", err)
	}
	second, err := NewTreeWitness("abc123", shuffled)
	if err != nil {
		t.Fatalf("witness: %v", err)
	}
	if first != second {
		t.Fatalf("dirty-set order changed the digest: %+v vs %+v", first, second)
	}
	if dirty[0].Path != "b.go" {
		t.Fatalf("NewTreeWitness sorted the caller's slice in place: %+v", dirty)
	}
	if first.DirtyCount != 3 {
		t.Fatalf("DirtyCount = %d, want 3", first.DirtyCount)
	}
	// The rename ORIGIN is inside the digest: losing it would let a rename pass as a
	// no-op even though the tree moved.
	noOrigin := []DirtyEntry{dirty[0], dirty[1], {Path: "c.go", Status: "R ", SHA256: "33"}}
	third, err := NewTreeWitness("abc123", noOrigin)
	if err != nil {
		t.Fatalf("witness: %v", err)
	}
	if third == first {
		t.Fatal("dropping a rename origin did not change the tree witness")
	}
}

// TestCheckpointRefusesHalfWitness pins the fail-closed producer edge: a hand-built
// witness (no HEAD SHA, or no dirty digest) binds the transcript to nothing and is
// refused at mint time rather than producing a checkpoint nobody can re-derive.
func TestCheckpointRefusesHalfWitness(t *testing.T) {
	l, err := Open(t.TempDir())
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	for name, tw := range map[string]TreeWitness{
		"no head":   {DirtySHA256: "deadbeef"},
		"no digest": {HeadSHA: "abc123"},
		"zero":      {},
	} {
		if _, err := l.Checkpoint("t", tw); err == nil {
			t.Fatalf("%s: minting over a half-built witness should be refused", name)
		}
	}
	good, err := NewTreeWitness("abc123", nil)
	if err != nil {
		t.Fatalf("witness: %v", err)
	}
	if _, err := l.Checkpoint("", good); err == nil {
		t.Fatal("minting without a trace should be refused")
	}
	if _, err := l.Checkpoint("t", good); err != nil {
		t.Fatalf("a clean tree is a legal checkpoint: %v", err)
	}
}

// mintOn mints a checkpoint on trace or fails the test.
func mintOn(t *testing.T, l *Ledger, trace string, tree TreeWitness) Checkpoint {
	t.Helper()
	cp, err := l.Checkpoint(trace, tree)
	if err != nil {
		t.Fatalf("mint checkpoint on %s: %v", trace, err)
	}
	return cp
}

// assertAxis fails unless err is a *CheckpointMismatch naming want — the point of the
// typed failure is that a caller can branch on the axis without parsing prose.
func assertAxis(t *testing.T, err error, want CheckpointAxis) {
	t.Helper()
	if err == nil {
		t.Fatalf("verify should have failed on the %s axis, got nil", want)
	}
	var mm *CheckpointMismatch
	if !errors.As(err, &mm) {
		t.Fatalf("verify error %v is not a *CheckpointMismatch", err)
	}
	if mm.Axis != want {
		t.Fatalf("verify named the %s axis, want %s (detail: %s)", mm.Axis, want, mm.Detail)
	}
	if !strings.Contains(err.Error(), string(want)) {
		t.Fatalf("rendered error %q does not name the %s axis", err, want)
	}
}

// dropMatching is the rewriteLedger mode that DELETES matching records instead of
// rewriting them.
const dropMatching = true

// rewriteLedger copies src to dst, applying edit to each decoded record. edit reports
// whether it matched; with dropMatching passed, a matched record is dropped instead of
// re-emitted. This is the "someone edited the append-only log" attack, performed on a
// copy so the source ledger stays intact for the other subtests.
func rewriteLedger(t *testing.T, src, dst string, edit func(map[string]json.RawMessage) bool, drop ...bool) {
	t.Helper()
	raw, err := os.ReadFile(src)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	dropIt := len(drop) > 0 && drop[0]
	var out bytes.Buffer
	matched := false
	for _, line := range bytes.Split(bytes.TrimSpace(raw), []byte("\n")) {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var rec map[string]json.RawMessage
		if err := json.Unmarshal(line, &rec); err != nil {
			t.Fatalf("decode ledger line: %v", err)
		}
		if edit(rec) {
			matched = true
			if dropIt {
				continue
			}
			re, err := json.Marshal(rec)
			if err != nil {
				t.Fatalf("re-encode ledger line: %v", err)
			}
			line = re
		}
		out.Write(line)
		out.WriteByte('\n')
	}
	if !matched {
		t.Fatal("rewriteLedger matched no record — the tamper did not happen")
	}
	if err := os.WriteFile(dst, out.Bytes(), 0o600); err != nil {
		t.Fatalf("write tampered ledger: %v", err)
	}
}

// openLedgerFile opens the ledger that owns the given ledger.jsonl path.
func openLedgerFile(t *testing.T, log string) *Ledger {
	t.Helper()
	l, err := Open(filepath.Dir(log))
	if err != nil {
		t.Fatalf("open tampered ledger: %v", err)
	}
	return l
}
