package logvault

// restore_test.go — the acceptance witness for the restore verb + drill (#2453):
// a restore that round-trips a source with 0 hash mismatches and a PASSING
// chained-journal re-verify (`fak audit verify` against the restored copy), the
// `--at` older-state reconstruction, the fail-closed postures (bitrot named, not
// silently copied; target discipline), and one drill invocation that journals a
// durable pass row. A backup nobody has restored from is a hypothesis; these
// tests are the restore path's standing proof.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// captureSourceWithJournal captures a source holding a plain log, a nested log,
// and a REAL hash-chained guard decision journal, so restore exercises the
// chained-journal re-verify path end to end. Returns the populated vault and the
// live source root.
func captureSourceWithJournal(t *testing.T) (*Vault, string) {
	t.Helper()
	srcDir := t.TempDir()
	writeFile(t, filepath.Join(srcDir, "loops.jsonl"), "row-a\nrow-b\n")
	writeFile(t, filepath.Join(srcDir, "sub", "notes.log"), "note\n")

	jpath := filepath.Join(srcDir, "guard-audit.jsonl")
	j, err := journal.Open(jpath)
	if err != nil {
		t.Fatal(err)
	}
	j.AppendAgentEvent("AGENT_SPAWN", "agent-1", "launch")
	j.AppendAgentEvent("AGENT_EXIT", "agent-1", "done")
	if err := j.Close(); err != nil {
		t.Fatal(err)
	}
	// The journal is sound at the SOURCE before the vault ever sees it.
	if n, err := journal.Verify(jpath); err != nil || n != 2 {
		t.Fatalf("source journal verify: n=%d err=%v, want 2 rows clean", n, err)
	}

	v := testVault(t, Source{ID: "s", Root: srcDir})
	if _, err := v.Capture(); err != nil {
		t.Fatal(err)
	}
	return v, srcDir
}

func TestRestoreRoundTripsCleanWithVerifiedJournal(t *testing.T) {
	v, srcDir := captureSourceWithJournal(t)
	to := filepath.Join(t.TempDir(), "restore-out")

	rep, err := v.Restore(RestoreOptions{Source: "s", To: to})
	if err != nil {
		t.Fatalf("restore: %v", err)
	}
	if !rep.OK() {
		t.Fatalf("restore not clean: problems=%v journal-failures=%d", rep.Problems, rep.JournalFailures())
	}
	if rep.Files != 3 {
		t.Fatalf("restored files=%d, want 3", rep.Files)
	}

	// Byte-for-byte identity against the live source — 0 hash mismatches means the
	// restored tree IS the source, not merely "some bytes".
	for _, rel := range []string{"loops.jsonl", "sub/notes.log", "guard-audit.jsonl"} {
		want := readFile(t, filepath.Join(srcDir, filepath.FromSlash(rel)))
		got := readFile(t, filepath.Join(to, filepath.FromSlash(rel)))
		if got != want {
			t.Fatalf("restored %s mismatch:\n got %q\nwant %q", rel, got, want)
		}
	}

	// The restored guard journal re-verified end to end during restore.
	var jc *JournalCheck
	for i := range rep.Journals {
		if rep.Journals[i].RelPath == "guard-audit.jsonl" {
			jc = &rep.Journals[i]
		}
	}
	if jc == nil {
		t.Fatal("restore did not re-verify the restored guard journal")
	}
	if jc.Kind != "decision-journal" || jc.Err != "" || jc.Rows != 2 {
		t.Fatalf("journal check = %+v, want a clean decision-journal of 2 rows", *jc)
	}

	// The literal acceptance bar: `fak audit verify` (journal.Verify) passes
	// against the RESTORED copy, proving the chain is sound after the round-trip.
	if n, err := journal.Verify(filepath.Join(to, "guard-audit.jsonl")); err != nil || n != 2 {
		t.Fatalf("restored journal verify: n=%d err=%v, want 2 rows clean", n, err)
	}
}

func TestRestoreAtReconstructsOlderPrefixState(t *testing.T) {
	srcDir := t.TempDir()
	log := filepath.Join(srcDir, "loops.jsonl")
	writeFile(t, log, "row-a\nrow-b\n")
	v := testVault(t, Source{ID: "s", Root: srcDir})
	if _, err := v.Capture(); err != nil {
		t.Fatal(err)
	}
	rows, err := ReadManifestRows(filepath.Join(v.Dir, ManifestName))
	if err != nil {
		t.Fatal(err)
	}
	seq1 := rows[len(rows)-1].Seq

	// Append-grow the log and re-capture: the mirror now holds all three rows.
	f, _ := os.OpenFile(log, os.O_APPEND|os.O_WRONLY, 0o644)
	f.WriteString("row-c\n")
	f.Close()
	future := time.Now().Add(2 * time.Second)
	os.Chtimes(log, future, future)
	if _, err := v.Capture(); err != nil {
		t.Fatal(err)
	}

	// Head restore: the current (grown) content.
	toNew := filepath.Join(t.TempDir(), "new")
	repNew, err := v.Restore(RestoreOptions{Source: "s", To: toNew})
	if err != nil || !repNew.OK() {
		t.Fatalf("head restore: err=%v ok=%v", err, repNew.OK())
	}
	if got := readFile(t, filepath.Join(toNew, "loops.jsonl")); got != "row-a\nrow-b\nrow-c\n" {
		t.Fatalf("head restore = %q, want all three rows", got)
	}

	// --at seq1: the older PREFIX, re-hash-verified against the chain (not the
	// current mirror content).
	toOld := filepath.Join(t.TempDir(), "old")
	repOld, err := v.Restore(RestoreOptions{Source: "s", To: toOld, At: seq1})
	if err != nil || !repOld.OK() {
		t.Fatalf("at-seq restore: err=%v ok=%v problems=%v", err, repOld.OK(), repOld.Problems)
	}
	if repOld.HeadSeq != seq1 {
		t.Fatalf("HeadSeq = %d, want %d", repOld.HeadSeq, seq1)
	}
	if got := readFile(t, filepath.Join(toOld, "loops.jsonl")); got != "row-a\nrow-b\n" {
		t.Fatalf("at-seq restore = %q, want the older 2-row prefix", got)
	}
}

func TestRestoreReportsUnprovableFileFailClosed(t *testing.T) {
	v, _ := captureSourceWithJournal(t)
	// Bitrot the mirror the way silent corruption would: same byte length,
	// different content. The manifest CHAIN still verifies (it attests the
	// ORIGINAL hash), but the mirror no longer re-hashes to it — restore must
	// name the file as a problem, never copy unproven bytes and call it done.
	writeFile(t, v.mirrorPath("s", "loops.jsonl"), "ROW-A\nrow-b\n")
	to := filepath.Join(t.TempDir(), "out")
	rep, err := v.Restore(RestoreOptions{Source: "s", To: to})
	if err != nil {
		t.Fatalf("restore: %v", err)
	}
	if rep.OK() {
		t.Fatal("restore over a bitrotten mirror must not report OK")
	}
	named := false
	for _, p := range rep.Problems {
		if p.RelPath == "loops.jsonl" {
			named = true
		}
	}
	if !named {
		t.Fatalf("expected loops.jsonl named as an unprovable file, got %v", rep.Problems)
	}
}

func TestRestoreTargetDiscipline(t *testing.T) {
	v, _ := captureSourceWithJournal(t)

	// Into the vault itself: always refused — Force does not override overlap.
	if _, err := v.Restore(RestoreOptions{Source: "s", To: v.Dir, Force: true}); err == nil {
		t.Fatal("restore into the vault must be refused even with Force")
	}

	// A non-empty existing target without Force: refused (fresh-directory contract).
	occupied := t.TempDir()
	writeFile(t, filepath.Join(occupied, "keep.txt"), "x\n")
	if _, err := v.Restore(RestoreOptions{Source: "s", To: occupied}); err == nil {
		t.Fatal("restore over a non-empty tree without Force must be refused")
	}
	// With the explicit Force grant it proceeds and round-trips clean.
	rep, err := v.Restore(RestoreOptions{Source: "s", To: occupied, Force: true})
	if err != nil || !rep.OK() {
		t.Fatalf("forced restore: err=%v ok=%v", err, rep.OK())
	}

	// An uncaptured source errors and names the captured set.
	if _, err := v.Restore(RestoreOptions{Source: "nope", To: filepath.Join(t.TempDir(), "x")}); err == nil {
		t.Fatal("restore of an uncaptured source must error")
	}
}

func TestDrillRestoresVerifiesAndJournalsPassRow(t *testing.T) {
	v, _ := captureSourceWithJournal(t)
	ledger := filepath.Join(t.TempDir(), "nightrun", "logvault-drill.jsonl")

	row, rep, err := v.Drill("", ledger) // "" picks the smallest captured source
	if err != nil {
		t.Fatalf("drill: %v", err)
	}
	if !row.Pass || !rep.OK() {
		t.Fatalf("drill did not pass: row=%+v problems=%v", row, rep.Problems)
	}
	if row.Source != "s" {
		t.Fatalf("drill source = %q, want s", row.Source)
	}
	if row.JournalsChecked < 1 || row.JournalsFailed != 0 {
		t.Fatalf("drill journals: checked=%d failed=%d, want >=1 checked / 0 failed", row.JournalsChecked, row.JournalsFailed)
	}

	// The durable drill row landed BOTH in the vault drill-log and the repo ledger.
	for _, p := range []string{filepath.Join(v.Dir, DrillLogName), ledger} {
		body := readFile(t, p)
		if !strings.Contains(body, DrillSchema) || !strings.Contains(body, `"pass":true`) {
			t.Fatalf("drill row missing / not-pass in %s: %q", p, body)
		}
	}
}

func TestSafeRestoreRelRefusesTraversal(t *testing.T) {
	refused := []string{"", "/etc/passwd", "../x", "a/../../b", "a/../b", "./x", `a\b`, `C:\x`, "a/b:c", "a/../"}
	for _, r := range refused {
		if safeRestoreRel(r) {
			t.Errorf("safeRestoreRel(%q) = true, want refused", r)
		}
	}
	allowed := []string{"a.jsonl", "sub/notes.log", "a/b/c.txt", "guard-audit.jsonl"}
	for _, r := range allowed {
		if !safeRestoreRel(r) {
			t.Errorf("safeRestoreRel(%q) = false, want allowed", r)
		}
	}
}
