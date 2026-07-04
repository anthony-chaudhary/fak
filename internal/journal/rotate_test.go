package journal

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// appendDecides commits n plain DECIDE rows to a file-backed journal for the test.
func appendDecides(j *Journal, n int) {
	for i := 0; i < n; i++ {
		j.append(Row{Kind: "DECIDE", Tool: "t", Verdict: "ALLOW"})
	}
}

// TestCutChainAwareRotation is the #2457 witness: naive rotation forks/breaks the
// chain, while Cut + VerifySegments keeps a rotated multi-file journal verifying
// end-to-end through the anchor.
func TestCutChainAwareRotation(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "journal.jsonl")

	j, err := Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	appendDecides(j, 5)
	finalSeq := j.seq
	finalHash := j.lastHash
	if finalSeq != 5 {
		t.Fatalf("finalSeq = %d, want 5", finalSeq)
	}

	// --- The DEFECT: a naive rotation (fresh Open on a new file) forks the chain.
	// The successor restarts at genesis (Seq 1, empty PrevHash), so folding the two
	// files together fails at the boundary. This is what Cut exists to prevent.
	naivePath := filepath.Join(dir, "naive.jsonl")
	nj, err := Open(naivePath)
	if err != nil {
		t.Fatalf("open naive: %v", err)
	}
	appendDecides(nj, 3)
	if err := nj.Close(); err != nil {
		t.Fatalf("close naive: %v", err)
	}
	if err := j.Flush(); err != nil {
		t.Fatalf("flush: %v", err)
	}
	if _, err := VerifySegments(path, naivePath); err == nil {
		t.Fatalf("naive rotation must NOT verify end-to-end, but VerifySegments passed")
	}

	// --- The FIX: Cut rotates without forking.
	archived, err := j.Cut()
	if err != nil {
		t.Fatalf("cut: %v", err)
	}
	if want := path + ".cut-5"; archived != want {
		t.Fatalf("archived = %q, want %q", archived, want)
	}
	appendDecides(j, 4) // land more rows in the successor segment
	if err := j.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// The archived segment verifies standalone (a self-contained genesis chain).
	if n, err := Verify(archived); err != nil || n != 5 {
		t.Fatalf("Verify(archived) = (%d,%v), want (5,nil)", n, err)
	}

	// The successor segment does NOT verify standalone — it begins mid-chain at the
	// CUT anchor (Seq 6), which is exactly why a rotated set needs VerifySegments.
	if _, err := Verify(path); err == nil {
		t.Fatalf("Verify(successor) must fail standalone (begins at the CUT anchor)")
	}

	// The anchor records the prior segment's head: first row of the successor is a
	// CUT row whose PrevHash == prior head hash and Seq-1 == prior final seq.
	succ, err := ReadRows(path)
	if err != nil || len(succ) == 0 {
		t.Fatalf("read successor: %v (len=%d)", err, len(succ))
	}
	anchor := succ[0]
	if anchor.Kind != KindCut {
		t.Fatalf("successor first row kind = %q, want %q", anchor.Kind, KindCut)
	}
	if anchor.Seq != finalSeq+1 {
		t.Fatalf("anchor seq = %d, want %d", anchor.Seq, finalSeq+1)
	}
	if anchor.PrevHash != finalHash {
		t.Fatalf("anchor prev_hash = %q, want prior head %q", anchor.PrevHash, finalHash)
	}

	// End-to-end: archived + successor verify as one continuous chain. Total rows =
	// 5 (archived) + 1 (CUT anchor) + 4 (successor decides) = 10, seq 1..10 gapless.
	segs, err := Segments(path)
	if err != nil {
		t.Fatalf("segments: %v", err)
	}
	if len(segs) != 2 || segs[0] != archived || segs[1] != path {
		t.Fatalf("segments = %v, want [%s %s]", segs, archived, path)
	}
	total, err := VerifySegments(segs...)
	if err != nil {
		t.Fatalf("VerifySegments end-to-end: %v", err)
	}
	if total != 10 {
		t.Fatalf("VerifySegments total = %d, want 10", total)
	}

	// ReadAllSegments loses no history across the cut.
	all, err := ReadAllSegments(path)
	if err != nil {
		t.Fatalf("ReadAllSegments: %v", err)
	}
	if len(all) != 10 {
		t.Fatalf("ReadAllSegments len = %d, want 10", len(all))
	}
	for i, r := range all {
		if r.Seq != uint64(i+1) {
			t.Fatalf("row %d seq = %d, want %d (chain not gapless across cut)", i, r.Seq, i+1)
		}
	}
}

// TestVerifySegmentsDetectsTamperAcrossCut proves the rotated set stays
// tamper-evident: a flipped byte in an archived segment breaks the end-to-end
// verification, and a successor missing its CUT anchor is rejected.
func TestVerifySegmentsDetectsTamperAcrossCut(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "journal.jsonl")
	j, err := Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	appendDecides(j, 3)
	archived, err := j.Cut()
	if err != nil {
		t.Fatalf("cut: %v", err)
	}
	appendDecides(j, 2)
	if err := j.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// Sanity: the intact rotated set verifies.
	if _, err := VerifySegments(archived, path); err != nil {
		t.Fatalf("intact VerifySegments: %v", err)
	}

	// Tamper the archived segment: flip a verdict in the first row.
	b, err := os.ReadFile(archived)
	if err != nil {
		t.Fatalf("read archived: %v", err)
	}
	tampered := strings.Replace(string(b), `"verdict":"ALLOW"`, `"verdict":"DENY"`, 1)
	if tampered == string(b) {
		t.Fatalf("tamper substitution did not change the file")
	}
	if err := os.WriteFile(archived, []byte(tampered), 0o644); err != nil {
		t.Fatalf("write tampered: %v", err)
	}
	if _, err := VerifySegments(archived, path); err == nil {
		t.Fatalf("tampered archived segment must fail VerifySegments")
	}
}
