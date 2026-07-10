package rsl

import (
	"path/filepath"
	"strings"
	"testing"
)

// mintChain stamps a sound, verifiable hash chain over the given content rows
// (Seq/PrevHash/Hash), the fixture builder for the pure Verify tests — the same
// shape cmd/fak's mintChain uses for the decision journal.
func mintChain(content []Row) []Row {
	out := make([]Row, 0, len(content))
	prev := ""
	for i, r := range content {
		r.Seq = uint64(i + 1)
		r.PrevHash = prev
		r.Hash = chainHash(prev, r)
		out = append(out, r)
		prev = r.Hash
	}
	return out
}

// ffOnlyHistory is a fast-forward-only trunk history: each transition continues
// the recorded head (old_sha == the prior new_sha) and never revisits a target.
func ffOnlyHistory() []Row {
	return mintChain([]Row{
		{Ref: "refs/heads/main", OldSHA: "A", NewSHA: "B"},
		{Ref: "refs/heads/main", OldSHA: "B", NewSHA: "C"},
		{Ref: "refs/heads/main", OldSHA: "C", NewSHA: "D"},
	})
}

func TestVerify_FastForwardOnlyHistoryPasses(t *testing.T) {
	n, err := Verify(ffOnlyHistory())
	if err != nil {
		t.Fatalf("fast-forward-only history must pass Verify, got err=%v", err)
	}
	if n != 3 {
		t.Fatalf("want 3 rows checked, got %d", n)
	}
}

func TestVerify_ForcePushGapFailsNamingRef(t *testing.T) {
	// A force-push: main advanced A->B->C, then was reset back to B and moved to X.
	// The forced transition's old_sha (B) no longer continues the recorded head (C).
	rows := mintChain([]Row{
		{Ref: "refs/heads/main", OldSHA: "A", NewSHA: "B"},
		{Ref: "refs/heads/main", OldSHA: "B", NewSHA: "C"},
		{Ref: "refs/heads/main", OldSHA: "B", NewSHA: "X"}, // non-ff: old_sha=B, head=C
	})
	n, err := Verify(rows)
	if err == nil {
		t.Fatalf("a non-fast-forward gap must fail Verify")
	}
	if !strings.Contains(err.Error(), "refs/heads/main") {
		t.Fatalf("error must name the offending ref, got %q", err.Error())
	}
	if !strings.Contains(err.Error(), "non-fast-forward") {
		t.Fatalf("error must classify the gap as non-fast-forward, got %q", err.Error())
	}
	if n != 2 {
		t.Fatalf("Verify should fail at the 3rd row (index 2), got n=%d", n)
	}
}

func TestVerify_RewindToPriorTargetFailsNamingRef(t *testing.T) {
	// A force-push that honestly records old_sha == head but rewinds new_sha back to
	// a commit the ref already held (B): contiguous, yet not fast-forward.
	rows := mintChain([]Row{
		{Ref: "refs/heads/main", OldSHA: "A", NewSHA: "B"},
		{Ref: "refs/heads/main", OldSHA: "B", NewSHA: "C"},
		{Ref: "refs/heads/main", OldSHA: "C", NewSHA: "B"}, // rewind: revisits prior target B
	})
	_, err := Verify(rows)
	if err == nil {
		t.Fatalf("a rewind to a prior target must fail Verify")
	}
	if !strings.Contains(err.Error(), "refs/heads/main") || !strings.Contains(err.Error(), "non-fast-forward") {
		t.Fatalf("error must name the ref and classify non-ff, got %q", err.Error())
	}
}

func TestVerify_TamperedRowDetected(t *testing.T) {
	// A sound chain whose middle row's new_sha is edited AFTER the hashes were
	// stamped: the hash no longer recomputes, so the chain breaks at that row.
	rows := ffOnlyHistory()
	rows[1].NewSHA = "TAMPERED"
	n, err := Verify(rows)
	if err == nil {
		t.Fatalf("a tampered row must fail Verify")
	}
	if !strings.Contains(err.Error(), "tampered") {
		t.Fatalf("error must name the tamper, got %q", err.Error())
	}
	if n != 1 {
		t.Fatalf("Verify should break at the edited row (index 1), got n=%d", n)
	}
}

func TestVerify_SequenceGapDetected(t *testing.T) {
	rows := ffOnlyHistory()
	rows = append(rows[:1], rows[2:]...) // drop the 2nd row: seq jumps 1 -> 3
	if _, err := Verify(rows); err == nil || !strings.Contains(err.Error(), "sequence gap") {
		t.Fatalf("a dropped row must fail as a sequence gap, got err=%v", err)
	}
}

func TestVerify_IndependentRefsDoNotCollide(t *testing.T) {
	// Two refs interleaved in one log: each is fast-forward-only on its own, so the
	// per-ref invariant must not conflate B(main) with B(release).
	rows := mintChain([]Row{
		{Ref: "refs/heads/main", OldSHA: "A", NewSHA: "B"},
		{Ref: "refs/heads/release", OldSHA: "P", NewSHA: "Q"},
		{Ref: "refs/heads/main", OldSHA: "B", NewSHA: "C"},
		{Ref: "refs/heads/release", OldSHA: "Q", NewSHA: "R"},
	})
	if _, err := Verify(rows); err != nil {
		t.Fatalf("two independent fast-forward refs must both pass, got %v", err)
	}
}

func TestVerify_EmptyLogIsSound(t *testing.T) {
	if n, err := Verify(nil); err != nil || n != 0 {
		t.Fatalf("empty log is trivially sound, got n=%d err=%v", n, err)
	}
}

func TestAppendThenVerifyRoundTrips(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rsl.jsonl")
	for _, tr := range [][2]string{{"A", "B"}, {"B", "C"}, {"C", "D"}} {
		if _, err := Append(path, Row{Ref: "refs/heads/main", OldSHA: tr[0], NewSHA: tr[1]}); err != nil {
			t.Fatalf("Append(%v): %v", tr, err)
		}
	}
	n, err := VerifyFile(path)
	if err != nil {
		t.Fatalf("a log built by Append must verify, got %v", err)
	}
	if n != 3 {
		t.Fatalf("want 3 appended rows, got %d", n)
	}
}

// stubSigner is a deterministic Signer for the seam test — no real key, just
// proof AppendSigned attributes and signs and that the signed row still verifies.
type stubSigner struct{ id string }

func (s stubSigner) Identity() string { return s.id }
func (s stubSigner) Sign(hash string) (string, error) {
	return "sig:" + hash[:8], nil
}

func TestAppendSigned_AttributesAndStillVerifies(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rsl.jsonl")
	row, err := AppendSigned(path, Row{Ref: "refs/heads/main", OldSHA: "A", NewSHA: "B"}, stubSigner{id: "op-key-1"})
	if err != nil {
		t.Fatalf("AppendSigned: %v", err)
	}
	if row.Signer != "op-key-1" || row.Sig == "" {
		t.Fatalf("signed row must carry identity + signature, got signer=%q sig=%q", row.Signer, row.Sig)
	}
	// The signature is over the hash and NOT part of the pre-image, so the chain
	// still verifies with the attribution chained in.
	if _, err := VerifyFile(path); err != nil {
		t.Fatalf("a signed log must still verify its hash chain, got %v", err)
	}
}
