package rsl

import (
	"fmt"
	"os"
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

func makeLinearChain(n int) []Row {
	content := make([]Row, n)
	for i := 0; i < n; i++ {
		content[i] = Row{
			Ref:    "refs/heads/main",
			OldSHA: fmt.Sprintf("sha%04d", i),
			NewSHA: fmt.Sprintf("sha%04d", i+1),
		}
	}
	return mintChain(content)
}

// BenchmarkChainHash measures SHA-256 hash chaining of a single row.
func BenchmarkChainHash(b *testing.B) {
	row := Row{
		Seq:    42,
		Ref:    "refs/heads/main",
		OldSHA: "0123456789abcdef0123456789abcdef01234567",
		NewSHA: "abcdef0123456789abcdef0123456789abcdef01",
		Signer: "operator@example.com",
	}
	prev := "fedcba9876543210fedcba9876543210fedcba98"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = chainHash(prev, row)
	}
}

// BenchmarkVerify measures in-memory verification across chain lengths.
func BenchmarkVerify(b *testing.B) {
	for _, size := range []int{10, 100, 1000} {
		rows := makeLinearChain(size)
		b.Run(fmt.Sprintf("%d_rows", size), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				n, err := Verify(rows)
				if err != nil || n != size {
					b.Fatalf("Verify failed: n=%d err=%v", n, err)
				}
			}
		})
	}
}

// BenchmarkVerify_MultiRef measures in-memory verification of interleaved refs.
func BenchmarkVerify_MultiRef(b *testing.B) {
	const totalRows = 100
	refs := []string{"refs/heads/main", "refs/heads/release", "refs/heads/dev", "refs/heads/feature"}
	content := make([]Row, totalRows)
	head := make([]int, len(refs))
	for i := 0; i < totalRows; i++ {
		refIdx := i % len(refs)
		r := refs[refIdx]
		oldS := fmt.Sprintf("%s-%04d", r, head[refIdx])
		head[refIdx]++
		newS := fmt.Sprintf("%s-%04d", r, head[refIdx])
		content[i] = Row{Ref: r, OldSHA: oldS, NewSHA: newS}
	}
	rows := mintChain(content)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n, err := Verify(rows)
		if err != nil || n != totalRows {
			b.Fatalf("Verify multi-ref: n=%d err=%v", n, err)
		}
	}
}

// BenchmarkVerifyFile measures file-backed read and verification end-to-end.
func BenchmarkVerifyFile(b *testing.B) {
	path := filepath.Join(b.TempDir(), "bench_verify.jsonl")
	rows := makeLinearChain(100)
	for _, r := range rows {
		if _, err := Append(path, r); err != nil {
			b.Fatalf("setup Append: %v", err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n, err := VerifyFile(path)
		if err != nil || n != 100 {
			b.Fatalf("VerifyFile failed: n=%d err=%v", n, err)
		}
	}
}

// BenchmarkReadRows measures disk scanning and JSON unmarshaling.
func BenchmarkReadRows(b *testing.B) {
	for _, size := range []int{10, 100} {
		path := filepath.Join(b.TempDir(), fmt.Sprintf("read_%d.jsonl", size))
		rows := makeLinearChain(size)
		for _, r := range rows {
			if _, err := Append(path, r); err != nil {
				b.Fatalf("setup Append: %v", err)
			}
		}

		b.Run(fmt.Sprintf("%d_rows", size), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				read, err := ReadRows(path)
				if err != nil || len(read) != size {
					b.Fatalf("ReadRows failed: len=%d err=%v", len(read), err)
				}
			}
		})
	}
}

// BenchmarkRecoverHead measures chain head discovery over existing logs.
func BenchmarkRecoverHead(b *testing.B) {
	for _, size := range []int{10, 100, 1000} {
		path := filepath.Join(b.TempDir(), fmt.Sprintf("recover_%d.jsonl", size))
		rows := makeLinearChain(size)
		for _, r := range rows {
			if _, err := Append(path, r); err != nil {
				b.Fatalf("setup Append: %v", err)
			}
		}

		b.Run(fmt.Sprintf("%d_rows", size), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				seq, hash, err := recoverHead(path)
				if err != nil || seq != uint64(size) || hash == "" {
					b.Fatalf("recoverHead failed: seq=%d err=%v", seq, err)
				}
			}
		})
	}
}

// BenchmarkAppend measures appending an observed ref transition to disk.
func BenchmarkAppend(b *testing.B) {
	path := filepath.Join(b.TempDir(), "bench_append.jsonl")
	for i := 0; i < 5; i++ {
		if _, err := Append(path, Row{
			Ref:    "refs/heads/main",
			OldSHA: fmt.Sprintf("sha%04d", i),
			NewSHA: fmt.Sprintf("sha%04d", i+1),
		}); err != nil {
			b.Fatalf("seed Append: %v", err)
		}
	}
	fi, err := os.Stat(path)
	if err != nil {
		b.Fatalf("stat: %v", err)
	}
	baseSize := fi.Size()
	nextRow := Row{
		Ref:    "refs/heads/main",
		OldSHA: "sha0005",
		NewSHA: "sha0006",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Append(path, nextRow); err != nil {
			b.Fatalf("Append failed: %v", err)
		}
		if err := os.Truncate(path, baseSize); err != nil {
			b.Fatalf("truncate failed: %v", err)
		}
	}
}

// BenchmarkAppendSigned measures appending an attributed, signed transition.
func BenchmarkAppendSigned(b *testing.B) {
	path := filepath.Join(b.TempDir(), "bench_append_signed.jsonl")
	signer := stubSigner{id: "signer-key-1"}
	for i := 0; i < 5; i++ {
		if _, err := AppendSigned(path, Row{
			Ref:    "refs/heads/main",
			OldSHA: fmt.Sprintf("sha%04d", i),
			NewSHA: fmt.Sprintf("sha%04d", i+1),
		}, signer); err != nil {
			b.Fatalf("seed AppendSigned: %v", err)
		}
	}
	fi, err := os.Stat(path)
	if err != nil {
		b.Fatalf("stat: %v", err)
	}
	baseSize := fi.Size()
	nextRow := Row{
		Ref:    "refs/heads/main",
		OldSHA: "sha0005",
		NewSHA: "sha0006",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := AppendSigned(path, nextRow, signer); err != nil {
			b.Fatalf("AppendSigned failed: %v", err)
		}
		if err := os.Truncate(path, baseSize); err != nil {
			b.Fatalf("truncate failed: %v", err)
		}
	}
}
