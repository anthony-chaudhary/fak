package rsl

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

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
	rows := mintChain([]Row{
		{Ref: "refs/heads/main", OldSHA: "A", NewSHA: "B"},
		{Ref: "refs/heads/main", OldSHA: "B", NewSHA: "C"},
		{Ref: "refs/heads/main", OldSHA: "B", NewSHA: "X"},
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
	rows := mintChain([]Row{
		{Ref: "refs/heads/main", OldSHA: "A", NewSHA: "B"},
		{Ref: "refs/heads/main", OldSHA: "B", NewSHA: "C"},
		{Ref: "refs/heads/main", OldSHA: "C", NewSHA: "B"},
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

func TestVerify_BrokenPrevHashDetected(t *testing.T) {
	rows := ffOnlyHistory()
	rows[1].PrevHash = "broken_prev"
	n, err := Verify(rows)
	if err == nil {
		t.Fatalf("broken prev_hash must fail Verify")
	}
	if !strings.Contains(err.Error(), "broken chain") {
		t.Fatalf("error must indicate broken chain, got %q", err.Error())
	}
	if n != 1 {
		t.Fatalf("Verify should break at index 1, got n=%d", n)
	}
}

func TestVerify_SequenceGapDetected(t *testing.T) {
	rows := ffOnlyHistory()
	rows = append(rows[:1], rows[2:]...)
	if _, err := Verify(rows); err == nil || !strings.Contains(err.Error(), "sequence gap") {
		t.Fatalf("a dropped row must fail as a sequence gap, got err=%v", err)
	}
}

func TestVerify_IndependentRefsDoNotCollide(t *testing.T) {
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

type stubSigner struct{ id string }

func (s stubSigner) Identity() string { return s.id }
func (s stubSigner) Sign(hash string) (string, error) {
	return "sig:" + hash[:8], nil
}

type errSigner struct{}

func (errSigner) Identity() string { return "err-key" }
func (errSigner) Sign(hash string) (string, error) {
	return "", fmt.Errorf("signature failure")
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
	if _, err := VerifyFile(path); err != nil {
		t.Fatalf("a signed log must still verify its hash chain, got %v", err)
	}
}

func TestAppendSigned_SignError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rsl.jsonl")
	_, err := AppendSigned(path, Row{Ref: "refs/heads/main", OldSHA: "A", NewSHA: "B"}, errSigner{})
	if err == nil {
		t.Fatalf("expected error from failing signer")
	}
	if !strings.Contains(err.Error(), "signature failure") {
		t.Fatalf("expected signature failure in error message, got %v", err)
	}
}

func TestReadRows_NonExistentFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nonexistent.jsonl")
	rows, err := ReadRows(path)
	if err != nil || rows != nil {
		t.Fatalf("non-existent file must return nil, nil; got rows=%v err=%v", rows, err)
	}
}

func TestRecoverHeadAndReadRows_TornLine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "torn.jsonl")
	content := "{\"seq\":1,\"ref\":\"refs/heads/main\",\"old_sha\":\"A\",\"new_sha\":\"B\",\"hash\":\"h1\"}\n{\"seq\":2,\"ref\":\"refs/heads/main\","
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write torn file: %v", err)
	}
	rows, err := ReadRows(path)
	if err != nil {
		t.Fatalf("ReadRows with torn line should not error, got %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("want 1 well-formed row, got %d", len(rows))
	}
	seq, hash, validOffset, err := recoverHead(path)
	if err != nil {
		t.Fatalf("recoverHead with torn line should not error, got %v", err)
	}
	if seq != 1 || hash != "h1" {
		t.Fatalf("recoverHead: want seq=1 hash=h1, got seq=%d hash=%q", seq, hash)
	}
	if want := int64(len("{\"seq\":1,\"ref\":\"refs/heads/main\",\"old_sha\":\"A\",\"new_sha\":\"B\",\"hash\":\"h1\"}\n")); validOffset != want {
		t.Fatalf("recoverHead: want validOffset=%d, got %d", want, validOffset)
	}
}

func TestReadRows_CorruptedMiddleLineFails(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corrupted_middle.jsonl")
	content := strings.Join([]string{
		`{"seq":1,"ref":"refs/heads/main","old_sha":"A","new_sha":"B","hash":"h1"}`,
		`{"seq":2,"ref":"refs/heads/main",corrupted_json_row`,
		`{"seq":3,"ref":"refs/heads/main","old_sha":"B","new_sha":"C","hash":"h2"}`,
	}, "\n") + "\n"

	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write corrupted file: %v", err)
	}

	rows, err := ReadRows(path)
	if err == nil {
		t.Fatalf("ReadRows must fail on non-terminal corrupted line, but returned %d rows and err=nil", len(rows))
	}
	if !strings.Contains(err.Error(), "corrupted") {
		t.Fatalf("ReadRows error must indicate corruption, got %v", err)
	}
	if _, _, _, err := recoverHead(path); err == nil {
		t.Fatalf("recoverHead must fail on non-terminal corrupted line, but returned err=nil")
	} else if !strings.Contains(err.Error(), "corrupted") {
		t.Fatalf("recoverHead error must indicate corruption, got %v", err)
	}
	if _, err := VerifyFile(path); err == nil {
		t.Fatalf("VerifyFile must fail on non-terminal corrupted line, but returned err=nil")
	} else if !strings.Contains(err.Error(), "corrupted") {
		t.Fatalf("VerifyFile error must indicate corruption, got %v", err)
	}
}

func TestAppend_TornTailTruncatedOnRestart(t *testing.T) {
	t.Run("TornFinalLineWithoutNewline", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "torn_no_nl.jsonl")
		r1, err := Append(path, Row{Ref: "refs/heads/main", OldSHA: "A", NewSHA: "B"})
		if err != nil {
			t.Fatalf("Append row 1: %v", err)
		}
		if r1.Seq != 1 {
			t.Fatalf("want seq 1, got %d", r1.Seq)
		}

		// Simulate crash mid-write producing unparseable bytes without a trailing newline.
		tornTail := []byte(`{"seq":2,"ref":"refs/heads/main",`)
		f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			t.Fatalf("open for torn tail: %v", err)
		}
		if _, err := f.Write(tornTail); err != nil {
			f.Close()
			t.Fatalf("write torn tail: %v", err)
		}
		f.Close()

		// Calling Append should truncate the torn tail and append row 2 cleanly.
		r2, err := Append(path, Row{Ref: "refs/heads/main", OldSHA: "B", NewSHA: "C"})
		if err != nil {
			t.Fatalf("Append row 2 after torn tail: %v", err)
		}
		if r2.Seq != 2 {
			t.Fatalf("want seq 2, got %d", r2.Seq)
		}
		if r2.PrevHash != r1.Hash {
			t.Fatalf("want prev_hash %q, got %q", r1.Hash, r2.PrevHash)
		}

		// VerifyFile must succeed without corruption errors.
		n, err := VerifyFile(path)
		if err != nil {
			t.Fatalf("VerifyFile after restart append: %v", err)
		}
		if n != 2 {
			t.Fatalf("want 2 verified rows, got %d", n)
		}

		// ReadRows must succeed without corruption errors.
		rows, err := ReadRows(path)
		if err != nil {
			t.Fatalf("ReadRows after restart append: %v", err)
		}
		if len(rows) != 2 {
			t.Fatalf("want 2 rows, got %d", len(rows))
		}
		if rows[0].Seq != 1 || rows[1].Seq != 2 {
			t.Fatalf("unexpected sequence numbers: %v, %v", rows[0].Seq, rows[1].Seq)
		}
	})

	t.Run("TornFinalLineWithNewline", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "torn_with_nl.jsonl")
		r1, err := Append(path, Row{Ref: "refs/heads/main", OldSHA: "A", NewSHA: "B"})
		if err != nil {
			t.Fatalf("Append row 1: %v", err)
		}

		// Simulate crash leaving unparseable JSON followed by a newline.
		tornTail := []byte("{\"seq\":2,\"ref\":\"refs/heads/main\",corrupted_json\n")
		f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			t.Fatalf("open for torn tail: %v", err)
		}
		if _, err := f.Write(tornTail); err != nil {
			f.Close()
			t.Fatalf("write torn tail: %v", err)
		}
		f.Close()

		// Calling Append should truncate the torn tail and append row 2 cleanly.
		r2, err := Append(path, Row{Ref: "refs/heads/main", OldSHA: "B", NewSHA: "C"})
		if err != nil {
			t.Fatalf("Append row 2 after torn tail: %v", err)
		}
		if r2.Seq != 2 {
			t.Fatalf("want seq 2, got %d", r2.Seq)
		}
		if r2.PrevHash != r1.Hash {
			t.Fatalf("want prev_hash %q, got %q", r1.Hash, r2.PrevHash)
		}

		// VerifyFile and ReadRows must succeed without corruption errors.
		n, err := VerifyFile(path)
		if err != nil {
			t.Fatalf("VerifyFile: %v", err)
		}
		if n != 2 {
			t.Fatalf("want 2 verified rows, got %d", n)
		}
		rows, err := ReadRows(path)
		if err != nil {
			t.Fatalf("ReadRows: %v", err)
		}
		if len(rows) != 2 {
			t.Fatalf("want 2 rows, got %d", len(rows))
		}
	})

	t.Run("TornTailAtGenesis", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "torn_genesis.jsonl")
		if err := os.WriteFile(path, []byte(`{"seq":1,"ref":"broken`), 0o644); err != nil {
			t.Fatalf("write torn genesis file: %v", err)
		}

		r1, err := Append(path, Row{Ref: "refs/heads/main", OldSHA: "A", NewSHA: "B"})
		if err != nil {
			t.Fatalf("Append row 1 on torn genesis: %v", err)
		}
		if r1.Seq != 1 {
			t.Fatalf("want seq 1, got %d", r1.Seq)
		}

		n, err := VerifyFile(path)
		if err != nil {
			t.Fatalf("VerifyFile: %v", err)
		}
		if n != 1 {
			t.Fatalf("want 1 verified row, got %d", n)
		}
		rows, err := ReadRows(path)
		if err != nil {
			t.Fatalf("ReadRows: %v", err)
		}
		if len(rows) != 1 {
			t.Fatalf("want 1 row, got %d", len(rows))
		}
	})
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
				seq, hash, _, err := recoverHead(path)
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
