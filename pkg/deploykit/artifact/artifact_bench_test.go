package artifact

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

const benchCommit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

// benchBody returns a deterministic few-KB candidate payload.
func benchBody() []byte {
	body := make([]byte, 4<<10)
	for i := range body {
		body[i] = byte('a' + i%26)
	}
	return body
}

// benchCandidate writes body to dir/candidate and returns a fully valid Candidate describing
// it: positive generation, a 40-hex commit, the file's real SHA-256 and size.
func benchCandidate(b *testing.B, dir string, body []byte) Candidate {
	b.Helper()
	path := filepath.Join(dir, "candidate")
	if err := os.WriteFile(path, body, 0o755); err != nil {
		b.Fatal(err)
	}
	sum := sha256.Sum256(body)
	return Candidate{
		Kind:       KindFetch,
		Path:       path,
		Generation: 7,
		Commit:     benchCommit,
		Digest:     hex.EncodeToString(sum[:]),
		Size:       int64(len(body)),
		AppVersion: "0.55.0-bench",
	}
}

// These benchmarks cover the hermetic production paths of the facade: VerifyBytes and
// StoreSlot, which check bytes and move files but execute nothing. Acquire and Fetch are
// deliberately out of scope: fetch.acquire smokes the downloaded binary, which runs an
// arbitrary executable and is not hermetic.

// BenchmarkVerifyBytes measures the byte-identity gate over a few-KB candidate file: the size
// and SHA-256 of the bytes at the candidate path against the recorded identity. It executes
// nothing.
func BenchmarkVerifyBytes(b *testing.B) {
	dir := b.TempDir()
	body := benchBody()
	path := filepath.Join(dir, "candidate")
	if err := os.WriteFile(path, body, 0o644); err != nil {
		b.Fatal(err)
	}
	sum := sha256.Sum256(body)
	digest := hex.EncodeToString(sum[:])
	if err := VerifyBytes(path, digest, int64(len(body))); err != nil {
		b.Fatalf("fixture VerifyBytes: %v", err)
	}
	if err := VerifyBytes(path, digest, int64(len(body))+1); !errors.Is(err, ErrSizeMismatch) {
		b.Fatalf("fixture wrong-size VerifyBytes err = %v, want ErrSizeMismatch", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := VerifyBytes(path, digest, int64(len(body))); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkStoreSlot measures storing a verified candidate into its immutable generation+
// digest slot beside a temp-dir target: the candidate bytes are re-checked, then the existing
// slot is re-verified and reused. The first store creates the slot; every measured iteration
// exercises the reuse path.
func BenchmarkStoreSlot(b *testing.B) {
	dir := b.TempDir()
	target := filepath.Join(dir, "app")
	if err := os.WriteFile(target, []byte("installed"), 0o644); err != nil {
		b.Fatal(err)
	}
	c := benchCandidate(b, dir, benchBody())
	slot, err := StoreSlot(target, c)
	if err != nil {
		b.Fatalf("fixture StoreSlot: %v", err)
	}
	want := filepath.Join(dir, "app.self-update-slots", "g00000000000000000007-"+c.Digest, "app")
	if slot != want {
		b.Fatalf("fixture StoreSlot = %s, want %s", slot, want)
	}
	if err := VerifyBytes(slot, c.Digest, c.Size); err != nil {
		b.Fatalf("fixture stored artifact: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if got, err := StoreSlot(target, c); err != nil || got != slot {
			b.Fatalf("iteration %d: StoreSlot = (%s, %v), want (%s, nil)", i, got, err, slot)
		}
	}
}
