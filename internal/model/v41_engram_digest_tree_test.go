package model

import (
	"bytes"
	"errors"
	"io"
	"math/rand"
	"strings"
	"testing"
)

// v41EngramEncodedFake serves rows from ReaderAt and names encoded as the
// extent they decode from, with an optional encoding name and size skew.
type v41EngramEncodedFake struct {
	io.ReaderAt
	encoded  []byte
	encoding string
	sizeSkew int64
}

func (f v41EngramEncodedFake) V41EngramEncodedRows(start, count int) (io.ReaderAt, int64, int64, string, error) {
	return bytes.NewReader(f.encoded), int64(start) * v41EngramQ2KEncodedRowBytes,
		int64(count)*v41EngramQ2KEncodedRowBytes + f.sizeSkew, f.encoding, nil
}

// TestV41EngramTreeDigestPinsScheme pins scheme sha256t1 to golden values (a
// leaf-size, domain or root-layout change must take a new scheme tag), proves
// the digest is independent of the pool size, binds every byte and the
// encoding name, and reports the lowest failing leaf's read error.
// fak-test:runtime fast est=1s lane=default
func TestV41EngramTreeDigestPinsScheme(t *testing.T) {
	const leaf = v41EngramTreeLeaf
	data := make([]byte, 2*leaf+3)
	for i := range data {
		data[i] = byte(i % 251)
	}
	golden := map[string]string{
		v41EngramEncodingRows: "sha256t1-rows:ff0ff7344bc1387327954ae907c6d4ae97df66b90a28ad4f74b8d6b65166d1cf",
		V41EngramEncodingQ2K:  "sha256t1-q2_k:9cfb5be6dba3078bb3cca463a7d73de0bfbd0d2731e97b61a71350cf5e8a5b1f",
	}
	for encoding, want := range golden {
		for _, workers := range []int{1, 2, 3, 16} {
			got, err := v41EngramTreeDigest(bytes.NewReader(data), 0, int64(len(data)), encoding, workers)
			if err != nil || got != want {
				t.Fatalf("encoding=%s workers=%d = %q, %v; want %q", encoding, workers, got, err, want)
			}
		}
	}
	if flat := v41EngramTestDigest(data); flat == golden[v41EngramEncodingRows] {
		t.Fatal("tree digest equals the flat digest of the same bytes")
	}

	flipped := append([]byte(nil), data...)
	flipped[len(flipped)-1] ^= 1
	if got, _ := v41EngramTreeDigest(bytes.NewReader(flipped), 0, int64(len(flipped)), v41EngramEncodingRows, 4); got == golden[v41EngramEncodingRows] {
		t.Fatal("one flipped byte in the short last leaf left the tree digest unchanged")
	}
	shifted, _ := v41EngramTreeDigest(bytes.NewReader(data), 1, int64(len(data))-1, v41EngramEncodingRows, 4)
	if shifted == golden[v41EngramEncodingRows] {
		t.Fatal("a shifted extent produced the same tree digest")
	}

	failing := v41EngramFailAtReader{inner: bytes.NewReader(data), failAt: leaf + 10}
	for _, workers := range []int{1, 3} {
		if _, err := v41EngramTreeDigest(failing, 0, int64(len(data)), v41EngramEncodingRows, workers); !errors.Is(err, errV41EngramAcquireTestRead) {
			t.Fatalf("workers=%d read failure = %v, want the injected error", workers, err)
		}
	}
	short := io.NewSectionReader(bytes.NewReader(data), 0, 2*leaf)
	if _, err := v41EngramTreeDigest(short, 0, int64(len(data)), v41EngramEncodingRows, 3); !errors.Is(err, io.EOF) {
		t.Fatalf("short extent = %v, want a short-read refusal", err)
	}
	if _, err := v41EngramTreeDigest(bytes.NewReader(data), 0, 0, v41EngramEncodingRows, 1); err == nil {
		t.Fatal("empty extent was digested")
	}
}

// TestV41EngramBindingDigestSchemes proves the verifier recomputes the scheme a
// binding names: flat and rows-tree bindings of the same bytes both admit, a
// value of one scheme under another scheme's tag is refused, an encoded-tree
// binding is refused for a reader that names no encoded rows, and an unknown or
// malformed scheme is refused as invalid before any byte is read.
// fak-test:runtime fast est=1s lane=default
func TestV41EngramBindingDigestSchemes(t *testing.T) {
	const rows = 40
	f32 := make([]byte, rows*V41EngramF32RowBytes)
	rand.New(rand.NewSource(13668)).Read(f32)
	size := int64(len(f32))
	shard := V41EngramShard{ID: "blk.1.engram_embd.weight", ExpectedID: "blk.1.engram_embd.weight", Size: size, Rows: rows}
	bind := V41EngramArtifactBinding{Shard: shard.ID, Size: size, Rows: rows, RowBytes: V41EngramF32RowBytes}
	flat := v41EngramTestDigest(f32)
	tree, err := v41EngramTreeDigest(bytes.NewReader(f32), 0, size, v41EngramEncodingRows, 1)
	if err != nil {
		t.Fatal(err)
	}
	flatHex := flat[strings.IndexByte(flat, ':')+1:]

	verify := func(digest string) error {
		b := bind
		b.Digest = digest
		_, err := NewV41VerifiedEngramRowSource(bytes.NewReader(f32), shard, b)
		return err
	}
	for _, d := range []string{flat, tree, "  " + strings.ToUpper(tree) + "\n"} {
		if err := verify(d); err != nil {
			t.Fatalf("digest %q refused: %v", d, err)
		}
	}
	var typed *V41EngramStreamError
	for _, d := range []string{"sha256t1-rows:" + flatHex, "sha256t1-q2_k:" + flatHex} {
		if err := verify(d); !errors.As(err, &typed) || typed.Kind != V41EngramStreamBindingMismatch {
			t.Fatalf("digest %q = %v, want a binding mismatch", d, err)
		}
	}
	for _, d := range []string{"sha256t2-rows:" + flatHex, "sha256t1-q9_k:" + flatHex, "md5:" + flatHex,
		"sha256t1-" + flatHex, flat[:len(flat)-1], "sha256:" + strings.Repeat("g", 64), ""} {
		counting := &v41EngramCountingReader{inner: bytes.NewReader(f32)}
		b := bind
		b.Digest = d
		if _, err := NewV41VerifiedEngramRowSource(counting, shard, b); !errors.As(err, &typed) || typed.Kind != V41EngramStreamGeometry {
			t.Fatalf("digest %q = %v, want an invalid-binding refusal", d, err)
		}
		if n := counting.reads.Load(); n != 0 {
			t.Fatalf("digest %q: refusal read %d chunks, want 0", d, n)
		}
	}
}

// TestV41AcquiredEngramBindsEncodedExtent proves acquisition over a reader that
// names its encoded rows binds a tree over exactly those bytes without reading a
// single served (decoded) row, the verifier admits it only for the same encoded
// bytes, a flat binding of the decoded rows stays verifiable, and an unknown
// encoding or a mis-sized encoded extent is refused.
// fak-test:runtime fast est=1s lane=default
func TestV41AcquiredEngramBindsEncodedExtent(t *testing.T) {
	const rows = 3000
	rng := rand.New(rand.NewSource(84))
	f32 := make([]byte, rows*V41EngramF32RowBytes)
	rng.Read(f32)
	encoded := make([]byte, rows*v41EngramQ2KEncodedRowBytes)
	rng.Read(encoded)
	size := int64(len(f32))
	shard := V41EngramShard{ID: "blk.1.engram_embd.weight", ExpectedID: "blk.1.engram_embd.weight", Size: size, Rows: rows}
	bind := V41EngramArtifactBinding{Shard: shard.ID, Size: size, Rows: rows, RowBytes: V41EngramF32RowBytes}

	served := &v41EngramCountingReader{inner: bytes.NewReader(f32)}
	src, err := NewV41AcquiredEngramRowSource(v41EngramEncodedFake{ReaderAt: served, encoded: encoded, encoding: V41EngramEncodingQ2K}, shard, bind)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	if n := served.reads.Load(); n != 0 {
		t.Fatalf("acquisition read %d served-row chunks, want 0 (the encoded extent is bound)", n)
	}
	got, _ := V41EngramBinding(src)
	want, err := v41EngramTreeDigest(bytes.NewReader(encoded), 0, int64(len(encoded)), V41EngramEncodingQ2K, 1)
	if err != nil || got.Digest != want {
		t.Fatalf("acquired digest %q, %v; want the q2_k tree over the encoded bytes %q", got.Digest, err, want)
	}

	verify := func(r io.ReaderAt, digest string) error {
		b := bind
		b.Digest = digest
		_, err := NewV41VerifiedEngramRowSource(r, shard, b)
		return err
	}
	same := v41EngramEncodedFake{ReaderAt: bytes.NewReader(f32), encoded: encoded, encoding: V41EngramEncodingQ2K}
	if err := verify(same, want); err != nil {
		t.Fatalf("verified route refuses the encoded binding acquisition produced: %v", err)
	}
	if err := verify(same, v41EngramTestDigest(f32)); err != nil {
		t.Fatalf("verified route refuses a flat binding of the decoded rows: %v", err)
	}
	var typed *V41EngramStreamError
	other := append([]byte(nil), encoded...)
	other[len(other)/2] ^= 0x40
	if err := verify(v41EngramEncodedFake{ReaderAt: bytes.NewReader(f32), encoded: other, encoding: V41EngramEncodingQ2K}, want); !errors.As(err, &typed) || typed.Kind != V41EngramStreamBindingMismatch {
		t.Fatalf("different encoded bytes under the same served rows = %v, want a binding mismatch", err)
	}
	for name, bad := range map[string]v41EngramEncodedFake{
		"unknown encoding": {ReaderAt: bytes.NewReader(f32), encoded: encoded, encoding: "q3_k"},
		"mis-sized extent": {ReaderAt: bytes.NewReader(f32), encoded: encoded, encoding: V41EngramEncodingQ2K, sizeSkew: 1},
	} {
		if _, err := NewV41AcquiredEngramRowSource(bad, shard, bind); !errors.As(err, &typed) || typed.Kind != V41EngramStreamBindingMismatch {
			t.Fatalf("%s: acquire = %v, want a binding mismatch", name, err)
		}
		if err := verify(bad, want); !errors.As(err, &typed) || typed.Kind != V41EngramStreamBindingMismatch {
			t.Fatalf("%s: verify = %v, want a binding mismatch", name, err)
		}
	}
}
