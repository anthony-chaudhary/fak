package model

// v41_engram_digest_tree.go — the versioned content-address schemes a V4.1
// Engram artifact binding carries (fak#13668).
//
// The flat "sha256:" scheme hashes the served row bytes as one stream. For the
// published Q2_K table the served rows are the f32 expansion, ~12x the
// checkpoint bytes, so binding two tables ran the whole 172 s strix3 load on
// ~3 cores (sha256 70-75%, Q2_K dequant 22-28%). Acquisition now binds a tree
// digest instead: fixed-size leaves hash on a bounded pool and a domain-separated
// root commits to the encoding, the extent size, the leaf size and every leaf in
// extent order. When the reader names the encoded bytes its rows decode from
// (V41EngramEncodedRows), the tree covers exactly those bytes, so a Q2_K table is
// hashed at checkpoint size and never dequantized to be bound. The served rows
// are a deterministic decode of the bound bytes, so the binding still pins them.
//
// The scheme is part of the digest string. The verifier recomputes the scheme a
// binding declares, so a value of one scheme is never compared against a value
// of another, and a flat "sha256:" binding stays verifiable. Changing the leaf
// size, the domain or the root layout requires a new scheme tag.

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
)

// V41EngramEncodingQ2K names Q2_K super-blocks: one 84-byte block per 256-wide
// Engram row ([16 scale][64 q][f16 d][f16 min]).
const V41EngramEncodingQ2K = "q2_k"

const (
	v41EngramDigestFlat       = "sha256"
	v41EngramDigestTreePrefix = "sha256t1-"
	v41EngramEncodingRows     = "rows"

	v41EngramTreeDomain     = "fak.v41.engram.tree1\x00"
	v41EngramTreeLeaf       = 1 << 20
	v41EngramTreeMaxWorkers = 16

	v41EngramQ2KEncodedRowBytes = 84
)

// v41EngramEncodedRowBytes is the closed set of encodings a tree digest may
// cover, with the encoded width of one row.
var v41EngramEncodedRowBytes = map[string]int{V41EngramEncodingQ2K: v41EngramQ2KEncodedRowBytes}

// V41EngramEncodedRows is implemented by a reader whose served rows are a
// deterministic decode of an encoded extent. It returns the reader and the
// [offset, offset+size) extent holding the encoded form of rows
// [start, start+count), and the encoding's name.
type V41EngramEncodedRows interface {
	V41EngramEncodedRows(start, count int) (r io.ReaderAt, offset, size int64, encoding string, err error)
}

// v41EngramParseDigest normalizes a binding digest and returns its scheme. It
// refuses an unknown scheme, an unknown tree encoding, and anything but 64 hex
// digits, so a malformed value is refused before any comparison.
func v41EngramParseDigest(digest string) (string, string, error) {
	d := strings.ToLower(strings.TrimSpace(digest))
	i := strings.LastIndexByte(d, ':')
	if i < 0 {
		return "", "", fmt.Errorf("binding digest %q names no scheme", digest)
	}
	scheme, sum := d[:i], d[i+1:]
	if scheme != v41EngramDigestFlat {
		enc, ok := strings.CutPrefix(scheme, v41EngramDigestTreePrefix)
		if _, known := v41EngramEncodedRowBytes[enc]; !ok || (enc != v41EngramEncodingRows && !known) {
			return "", "", fmt.Errorf("binding digest %q names no known scheme", digest)
		}
	}
	if len(sum) != sha256.Size*2 {
		return "", "", fmt.Errorf("binding digest %q is not a sha256 content address", digest)
	}
	if _, err := hex.DecodeString(sum); err != nil {
		return "", "", fmt.Errorf("binding digest %q is not hex", digest)
	}
	return d, scheme, nil
}

// v41EngramEncodedExtent resolves the encoded extent behind binding's rows. ok is
// false when the reader names no encoded form; a reader that names one must name
// a known encoding whose extent is exactly Rows encoded rows.
func v41EngramEncodedExtent(reader io.ReaderAt, binding V41EngramArtifactBinding, rowBytes int) (r io.ReaderAt, offset, size int64, encoding string, ok bool, err error) {
	enc, ok := reader.(V41EngramEncodedRows)
	if !ok {
		return nil, 0, 0, "", false, nil
	}
	if binding.Offset%int64(rowBytes) != 0 {
		return nil, 0, 0, "", true, fmt.Errorf("binding offset %d is not row-aligned (%d bytes)", binding.Offset, rowBytes)
	}
	r, offset, size, encoding, err = enc.V41EngramEncodedRows(int(binding.Offset/int64(rowBytes)), binding.Rows)
	if err != nil {
		return nil, 0, 0, "", true, err
	}
	per, known := v41EngramEncodedRowBytes[encoding]
	if !known {
		return nil, 0, 0, "", true, fmt.Errorf("encoded rows name unknown encoding %q", encoding)
	}
	if r == nil || offset < 0 || size != int64(binding.Rows)*int64(per) {
		return nil, 0, 0, "", true, fmt.Errorf("encoded %s extent offset=%d size=%d is not %d rows * %d bytes", encoding, offset, size, binding.Rows, per)
	}
	return r, offset, size, encoding, true, nil
}

// v41EngramAcquireDigest is the digest acquisition binds: the tree over the
// encoded extent when the reader names one, else the tree over the served rows.
func v41EngramAcquireDigest(reader io.ReaderAt, binding V41EngramArtifactBinding, rowBytes int) (string, error) {
	r, off, size, encoding, ok, err := v41EngramEncodedExtent(reader, binding, rowBytes)
	if err != nil {
		return "", err
	}
	if !ok {
		r, off, size, encoding = reader, binding.Offset, binding.Size, v41EngramEncodingRows
	}
	return v41EngramTreeDigest(r, off, size, encoding, v41EngramTreeWorkers(size))
}

// v41EngramBindingDigest recomputes, from reader, the digest of the scheme the
// binding declares. A tree-over-encoding scheme refuses a reader that cannot name
// that encoding rather than falling back to another scheme.
func v41EngramBindingDigest(reader io.ReaderAt, binding V41EngramArtifactBinding, scheme string, rowBytes int) (string, error) {
	if scheme == v41EngramDigestFlat {
		return v41EngramDigestExtent(reader, binding.Offset, binding.Size)
	}
	want := strings.TrimPrefix(scheme, v41EngramDigestTreePrefix)
	if want == v41EngramEncodingRows {
		return v41EngramTreeDigest(reader, binding.Offset, binding.Size, want, v41EngramTreeWorkers(binding.Size))
	}
	r, off, size, encoding, ok, err := v41EngramEncodedExtent(reader, binding, rowBytes)
	if err != nil {
		return "", err
	}
	if !ok || encoding != want {
		return "", fmt.Errorf("binding digest scheme %s needs %s-encoded rows; reader names %q", scheme, want, encoding)
	}
	return v41EngramTreeDigest(r, off, size, encoding, v41EngramTreeWorkers(size))
}

// v41EngramTreeWorkers sizes the leaf pool: one worker per available core up to
// a fixed cap, never more than the extent has leaves. Host memory is one leaf
// buffer per worker (16 MiB at the cap).
func v41EngramTreeWorkers(size int64) int {
	leaves := (size + v41EngramTreeLeaf - 1) / v41EngramTreeLeaf
	w := runtime.GOMAXPROCS(0)
	if w > v41EngramTreeMaxWorkers {
		w = v41EngramTreeMaxWorkers
	}
	if int64(w) > leaves {
		w = int(leaves)
	}
	if w < 1 {
		w = 1
	}
	return w
}

// v41EngramTreeDigest is scheme sha256t1-<encoding> over [offset, offset+size):
// leaf k is sha256 of bytes [k*L, min((k+1)*L, size)) with L = 1 MiB, and the
// root is sha256(domain || encoding || 0x00 || u64le(size) || u64le(L) || leaf
// digests in order). Leaves hash on up to workers goroutines. The reported
// read error is the one at the lowest failing leaf, which is the serial loop's.
func v41EngramTreeDigest(reader io.ReaderAt, offset, size int64, encoding string, workers int) (string, error) {
	if size <= 0 {
		return "", fmt.Errorf("extent size %d must be positive", size)
	}
	leaves := (size + v41EngramTreeLeaf - 1) / v41EngramTreeLeaf
	if workers < 1 {
		workers = 1
	}
	if int64(workers) > leaves {
		workers = int(leaves)
	}
	sums := make([][sha256.Size]byte, leaves)
	var (
		next    atomic.Int64
		failed  atomic.Bool
		mu      sync.Mutex
		errLeaf int64 = -1
		leafErr error
		wg      sync.WaitGroup
	)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			buf := make([]byte, v41EngramTreeLeaf)
			for !failed.Load() {
				k := next.Add(1) - 1
				if k >= leaves {
					return
				}
				at := offset + k*v41EngramTreeLeaf
				n := size - k*v41EngramTreeLeaf
				if n > v41EngramTreeLeaf {
					n = v41EngramTreeLeaf
				}
				got, err := reader.ReadAt(buf[:n], at)
				if int64(got) != n {
					if err == nil {
						err = io.ErrUnexpectedEOF
					}
					mu.Lock()
					if errLeaf < 0 || k < errLeaf {
						errLeaf, leafErr = k, fmt.Errorf("hashing extent at offset %d: read %d want %d: %w", at, got, n, err)
					}
					mu.Unlock()
					failed.Store(true)
					return
				}
				sums[k] = sha256.Sum256(buf[:n])
			}
		}()
	}
	wg.Wait()
	if leafErr != nil {
		return "", leafErr
	}
	root := sha256.New()
	root.Write([]byte(v41EngramTreeDomain))
	root.Write([]byte(encoding))
	var hdr [17]byte
	binary.LittleEndian.PutUint64(hdr[1:9], uint64(size))
	binary.LittleEndian.PutUint64(hdr[9:], uint64(v41EngramTreeLeaf))
	root.Write(hdr[:])
	for i := range sums {
		root.Write(sums[i][:])
	}
	return v41EngramDigestTreePrefix + encoding + ":" + hex.EncodeToString(root.Sum(nil)), nil
}
