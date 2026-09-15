package model

// v41_engram_binding.go — the verified artifact binding for the V4.1 Engram
// packed-row source (#12909 integration prerequisite, parent #12640).
//
// WHY THIS EXISTS. NewV41PackedEngramRowSource (v41_engram_stream.go) checks only
// that its two caller-supplied identity STRINGS are equal and that the declared
// extent is internally consistent. It never looks at the backing bytes: the
// `reader io.ReaderAt` is whatever the caller handed in, and a caller that paired
// the layer-1 shard NAME with the layer-2 shard's BYTES would pass every existing
// check and silently read the wrong rows. The #12937 review recorded this as the
// blocker to removing native admission refusal: before V4.1 native execution
// stops refusing, each opened reader/range must be bound to the pinned
// checkpoint-index shard — its name, exact extent, and an immutable artifact
// digest established at acquisition.
//
// WHAT THIS IS. V41EngramArtifactBinding is that observation, carried separately
// from the index-requested identity so the two cannot be confused: the binding is
// what the loader SAW when it opened the artifact (name, [offset,size) extent, row
// count, and a sha256 content address over exactly those bytes). The constructor
// re-derives that content address from the reader and refuses on any disagreement,
// so the identity of the bytes — not just a string a caller typed — gates admission.
//
// COST. Verification hashes the DECLARED EXTENT once, at construction. For the
// published ~203 GB Engram table that is a one-time acquisition-time cost, not a
// per-read cost: ReadRows afterwards is unchanged, so bounded streaming is
// preserved. Tiny fixtures hash a miniature shard, which is what makes the
// negative case a deterministic, in-process witness.
//
// WHAT THIS IS NOT. It does not make the prepared-reader path (NewV41PackedEngramRowSource)
// verified — that constructor is retained byte-identically for callers that have
// already authenticated their reader under a documented preverified contract. This
// file adds the verifiable route; wiring it into the loader is the loader leaf's
// change, not this one.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"strings"
)

// V41EngramArtifactBinding is the immutable observation a loader makes when it
// opens one packed Engram shard: WHICH artifact (Shard), WHAT extent of it
// ([Offset,Offset+Size)), how many packed rows that extent holds (Rows), and a
// sha256 content address over exactly those extent bytes (Digest). It is carried
// separately from V41EngramShard's index-requested identity so observed and
// requested identity cannot be conflated.
//
// Rows is the packed-row count (V41EngramPackedRowBytes each), so Size must equal
// Rows*V41EngramPackedRowBytes. Digest is "sha256:<hex>" over the extent bytes.
type V41EngramArtifactBinding struct {
	Shard  string
	Offset int64
	Size   int64
	Rows   int
	Digest string
}

// validate checks the binding's internal consistency and normalizes its digest.
// A zero or malformed digest is refused here rather than allowed to reach a
// comparison that would then reject a correct artifact for the wrong reason.
func (b V41EngramArtifactBinding) validate() error {
	if strings.TrimSpace(b.Shard) == "" {
		return fmt.Errorf("binding declares no shard name")
	}
	if b.Offset < 0 || b.Size <= 0 || b.Rows <= 0 {
		return fmt.Errorf("binding extent offset=%d size=%d rows=%d", b.Offset, b.Size, b.Rows)
	}
	if b.Size != int64(b.Rows)*V41EngramPackedRowBytes {
		return fmt.Errorf("binding size=%d is not rows=%d * %d packed bytes", b.Size, b.Rows, V41EngramPackedRowBytes)
	}
	digest := strings.ToLower(strings.TrimSpace(b.Digest))
	const prefix = "sha256:"
	if !strings.HasPrefix(digest, prefix) || len(digest) != len(prefix)+64 {
		return fmt.Errorf("binding digest %q is not a sha256 content address", b.Digest)
	}
	if _, err := hex.DecodeString(digest[len(prefix):]); err != nil {
		return fmt.Errorf("binding digest %q is not hex", b.Digest)
	}
	return nil
}

// v41EngramDigestExtent hashes exactly the declared extent of reader. It reads in
// bounded chunks so a large declared extent does not require a full-size buffer.
func v41EngramDigestExtent(reader io.ReaderAt, offset, size int64) (string, error) {
	if size <= 0 {
		return "", fmt.Errorf("extent size %d must be positive", size)
	}
	h := sha256.New()
	const chunk = 1 << 20
	buf := make([]byte, chunk)
	remaining := size
	at := offset
	for remaining > 0 {
		n := int64(chunk)
		if n > remaining {
			n = remaining
		}
		got, err := reader.ReadAt(buf[:n], at)
		if int64(got) != n {
			if err == nil {
				err = io.ErrUnexpectedEOF
			}
			return "", fmt.Errorf("hashing extent at offset %d: read %d want %d: %w", at, got, n, err)
		}
		h.Write(buf[:got])
		at += int64(got)
		remaining -= int64(got)
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}

// NewV41VerifiedPackedEngramRowSource returns a packed 264-byte row reader whose
// backing bytes have been verified against a loader-established artifact binding.
// It is the fail-closed route for removing native admission refusal: the shard
// name, extent, and row count must agree with the binding, the shard's observed
// identity must match the requested identity, and the extent's content address
// must equal the binding's digest. Any disagreement is refused before a single row
// can be served, so a wrong shard/range cannot reach Engram projection or
// Prefill/Step.
func NewV41VerifiedPackedEngramRowSource(reader io.ReaderAt, shard V41EngramShard, binding V41EngramArtifactBinding) (V41EngramRowSource, error) {
	if err := binding.validate(); err != nil {
		return nil, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: -1, Shard: shard.ID,
			Err: fmt.Errorf("invalid artifact binding: %w", err)}
	}
	// The reader-side geometry checks (nil reader, packed extent, requested==observed
	// name) are the prepared-reader contract; reuse them rather than restating them.
	base, err := NewV41PackedEngramRowSource(reader, shard)
	if err != nil {
		return nil, err
	}
	// The observed extent must be the bound extent. A binding for a different
	// shard, a different offset, or a different row count is a mis-paired reader.
	if shard.ID != binding.Shard || shard.Offset != binding.Offset || shard.Size != binding.Size || shard.Rows != binding.Rows {
		return nil, &V41EngramStreamError{Kind: V41EngramStreamShardMismatch, Row: -1, Shard: shard.ID,
			Err: fmt.Errorf("observed extent name=%q offset=%d size=%d rows=%d does not match binding name=%q offset=%d size=%d rows=%d",
				shard.ID, shard.Offset, shard.Size, shard.Rows, binding.Shard, binding.Offset, binding.Size, binding.Rows)}
	}
	observed, err := v41EngramDigestExtent(reader, binding.Offset, binding.Size)
	if err != nil {
		return nil, &V41EngramStreamError{Kind: V41EngramStreamBindingMismatch, Row: -1, Shard: shard.ID,
			Err: fmt.Errorf("cannot verify shard bytes: %w", err)}
	}
	if observed != strings.ToLower(strings.TrimSpace(binding.Digest)) {
		return nil, &V41EngramStreamError{Kind: V41EngramStreamBindingMismatch, Row: -1, Shard: shard.ID,
			Err: fmt.Errorf("shard bytes digest %s does not match bound artifact %s", observed, binding.Digest)}
	}
	return &v41VerifiedPackedEngramRowSource{inner: base, binding: binding}, nil
}

// v41VerifiedPackedEngramRowSource is a V41EngramRowSource that only exists once
// its backing bytes have been verified. It delegates reads to the prepared source
// so per-read behavior (bounds checks, short-read refusal, accounting) is exactly
// the unverified path's; only the constructor's admission gate differs.
type v41VerifiedPackedEngramRowSource struct {
	inner   V41EngramRowSource
	binding V41EngramArtifactBinding
}

func (*v41VerifiedPackedEngramRowSource) RowBytes() int { return V41EngramPackedRowBytes }

func (s *v41VerifiedPackedEngramRowSource) ReadRows(start, count int, dst []byte) (int, error) {
	return s.inner.ReadRows(start, count, dst)
}

// V41EngramBinding reports the verified artifact binding a source was admitted
// under, and whether the source carries one at all. A loader uses it to prove that
// every source wired into the forward is verified: an implementation that is not
// *v41VerifiedPackedEngramRowSource reports ok=false.
func V41EngramBinding(src V41EngramRowSource) (V41EngramArtifactBinding, bool) {
	if v, ok := src.(*v41VerifiedPackedEngramRowSource); ok {
		return v.binding, true
	}
	return V41EngramArtifactBinding{}, false
}
