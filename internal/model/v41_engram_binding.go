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
// Rows is the row count and RowBytes is the per-row width, so Size must equal
// Rows*RowBytes. Digest is "<scheme>:<hex>" (v41_engram_digest_tree.go):
// "sha256" over the extent bytes, "sha256t1-rows" a tree over them, or
// "sha256t1-q2_k" a tree over the Q2_K blocks the extent's rows decode from.
//
// RowBytes makes the binding dialect-parameterized. The published V4.1 artifact's
// Engram table is stored as Q2_K (one super-block per row), so the loader's
// V41EngramQ2KOpen yields DEQUANTIZED f32 rows (qkK*4 = 1024 bytes), while the
// raw-I8 converter dialect yields packed 264-byte rows (256 E4M3 + 8 E8M0). A
// zero RowBytes means the packed dialect, preserving bindings written before the
// width was explicit.
type V41EngramArtifactBinding struct {
	Shard    string
	Offset   int64
	Size     int64
	Rows     int
	Digest   string
	RowBytes int
}

// rowWidth resolves the effective per-row width, defaulting a zero RowBytes to
// the packed dialect. It refuses any width the forward cannot consume, so an
// unsupported dialect cannot be admitted by omission.
func (b V41EngramArtifactBinding) rowWidth() (int, error) {
	rb := b.RowBytes
	if rb == 0 {
		rb = V41EngramPackedRowBytes
	}
	if rb != V41EngramPackedRowBytes && rb != V41EngramF32RowBytes {
		return 0, fmt.Errorf("binding row width %d is not a supported Engram row width (%d packed, %d f32)",
			b.RowBytes, V41EngramPackedRowBytes, V41EngramF32RowBytes)
	}
	return rb, nil
}

// validate checks the binding's internal consistency and returns its normalized
// digest and scheme. A zero, malformed or unknown-scheme digest is refused here
// rather than allowed to reach a comparison that would then reject a correct
// artifact for the wrong reason.
func (b V41EngramArtifactBinding) validate() (digest, scheme string, err error) {
	if err := b.validateExtent(); err != nil {
		return "", "", err
	}
	return v41EngramParseDigest(b.Digest)
}

func (b V41EngramArtifactBinding) validateExtent() error {
	if strings.TrimSpace(b.Shard) == "" {
		return fmt.Errorf("binding declares no shard name")
	}
	rowBytes, err := b.rowWidth()
	if err != nil {
		return err
	}
	if b.Offset < 0 || b.Size <= 0 || b.Rows <= 0 {
		return fmt.Errorf("binding extent offset=%d size=%d rows=%d", b.Offset, b.Size, b.Rows)
	}
	if b.Size != int64(b.Rows)*int64(rowBytes) {
		return fmt.Errorf("binding size=%d is not rows=%d * %d bytes", b.Size, b.Rows, rowBytes)
	}
	return nil
}

// v41EngramDigestExtent hashes exactly the declared extent of reader. It reads in
// bounded chunks so a large declared extent does not require a full-size buffer.
// The chunk reads run ahead on a bounded worker pool (v41_engram_digest.go)
// while one goroutine hashes them in extent order, so the digest is the serial
// form's byte-for-byte.
func v41EngramDigestExtent(reader io.ReaderAt, offset, size int64) (string, error) {
	return v41EngramDigestExtentWorkers(reader, offset, size, v41EngramDigestWorkers(size))
}

// NewV41VerifiedEngramRowSource returns a row reader over the binding's declared
// dialect whose backing bytes have been verified against a loader-established
// artifact binding. RowBytes() reports the bound width (264 packed or 1024 f32),
// so a caller cannot pair a packed reader with an f32 binding by accident. The
// shard name, extent, and row count must agree with the binding, the shard's
// observed identity must match the requested identity, and the extent's content
// address, recomputed under the scheme the binding's digest names, must equal
// the binding's digest. Any disagreement is refused before a single row can be
// served, so a wrong shard/range cannot reach Engram projection or Prefill/Step.
func NewV41VerifiedEngramRowSource(reader io.ReaderAt, shard V41EngramShard, binding V41EngramArtifactBinding) (V41EngramRowSource, error) {
	want, scheme, err := binding.validate()
	if err != nil {
		return nil, &V41EngramStreamError{Kind: V41EngramStreamGeometry, Row: -1, Shard: shard.ID,
			Err: fmt.Errorf("invalid artifact binding: %w", err)}
	}
	base, rowBytes, err := v41BoundEngramBase(reader, shard, binding)
	if err != nil {
		return nil, err
	}
	observed, err := v41EngramBindingDigest(reader, binding, scheme, rowBytes)
	if err != nil {
		return nil, &V41EngramStreamError{Kind: V41EngramStreamBindingMismatch, Row: -1, Shard: shard.ID,
			Err: fmt.Errorf("cannot verify shard bytes: %w", err)}
	}
	if observed != want {
		return nil, &V41EngramStreamError{Kind: V41EngramStreamBindingMismatch, Row: -1, Shard: shard.ID,
			Err: fmt.Errorf("shard bytes digest %s does not match bound artifact %s", observed, binding.Digest)}
	}
	return &v41VerifiedEngramRowSource{inner: base, binding: binding, rowBytes: rowBytes}, nil
}

// v41BoundEngramBase runs the geometry half of admission shared by the verified
// and acquired constructors: the prepared-reader contract plus the observed
// extent agreeing with the bound one. It reads no payload bytes.
func v41BoundEngramBase(reader io.ReaderAt, shard V41EngramShard, binding V41EngramArtifactBinding) (V41EngramRowSource, int, error) {
	rowBytes, _ := binding.rowWidth()
	// The reader-side geometry checks (nil reader, packed extent, requested==observed
	// name) are the prepared-reader contract; reuse them rather than restating them.
	base, err := newV41PreparedEngramRowSource(reader, shard, rowBytes)
	if err != nil {
		return nil, 0, err
	}
	// The observed extent must be the bound extent. A binding for a different
	// shard, a different offset, or a different row count is a mis-paired reader.
	if shard.ID != binding.Shard || shard.Offset != binding.Offset || shard.Size != binding.Size || shard.Rows != binding.Rows {
		return nil, 0, &V41EngramStreamError{Kind: V41EngramStreamShardMismatch, Row: -1, Shard: shard.ID,
			Err: fmt.Errorf("observed extent name=%q offset=%d size=%d rows=%d does not match binding name=%q offset=%d size=%d rows=%d",
				shard.ID, shard.Offset, shard.Size, shard.Rows, binding.Shard, binding.Offset, binding.Size, binding.Rows)}
	}
	return base, rowBytes, nil
}

// NewV41VerifiedPackedEngramRowSource is the packed-264 dialect entry point. It
// keeps the exact behaviour of the original constructor (the binding must be a
// packed-dialect binding) so existing callers are byte-for-byte unchanged.
func NewV41VerifiedPackedEngramRowSource(reader io.ReaderAt, shard V41EngramShard, binding V41EngramArtifactBinding) (V41EngramRowSource, error) {
	return NewV41VerifiedEngramRowSource(reader, shard, binding)
}

// v41VerifiedEngramRowSource is a V41EngramRowSource that only exists once
// its backing bytes have been verified. It delegates reads to the prepared source
// so per-read behavior (bounds checks, short-read refusal, accounting) is exactly
// the unverified path's; only the constructor's admission gate differs.
type v41VerifiedEngramRowSource struct {
	inner    V41EngramRowSource
	binding  V41EngramArtifactBinding
	rowBytes int
}

func (s *v41VerifiedEngramRowSource) RowBytes() int { return s.rowBytes }

func (s *v41VerifiedEngramRowSource) ReadRows(start, count int, dst []byte) (int, error) {
	return s.inner.ReadRows(start, count, dst)
}

// V41EngramBinding reports the verified artifact binding a source was admitted
// under, and whether the source carries one at all. A loader uses it to prove that
// every source wired into the forward is verified: an implementation that is not
// *v41VerifiedEngramRowSource reports ok=false.
func V41EngramBinding(src V41EngramRowSource) (V41EngramArtifactBinding, bool) {
	if v, ok := src.(*v41VerifiedEngramRowSource); ok {
		return v.binding, true
	}
	return V41EngramArtifactBinding{}, false
}
