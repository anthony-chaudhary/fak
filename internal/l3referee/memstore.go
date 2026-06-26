package l3referee

import (
	"crypto/sha256"
	"encoding/hex"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// MemL3 is a fake, in-memory stand-in for an external L3 KV store — NO RDMA/CAMA
// dependency. It exists to drive the referee's tests and to demonstrate the
// store-agnostic seam: a real L3 store (CAMA) swaps in behind the same Set/Get
// shape. The store records each page's content digest at write (the Ref.Digest
// the referee verifies a read against) and recomputes it at read, so a test can
// simulate corruption by handing back tampered bytes under the same key.
//
// This is the data-plane fake; the referee never touches it on the hot path. The
// referee only ever sees the Ref it returns and the digest Get reports.
type MemL3 struct {
	pages map[string][]byte // recorded digest -> page bytes
}

// NewMemL3 returns an empty fake L3 store.
func NewMemL3() *MemL3 { return &MemL3{pages: map[string][]byte{}} }

// digest mirrors blob.Digest's scheme (sha256 hex) so a recorded Ref.Digest is
// byte-identical to what fak's content-addressed store records — kept local to
// keep this leaf abi-only (no blob import in non-test code).
func digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// Set stores page and returns the Ref fak would record. Its Digest is the content
// address the referee later verifies a read against.
func (m *MemL3) Set(page []byte) abi.Ref {
	d := digest(page)
	cp := make([]byte, len(page))
	copy(cp, page)
	m.pages[d] = cp
	return abi.Ref{Kind: abi.RefBlob, Digest: d, Len: int64(len(page))}
}

// Get returns the stored page for ref AND the digest of the bytes actually
// returned. An honest store returns matching bytes; a Corrupt'd store returns a
// page whose recomputed digest no longer matches ref.Digest.
func (m *MemL3) Get(ref abi.Ref) (page []byte, pageDigest string, ok bool) {
	b, ok := m.pages[ref.Digest]
	if !ok {
		return nil, "", false
	}
	return b, digest(b), true
}

// Corrupt overwrites the bytes stored under ref.Digest WITHOUT changing the key —
// the exact hash-collision / mis-computed-prefix failure G1 exists to catch. A
// subsequent Get then returns a page whose recomputed digest ≠ ref.Digest.
func (m *MemL3) Corrupt(ref abi.Ref, tampered []byte) {
	cp := make([]byte, len(tampered))
	copy(cp, tampered)
	m.pages[ref.Digest] = cp
}
