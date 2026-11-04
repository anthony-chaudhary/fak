// kv_transfer_checksum.go — optional adler32 integrity sidecar over KV page-index vectors.
//
// ADAPTED from SGLang's optional disaggregation KV-transfer checksum
// (sgl-project/sglang, python/sglang/srt/disaggregation/checksum.py @1b78083,
// Apache-2.0, upstream PR #39500): a per-request adler32 computed at the transfer
// source over the KV page indices and re-checked at the destination, default-off,
// with health-check traffic excluded. Upstream hashes the index vector rather
// than the full block payloads so a corrupted or reordered page landing is caught
// cheaply at the receiver. This file re-expresses that design in idiomatic Go
// over fak physical page IDs; no upstream code is copied.
//
// Default-off posture: the zero value carries no checksum. A disabled record
// verifies any vector and marshals to "{}", so attaching one to a transfer
// descriptor changes neither behavior nor wire until a transfer path opts in by
// constructing a record at the source. Callers must not attach a checksum to
// health-check or probe transfers (upstream excludes them); pass the zero value.
package ctxmmu

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/adler32"
)

var (
	// ErrKVTransferChecksumMismatch is returned when a destination-side
	// page-index vector fails the source-computed adler32 check.
	ErrKVTransferChecksumMismatch = errors.New("ctxmmu: kv transfer page-index checksum mismatch")

	// ErrKVTransferChecksumMalformed is returned when a wire record is neither
	// the disabled "{}" shape nor a complete enabled record.
	ErrKVTransferChecksumMalformed = errors.New("ctxmmu: kv transfer checksum record malformed")
)

// encodePageIndices returns the canonical byte form of a page-index vector:
// a little-endian uint64 element count followed by each page ID as a
// little-endian uint64. The count prefix keeps distinct vectors such as
// [1,2] and [12] from encoding to the same byte stream.
func encodePageIndices(pages []uint64) []byte {
	buf := make([]byte, 8+8*len(pages))
	binary.LittleEndian.PutUint64(buf[:8], uint64(len(pages)))
	for i, p := range pages {
		binary.LittleEndian.PutUint64(buf[8+i*8:], p)
	}
	return buf
}

// PageIndexAdler32 returns the zlib adler32 checksum of the canonical encoding
// of pages. It is the closed source/destination contract: both sides reduce
// their page-index vector through this one function before comparing.
func PageIndexAdler32(pages []uint64) uint32 {
	return adler32.Checksum(encodePageIndices(pages))
}

// KVTransferChecksum is the optional per-transfer integrity sidecar. The source
// constructs it over the page indices it ships; the record crosses the wire
// with the transfer metadata; the destination re-checks its received page
// indices against it. State is unexported so every live record is either the
// disabled zero value or one produced by NewKVTransferChecksum or UnmarshalJSON.
type KVTransferChecksum struct {
	enabled bool
	sum     uint32
	pages   int
}

// NewKVTransferChecksum computes the source-side checksum over the page-index
// vector being shipped. An empty vector yields a well-defined enabled record
// over the bare count prefix.
func NewKVTransferChecksum(pages []uint64) KVTransferChecksum {
	return KVTransferChecksum{enabled: true, sum: PageIndexAdler32(pages), pages: len(pages)}
}

// Enabled reports whether the record carries a source-computed checksum. The
// zero value is disabled: default-off leaves both verification and wire unchanged.
func (c KVTransferChecksum) Enabled() bool { return c.enabled }

// Sum returns the source-computed adler32 (meaningful only when Enabled).
func (c KVTransferChecksum) Sum() uint32 { return c.sum }

// PageCount returns the number of page indices the source summed (meaningful
// only when Enabled).
func (c KVTransferChecksum) PageCount() int { return c.pages }

// Verify re-checks the destination-side page-index vector against the source
// sum. A disabled record verifies any vector (the default-off no-op). A length
// difference or an adler32 difference refuses with ErrKVTransferChecksumMismatch.
func (c KVTransferChecksum) Verify(pages []uint64) error {
	if !c.enabled {
		return nil
	}
	if len(pages) != c.pages {
		return fmt.Errorf("%w: received %d pages, source recorded %d", ErrKVTransferChecksumMismatch, len(pages), c.pages)
	}
	if got := PageIndexAdler32(pages); got != c.sum {
		return fmt.Errorf("%w: source adler32 %#010x, recomputed %#010x over %d pages", ErrKVTransferChecksumMismatch, c.sum, got, len(pages))
	}
	return nil
}

// MarshalJSON renders the wire sidecar. A disabled record marshals to "{}" —
// byte-identical to carrying nothing — so default-off leaves the transfer wire
// unchanged relative to the pre-feature baseline.
func (c KVTransferChecksum) MarshalJSON() ([]byte, error) {
	if !c.enabled {
		return []byte("{}"), nil
	}
	return []byte(fmt.Sprintf(`{"adler32":%d,"pages":%d}`, c.sum, c.pages)), nil
}

// UnmarshalJSON parses a wire sidecar. "{}" decodes to the disabled record; any
// other shape must carry both fields consistently, else it refuses with
// ErrKVTransferChecksumMalformed rather than silently downgrading to disabled.
func (c *KVTransferChecksum) UnmarshalJSON(b []byte) error {
	var raw struct {
		Adler32 *uint32 `json:"adler32"`
		Pages   *int    `json:"pages"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	if raw.Adler32 == nil && raw.Pages == nil {
		*c = KVTransferChecksum{}
		return nil
	}
	if raw.Adler32 == nil || raw.Pages == nil || *raw.Pages < 0 {
		return ErrKVTransferChecksumMalformed
	}
	*c = KVTransferChecksum{enabled: true, sum: *raw.Adler32, pages: *raw.Pages}
	return nil
}
