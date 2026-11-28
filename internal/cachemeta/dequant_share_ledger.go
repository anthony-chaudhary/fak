package cachemeta

import (
	"errors"
	"sync"
)

// dequant_share_ledger.go — expand-once-share-many dequant ledger (sglang PD
// dequant-once-share, sgl-project/sglang PR #40421,
// python/sglang/kernels/ops/attention/dsv4/dequant_k_cache.py +
// python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py
//
// The upstream change: when KV crosses a prefill->decode boundary in compressed
// (quantized) form under context parallelism, every CP chunk otherwise
// re-dequantizes its own slice of the same quantized K cache — N chunks pay N
// dequant passes over the same bytes. The change dequantizes ONCE at the
// receiving stage and hands the single expanded buffer to all concurrent
// consumers in the same pass.
//
// This file ports that contract as a pure-Go, no-hardware receiving-stage
// expansion ledger. It is the deterministic bookkeeping half of the change:
//
//   - a quantized KV slice is identified by a source digest (the identity of
//     the compressed bytes it expands from — the same role the upstream
//     quantized K cache tensor plays);
//   - the FIRST consumer to request a digest triggers exactly one expansion
//     (DequantShareLedger.Expansions stays 1 no matter how many consumers
//     follow);
//   - every consumer of that digest is handed the SAME expanded buffer (one
//     shared []float32, not N copies), with a reference count on the entry;
//   - the entry is released only when the LAST consumer releases it
//     (DequantShareEntry.Released flips at refcount 0);
//   - a double-release of a consumer is refused fail-closed
//     (ErrDequantShareDoubleRelease) and leaves the ledger state untouched.
//
// The fence (matching the package's no-payloads stance, doc.go): the ledger
// owns no transfer engine and moves no bytes across a wire. The expansion
// itself is a plain host-side dequant (int8 codes x per-block scale -> f32),
// so the "expand once" witness is real arithmetic, not a counter alone. The
// transfer-engine side (who ships the quantized slice, when) is out of scope
// here; this is the receiving-stage share ledger only.

// DequantShare errors.
var (
	// ErrDequantShareDoubleRelease is returned when a consumer is released
	// more than once. Fail-closed: the second release is refused and the
	// entry's reference count is not decremented again.
	ErrDequantShareDoubleRelease = errors.New("cachemeta: dequant share consumer already released")
	// ErrDequantShareEmpty is returned when a quantized slice has no codes to
	// expand. An empty slice has no expansion to share.
	ErrDequantShareEmpty = errors.New("cachemeta: dequant share slice has no codes")
	// ErrDequantShareBadScale is returned when a per-block scale is not
	// positive. A non-positive scale would expand codes into a meaningless
	// (zero or sign-flipped) buffer, so the expansion is refused.
	ErrDequantShareBadScale = errors.New("cachemeta: dequant share scale must be positive")
)

// DequantShareEntry is one expanded quantized KV slice held for sharing. It
// is created exactly once per source digest (the expand-once half of the
// contract) and is released when its reference count reaches zero (the
// share-many / last-use half).
type DequantShareEntry struct {
	// SourceDigest identifies the quantized slice this entry expanded from.
	// Two consumers with the same digest share this one entry.
	SourceDigest string
	// Expanded is the single expanded (dequantized) buffer shared by every
	// consumer of this entry. All consumers read the same slice.
	Expanded []float32
	// RefCount is the number of live consumers holding this entry. It is
	// decremented by Release and the entry is released at 0.
	RefCount int
	// Released is true once RefCount reached 0. It is the last-use witness:
	// it flips only when the final consumer releases.
	Released bool
}

// DequantShareConsumer is a single consumer's handle on a shared expanded
// buffer. It is what Acquire returns and what Release consumes.
type DequantShareConsumer struct {
	// entry is the shared entry this consumer holds a reference to.
	entry *DequantShareEntry
	// released is true once this consumer has been released. A second
	// Release is refused fail-closed.
	released bool
}

// DequantShareLedger is the receiving-stage expansion ledger. It maps source
// digests to their single shared expanded entry and counts how many
// expansions it has performed. It is safe for concurrent use: the whole
// point of the ledger is that N concurrent consumers in the same pass all
// land on the one expansion.
type DequantShareLedger struct {
	mu         sync.Mutex
	entries    map[string]*DequantShareEntry
	Expansions int // total expansions performed (one per distinct digest)
	Released   int // total entries released at last use
	Acquires   int // total successful consumer acquisitions
	Releases   int // total successful consumer releases
}

// NewDequantShareLedger returns an empty ledger.
func NewDequantShareLedger() *DequantShareLedger {
	return &DequantShareLedger{entries: make(map[string]*DequantShareEntry)}
}

// DequantShareSlice is a quantized KV slice arriving at the receiving stage:
// int8 codes plus one positive per-block scale, in the shape of the upstream
// quantized K cache (code + scale per block).
type DequantShareSlice struct {
	// Digest is the identity of the quantized bytes (the share key).
	Digest string
	// Codes are the quantized values, one int8 per position.
	Codes []int8
	// Scales is the per-block positive scale; the slice is len(Scales) blocks
	// of len(Codes)/len(Scales) codes each, and Expanded[i] =
	// float32(Codes[i]) * Scales[i/blockSize]. len(Scales) must divide
	// len(Codes) and every scale must be > 0.
	Scales []float32
}

// dequantShareExpand performs the host-side expansion: int8 codes times their
// per-block scale, into a fresh []float32. The slice is laid out as
// len(scales) blocks, each covering len(codes)/len(scales) consecutive codes;
// code i takes the scale of its block, scales[i/blockSize]. This is the real
// arithmetic the "expand once" witness stands on — a second consumer of the
// same digest never re-runs it.
func dequantShareExpand(codes []int8, scales []float32) ([]float32, error) {
	if len(codes) == 0 {
		return nil, ErrDequantShareEmpty
	}
	if len(scales) == 0 || len(codes)%len(scales) != 0 {
		return nil, ErrDequantShareBadScale
	}
	for _, s := range scales {
		if s <= 0 {
			return nil, ErrDequantShareBadScale
		}
	}
	blockSize := len(codes) / len(scales)
	out := make([]float32, len(codes))
	for i, c := range codes {
		out[i] = float32(c) * scales[i/blockSize]
	}
	return out, nil
}

// Acquire hands a consumer the expanded buffer for the given quantized slice.
// On the first request for a digest it performs the single expansion and
// records it; every later request for the same digest is handed the SAME
// expanded buffer with no further expansion. The returned consumer holds one
// reference on the entry and must be Released exactly once.
func (l *DequantShareLedger) Acquire(slice DequantShareSlice) (*DequantShareConsumer, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	entry, ok := l.entries[slice.Digest]
	if !ok {
		expanded, err := dequantShareExpand(slice.Codes, slice.Scales)
		if err != nil {
			return nil, err
		}
		entry = &DequantShareEntry{SourceDigest: slice.Digest, Expanded: expanded}
		l.entries[slice.Digest] = entry
		l.Expansions++
	}
	entry.RefCount++
	l.Acquires++
	return &DequantShareConsumer{entry: entry}, nil
}

// Release drops one consumer's reference on its entry. When it is the LAST
// reference the entry is released (Released flips, RefCount hits 0) — the
// last-use release the contract pins. Releasing the same consumer twice is
// refused fail-closed with ErrDequantShareDoubleRelease and changes nothing.
func (l *DequantShareLedger) Release(c *DequantShareConsumer) error {
	if c == nil {
		return ErrDequantShareDoubleRelease
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	if c.released {
		return ErrDequantShareDoubleRelease
	}
	c.released = true
	e := c.entry
	e.RefCount--
	l.Releases++
	if e.RefCount == 0 {
		e.Released = true
		l.Released++
	}
	return nil
}

// Entry exposes the shared entry for a digest so an observer can read its
// reference count and released state without holding a consumer. It returns
// false when the digest has no (or a fully released) entry.
func (l *DequantShareLedger) Entry(digest string) (*DequantShareEntry, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	e, ok := l.entries[digest]
	return e, ok
}
