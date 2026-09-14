package model

import (
	"fmt"
	"sync"
)

// expert_sparse_overlay.go — the disk-backed Engram sparse-overlay expert layout with per-rank row
// copies (#13030, parent #13008, the MoE disk-backed expert serving spine).
//
// The gap this closes. ExpertCheckpointTier reads WHOLE expert strides: AddShardData indexes one
// {offset, [E,out,in], quant geometry} stride per expert and fault reads exactly one expert's
// stride bytes. That is the right unit when the rank can use every weight of an expert it routes
// to, but a rank at scale materializes only the ROWS its own expert band can route to. The
// per-layer Engram hash cache (v41_engram_stream.go) gathers scattered ROWS but has no notion of a
// rank band and no write-back over a shared backing file. Neither path can express "this rank's
// sparse row subset over one shared shard".
//
// THE THROUGH-LINE. An overlay is a per-rank sparse row set over one shared checkpoint shard: a
// row-presence BITMAP naming which experts of a fused tensor this rank materializes, plus the raw
// stride bytes of each present expert. Registering an overlay is ADDITIVE and default-absent:
// with no overlay registered the tier faults the historical dense stride exactly as before, so
// every current caller and NewExpertCheckpointTier(hostBytes) stay byte-identical. When an overlay
// IS registered, fault serves an overlay-present expert from its copied rows and falls through to
// the dense stride for an expert outside the overlay, byte-for-byte. ExpertCheckpointStats.BytesRead
// reflects the overlay row bytes actually moved, so the offload win is a measured number rather than
// an assertion.
//
// WHAT THIS IS NOT (gold-plating boundary, #13030): no writer and no new on-disk format — the
// overlay is an in-memory index + row copy over an EXISTING shard, registered by whoever loaded it;
// no EPLB all-to-all placement or multi-node shipping; no physical measurement. The bitmap is an
// in-memory presence index, never a persisted artifact.

// ExpertSparseOverlay is one rank's sparse row set over a single fused checkpoint tensor: a
// row-presence bitmap over expert indices plus the raw stride bytes of every present expert.
// Register it with ExpertCheckpointTier.SetExpertSparseOverlay. A nil *ExpertSparseOverlay is the
// default and means "no overlay" — the tier stays byte-identical to the dense-stride path.
//
// The overlay is immutable once registered: every accessor reads the bitmap and rows under one
// lock, and nothing mutates either after construction.
type ExpertSparseOverlay struct {
	// fused is the checkpoint's own fused-tensor name this overlay bands (e.g.
	// `blk.3.ffn_gate_exps.weight`). fault matches it against the indexed entry's fused tensor, so
	// an overlay never serves a name outside the tensor it was registered for.
	fused string

	mu      sync.RWMutex
	present map[string][]byte // canonical expert name -> raw stride bytes (present experts only)
	bitmap  []uint64          // row-presence bitmap over expert indices of fused
	rows    int               // expert count of fused (bitmap width)
}

// NewExpertSparseOverlay returns an empty overlay for the fused tensor named fused with rows expert
// slots. fused must be non-empty and rows positive; a malformed overlay is refused at construction
// rather than serving a half-built band.
func NewExpertSparseOverlay(fused string, rows int) (*ExpertSparseOverlay, error) {
	if fused == "" {
		return nil, fmt.Errorf("%w: sparse overlay declares no fused tensor", ErrGGUFExpertMetadata)
	}
	if rows <= 0 {
		return nil, fmt.Errorf("%w: sparse overlay for %s declares %d expert rows", ErrGGUFExpertMetadata, fused, rows)
	}
	return &ExpertSparseOverlay{
		fused:   fused,
		present: make(map[string][]byte),
		bitmap:  make([]uint64, (rows+63)/64),
		rows:    rows,
	}, nil
}

// Rows is the fused tensor's expert count the bitmap spans.
func (o *ExpertSparseOverlay) Rows() int {
	if o == nil {
		return 0
	}
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.rows
}

// Fused names the fused tensor this overlay bands.
func (o *ExpertSparseOverlay) Fused() string {
	if o == nil {
		return ""
	}
	return o.fused
}

// SetRow copies raw stride bytes for the canonical expert name at bit index into the overlay and
// sets that bit in the row-presence bitmap. index must be in [0, rows); raw must be non-empty.
// It is how a loader materializes one rank's sparse rows over the shared shard.
func (o *ExpertSparseOverlay) SetRow(index int, name string, raw []byte) error {
	if o == nil {
		return fmt.Errorf("%w: nil sparse overlay", ErrGGUFExpertMetadata)
	}
	if index < 0 || index >= o.rows {
		return fmt.Errorf("%w: sparse overlay row %d outside [0,%d)", ErrGGUFExpertMetadata, index, o.rows)
	}
	if name == "" {
		return fmt.Errorf("%w: sparse overlay row %d declares no expert name", ErrGGUFExpertMetadata, index)
	}
	if len(raw) == 0 {
		return fmt.Errorf("%w: sparse overlay row %s declares no bytes", ErrGGUFExpertMetadata, name)
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	cp := make([]byte, len(raw))
	copy(cp, raw)
	o.present[name] = cp
	o.bitmap[index/64] |= 1 << uint(index%64)
	return nil
}

// RowPresent reports whether the row-presence bitmap marks expert index present.
func (o *ExpertSparseOverlay) RowPresent(index int) bool {
	if o == nil || index < 0 || index >= o.rows {
		return false
	}
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.bitmap[index/64]&(1<<uint(index%64)) != 0
}

// PresentRows counts the set bits in the row-presence bitmap — the number of experts this rank
// materializes.
func (o *ExpertSparseOverlay) PresentRows() int {
	if o == nil {
		return 0
	}
	o.mu.RLock()
	defer o.mu.RUnlock()
	n := 0
	for _, w := range o.bitmap {
		n += popcount64(w)
	}
	return n
}

// row returns a COPY of the raw stride bytes for name when it is bitmap-present, or (nil,false)
// when the expert is outside this rank's overlay (the fall-through signal). A copy is returned so
// a caller can never mutate the overlay's retained rows through the buffer it was handed.
func (o *ExpertSparseOverlay) row(name string) ([]byte, bool) {
	if o == nil {
		return nil, false
	}
	o.mu.RLock()
	defer o.mu.RUnlock()
	raw, ok := o.present[name]
	if !ok {
		return nil, false
	}
	cp := make([]byte, len(raw))
	copy(cp, raw)
	return cp, true
}

// SetExpertSparseOverlay registers a per-rank sparse overlay for the fused tensor it bands. It is
// the loader's hand-off; registration is ADDITIVE and default-absent, so a tier with no overlay
// stays byte-identical to the dense-stride path. An overlay naming a fused tensor this tier does
// not index is refused, so a mis-registered band cannot silently serve nothing.
func (t *ExpertCheckpointTier) SetExpertSparseOverlay(o *ExpertSparseOverlay) error {
	if t == nil {
		return fmt.Errorf("%w: nil tier", ErrGGUFExpertMetadata)
	}
	if o == nil {
		return fmt.Errorf("%w: nil sparse overlay", ErrGGUFExpertMetadata)
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	known := false
	for _, entry := range t.index {
		if entry.fused == o.fused {
			known = true
			break
		}
	}
	if !known {
		return fmt.Errorf("%w: sparse overlay bands %s, which no indexed shard carries", ErrGGUFExpertMetadata, o.fused)
	}
	if t.overlays == nil {
		t.overlays = map[string]*ExpertSparseOverlay{}
	}
	t.overlays[o.fused] = o
	return nil
}

// popcount64 is the set-bit count of one bitmap word.
func popcount64(w uint64) int {
	n := 0
	for w != 0 {
		w &= w - 1
		n++
	}
	return n
}
