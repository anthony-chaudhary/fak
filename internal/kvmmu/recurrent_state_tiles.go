// Recurrent-state tiles: the delta/linear-attention recurrent state of
// hybrid-attention models as addressable cache objects with their own key,
// layout descriptor, and group membership, listed alongside ordinary KV groups.
//
// Provenance (ADAPT; no upstream code copied): LMCache's Kimi-K3.1 KDA state
// tiling (PR #6318) tiles the per-layer recurrent state of hybrid-attention
// models into chunks stored and restored alongside ordinary KV groups under a
// unified mamba-view layout, editing the KV-cache-group descriptors so the
// small recurrent state is a first-class cacheable object with its own key and
// layout metadata. Source: lmcache/lmcache,
// lmcache/integration/vllm/kv_cache_group_edits.py +
// tests/v1/test_vllm_mamba_unified_view_layouts.py @ 3e47dfbe6c4d (Apache-2.0);
// study note fak-private docs/research/STUDY-BORROW-WAVE-H200-LATEST-2026-11-06.md.
// Component only: software evidence, no hardware claim.
package kvmmu

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// Structural sentinels for state-tile misuse. Callers assert these with
// errors.Is, never prose.
var (
	// ErrEmptyModelID refuses a tile plan without a stable model identity.
	ErrEmptyModelID = errors.New("kvmmu: state tile plan requires a non-empty model id")
	// ErrInvalidChunks refuses a tiling factor below one.
	ErrInvalidChunks = errors.New("kvmmu: state tile plan requires at least one chunk per layer")
	// ErrUnsupportedStateDtype refuses a state dtype the layout contract does not account.
	ErrUnsupportedStateDtype = errors.New("kvmmu: unsupported recurrent state dtype")
	// ErrNonPositiveGeometry refuses zero/negative head geometry or an empty layout.
	ErrNonPositiveGeometry = errors.New("kvmmu: recurrent state geometry must be positive")
	// ErrStateTileOverflow refuses a tile payload whose element or byte count overflows.
	ErrStateTileOverflow = errors.New("kvmmu: recurrent state tile payload overflow")
)

// StateTileLayout is the mamba-view layout descriptor of one recurrent-state
// tile: nV value heads, each holding a KeyHeadDim x ValueHeadDim recurrent
// matrix row block. Byte counts cover the payload only; a valid layout proves
// addressable arithmetic, not memory fit.
type StateTileLayout struct {
	Dtype        string // normalized state dtype ("f32", "f16", "bf16")
	ElementBytes int    // bytes per state element
	Heads        int    // value heads (nV)
	KeyHeadDim   int    // key head dimension
	ValueHeadDim int    // value head dimension
	RowElements  int    // KeyHeadDim * ValueHeadDim
	Elements     int    // Heads * RowElements
	Bytes        int64  // Elements * ElementBytes
}

// NewStateTileLayout derives the immutable layout of one recurrent-state tile
// without allocating state. Empty dtype means f32; f16/bf16 are accounting
// contracts, not permission for a consumer to change its realized storage.
func NewStateTileLayout(dtype string, heads, keyHeadDim, valueHeadDim int) (StateTileLayout, error) {
	d := strings.ToLower(strings.TrimSpace(dtype))
	if d == "" {
		d = "f32"
	}
	elementBytes := 0
	switch d {
	case "f32":
		elementBytes = 4
	case "f16", "bf16":
		elementBytes = 2
	default:
		return StateTileLayout{}, fmt.Errorf("%w: %q", ErrUnsupportedStateDtype, dtype)
	}
	if heads <= 0 || keyHeadDim <= 0 || valueHeadDim <= 0 {
		return StateTileLayout{}, ErrNonPositiveGeometry
	}
	row, rowOK := checkedIntMul(keyHeadDim, valueHeadDim)
	elements, elOK := checkedIntMul(heads, row)
	if !rowOK || !elOK {
		return StateTileLayout{}, ErrStateTileOverflow
	}
	bytes, bytesOK := checkedInt64Mul(int64(elements), int64(elementBytes))
	if !bytesOK {
		return StateTileLayout{}, ErrStateTileOverflow
	}
	return StateTileLayout{
		Dtype: d, ElementBytes: elementBytes,
		Heads: heads, KeyHeadDim: keyHeadDim, ValueHeadDim: valueHeadDim,
		RowElements: row, Elements: elements, Bytes: bytes,
	}, nil
}

// Digest renders the layout's stable content identity: hex sha256 over the
// canonical field encoding, so two geometrically equal layouts share a digest
// and any field difference does not.
func (l StateTileLayout) Digest() string {
	buf := make([]byte, 0, len(l.Dtype)+1+5*4)
	buf = append(buf, l.Dtype...)
	buf = append(buf, 0)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(l.ElementBytes))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(l.Heads))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(l.KeyHeadDim))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(l.ValueHeadDim))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(l.RowElements))
	sum := sha256.Sum256(buf)
	return hex.EncodeToString(sum[:])
}

// RecurrentStateTile is one addressable chunk of a recurrent layer's
// accumulated state — a first-class cacheable object with its own key, layout
// descriptor, and group membership, restorable alongside ordinary KV groups.
type RecurrentStateTile struct {
	Key    string             // stable cache key for this tile
	Layer  int                // decoder layer index
	Chunk  int                // tile index within the layer's state (0..chunks-1)
	Layout StateTileLayout    // mamba-view layout descriptor
	Group  model.KVLayerGroup // group membership (KVGroupRecurrent)
}

// StateTilePlan addresses every recurrent layer's state as a fixed grid of
// tiles in one contiguous slab: recurrent layers in ascending index order,
// chunks in order within a layer. The slab is the unified mamba-view region
// the state tiles are stored and restored in, alongside the ordinary KV groups.
type StateTilePlan struct {
	modelID string
	cfg     model.Config
	layout  StateTileLayout
	chunks  int
	layers  []int       // recurrent layer indices, ascending
	rank    map[int]int // layer -> position among the recurrent layers
	tiles   int         // total tile count (len(layers) * chunks)
}

// NewStateTilePlan builds the tile plan for cfg's recurrent layers under the
// given layout and tiling factor. A model with no recurrent layers yields an
// empty (valid) plan: zero tiles, zero-byte slab.
// Precondition: modelID identifies the checkpoint; layout is a layout NewStateTileLayout accepted.
// Postcondition: every (layer, chunk) pair of the plan has a unique slab address and key.
func NewStateTilePlan(modelID string, cfg model.Config, layout StateTileLayout, chunksPerLayer int) (*StateTilePlan, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, ErrEmptyModelID
	}
	if chunksPerLayer < 1 {
		return nil, ErrInvalidChunks
	}
	if layout.Bytes <= 0 {
		return nil, ErrNonPositiveGeometry
	}
	var layers []int
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.KVLayerGroupOf(l) == model.KVGroupRecurrent {
			layers = append(layers, l)
		}
	}
	tiles, ok := checkedIntMul(len(layers), chunksPerLayer)
	if !ok {
		return nil, ErrStateTileOverflow
	}
	if tiles > 0 && layout.Bytes > int64(maxInt64)/int64(tiles) {
		return nil, ErrStateTileOverflow
	}
	p := &StateTilePlan{
		modelID: modelID, cfg: cfg, layout: layout,
		chunks: chunksPerLayer, layers: layers, tiles: tiles,
		rank: make(map[int]int, len(layers)),
	}
	for i, l := range layers {
		p.rank[l] = i
	}
	return p, nil
}

// TileCount is the number of addressable tiles in the slab.
func (p *StateTilePlan) TileCount() int { return p.tiles }

// SlabBytes is the total byte footprint of the tile slab.
func (p *StateTilePlan) SlabBytes() int64 { return int64(p.tiles) * p.layout.Bytes }

// Address returns the slab offset of the tile (layer, chunk).
// Precondition: plan is non-nil.
// Postcondition: ok=false when the layer is not one of the plan's recurrent layers or the
// chunk is outside [0, chunks); the offset is always a multiple of the tile byte size.
func (p *StateTilePlan) Address(layer, chunk int) (int64, bool) {
	rank, isRecurrent := p.rank[layer]
	if !isRecurrent || chunk < 0 || chunk >= p.chunks {
		return 0, false
	}
	return int64(rank*p.chunks+chunk) * p.layout.Bytes, true
}

// Decode resolves a slab offset back to (layer, chunk).
// Precondition: plan is non-nil.
// Postcondition: ok=false for negative offsets, offsets that do not align to a tile boundary,
// and offsets at or past the slab end; a decoded pair always re-encodes to the same offset.
func (p *StateTilePlan) Decode(offset int64) (int, int, bool) {
	if len(p.layers) == 0 || p.layout.Bytes <= 0 {
		return 0, 0, false
	}
	if offset < 0 || offset >= int64(p.tiles)*p.layout.Bytes {
		return 0, 0, false
	}
	idx, rem := offset/p.layout.Bytes, offset%p.layout.Bytes
	if rem != 0 {
		return 0, 0, false
	}
	rank, chunk := int(idx)/p.chunks, int(idx)%p.chunks
	if rank < 0 || rank >= len(p.layers) {
		return 0, 0, false
	}
	return p.layers[rank], chunk, true
}

// Tile returns the addressable tile record at (layer, chunk) — its own key,
// layout descriptor, and group membership — or (zero, false) when the
// coordinates are outside the plan.
func (p *StateTilePlan) Tile(layer, chunk int) (RecurrentStateTile, bool) {
	if _, ok := p.Address(layer, chunk); !ok {
		return RecurrentStateTile{}, false
	}
	return RecurrentStateTile{
		Key:    p.tileKey(layer, chunk),
		Layer:  layer,
		Chunk:  chunk,
		Layout: p.layout,
		Group:  model.KVGroupRecurrent,
	}, true
}

// tileKey renders the tile's stable cache key: model identity, layer, chunk,
// and the layout digest, so a different geometry or tiling can never collide
// with this tile's key.
func (p *StateTilePlan) tileKey(layer, chunk int) string {
	return fmt.Sprintf("kvmmu/recstate/v1/%s/l%d/c%d/%s", p.modelID, layer, chunk, p.layout.Digest())
}

// ListingKind names the kind of cacheable object a listing row describes.
type ListingKind string

const (
	// ListingKVSpan marks a token-indexed K/V span shared by every layer in the group.
	ListingKVSpan ListingKind = "kv-token-span"
	// ListingStateTile marks an O(1) recurrent-state chunk addressed as its own tile.
	ListingStateTile ListingKind = "recurrent-state-tile"
)

// CacheableEntry is one row of the unified cacheable-object listing: ordinary
// KV groups appear as token spans, the recurrent group as its state tiles —
// each a first-class object carrying its own key and layout.
type CacheableEntry struct {
	Group  model.KVLayerGroup  // the KV layer group this object belongs to
	Kind   ListingKind         // token span vs state tile
	Layers []int               // token span: the layers sharing this group's span
	Tile   *RecurrentStateTile // state tile: the tile itself (nil for token spans)
}

// CacheableListing is the unified view of everything the hybrid cache stores
// and restores: the ordinary KV groups as token spans, plus the recurrent
// state tiles alongside them, in deterministic order (full group, window
// group, then state tiles by layer and chunk).
func (p *StateTilePlan) CacheableListing() []CacheableEntry {
	var out []CacheableEntry
	var full, window []int
	for l := 0; l < p.cfg.NumLayers; l++ {
		switch p.cfg.KVLayerGroupOf(l) {
		case model.KVGroupFull:
			full = append(full, l)
		case model.KVGroupSlidingWindow:
			window = append(window, l)
		}
	}
	if len(full) > 0 {
		out = append(out, CacheableEntry{Group: model.KVGroupFull, Kind: ListingKVSpan, Layers: full})
	}
	if len(window) > 0 {
		out = append(out, CacheableEntry{Group: model.KVGroupSlidingWindow, Kind: ListingKVSpan, Layers: window})
	}
	for _, l := range p.layers {
		for c := 0; c < p.chunks; c++ {
			tile, _ := p.Tile(l, c)
			out = append(out, CacheableEntry{Group: model.KVGroupRecurrent, Kind: ListingStateTile, Tile: &tile})
		}
	}
	return out
}

const (
	maxInt   = int(^uint(0) >> 1)
	maxInt64 = int64(^uint(0) >> 1)
)

// checkedIntMul multiplies two ints, reporting overflow instead of wrapping.
func checkedIntMul(a, b int) (int, bool) {
	if a < 0 || b < 0 || (b != 0 && a > maxInt/b) {
		return 0, false
	}
	return a * b, true
}

// checkedInt64Mul multiplies two int64s, reporting overflow instead of wrapping.
func checkedInt64Mul(a, b int64) (int64, bool) {
	if a < 0 || b < 0 || (b != 0 && a > maxInt64/b) {
		return 0, false
	}
	return a * b, true
}
