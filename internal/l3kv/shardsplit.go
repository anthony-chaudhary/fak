package l3kv

// shardsplit.go — write-side object-cap sharding by layer group (#5264).
//
// Why this leaf exists. A backend that puts whole KV objects (the abi.KVBackend
// seam) has a hard per-object byte ceiling — Bigtable refuses a row past ~256 MB,
// an HTTP blob pool past its own limit. A single logical KV object for a large
// model is many layers wide and can exceed that ceiling, so one serial
// all-or-nothing put is refused by the backend and the whole demote is lost.
//
// The fix (borrowed clean-room from the LMCache study, issue #5264): split the one
// oversized object into contiguous LAYER-GROUP shards, each a separate keyed object
// whose byte size is at or under the backend's ceiling, then put the shards
// independently. This turns one refused put into several accepted, independently
// retryable puts, and lets a reader page back only the layer groups it needs.
//
// This file is the deterministic core only: given each layer's serialized byte
// size and the backend's per-object byte ceiling, it returns the minimal set of
// contiguous layer groups (shards) whose byte totals each stay at or under the
// ceiling. It touches no network and no device; the keyed put of each shard and
// the read-side reassembly are the wiring follow-on, not this seam.

import (
	"errors"
	"fmt"
)

// LayerShard is one contiguous group of whole layers that a single keyed object
// holds. FirstLayer and LastLayer are inclusive indices into the ordered layer
// list; Bytes is the summed serialized byte size of those layers, which the
// split guarantees is at or under the requested per-object ceiling.
type LayerShard struct {
	FirstLayer int   // index of the first layer in this shard (inclusive)
	LastLayer  int   // index of the last layer in this shard (inclusive)
	Bytes      int64 // summed byte size of the layers in this shard, at or under the ceiling
}

// Layers returns how many whole layers this shard spans.
func (s LayerShard) Layers() int { return s.LastLayer - s.FirstLayer + 1 }

var (
	// errBadCeiling is the misuse fault for a non-positive per-object byte
	// ceiling: nothing fits under a zero or negative limit, so the request is
	// refused rather than looped forever making empty shards.
	errBadCeiling = errors.New("l3kv: per-object byte ceiling must be positive")
	// errNegativeLayer is the misuse fault for a layer reporting a byte size
	// below zero — a size that cannot be measured, refused up front.
	errNegativeLayer = errors.New("l3kv: layer byte size is negative")
	// errLayerOverCeiling is the fail-closed fault the split raises when a
	// SINGLE layer is already larger than the whole per-object ceiling. A shard
	// boundary can only fall BETWEEN layers, so one over-ceiling layer cannot be
	// made to fit and the object is honestly unshardable at this ceiling — the
	// caller is TOLD, never handed a shard that the backend will refuse.
	errLayerOverCeiling = errors.New("l3kv: a single layer exceeds the per-object byte ceiling — unshardable")
)

// SplitLayerShards groups the ordered per-layer byte sizes into the minimal set
// of contiguous shards whose byte totals each stay at or under capBytes, and
// returns the shard boundaries in layer order. The grouping is greedy — extend
// the current shard with the next layer whenever it still fits, otherwise open a
// fresh shard — which yields the fewest possible contiguous groups because a
// boundary is only ever opened when the next layer would overflow the current
// one.
//
// It is deterministic and side-effect-free: same inputs, same shards, no clock,
// no network, no device.
//
// Fail-closed cases, all typed:
//   - capBytes at or below zero → errBadCeiling (nothing fits).
//   - any layerBytes entry below zero → errNegativeLayer.
//   - any single layer larger than capBytes → errLayerOverCeiling, because a
//     boundary can only fall between layers and one over-ceiling layer can never
//     be split to fit.
//
// A layer whose size exactly equals capBytes is allowed and forms (or ends) its
// own shard — the ceiling is inclusive. Zero input layers yield zero shards and
// no error.
func SplitLayerShards(layerBytes []int64, capBytes int64) ([]LayerShard, error) {
	if capBytes <= 0 {
		return nil, fmt.Errorf("%w (got %d)", errBadCeiling, capBytes)
	}
	var shards []LayerShard
	for i, sz := range layerBytes {
		if sz < 0 {
			return nil, fmt.Errorf("%w: layer %d reports %d bytes", errNegativeLayer, i, sz)
		}
		if sz > capBytes {
			return nil, fmt.Errorf("%w: layer %d is %d bytes, ceiling is %d", errLayerOverCeiling, i, sz, capBytes)
		}
		n := len(shards)
		if n == 0 || shards[n-1].Bytes+sz > capBytes {
			// The next layer overflows the open shard (or there is none yet):
			// close the boundary here and start a fresh shard on this layer.
			shards = append(shards, LayerShard{FirstLayer: i, LastLayer: i, Bytes: sz})
			continue
		}
		// The next layer still fits under the ceiling — fold it into the open shard.
		shards[n-1].Bytes += sz
		shards[n-1].LastLayer = i
	}
	return shards, nil
}

// SplitUniformLayers is the equal-size convenience over SplitLayerShards: it
// shards a logical KV object of layers layers, each perLayerBytes bytes, under
// the per-object ceiling capBytes. It is the direct answer to "total layers,
// per-layer byte size, per-object ceiling" and shares every fail-closed rule of
// the slice form (a single layer over the ceiling is still refused).
func SplitUniformLayers(layers int, perLayerBytes, capBytes int64) ([]LayerShard, error) {
	if layers < 0 {
		return nil, fmt.Errorf("%w: layer count is negative (%d)", errNegativeLayer, layers)
	}
	sizes := make([]int64, layers)
	for i := range sizes {
		sizes[i] = perLayerBytes
	}
	return SplitLayerShards(sizes, capBytes)
}
