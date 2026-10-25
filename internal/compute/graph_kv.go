package compute

// GraphKVDefaultPositions is the fixed device-KV capacity, in positions, a capturing
// session preallocates until a host raises it to its resolved context with
// SetCUDAGraphKVCapacity.
const GraphKVDefaultPositions = 1024

// Graph KV prealloc decisions. GraphKVPreallocated is the only one that fixes capacity;
// the others leave the store growable and the session uncaptured.
const (
	GraphKVPreallocated  = "preallocated"
	GraphKVGraphOff      = "graph-off"
	GraphKVNotEligible   = "not-capture-eligible"
	GraphKVDoesNotFit    = "prealloc-exceeds-free-device-memory"
	GraphKVEmptyGeometry = "empty-kv-geometry"
)

// GraphKVPreallocBytes is the device memory one capturing session preallocates for
// positions: K, pre-RoPE K and V F32 rows for each of cfg.NumLayers written rows. A Qwen
// hybrid's KVConfig counts only its compacted full-attention rows, so it is charged for
// the rows it writes, not for every model layer.
func GraphKVPreallocBytes(cfg KVConfig, positions int) int64 {
	if positions <= 0 || cfg.NumLayers <= 0 || cfg.NumKVHeads <= 0 || cfg.HeadDim <= 0 {
		return 0
	}
	return 3 * int64(cfg.NumLayers) * int64(positions) * int64(cfg.NumKVHeads) * int64(cfg.HeadDim) * int64(F32.Bytes())
}

// GraphKVPrealloc is the fixed-capacity decision for one session's device KV.
type GraphKVPrealloc struct {
	Preallocate bool
	Positions   int
	Bytes       int64
	Reason      string
}

// GraphKVPreallocFor decides whether a session's device KV takes the fixed graph
// capacity. It preallocates only when graphs are on, the session is capture-eligible and
// the bytes fit the free device memory; free < 0 (unknown) fails open. A session that is
// refused stays growable and runs uncaptured instead of failing its allocation.
func GraphKVPreallocFor(graphOn bool, cfg KVConfig, positions int, free int64) GraphKVPrealloc {
	prealloc := GraphKVPrealloc{Positions: positions, Bytes: GraphKVPreallocBytes(cfg, positions)}
	switch {
	case !graphOn:
		prealloc.Reason = GraphKVGraphOff
	case !cfg.GraphCapture:
		prealloc.Reason = GraphKVNotEligible
	case prealloc.Bytes <= 0:
		prealloc.Reason = GraphKVEmptyGeometry
	case free >= 0 && prealloc.Bytes > free:
		prealloc.Reason = GraphKVDoesNotFit
	default:
		prealloc.Preallocate, prealloc.Reason = true, GraphKVPreallocated
	}
	return prealloc
}
