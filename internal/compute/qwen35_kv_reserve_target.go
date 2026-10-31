package compute

// qwen35KVReserveChunkPositions matches the decode-batch growth chunk
// (qwen35DecodeBatchKVChunk): prefill and decode reserve the same size classes.
const qwen35KVReserveChunkPositions = 256

// qwen35KVReserveTarget rounds a growable KV reservation of needed elements up to a
// whole 256-position chunk. An exact prefill reservation had a per-request size, so
// each request's freed KV landed in its own never-reused pool bucket; chunked sizes
// recur across requests and decode no longer reallocates at the first step. A fixed
// graph store (maxPos > 0), a non-positive stride, or a rounding past the C int bound
// keeps the exact size.
func qwen35KVReserveTarget(needed, stride, maxPos int) int {
	if needed <= 0 || stride <= 0 || maxPos > 0 {
		return needed
	}
	chunk := int64(qwen35KVReserveChunkPositions) * int64(stride)
	rounded := (int64(needed) + chunk - 1) / chunk * chunk
	if rounded > qwen35GDNMaxCInt {
		return needed
	}
	return int(rounded)
}
