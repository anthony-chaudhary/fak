package model

import "github.com/anthony-chaudhary/fak/internal/compute"

const hostPositionBytes = 8

// ResidentBytes reports the deterministic payload owned by this complete prefix
// snapshot. It reads tensor metadata only; it never copies device data to the host.
func (p *PrefixSnapshot) ResidentBytes() int64 {
	if p == nil || p.Cache == nil {
		return 0
	}
	bytes := p.Cache.residentBytes()
	if p.halKV != nil {
		bytes += p.halKV.ResidentBytes()
		bytes += p.halLineage.metadataBytes()
	}
	if p.qwen35 != nil {
		for i := range p.qwen35.layers {
			bytes += tensorResidentBytes(p.qwen35.layers[i].conv)
			bytes += tensorResidentBytes(p.qwen35.layers[i].recurrent)
		}
	}
	bytes += p.v41ResidentBytes()
	return bytes
}

// v41ResidentBytes reports the deterministic payload held by this snapshot's
// V4.1 continuation state: the committed token history plus every temporal
// attention state's window ring, incomplete compressor group, shared
// publications, and selections. It is a pure byte count over owned slices; a
// snapshot with no V4.1 state reports 0, so it never inflates the ledger for a
// non-V4.1 prefix (fak#13342).
func (p *PrefixSnapshot) v41ResidentBytes() int64 {
	if p == nil || p.v41 == nil {
		return 0
	}
	bytes := int64(len(p.v41.history)) * hostPositionBytes
	bytes += p.v41.attn.residentBytes()
	for _, layer := range p.v41.layers {
		bytes += layer.residentBytes()
	}
	return bytes
}

// residentBytes reports the owned payload of one temporal attention state.
func (s *V41AttentionState) residentBytes() int64 {
	if s == nil {
		return 0
	}
	bytes := v41RowsBytes(s.window) + v41RowsBytes(s.partialKV) + v41RowsBytes(s.partialInputs)
	bytes += int64(len(s.partialPositions)) * hostPositionBytes
	bytes += v41PublicationBytes(s.kvPublications) + v41PublicationBytes(s.indexPublications)
	bytes += int64(len(s.candidates))
	for _, row := range s.topk {
		bytes += int64(len(row)) * int32Bytes
	}
	return bytes
}

const int32Bytes = 4

func v41RowsBytes(rows [][]float32) int64 {
	var n int64
	for _, row := range rows {
		n += int64(len(row))
	}
	return n * int64(compute.F32.Bytes())
}

func v41PublicationBytes(in map[v41AttentionPublicationKey][]float32) int64 {
	var n int64
	for _, row := range in {
		n += int64(len(row))
	}
	return n * int64(compute.F32.Bytes())
}

func tensorResidentBytes(t compute.Tensor) int64 {
	if t.Buf() == nil {
		return 0
	}
	return int64(t.Numel()) * int64(t.Dtype.Bytes())
}

func (c *KVCache) residentBytes() int64 {
	if c == nil {
		return 0
	}
	var f32, f64 int64
	for i := range c.K {
		f32 += int64(len(c.K[i]) + len(c.Kraw[i]) + len(c.V[i]))
	}
	if c.linear != nil {
		for i := range c.linear.layers {
			f32 += int64(len(c.linear.layers[i].conv) + len(c.linear.layers[i].recurrent))
		}
	}
	if c.glm != nil {
		for i := range c.glm.K {
			f32 += int64(len(c.glm.K[i]) + len(c.glm.Kraw[i]) + len(c.glm.V[i]))
			f64 += int64(len(c.glm.IndexK[i]) + len(c.glm.IndexKraw[i]))
		}
	}
	if c.msa != nil {
		for i := range c.msa.IndexK {
			f32 += int64(len(c.msa.IndexK[i]) + len(c.msa.IndexKraw[i]))
		}
	}
	return f32*int64(compute.F32.Bytes()) + f64*8 + int64(len(c.pos))*hostPositionBytes + c.lineage.metadataBytes()
}

// TokenLineageMetadataBytes reports the exact compact lineage payload included
// in this prefix snapshot's resident-byte receipt.
func (p *PrefixSnapshot) TokenLineageMetadataBytes() int64 {
	if p == nil || p.Cache == nil {
		return 0
	}
	return p.Cache.lineage.metadataBytes() + p.halLineage.metadataBytes()
}

// NewHostPrefixSnapshotForTest constructs an independently owned host snapshot for
// cross-package cache-budget tests without exposing snapshot internals in production.
func NewHostPrefixSnapshotForTest(cache *KVCache) *PrefixSnapshot {
	if cache == nil {
		return nil
	}
	return &PrefixSnapshot{Cache: cache, Tokens: cache.Len()}
}
