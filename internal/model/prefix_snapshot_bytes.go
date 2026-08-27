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
	return bytes
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
