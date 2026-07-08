package model

import (
	"bytes"
	"encoding/binary"
	"fmt"
)

// spanSerMagic tags a serialized KV span so a decoder can reject foreign bytes.
const spanSerMagic uint32 = 0x4C334B56 // "L3KV"

// SerializeSpan serializes the [from, from+n) span of this cache into a portable,
// self-describing byte blob: a small header, the span's absolute positions, and the
// pre-RoPE Kraw rows and V rows of every layer. It is the byte-source a durable
// off-box L3 KV backend (internal/l3kv) stages on a demote so the span survives
// off-box instead of being dropped — the producer the residency seam (#638) lacked.
//
// Kraw (pre-RoPE keys), NOT post-RoPE K, is serialized: post-RoPE K is bound to the
// entry's absolute position, so a span restored at a different position must be
// re-rotated in a SINGLE rotation from Kraw to stay bit-exact (composing two
// rotations drifts ~1e-6 and can flip a greedy token — see NewKVCache). V is never
// rotated, so it is carried verbatim.
//
// It supports only a plain softmax-KV cache. A recurrent/hybrid Gated-DeltaNet cache
// has no per-token rows to serialize (CanEvict reports it), and the GLM-MoE-DSA /
// MiniMax-M3 caches carry sidecar state this does not move — all return a typed error
// so the caller surfaces a FAULT (and retains the live span) rather than staging a
// partial span that would restore wrong.
func (c *KVCache) SerializeSpan(from, n int) ([]byte, error) {
	if err := c.CanEvict(); err != nil {
		return nil, err
	}
	if c.glm != nil || c.msa != nil {
		return nil, fmt.Errorf("model: SerializeSpan supports only a plain softmax-KV cache (sidecar cache state present)")
	}
	if from < 0 || n <= 0 || from+n > len(c.pos) {
		return nil, fmt.Errorf("model: SerializeSpan span [%d,%d) out of range [0,%d)", from, from+n, len(c.pos))
	}
	w := c.kvStride()
	end := from + n
	layers := c.cfg.NumLayers

	buf := new(bytes.Buffer)
	if err := binary.Write(buf, binary.LittleEndian, []uint32{spanSerMagic, uint32(layers), uint32(w), uint32(n)}); err != nil {
		return nil, fmt.Errorf("model: SerializeSpan header: %w", err)
	}
	posU := make([]uint32, n)
	for i, p := range c.pos[from:end] {
		posU[i] = uint32(p)
	}
	if err := binary.Write(buf, binary.LittleEndian, posU); err != nil {
		return nil, fmt.Errorf("model: SerializeSpan positions: %w", err)
	}
	for l := 0; l < layers; l++ {
		if len(c.Kraw[l]) < end*w || len(c.V[l]) < end*w {
			return nil, fmt.Errorf("model: SerializeSpan layer %d cache shorter than span end %d", l, end*w)
		}
		if err := binary.Write(buf, binary.LittleEndian, c.Kraw[l][from*w:end*w]); err != nil {
			return nil, fmt.Errorf("model: SerializeSpan Kraw L%d: %w", l, err)
		}
		if err := binary.Write(buf, binary.LittleEndian, c.V[l][from*w:end*w]); err != nil {
			return nil, fmt.Errorf("model: SerializeSpan V L%d: %w", l, err)
		}
	}
	return buf.Bytes(), nil
}

// StageSpanBytes lets the in-process KV backend act as the byte-source a durable
// off-box L3 KV backend (internal/l3kv) stages from. It is additive to the
// abi.KVBackend seam — reachable by type assertion exactly like CanEvict — so a
// backend that does not implement it simply cannot stage (the L3 wrapper reports a
// typed FAULT and retains the span). It delegates to KVCache.SerializeSpan, so a
// cache variant that cannot be serialized surfaces the same typed error.
func (b kvBackend) StageSpanBytes(from, n int) ([]byte, error) {
	return b.s.Cache.SerializeSpan(from, n)
}
