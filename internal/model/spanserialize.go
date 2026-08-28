package model

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
)

// spanSerMagicV2 intentionally differs from the old write-only "L3KV" image.
// Cache payloads are regenerable, so an old image is refused instead of guessed
// into the stricter live-install contract.
const spanSerMagicV2 = "FAKL3SP2"

const spanSerHeaderBytes = len(spanSerMagicV2) + 4 + 32 + 32 + 5*4

type serializedSpan struct {
	from           int
	n              int
	originalLen    int
	survivorDigest [32]byte
	positions      []int
	lineage        []uint32
	kraw           [][]float32
	v              [][]float32
}

// SerializeSpan serializes the exact state required to reverse one later
// [from,from+n) eviction. It includes the original geometry and token lineage so
// a payload cannot be installed into a merely shape-compatible unrelated cache.
func (c *KVCache) SerializeSpan(from, n int) ([]byte, error) {
	if c == nil {
		return nil, fmt.Errorf("model: SerializeSpan nil cache")
	}
	if err := c.CanEvict(); err != nil {
		return nil, err
	}
	if c.glm != nil || c.msa != nil {
		return nil, fmt.Errorf("model: SerializeSpan supports only a plain softmax-KV cache (sidecar cache state present)")
	}
	if from < 0 || n <= 0 || from+n > len(c.pos) {
		return nil, fmt.Errorf("model: SerializeSpan span [%d,%d) out of range [0,%d)", from, from+n, len(c.pos))
	}
	if c.lineage.fault != "" || len(c.lineage.ids) != len(c.pos) {
		return nil, fmt.Errorf("model: SerializeSpan requires exact token lineage for every resident position")
	}
	w := c.kvStride()
	if w <= 0 || c.cfg.NumLayers <= 0 {
		return nil, fmt.Errorf("model: SerializeSpan invalid cache geometry layers=%d stride=%d", c.cfg.NumLayers, w)
	}
	for i, p := range c.pos {
		if p != i {
			return nil, fmt.Errorf("model: SerializeSpan requires canonical positions: index %d carries %d", i, p)
		}
	}
	rowFloats, ok := checkedSpanProduct(n, w, c.cfg.NumLayers, 2)
	if !ok {
		return nil, fmt.Errorf("model: SerializeSpan size overflow")
	}
	body64 := int64(spanSerHeaderBytes) + int64(n)*8 + int64(rowFloats)*4
	maxInt := int64(^uint(0) >> 1)
	if body64 < int64(spanSerHeaderBytes) || body64 > maxInt-int64(sha256.Size) {
		return nil, fmt.Errorf("model: SerializeSpan size overflow")
	}
	bodyBytes := int(body64)
	payload := make([]byte, bodyBytes+sha256.Size)
	off := 0
	putBytes := func(src []byte) { copy(payload[off:], src); off += len(src) }
	putU32 := func(v uint32) { binary.LittleEndian.PutUint32(payload[off:], v); off += 4 }
	putF32 := func(rows []float32) {
		for _, v := range rows {
			putU32(math.Float32bits(v))
		}
	}
	putBytes([]byte(spanSerMagicV2))
	putU32(2)
	cfgDigest, err := spanConfigDigest(c.cfg)
	if err != nil {
		return nil, fmt.Errorf("model: SerializeSpan config identity: %w", err)
	}
	putBytes(cfgDigest[:])
	survivorDigest, err := spanPostEvictionDigest(c, from, n)
	if err != nil {
		return nil, fmt.Errorf("model: SerializeSpan survivor identity: %w", err)
	}
	putBytes(survivorDigest[:])
	putU32(uint32(c.cfg.NumLayers))
	putU32(uint32(w))
	putU32(uint32(from))
	putU32(uint32(n))
	putU32(uint32(len(c.pos)))
	for _, p := range c.pos[from : from+n] {
		putU32(uint32(p))
	}
	for _, id := range c.lineage.ids[from : from+n] {
		putU32(id)
	}
	end := from + n
	for l := 0; l < c.cfg.NumLayers; l++ {
		if len(c.Kraw[l]) != len(c.pos)*w || len(c.V[l]) != len(c.pos)*w || len(c.K[l]) != len(c.pos)*w {
			return nil, fmt.Errorf("model: SerializeSpan layer %d row geometry mismatch", l)
		}
		putF32(c.Kraw[l][from*w : end*w])
		putF32(c.V[l][from*w : end*w])
	}
	if off != bodyBytes {
		return nil, fmt.Errorf("model: SerializeSpan internal size mismatch")
	}
	sum := sha256.Sum256(payload[:bodyBytes])
	copy(payload[bodyBytes:], sum[:])
	return payload, nil
}

// RestoreSpan installs a version-2 span image as the exact reverse of one
// eviction. Parsing, checksum/config/lineage validation, and all allocations
// complete before any field on c is changed. The caller owns exclusive cache
// mutation; concurrent demand coordination remains a higher-level follow-on.
func (c *KVCache) RestoreSpan(payload []byte) (int, error) {
	if c == nil {
		return 0, fmt.Errorf("model: RestoreSpan nil cache")
	}
	if err := c.CanEvict(); err != nil {
		return 0, err
	}
	if c.glm != nil || c.msa != nil {
		return 0, fmt.Errorf("model: RestoreSpan supports only a plain softmax-KV cache (sidecar cache state present)")
	}
	span, err := decodeSerializedSpan(c.cfg, payload)
	if err != nil {
		return 0, err
	}
	currentLen := len(c.pos)
	if span.originalLen-span.n != currentLen || span.from < 0 || span.from > currentLen {
		return 0, fmt.Errorf("model: RestoreSpan is not the reverse of current cache geometry: current=%d image=%d-%d from=%d", currentLen, span.originalLen, span.n, span.from)
	}
	if c.lineage.fault != "" || len(c.lineage.ids) != currentLen {
		return 0, fmt.Errorf("model: RestoreSpan current cache has incomplete token lineage")
	}
	w := c.kvStride()
	for i, p := range c.pos {
		if p != i {
			return 0, fmt.Errorf("model: RestoreSpan current position %d carries %d, want canonical", i, p)
		}
	}
	for i, p := range span.positions {
		if p != span.from+i {
			return 0, fmt.Errorf("model: RestoreSpan image position %d carries %d, want %d", i, p, span.from+i)
		}
	}
	currentDigest, err := spanCacheDigest(c)
	if err != nil {
		return 0, fmt.Errorf("model: RestoreSpan current survivor identity: %w", err)
	}
	if currentDigest != span.survivorDigest {
		return 0, fmt.Errorf("model: RestoreSpan current cache does not match staged post-eviction survivor state")
	}
	fullLineage := make([]uint32, 0, span.originalLen)
	fullLineage = append(fullLineage, c.lineage.ids[:span.from]...)
	fullLineage = append(fullLineage, span.lineage...)
	fullLineage = append(fullLineage, c.lineage.ids[span.from:]...)

	newK := make([][]float32, c.cfg.NumLayers)
	newKraw := make([][]float32, c.cfg.NumLayers)
	newV := make([][]float32, c.cfg.NumLayers)
	for l := 0; l < c.cfg.NumLayers; l++ {
		if len(c.K[l]) != currentLen*w || len(c.Kraw[l]) != currentLen*w || len(c.V[l]) != currentLen*w {
			return 0, fmt.Errorf("model: RestoreSpan current layer %d row geometry mismatch", l)
		}
		newK[l] = make([]float32, span.originalLen*w)
		newKraw[l] = make([]float32, span.originalLen*w)
		newV[l] = make([]float32, span.originalLen*w)
		prefix := span.from * w
		insertEnd := (span.from + span.n) * w
		copy(newK[l][:prefix], c.K[l][:prefix])
		copy(newKraw[l][:prefix], c.Kraw[l][:prefix])
		copy(newV[l][:prefix], c.V[l][:prefix])
		copy(newKraw[l][prefix:insertEnd], span.kraw[l])
		copy(newV[l][prefix:insertEnd], span.v[l])
		copy(newKraw[l][insertEnd:], c.Kraw[l][prefix:])
		copy(newV[l][insertEnd:], c.V[l][prefix:])
		for i := span.from; i < span.originalLen; i++ {
			dst := newK[l][i*w : (i+1)*w]
			copy(dst, newKraw[l][i*w:(i+1)*w])
			if c.cfg.Alibi {
				continue
			}
			cos, sin := ropeRowForLayer(c.cfg, l, i)
			for h := 0; h < c.cfg.NumKVHeads; h++ {
				applyRopeRow(dst[h*c.cfg.HeadDim:(h+1)*c.cfg.HeadDim], cos, sin)
			}
		}
	}
	newPos := make([]int, span.originalLen)
	for i := range newPos {
		newPos[i] = i
	}
	// Publish only after the complete replacement is ready.
	c.K, c.Kraw, c.V, c.pos = newK, newKraw, newV, newPos
	c.lineage = tokenLineage{ids: fullLineage}
	return span.n, nil
}

func decodeSerializedSpan(cfg Config, payload []byte) (serializedSpan, error) {
	var out serializedSpan
	if len(payload) < spanSerHeaderBytes+sha256.Size {
		return out, fmt.Errorf("model: RestoreSpan truncated or legacy span image")
	}
	body := payload[:len(payload)-sha256.Size]
	want := sha256.Sum256(body)
	if !equalSpanBytes(want[:], payload[len(body):]) {
		return out, fmt.Errorf("model: RestoreSpan checksum mismatch")
	}
	off := 0
	readBytes := func(n int) ([]byte, bool) {
		if n < 0 || off > len(body)-n {
			return nil, false
		}
		b := body[off : off+n]
		off += n
		return b, true
	}
	readU32 := func() (uint32, bool) {
		b, ok := readBytes(4)
		if !ok {
			return 0, false
		}
		return binary.LittleEndian.Uint32(b), true
	}
	magic, _ := readBytes(len(spanSerMagicV2))
	if string(magic) != spanSerMagicV2 {
		return out, fmt.Errorf("model: RestoreSpan unsupported legacy or foreign span image")
	}
	version, ok := readU32()
	if !ok || version != 2 {
		return out, fmt.Errorf("model: RestoreSpan unsupported span image version %d", version)
	}
	encodedCfg, ok := readBytes(32)
	if !ok {
		return out, fmt.Errorf("model: RestoreSpan truncated config identity")
	}
	cfgDigest, err := spanConfigDigest(cfg)
	if err != nil || !equalSpanBytes(encodedCfg, cfgDigest[:]) {
		return out, fmt.Errorf("model: RestoreSpan config identity mismatch")
	}
	survivorDigest, ok := readBytes(32)
	if !ok {
		return out, fmt.Errorf("model: RestoreSpan truncated survivor identity")
	}
	copy(out.survivorDigest[:], survivorDigest)
	layersU, ok1 := readU32()
	strideU, ok2 := readU32()
	fromU, ok3 := readU32()
	nU, ok4 := readU32()
	originalU, ok5 := readU32()
	if !ok1 || !ok2 || !ok3 || !ok4 || !ok5 {
		return serializedSpan{}, fmt.Errorf("model: RestoreSpan truncated geometry")
	}
	layers, stride, from, n, originalLen := int(layersU), int(strideU), int(fromU), int(nU), int(originalU)
	if layers != cfg.NumLayers || stride != cfg.NumKVHeads*cfg.HeadDim || layers <= 0 || stride <= 0 || n <= 0 || originalLen < n || from > originalLen-n {
		return serializedSpan{}, fmt.Errorf("model: RestoreSpan invalid image geometry layers=%d stride=%d from=%d n=%d original=%d", layers, stride, from, n, originalLen)
	}
	rowFloats, sizeOK := checkedSpanProduct(n, stride, layers, 2)
	expected64 := int64(spanSerHeaderBytes) + int64(n)*8 + int64(rowFloats)*4
	if !sizeOK || expected64 != int64(len(body)) {
		return serializedSpan{}, fmt.Errorf("model: RestoreSpan image length does not match geometry")
	}
	out.from, out.n, out.originalLen = from, n, originalLen
	out.positions = make([]int, n)
	out.lineage = make([]uint32, n)
	out.kraw = make([][]float32, layers)
	out.v = make([][]float32, layers)
	for i := range out.positions {
		v, _ := readU32()
		out.positions[i] = int(v)
	}
	for i := range out.lineage {
		out.lineage[i], _ = readU32()
	}
	for l := 0; l < layers; l++ {
		out.kraw[l] = make([]float32, n*stride)
		out.v[l] = make([]float32, n*stride)
		for i := range out.kraw[l] {
			v, _ := readU32()
			out.kraw[l][i] = math.Float32frombits(v)
		}
		for i := range out.v[l] {
			v, _ := readU32()
			out.v[l][i] = math.Float32frombits(v)
		}
	}
	if off != len(body) {
		return serializedSpan{}, fmt.Errorf("model: RestoreSpan trailing bytes")
	}
	return out, nil
}

func spanConfigDigest(cfg Config) ([32]byte, error) {
	b, err := json.Marshal(cfg)
	if err != nil {
		return [32]byte{}, err
	}
	return sha256.Sum256(b), nil
}

func spanCacheDigest(c *KVCache) ([32]byte, error) {
	return spanDigestRows(c, -1, 0)
}

// spanPostEvictionDigest hashes the exact state Evict(from,n) must produce,
// without cloning the cache or mutating it. K rows after the removed span are
// rederived from Kraw at their compacted position using the same one-pass RoPE
// rule as KVCache.Evict.
func spanPostEvictionDigest(c *KVCache, from, n int) ([32]byte, error) {
	return spanDigestRows(c, from, n)
}

func spanDigestRows(c *KVCache, from, n int) ([32]byte, error) {
	var zero [32]byte
	if c == nil || c.lineage.fault != "" || len(c.lineage.ids) != len(c.pos) {
		return zero, fmt.Errorf("incomplete cache identity")
	}
	remove := from >= 0
	if remove && (n <= 0 || from+n > len(c.pos)) {
		return zero, fmt.Errorf("invalid removed span")
	}
	w := c.kvStride()
	h := sha256.New()
	writeU32 := func(v uint32) {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], v)
		_, _ = h.Write(b[:])
	}
	writeF32 := func(rows []float32) {
		for _, v := range rows {
			writeU32(math.Float32bits(v))
		}
	}
	cfgDigest, err := spanConfigDigest(c.cfg)
	if err != nil {
		return zero, err
	}
	_, _ = h.Write(cfgDigest[:])
	newLen := len(c.pos)
	if remove {
		newLen -= n
	}
	writeU32(uint32(newLen))
	for i := 0; i < newLen; i++ {
		writeU32(uint32(i))
	}
	for i, id := range c.lineage.ids {
		if remove && i >= from && i < from+n {
			continue
		}
		writeU32(id)
	}
	for l := 0; l < c.cfg.NumLayers; l++ {
		if len(c.K[l]) != len(c.pos)*w || len(c.Kraw[l]) != len(c.pos)*w || len(c.V[l]) != len(c.pos)*w {
			return zero, fmt.Errorf("layer %d row geometry mismatch", l)
		}
		for old := 0; old < len(c.pos); old++ {
			if remove && old >= from && old < from+n {
				continue
			}
			newPos := old
			if remove && old >= from+n {
				newPos -= n
			}
			if newPos == old {
				writeF32(c.K[l][old*w : (old+1)*w])
				continue
			}
			row := append([]float32(nil), c.Kraw[l][old*w:(old+1)*w]...)
			if !c.cfg.Alibi {
				cos, sin := ropeRowForLayer(c.cfg, l, newPos)
				for head := 0; head < c.cfg.NumKVHeads; head++ {
					applyRopeRow(row[head*c.cfg.HeadDim:(head+1)*c.cfg.HeadDim], cos, sin)
				}
			}
			writeF32(row)
		}
		for i := 0; i < len(c.pos); i++ {
			if !remove || i < from || i >= from+n {
				writeF32(c.Kraw[l][i*w : (i+1)*w])
			}
		}
		for i := 0; i < len(c.pos); i++ {
			if !remove || i < from || i >= from+n {
				writeF32(c.V[l][i*w : (i+1)*w])
			}
		}
	}
	copy(zero[:], h.Sum(nil))
	return zero, nil
}

func checkedSpanProduct(values ...int) (int, bool) {
	n := 1
	max := int(^uint(0) >> 1)
	for _, v := range values {
		if v < 0 || (v != 0 && n > max/v) {
			return 0, false
		}
		n *= v
	}
	return n, true
}

func equalSpanBytes(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	var diff byte
	for i := range a {
		diff |= a[i] ^ b[i]
	}
	return diff == 0
}

// StageSpanBytes exposes strict serialization to the durable L3 wrapper.
func (b kvBackend) StageSpanBytes(from, n int) ([]byte, error) {
	if b.s == nil {
		return nil, fmt.Errorf("model: StageSpanBytes nil session")
	}
	b.s.cacheGeometryMu.RLock()
	defer b.s.cacheGeometryMu.RUnlock()
	if b.s.Backend != nil || b.s.halKV != nil {
		return nil, fmt.Errorf("model: StageSpanBytes supports only a host-owned KV cache; HAL/device session refused")
	}
	if b.s.Cache == nil {
		return nil, fmt.Errorf("model: StageSpanBytes session has no host KV cache")
	}
	return b.s.Cache.SerializeSpan(from, n)
}

// RestoreSpanBytes is the optional materialization capability consumed by l3kv.
func (b kvBackend) RestoreSpanBytes(payload []byte) (int, error) {
	if b.s == nil {
		return 0, fmt.Errorf("model: RestoreSpanBytes nil session")
	}
	b.s.cacheGeometryMu.Lock()
	defer b.s.cacheGeometryMu.Unlock()
	if b.s.Backend != nil || b.s.halKV != nil {
		return 0, fmt.Errorf("model: RestoreSpanBytes supports only a host-owned KV cache; HAL/device session refused")
	}
	if b.s.Cache == nil {
		return 0, fmt.Errorf("model: RestoreSpanBytes session has no host KV cache")
	}
	return b.s.Cache.RestoreSpan(payload)
}
