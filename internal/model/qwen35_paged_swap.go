package model

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

const qwenHybridPagedSwapMagic = "FAKQHPS1"

const (
	qwenHybridSwapEncodeRecovery = "rebuild an intact Qwen hybrid KV cache, then retry with a positive block size and smaller token window"
	qwenHybridSwapDecodeRecovery = "re-serialize the cache with the matching Qwen hybrid config, then retry"
)

// QwenHybridKVCacheToHost serializes a Qwen hybrid host cache for scheduler swap.
// Only full-attention layers occupy token-indexed page rows; linear-attention
// convolution and recurrent matrices are carried exactly in a fixed-state sidecar.
func QwenHybridKVCacheToHost(c *KVCache, blockTokens int) (_ []byte, err error) {
	defer func() {
		err = qwenHybridSwapErrorWithRecovery(err, qwenHybridSwapEncodeRecovery)
	}()
	if c == nil || !c.cfg.IsQwen35Hybrid() {
		return nil, errors.New("model: Qwen hybrid swap requires a Qwen hybrid KV cache")
	}
	if blockTokens <= 0 {
		return nil, errors.New("model: Qwen hybrid swap block size must be positive")
	}
	cfg := c.cfg
	stride := c.kvStride()
	full := qwenFullAttentionLayers(cfg)
	n := c.Len()
	for _, l := range full {
		if l >= len(c.K) || l >= len(c.Kraw) || l >= len(c.V) {
			return nil, fmt.Errorf("model: Qwen hybrid swap layer %d plane inventory mismatch", l)
		}
		want := n * stride
		if len(c.K[l]) != want || len(c.Kraw[l]) != want || len(c.V[l]) != want {
			return nil, fmt.Errorf("model: Qwen hybrid swap layer %d row geometry mismatch", l)
		}
	}
	if c.linear == nil || len(c.linear.layers) != cfg.NumLayers {
		return nil, errors.New("model: Qwen hybrid swap linear layer inventory mismatch")
	}
	blocks := qwenSwapCeilDiv(n, blockTokens)
	bodyBytes, err := qwenHybridSwapBodyBytes(c, blockTokens, full, blocks, stride)
	if err != nil {
		return nil, err
	}
	if bodyBytes > int(^uint(0)>>1)-sha256.Size {
		return nil, errors.New("model: Qwen hybrid swap size overflow")
	}
	payload := make([]byte, bodyBytes+sha256.Size)
	e := qwenSwapEncoder{buf: payload[:bodyBytes]}
	e.bytes([]byte(qwenHybridPagedSwapMagic))
	e.u32(1)
	e.config(cfg)
	e.integer(blockTokens)
	e.integer(n)
	e.ints(c.pos)
	e.ints(full)
	e.integer(blocks)
	for b := 0; b < blocks; b++ {
		for _, l := range full {
			for _, plane := range [][][]float32{c.K, c.Kraw, c.V} {
				for off := 0; off < blockTokens; off++ {
					pos := b*blockTokens + off
					if pos < n {
						e.f32raw(plane[l][pos*stride : (pos+1)*stride])
					} else {
						e.zeros(stride)
					}
				}
			}
		}
	}
	e.integer(len(c.linear.layers))
	for l := range c.linear.layers {
		st := c.linear.layers[l]
		e.f32rows(st.conv)
		e.f32rows(st.recurrent)
	}
	if e.err != nil {
		return nil, e.err
	}
	if e.off != bodyBytes {
		return nil, errors.New("model: internal Qwen hybrid swap size mismatch")
	}
	sum := sha256.Sum256(payload[:bodyBytes])
	copy(payload[bodyBytes:], sum[:])
	return payload, nil
}

// QwenHybridKVCacheFromHost validates and restores a Qwen swap blob into a fresh cache.
// The destination remains private until every validation passes, so refusal cannot
// partially mutate or publish a live cache.
func QwenHybridKVCacheFromHost(cfg Config, data []byte) (_ *KVCache, err error) {
	defer func() {
		err = qwenHybridSwapErrorWithRecovery(err, qwenHybridSwapDecodeRecovery)
	}()
	if !cfg.IsQwen35Hybrid() {
		return nil, errors.New("model: Qwen hybrid restore requires a Qwen hybrid config")
	}
	if len(data) < sha256.Size {
		return nil, errors.New("model: truncated Qwen hybrid swap blob")
	}
	body, got := data[:len(data)-sha256.Size], data[len(data)-sha256.Size:]
	want := sha256.Sum256(body)
	if !bytes.Equal(got, want[:]) {
		return nil, errors.New("model: Qwen hybrid swap checksum mismatch")
	}
	d := qwenSwapDecoder{buf: body}
	if string(d.bytes()) != qwenHybridPagedSwapMagic {
		return nil, errors.New("model: invalid Qwen hybrid swap blob")
	}
	if v := d.u32(); v != 1 {
		return nil, fmt.Errorf("model: unsupported Qwen hybrid swap version %d", v)
	}
	if err := d.config(cfg); err != nil {
		return nil, err
	}
	blockTokens, n := d.integer(), d.integer()
	pos, full := d.ints(), d.ints()
	blocks := d.integer()
	stride := cfg.NumKVHeads * cfg.HeadDim
	expectedFull := qwenFullAttentionLayers(cfg)
	if blockTokens <= 0 || n < 0 || len(pos) != n || !qwenSwapEqualInts(full, expectedFull) || blocks != qwenSwapCeilDiv(n, blockTokens) {
		return nil, errors.New("model: Qwen hybrid swap geometry mismatch")
	}
	rowFloats, ok := qwenSwapSizeProduct(n, stride)
	if !ok {
		return nil, errors.New("model: Qwen hybrid swap geometry mismatch")
	}
	pageFloats, ok := qwenSwapSizeProduct(blocks, len(full), 3, blockTokens, stride)
	if !ok {
		return nil, errors.New("model: Qwen hybrid swap geometry mismatch")
	}
	if pageFloats > d.remaining()/4 {
		return nil, io.ErrUnexpectedEOF
	}
	out := newQwenHybridSwapCache(cfg)
	out.pos = pos
	for _, l := range full {
		out.K[l] = make([]float32, rowFloats)
		out.Kraw[l] = make([]float32, rowFloats)
		out.V[l] = make([]float32, rowFloats)
	}
	for b := 0; b < blocks; b++ {
		rowStart := b * blockTokens
		live := min(blockTokens, n-rowStart)
		liveFloats := live * stride
		paddingFloats := (blockTokens - live) * stride
		for _, l := range full {
			for _, dst := range [][]float32{out.K[l], out.Kraw[l], out.V[l]} {
				start := rowStart * stride
				d.f32rawInto(dst[start : start+liveFloats])
				d.skipF32(paddingFloats)
			}
		}
	}
	layerCount := d.integer()
	if layerCount != cfg.NumLayers {
		return nil, errors.New("model: Qwen hybrid swap linear layer count mismatch")
	}
	_, _, _, _, _, _, convDim := cfg.linearAttnDims()
	for l := 0; l < layerCount; l++ {
		conv := d.f32rows()
		rec := d.f32rows()
		if cfg.isLinearAttnLayer(l) {
			if len(rec) != cfg.LinearNumValueHeads {
				return nil, fmt.Errorf("model: Qwen hybrid swap recurrent geometry mismatch at layer %d", l)
			}
			for _, row := range rec {
				if len(row) != cfg.LinearKeyHeadDim*cfg.LinearValueHeadDim {
					return nil, fmt.Errorf("model: Qwen hybrid swap recurrent row mismatch at layer %d", l)
				}
			}
			if len(conv) > max(0, cfg.LinearConvKernelDim-1) {
				return nil, fmt.Errorf("model: Qwen hybrid swap conv window mismatch at layer %d", l)
			}
			for _, row := range conv {
				if len(row) != convDim {
					return nil, fmt.Errorf("model: Qwen hybrid swap conv row mismatch at layer %d", l)
				}
			}
		} else if len(conv) != 0 || len(rec) != 0 {
			return nil, fmt.Errorf("model: Qwen hybrid swap state on full-attention layer %d", l)
		}
		out.linear.layers[l].conv, out.linear.layers[l].recurrent = conv, rec
	}
	if d.err != nil {
		return nil, d.err
	}
	if d.remaining() != 0 {
		return nil, errors.New("model: trailing Qwen hybrid swap bytes")
	}
	return out, nil
}

func qwenHybridSwapErrorWithRecovery(err error, recovery string) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("%w; recovery: %s", err, recovery)
}

func newQwenHybridSwapCache(cfg Config) *KVCache {
	leanCfg := cfg
	leanCfg.LinearNumValueHeads = 0
	out := NewKVCache(leanCfg)
	out.cfg = cfg
	return out
}

func qwenFullAttentionLayers(cfg Config) []int {
	var out []int
	for l := 0; l < cfg.NumLayers; l++ {
		if !cfg.isLinearAttnLayer(l) {
			out = append(out, l)
		}
	}
	return out
}
func qwenSwapCeilDiv(n, d int) int {
	if n == 0 {
		return 0
	}
	return 1 + (n-1)/d
}
func qwenSwapEqualInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func qwenHybridSwapBodyBytes(c *KVCache, blockTokens int, full []int, blocks, stride int) (int, error) {
	s := qwenSwapSizer{}
	s.bytes(len(qwenHybridPagedSwapMagic))
	s.add(4) // format version
	for range 9 {
		s.add(8)
	}
	s.add(8) // layer-type count
	for _, layerType := range c.cfg.LayerTypes {
		s.bytes(len(layerType))
	}
	s.add(16) // blockTokens and live token count
	s.ints(c.pos)
	s.ints(full)
	s.add(8) // padded block count
	pageFloats, ok := qwenSwapSizeProduct(blocks, len(full), 3, blockTokens, stride)
	if !ok {
		return 0, errors.New("model: Qwen hybrid swap size overflow")
	}
	pageBytes, ok := qwenSwapSizeProduct(pageFloats, 4)
	if !ok {
		return 0, errors.New("model: Qwen hybrid swap size overflow")
	}
	s.add(pageBytes)
	s.add(8) // linear layer count
	for l := range c.linear.layers {
		s.f32rows(c.linear.layers[l].conv)
		s.f32rows(c.linear.layers[l].recurrent)
	}
	if s.err != nil {
		return 0, s.err
	}
	return s.n, nil
}

type qwenSwapSizer struct {
	n   int
	err error
}

func (s *qwenSwapSizer) add(n int) {
	if s.err != nil {
		return
	}
	if n < 0 || n > int(^uint(0)>>1)-s.n {
		s.err = errors.New("model: Qwen hybrid swap size overflow")
		return
	}
	s.n += n
}

func (s *qwenSwapSizer) bytes(n int) {
	s.add(8)
	s.add(n)
}

func (s *qwenSwapSizer) ints(v []int) {
	s.add(8)
	n, ok := qwenSwapSizeProduct(len(v), 8)
	if !ok {
		s.err = errors.New("model: Qwen hybrid swap size overflow")
		return
	}
	s.add(n)
}

func (s *qwenSwapSizer) f32rows(rows [][]float32) {
	s.add(8)
	for _, row := range rows {
		s.add(8)
		n, ok := qwenSwapSizeProduct(len(row), 4)
		if !ok {
			s.err = errors.New("model: Qwen hybrid swap size overflow")
			return
		}
		s.add(n)
	}
}

func qwenSwapSizeProduct(values ...int) (int, bool) {
	product := 1
	maxInt := int(^uint(0) >> 1)
	for _, value := range values {
		if value < 0 || value != 0 && product > maxInt/value {
			return 0, false
		}
		product *= value
	}
	return product, true
}

type qwenSwapEncoder struct {
	buf []byte
	off int
	err error
}

func (e *qwenSwapEncoder) take(n int) []byte {
	if e.err != nil {
		return nil
	}
	if n < 0 || n > len(e.buf)-e.off {
		e.err = io.ErrShortBuffer
		return nil
	}
	out := e.buf[e.off : e.off+n]
	e.off += n
	return out
}

func (e *qwenSwapEncoder) u32(v uint32) {
	if b := e.take(4); b != nil {
		binary.LittleEndian.PutUint32(b, v)
	}
}

func (e *qwenSwapEncoder) integer(v int) {
	if v < 0 {
		e.err = errors.New("model: negative Qwen hybrid swap value")
		return
	}
	if b := e.take(8); b != nil {
		binary.LittleEndian.PutUint64(b, uint64(v))
	}
}
func (e *qwenSwapEncoder) bytes(v []byte) {
	e.integer(len(v))
	if b := e.take(len(v)); b != nil {
		copy(b, v)
	}
}
func (e *qwenSwapEncoder) ints(v []int) {
	e.integer(len(v))
	for _, x := range v {
		e.integer(x)
	}
}
func (e *qwenSwapEncoder) f32raw(v []float32) {
	n, ok := qwenSwapSizeProduct(len(v), 4)
	if !ok {
		e.err = errors.New("model: Qwen hybrid swap size overflow")
		return
	}
	b := e.take(n)
	if b == nil {
		return
	}
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(x))
	}
}
func (e *qwenSwapEncoder) zeros(n int) {
	bytes, ok := qwenSwapSizeProduct(n, 4)
	if !ok {
		e.err = errors.New("model: Qwen hybrid swap size overflow")
		return
	}
	e.take(bytes) // The exact-sized destination is zero-initialized.
}
func (e *qwenSwapEncoder) f32rows(v [][]float32) {
	e.integer(len(v))
	for _, r := range v {
		e.integer(len(r))
		e.f32raw(r)
	}
}
func (e *qwenSwapEncoder) config(c Config) {
	for _, v := range []int{c.NumLayers, c.NumKVHeads, c.HeadDim, c.LinearNumKeyHeads, c.LinearNumValueHeads, c.LinearKeyHeadDim, c.LinearValueHeadDim, c.LinearConvKernelDim, c.FullAttentionInterval} {
		e.integer(v)
	}
	e.integer(len(c.LayerTypes))
	for _, s := range c.LayerTypes {
		e.bytes([]byte(s))
	}
}

type qwenSwapDecoder struct {
	buf []byte
	off int
	err error
}

func (d *qwenSwapDecoder) remaining() int { return len(d.buf) - d.off }

func (d *qwenSwapDecoder) take(n int) []byte {
	if d.err != nil {
		return nil
	}
	if n < 0 || n > d.remaining() {
		d.err = io.ErrUnexpectedEOF
		return nil
	}
	out := d.buf[d.off : d.off+n]
	d.off += n
	return out
}
func (d *qwenSwapDecoder) u32() uint32 {
	b := d.take(4)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(b)
}
func (d *qwenSwapDecoder) integer() int {
	b := d.take(8)
	if b == nil {
		return 0
	}
	v := binary.LittleEndian.Uint64(b)
	if v > uint64(^uint(0)>>1) {
		d.err = errors.New("model: Qwen hybrid swap integer overflow")
		return 0
	}
	return int(v)
}
func (d *qwenSwapDecoder) bytes() []byte { return d.take(d.integer()) }
func (d *qwenSwapDecoder) ints() []int {
	n := d.integer()
	if n < 0 || n > d.remaining()/8 {
		d.err = io.ErrUnexpectedEOF
		return nil
	}
	out := make([]int, n)
	b := d.take(n * 8)
	for i := range out {
		v := binary.LittleEndian.Uint64(b[i*8:])
		if v > uint64(^uint(0)>>1) {
			d.err = errors.New("model: Qwen hybrid swap integer overflow")
			return nil
		}
		out[i] = int(v)
	}
	return out
}
func (d *qwenSwapDecoder) f32raw(n int) []float32 {
	if n < 0 || n > d.remaining()/4 {
		d.err = io.ErrUnexpectedEOF
		return nil
	}
	out := make([]float32, n)
	d.f32rawInto(out)
	return out
}
func (d *qwenSwapDecoder) f32rawInto(out []float32) {
	b := d.take(len(out) * 4)
	if b == nil {
		return
	}
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
}
func (d *qwenSwapDecoder) skipF32(n int) {
	if n < 0 || n > d.remaining()/4 {
		d.err = io.ErrUnexpectedEOF
		return
	}
	d.take(n * 4)
}
func (d *qwenSwapDecoder) f32rows() [][]float32 {
	n := d.integer()
	if n < 0 || n > d.remaining()/8 {
		d.err = io.ErrUnexpectedEOF
		return nil
	}
	out := make([][]float32, n)
	for i := range out {
		out[i] = d.f32raw(d.integer())
	}
	return out
}
func (d *qwenSwapDecoder) config(c Config) error {
	vals := []int{c.NumLayers, c.NumKVHeads, c.HeadDim, c.LinearNumKeyHeads, c.LinearNumValueHeads, c.LinearKeyHeadDim, c.LinearValueHeadDim, c.LinearConvKernelDim, c.FullAttentionInterval}
	for _, want := range vals {
		if got := d.integer(); got != want {
			return errors.New("model: Qwen hybrid swap config mismatch")
		}
	}
	n := d.integer()
	if n != len(c.LayerTypes) {
		return errors.New("model: Qwen hybrid swap layer map mismatch")
	}
	for _, want := range c.LayerTypes {
		if string(d.bytes()) != want {
			return errors.New("model: Qwen hybrid swap layer map mismatch")
		}
	}
	return d.err
}
