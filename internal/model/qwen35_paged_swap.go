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

// QwenHybridKVCacheToHost serializes a Qwen hybrid host cache for scheduler swap.
// Only full-attention layers occupy token-indexed page rows; linear-attention
// convolution and recurrent matrices are carried exactly in a fixed-state sidecar.
func QwenHybridKVCacheToHost(c *KVCache, blockTokens int) ([]byte, error) {
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
		want := n * stride
		if len(c.K[l]) != want || len(c.Kraw[l]) != want || len(c.V[l]) != want {
			return nil, fmt.Errorf("model: Qwen hybrid swap layer %d row geometry mismatch", l)
		}
	}
	var payload bytes.Buffer
	e := qwenSwapEncoder{w: &payload}
	e.bytes([]byte(qwenHybridPagedSwapMagic))
	e.u32(1)
	e.config(cfg)
	e.integer(blockTokens)
	e.integer(n)
	e.ints(c.pos)
	e.ints(full)
	blocks := 0
	if n > 0 {
		blocks = (n + blockTokens - 1) / blockTokens
	}
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
	if c.linear == nil {
		return nil, errors.New("model: Qwen hybrid swap missing linear state")
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
	sum := sha256.Sum256(payload.Bytes())
	return append(payload.Bytes(), sum[:]...), nil
}

// QwenHybridKVCacheFromHost validates and restores a Qwen swap blob into a fresh cache.
// Validation completes before NewKVCache allocates the destination, so refusal cannot
// partially mutate a live cache or leak paged allocations.
func QwenHybridKVCacheFromHost(cfg Config, data []byte) (*KVCache, error) {
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
	d := qwenSwapDecoder{r: bytes.NewReader(body)}
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
	k := make([][]float32, len(full))
	kr := make([][]float32, len(full))
	v := make([][]float32, len(full))
	for b := 0; b < blocks; b++ {
		for fi := range full {
			for pi, dst := range [][][]float32{k, kr, v} {
				_ = pi
				rows := d.f32raw(blockTokens * stride)
				live := n - b*blockTokens
				if live > blockTokens {
					live = blockTokens
				}
				if live < 0 {
					live = 0
				}
				dst[fi] = append(dst[fi], rows[:live*stride]...)
			}
		}
	}
	layerCount := d.integer()
	if layerCount != cfg.NumLayers {
		return nil, errors.New("model: Qwen hybrid swap linear layer count mismatch")
	}
	conv := make([][][]float32, layerCount)
	rec := make([][][]float32, layerCount)
	_, _, _, _, _, _, convDim := cfg.linearAttnDims()
	for l := 0; l < layerCount; l++ {
		conv[l] = d.f32rows()
		rec[l] = d.f32rows()
		if cfg.isLinearAttnLayer(l) {
			if len(rec[l]) != cfg.LinearNumValueHeads {
				return nil, fmt.Errorf("model: Qwen hybrid swap recurrent geometry mismatch at layer %d", l)
			}
			for _, row := range rec[l] {
				if len(row) != cfg.LinearKeyHeadDim*cfg.LinearValueHeadDim {
					return nil, fmt.Errorf("model: Qwen hybrid swap recurrent row mismatch at layer %d", l)
				}
			}
			if len(conv[l]) > max(0, cfg.LinearConvKernelDim-1) {
				return nil, fmt.Errorf("model: Qwen hybrid swap conv window mismatch at layer %d", l)
			}
			for _, row := range conv[l] {
				if len(row) != convDim {
					return nil, fmt.Errorf("model: Qwen hybrid swap conv row mismatch at layer %d", l)
				}
			}
		} else if len(conv[l]) != 0 || len(rec[l]) != 0 {
			return nil, fmt.Errorf("model: Qwen hybrid swap state on full-attention layer %d", l)
		}
	}
	if d.err != nil {
		return nil, d.err
	}
	if d.r.Len() != 0 {
		return nil, errors.New("model: trailing Qwen hybrid swap bytes")
	}
	out := NewKVCache(cfg)
	out.pos = append([]int(nil), pos...)
	for fi, l := range full {
		out.K[l], out.Kraw[l], out.V[l] = k[fi], kr[fi], v[fi]
	}
	for l := range out.linear.layers {
		out.linear.layers[l].conv, out.linear.layers[l].recurrent = conv[l], rec[l]
	}
	return out, nil
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
	return (n + d - 1) / d
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

type qwenSwapEncoder struct {
	w   io.Writer
	err error
}

func (e *qwenSwapEncoder) write(v any) {
	if e.err == nil {
		e.err = binary.Write(e.w, binary.LittleEndian, v)
	}
}
func (e *qwenSwapEncoder) u32(v uint32) { e.write(v) }
func (e *qwenSwapEncoder) integer(v int) {
	if v < 0 {
		e.err = errors.New("model: negative Qwen hybrid swap value")
		return
	}
	e.write(uint64(v))
}
func (e *qwenSwapEncoder) bytes(v []byte) {
	e.integer(len(v))
	if e.err == nil {
		_, e.err = e.w.Write(v)
	}
}
func (e *qwenSwapEncoder) ints(v []int) {
	e.integer(len(v))
	for _, x := range v {
		e.integer(x)
	}
}
func (e *qwenSwapEncoder) f32raw(v []float32) {
	for _, x := range v {
		e.write(math.Float32bits(x))
	}
}
func (e *qwenSwapEncoder) zeros(n int) {
	for i := 0; i < n; i++ {
		e.write(uint32(0))
	}
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
	r   *bytes.Reader
	err error
}

func (d *qwenSwapDecoder) take(n int) []byte {
	if d.err != nil {
		return nil
	}
	if n < 0 || n > d.r.Len() {
		d.err = io.ErrUnexpectedEOF
		return nil
	}
	b := make([]byte, n)
	_, d.err = io.ReadFull(d.r, b)
	return b
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
	if n < 0 || n > d.r.Len()/8 {
		d.err = io.ErrUnexpectedEOF
		return nil
	}
	out := make([]int, n)
	for i := range out {
		out[i] = d.integer()
	}
	return out
}
func (d *qwenSwapDecoder) f32raw(n int) []float32 {
	if n < 0 || n > d.r.Len()/4 {
		d.err = io.ErrUnexpectedEOF
		return nil
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(d.u32())
	}
	return out
}
func (d *qwenSwapDecoder) f32rows() [][]float32 {
	n := d.integer()
	if n < 0 || n > d.r.Len()/8 {
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
