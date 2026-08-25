package model

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const (
	hostPrefixSnapshotMagic   = "FAKHPS"
	hostPrefixSnapshotVersion = uint16(1)
	hostPrefixSnapshotHeader  = len(hostPrefixSnapshotMagic) + 2 + sha256.Size
)

var (
	ErrHostPrefixSnapshotIntegrity = errors.New("model: host prefix snapshot integrity check failed")
	ErrHostPrefixSnapshotVersion   = errors.New("model: unsupported host prefix snapshot version")
	ErrHostPrefixSnapshotScope     = errors.New("model: host prefix snapshot model configuration mismatch")
)

// MarshalBinary returns the deterministic, versioned wire image of the complete
// host-owned prefix. Every cache row, attention K/Kraw/V value, position, and
// Qwen3.5/3.6 convolution/recurrent tensor is encoded by IEEE bit pattern.
func (h *HostPrefixSnapshot) MarshalBinary() ([]byte, error) {
	if h == nil || h.cache == nil || h.backend == nil {
		return nil, errors.New("model: cannot serialize an incomplete host prefix snapshot")
	}
	if err := h.validate(); err != nil {
		return nil, err
	}
	var payload snapshotEncoder
	cfg, err := json.Marshal(h.cache.cfg)
	if err != nil {
		return nil, fmt.Errorf("model: encode host prefix configuration: %w", err)
	}
	payload.bytes(cfg)
	payload.integer(h.tokens)
	payload.cache(h.cache)
	payload.kv(h.kv)
	payload.boolean(h.qwen35 != nil)
	if h.qwen35 != nil {
		payload.count(len(h.qwen35.layers))
		for i := range h.qwen35.layers {
			payload.tensor(h.qwen35.layers[i].conv)
			payload.tensor(h.qwen35.layers[i].recurrent)
		}
	}
	if payload.err != nil {
		return nil, payload.err
	}
	body := payload.Bytes()
	sum := sha256.Sum256(body)
	out := make([]byte, 0, hostPrefixSnapshotHeader+len(body))
	out = append(out, hostPrefixSnapshotMagic...)
	var version [2]byte
	binary.BigEndian.PutUint16(version[:], hostPrefixSnapshotVersion)
	out = append(out, version[:]...)
	out = append(out, sum[:]...)
	out = append(out, body...)
	return out, nil
}

// DecodeHostPrefixSnapshot verifies and decodes a host image for expected, then
// binds its non-serializable physical handles to backend. A configuration mismatch
// is a scope fault: bytes produced for another model geometry are never installed.
func DecodeHostPrefixSnapshot(data []byte, backend compute.Backend, expected Config) (*HostPrefixSnapshot, error) {
	if backend == nil {
		return nil, errors.New("model: cannot decode host prefix snapshot without a backend")
	}
	if len(data) < hostPrefixSnapshotHeader {
		return nil, fmt.Errorf("%w: truncated header", ErrHostPrefixSnapshotIntegrity)
	}
	if string(data[:len(hostPrefixSnapshotMagic)]) != hostPrefixSnapshotMagic {
		return nil, fmt.Errorf("%w: unknown magic", ErrHostPrefixSnapshotIntegrity)
	}
	versionAt := len(hostPrefixSnapshotMagic)
	if version := binary.BigEndian.Uint16(data[versionAt : versionAt+2]); version != hostPrefixSnapshotVersion {
		return nil, fmt.Errorf("%w %d (this build reads v%d)", ErrHostPrefixSnapshotVersion, version, hostPrefixSnapshotVersion)
	}
	wantSum := data[versionAt+2 : hostPrefixSnapshotHeader]
	body := data[hostPrefixSnapshotHeader:]
	gotSum := sha256.Sum256(body)
	if !bytes.Equal(wantSum, gotSum[:]) {
		return nil, ErrHostPrefixSnapshotIntegrity
	}

	d := snapshotDecoder{data: body}
	cfgBytes := d.bytes()
	wantCfg, err := json.Marshal(expected)
	if err != nil {
		return nil, fmt.Errorf("model: encode expected host prefix configuration: %w", err)
	}
	if d.err == nil && !bytes.Equal(cfgBytes, wantCfg) {
		return nil, ErrHostPrefixSnapshotScope
	}
	// The caller's already-loaded Config is the install authority. Re-decoding its
	// JSON would run Config's compatibility/default derivation a second time and can
	// change an already-normalized geometry; the byte comparison above is the scope
	// proof, so bind the exact expected value instead.
	cfg := expected
	tokens := d.integer()
	cache := d.cache(cfg)
	kv := d.kv()
	hasQwen := d.boolean()
	var qwen *hostQwen35State
	if hasQwen {
		gdn, ok := backend.(Qwen35GDNBackend)
		if !ok || gdn.Qwen35GDNPath() != Qwen35GDNCUDAPath {
			return nil, fmt.Errorf("model: backend %q cannot restore Qwen3.5/3.6 recurrent state", backend.Name())
		}
		layerCount := d.count()
		if d.err == nil && layerCount != cfg.NumLayers {
			d.err = fmt.Errorf("Qwen recurrent-state layers=%d, want %d", layerCount, cfg.NumLayers)
			layerCount = 0
		}
		qwen = &hostQwen35State{backend: gdn, layers: make([]hostQwen35LayerState, layerCount)}
		for i := range qwen.layers {
			qwen.layers[i].conv = d.tensor()
			qwen.layers[i].recurrent = d.tensor()
		}
	}
	if d.err != nil {
		return nil, fmt.Errorf("model: decode host prefix snapshot: %w", d.err)
	}
	if d.off != len(d.data) {
		return nil, fmt.Errorf("model: decode host prefix snapshot: %d trailing bytes", len(d.data)-d.off)
	}
	out := &HostPrefixSnapshot{cache: cache, kv: kv, qwen35: qwen, backend: backend, tokens: tokens}
	if err := out.validate(); err != nil {
		out.Close()
		return nil, err
	}
	return out, nil
}

func (h *HostPrefixSnapshot) validate() error {
	if h == nil || h.cache == nil || h.backend == nil || h.tokens < 0 {
		return errors.New("model: incomplete host prefix snapshot")
	}
	if err := h.kv.Validate(); err != nil {
		return fmt.Errorf("model: invalid host prefix attention KV: %w", err)
	}
	if len(h.kv.Pos) != h.tokens {
		return fmt.Errorf("model: host prefix positions=%d, want tokens=%d", len(h.kv.Pos), h.tokens)
	}
	layers := h.cache.cfg.NumLayers
	if len(h.cache.K) != layers || len(h.cache.Kraw) != layers || len(h.cache.V) != layers {
		return fmt.Errorf("model: host prefix model-cache layers K/Kraw/V=%d/%d/%d, want %d",
			len(h.cache.K), len(h.cache.Kraw), len(h.cache.V), layers)
	}
	if h.cache.linear != nil && len(h.cache.linear.layers) != layers {
		return fmt.Errorf("model: host prefix linear-cache layers=%d, want %d", len(h.cache.linear.layers), layers)
	}
	if h.cache.glm != nil && (len(h.cache.glm.K) != layers || len(h.cache.glm.Kraw) != layers || len(h.cache.glm.V) != layers ||
		len(h.cache.glm.IndexK) != layers || len(h.cache.glm.IndexKraw) != layers) {
		return errors.New("model: host prefix GLM cache layer geometry mismatch")
	}
	if h.cache.msa != nil && (len(h.cache.msa.IndexK) != layers || len(h.cache.msa.IndexKraw) != layers) {
		return errors.New("model: host prefix MiniMax cache layer geometry mismatch")
	}
	if h.cache.cfg.IsQwen35Hybrid() != (h.qwen35 != nil) {
		return errors.New("model: host prefix Qwen recurrent-state presence does not match model configuration")
	}
	if h.qwen35 != nil {
		if h.qwen35.backend == nil || len(h.qwen35.layers) != h.cache.cfg.NumLayers {
			return errors.New("model: incomplete host prefix Qwen recurrent state")
		}
		for layer := range h.qwen35.layers {
			for _, tensor := range []hostTensor{h.qwen35.layers[layer].conv, h.qwen35.layers[layer].recurrent} {
				want := 1
				if len(tensor.shape) == 0 {
					want = 0
				}
				for _, dim := range tensor.shape {
					if dim < 0 || (dim > 0 && want > math.MaxInt/dim) {
						return fmt.Errorf("model: invalid host prefix tensor shape at layer %d", layer)
					}
					want *= dim
				}
				if want != len(tensor.data) {
					return fmt.Errorf("model: host prefix tensor at layer %d has %d values, want %d", layer, len(tensor.data), want)
				}
			}
		}
	}
	return nil
}

type snapshotEncoder struct {
	bytes.Buffer
	err error
}

func (e *snapshotEncoder) u64(v uint64) {
	e.write(v)
}

func (e *snapshotEncoder) write(v any) {
	if e.err != nil {
		return
	}
	e.err = binary.Write(&e.Buffer, binary.BigEndian, v)
}

func (e *snapshotEncoder) u32(v uint32) {
	e.write(v)
}

func (e *snapshotEncoder) integer(v int) { e.u64(uint64(int64(v))) }

func (e *snapshotEncoder) count(v int) { e.u64(uint64(v)) }

func (e *snapshotEncoder) boolean(v bool) {
	if v {
		e.WriteByte(1)
	} else {
		e.WriteByte(0)
	}
}

func (e *snapshotEncoder) bytes(v []byte) {
	e.count(len(v))
	if e.err == nil {
		_, e.err = e.Write(v)
	}
}

func (e *snapshotEncoder) ints(v []int) {
	e.count(len(v))
	for _, x := range v {
		e.integer(x)
	}
}

func (e *snapshotEncoder) f32(v []float32) {
	e.count(len(v))
	for _, x := range v {
		e.u32(math.Float32bits(x))
	}
}

func (e *snapshotEncoder) f64(v []float64) {
	e.count(len(v))
	for _, x := range v {
		e.u64(math.Float64bits(x))
	}
}

func (e *snapshotEncoder) f32rows(v [][]float32) {
	e.count(len(v))
	for _, row := range v {
		e.f32(row)
	}
}

func (e *snapshotEncoder) f64rows(v [][]float64) {
	e.count(len(v))
	for _, row := range v {
		e.f64(row)
	}
}

func (e *snapshotEncoder) cache(c *KVCache) {
	e.ints(c.pos)
	e.f32rows(c.K)
	e.f32rows(c.Kraw)
	e.f32rows(c.V)
	e.boolean(c.linear != nil)
	if c.linear != nil {
		e.count(len(c.linear.layers))
		for _, layer := range c.linear.layers {
			e.f32rows(layer.recurrent)
			e.f32rows(layer.conv)
		}
	}
	e.boolean(c.glm != nil)
	if c.glm != nil {
		e.f32rows(c.glm.K)
		e.f32rows(c.glm.Kraw)
		e.f32rows(c.glm.V)
		e.f64rows(c.glm.IndexK)
		e.f64rows(c.glm.IndexKraw)
	}
	e.boolean(c.msa != nil)
	if c.msa != nil {
		e.f32rows(c.msa.IndexK)
		e.f32rows(c.msa.IndexKraw)
	}
}

func (e *snapshotEncoder) kv(kv compute.KVHostSnapshot) {
	e.integer(kv.Config.NumLayers)
	e.integer(kv.Config.NumKVHeads)
	e.integer(kv.Config.HeadDim)
	e.u64(math.Float64bits(kv.Config.RopeTheta))
	e.u64(uint64(kv.Config.Precision))
	e.ints(kv.Config.WindowPerLayer)
	e.ints(kv.Pos)
	e.f32rows(kv.K)
	e.f32rows(kv.KRaw)
	e.f32rows(kv.V)
}

func (e *snapshotEncoder) tensor(t hostTensor) {
	e.ints(t.shape)
	e.f32(t.data)
}

type snapshotDecoder struct {
	data []byte
	off  int
	err  error
}

func (d *snapshotDecoder) take(n int) []byte {
	if d.err != nil {
		return nil
	}
	if n < 0 || n > len(d.data)-d.off {
		d.err = errors.New("truncated or oversized field")
		return nil
	}
	b := d.data[d.off : d.off+n]
	d.off += n
	return b
}

func (d *snapshotDecoder) u64() uint64 {
	b := d.take(8)
	if b == nil {
		return 0
	}
	return binary.BigEndian.Uint64(b)
}

func (d *snapshotDecoder) u32() uint32 {
	b := d.take(4)
	if b == nil {
		return 0
	}
	return binary.BigEndian.Uint32(b)
}

func (d *snapshotDecoder) count() int {
	v := d.u64()
	if d.err == nil && v > uint64(math.MaxInt) {
		d.err = errors.New("field count exceeds platform int")
		return 0
	}
	return int(v)
}

func (d *snapshotDecoder) integer() int {
	v := int64(d.u64())
	if d.err == nil && (int64(int(v)) != v) {
		d.err = errors.New("integer exceeds platform int")
		return 0
	}
	return int(v)
}

func (d *snapshotDecoder) boolean() bool {
	b := d.take(1)
	if b == nil {
		return false
	}
	if b[0] > 1 {
		d.err = errors.New("invalid boolean field")
	}
	return b[0] == 1
}

func (d *snapshotDecoder) bytes() []byte { return append([]byte(nil), d.take(d.count())...) }

func (d *snapshotDecoder) ints() []int {
	n := d.count()
	if d.err != nil || n > (len(d.data)-d.off)/8 {
		if d.err == nil {
			d.err = errors.New("oversized integer vector")
		}
		return nil
	}
	out := make([]int, n)
	for i := range out {
		out[i] = d.integer()
	}
	return out
}

func (d *snapshotDecoder) f32() []float32 {
	n := d.count()
	if d.err != nil || n > (len(d.data)-d.off)/4 {
		if d.err == nil {
			d.err = errors.New("oversized float32 vector")
		}
		return nil
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(d.u32())
	}
	return out
}

func (d *snapshotDecoder) f64() []float64 {
	n := d.count()
	if d.err != nil || n > (len(d.data)-d.off)/8 {
		if d.err == nil {
			d.err = errors.New("oversized float64 vector")
		}
		return nil
	}
	out := make([]float64, n)
	for i := range out {
		out[i] = math.Float64frombits(d.u64())
	}
	return out
}

func (d *snapshotDecoder) f32rows() [][]float32 {
	n := d.count()
	if d.err != nil || n > (len(d.data)-d.off)/8 {
		if d.err == nil {
			d.err = errors.New("oversized float32 row vector")
		}
		return nil
	}
	out := make([][]float32, n)
	for i := range out {
		out[i] = d.f32()
	}
	return out
}

func (d *snapshotDecoder) f64rows() [][]float64 {
	n := d.count()
	if d.err != nil || n > (len(d.data)-d.off)/8 {
		if d.err == nil {
			d.err = errors.New("oversized float64 row vector")
		}
		return nil
	}
	out := make([][]float64, n)
	for i := range out {
		out[i] = d.f64()
	}
	return out
}

func (d *snapshotDecoder) cache(cfg Config) *KVCache {
	c := &KVCache{cfg: cfg, pos: d.ints(), K: d.f32rows(), Kraw: d.f32rows(), V: d.f32rows()}
	if d.boolean() {
		n := d.count()
		if d.err == nil && n != cfg.NumLayers {
			d.err = fmt.Errorf("linear-cache layers=%d, want %d", n, cfg.NumLayers)
			n = 0
		}
		c.linear = &linearAttnCache{layers: make([]linearAttnLayerState, n)}
		for i := range c.linear.layers {
			c.linear.layers[i].recurrent = d.f32rows()
			c.linear.layers[i].conv = d.f32rows()
		}
	}
	if d.boolean() {
		c.glm = &glmDsaKVCache{K: d.f32rows(), Kraw: d.f32rows(), V: d.f32rows(), IndexK: d.f64rows(), IndexKraw: d.f64rows()}
	}
	if d.boolean() {
		c.msa = &minimaxKVCache{IndexK: d.f32rows(), IndexKraw: d.f32rows()}
	}
	return c
}

func (d *snapshotDecoder) kv() compute.KVHostSnapshot {
	return compute.KVHostSnapshot{
		Config: compute.KVConfig{
			NumLayers: d.integer(), NumKVHeads: d.integer(), HeadDim: d.integer(),
			RopeTheta: math.Float64frombits(d.u64()), Precision: compute.KVPrecision(d.u64()),
			WindowPerLayer: d.ints(),
		},
		Pos: d.ints(), K: d.f32rows(), KRaw: d.f32rows(), V: d.f32rows(),
	}
}

func (d *snapshotDecoder) tensor() hostTensor {
	return hostTensor{shape: d.ints(), data: d.f32()}
}
