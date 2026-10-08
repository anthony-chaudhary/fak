package model

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
)

const (
	cpuPrefixSnapshotMagic   = "FAKCPS"
	cpuPrefixSnapshotVersion = uint16(1)
	cpuPrefixSnapshotHeader  = len(cpuPrefixSnapshotMagic) + 2 + sha256.Size
)

var (
	ErrCPUPrefixSnapshotIntegrity = errors.New("model: CPU prefix snapshot integrity check failed")
	ErrCPUPrefixSnapshotVersion   = errors.New("model: unsupported CPU prefix snapshot version")
	ErrCPUPrefixSnapshotScope     = errors.New("model: CPU prefix snapshot model configuration mismatch")
	ErrCPUPrefixSnapshotState     = errors.New("model: CPU prefix snapshot contains unsupported continuation state")
)

// HostDiskSerializable reports whether p has no private continuation state
// omitted by restart persistence. Exported placement and policy fields belong
// to the outer persistence envelope; codec-specific validation still runs when
// its payload is encoded.
func (p *PrefixSnapshot) HostDiskSerializable() bool {
	if p == nil || p.Cache == nil || p.Tokens < 0 {
		return false
	}
	if p.v41 != nil || p.v41Tokens != 0 || p.hasV41Tokens || p.v41DeviceIdentity != nil || p.hasV41DeviceIdentity {
		return false
	}
	if p.captureTargetHidden || len(p.targetHidden) != 0 || len(p.targetHiddenTokens) != 0 {
		return false
	}
	if p.Backend != nil {
		return p.halKV != nil
	}
	if p.halKV != nil || p.qwen35 != nil || len(p.halLineage.ids) != 0 || p.halLineage.fault != "" {
		return false
	}
	return true
}

// MarshalCPU returns a deterministic, versioned image of a complete backend-nil
// prefix. State that is not represented by KVCache is refused rather than omitted.
func (p *PrefixSnapshot) MarshalCPU() ([]byte, error) {
	if err := validateCPUPrefixSnapshot(p); err != nil {
		return nil, err
	}

	var payload snapshotEncoder
	cfg, err := json.Marshal(p.Cache.cfg)
	if err != nil {
		return nil, fmt.Errorf("model: encode CPU prefix configuration: %w", err)
	}
	payload.bytes(cfg)
	payload.integer(p.Tokens)
	payload.integer(p.DenseGPULayers)
	payload.integer(p.GPULayers)
	payload.u64(uint64(p.ExecutionPolicy))
	payload.cache(p.Cache)
	encodeCPUCachePrecision(&payload, p.Cache)
	if payload.err != nil {
		return nil, payload.err
	}

	body := payload.Bytes()
	sum := sha256.Sum256(body)
	out := make([]byte, 0, cpuPrefixSnapshotHeader+len(body))
	out = append(out, cpuPrefixSnapshotMagic...)
	var version [2]byte
	binary.BigEndian.PutUint16(version[:], cpuPrefixSnapshotVersion)
	out = append(out, version[:]...)
	out = append(out, sum[:]...)
	out = append(out, body...)
	return out, nil
}

// DecodeCPUPrefixSnapshot verifies data against expected and returns an
// independently owned backend-nil prefix suitable for PrefixSnapshot.Restore.
func DecodeCPUPrefixSnapshot(data []byte, expected Config) (*PrefixSnapshot, error) {
	if len(data) < cpuPrefixSnapshotHeader {
		return nil, fmt.Errorf("%w: truncated header", ErrCPUPrefixSnapshotIntegrity)
	}
	if string(data[:len(cpuPrefixSnapshotMagic)]) != cpuPrefixSnapshotMagic {
		return nil, fmt.Errorf("%w: unknown magic", ErrCPUPrefixSnapshotIntegrity)
	}
	versionAt := len(cpuPrefixSnapshotMagic)
	if version := binary.BigEndian.Uint16(data[versionAt : versionAt+2]); version != cpuPrefixSnapshotVersion {
		return nil, fmt.Errorf("%w %d (this build reads v%d)", ErrCPUPrefixSnapshotVersion, version, cpuPrefixSnapshotVersion)
	}
	wantSum := data[versionAt+2 : cpuPrefixSnapshotHeader]
	body := data[cpuPrefixSnapshotHeader:]
	gotSum := sha256.Sum256(body)
	if !bytes.Equal(wantSum, gotSum[:]) {
		return nil, ErrCPUPrefixSnapshotIntegrity
	}

	d := snapshotDecoder{data: body}
	cfgBytes := d.bytes()
	wantCfg, err := json.Marshal(expected)
	if err != nil {
		return nil, fmt.Errorf("model: encode expected CPU prefix configuration: %w", err)
	}
	if d.err == nil && !bytes.Equal(cfgBytes, wantCfg) {
		return nil, ErrCPUPrefixSnapshotScope
	}
	tokens := d.integer()
	denseGPULayers := d.integer()
	gpuLayers := d.integer()
	executionPolicy := ExecutionPolicy(d.u64())
	cache := d.cache(expected)
	decodeCPUCachePrecision(&d, cache)
	if d.err != nil {
		return nil, fmt.Errorf("model: decode CPU prefix snapshot: %w", d.err)
	}
	if d.off != len(d.data) {
		return nil, fmt.Errorf("model: decode CPU prefix snapshot: %d trailing bytes", len(d.data)-d.off)
	}

	out := &PrefixSnapshot{
		Cache: cache, Tokens: tokens, DenseGPULayers: denseGPULayers,
		GPULayers: gpuLayers, ExecutionPolicy: executionPolicy,
	}
	if err := validateCPUPrefixSnapshot(out); err != nil {
		out.Close()
		return nil, err
	}
	return out, nil
}

func encodeCPUCachePrecision(e *snapshotEncoder, c *KVCache) {
	precision := c.prec
	if precision == "" {
		precision = KVPrecisionFP32
	}
	e.bytes([]byte(precision))
	if precision != KVPrecisionQ8_0 {
		return
	}
	encodePacked := func(rows []kvPackedRow) {
		e.count(len(rows))
		for i := range rows {
			e.integer(rows[i].width)
			codes := make([]byte, len(rows[i].codes))
			for j := range rows[i].codes {
				codes[j] = byte(rows[i].codes[j])
			}
			e.bytes(codes)
			e.f32(rows[i].scales)
		}
	}
	encodePacked(c.kQ8)
	encodePacked(c.vQ8)
}

func decodeCPUCachePrecision(d *snapshotDecoder, c *KVCache) {
	precision := KVPrecision(string(d.bytes()))
	switch precision {
	case KVPrecisionFP32:
		c.prec = KVPrecisionFP32
	case KVPrecisionQ8_0:
		c.prec = KVPrecisionQ8_0
		decodePacked := func() []kvPackedRow {
			n := d.count()
			if d.err != nil || n > (len(d.data)-d.off)/8 {
				if d.err == nil {
					d.err = errors.New("oversized packed-cache layer vector")
				}
				return nil
			}
			out := make([]kvPackedRow, n)
			for i := range out {
				out[i].width = d.integer()
				codes := d.bytes()
				out[i].codes = make([]int8, len(codes))
				for j := range codes {
					out[i].codes[j] = int8(codes[j])
				}
				out[i].scales = d.f32()
			}
			return out
		}
		c.kQ8 = decodePacked()
		c.vQ8 = decodePacked()
	default:
		d.err = fmt.Errorf("unsupported KV precision %q", precision)
	}
}

func validateCPUPrefixSnapshot(p *PrefixSnapshot) error {
	if p == nil || p.Cache == nil || p.Tokens < 0 {
		return errors.New("model: incomplete CPU prefix snapshot")
	}
	if p.Backend != nil || p.halKV != nil || p.qwen35 != nil || len(p.halLineage.ids) != 0 || p.halLineage.fault != "" {
		return fmt.Errorf("%w: device/HAL state", ErrCPUPrefixSnapshotState)
	}
	if p.v41 != nil || p.hasV41Tokens || p.hasV41DeviceIdentity {
		return fmt.Errorf("%w: DeepSeek V4.1 state", ErrCPUPrefixSnapshotState)
	}
	if p.captureTargetHidden || len(p.targetHidden) != 0 || len(p.targetHiddenTokens) != 0 {
		return fmt.Errorf("%w: captured MTP hidden history", ErrCPUPrefixSnapshotState)
	}
	if p.ExecutionPolicy != ExecutionPolicyPortable && p.ExecutionPolicy != ExecutionPolicyDeviceOnly {
		return fmt.Errorf("model: invalid CPU prefix execution policy %d", p.ExecutionPolicy)
	}
	c := p.Cache
	if !c.cfg.KVPrefixReuseSupported() {
		return fmt.Errorf("%w: architecture %s", ErrCPUPrefixSnapshotState, c.cfg.archFamilyKey())
	}
	if p.Tokens != c.Len() {
		return fmt.Errorf("model: CPU prefix tokens=%d, want cache positions=%d", p.Tokens, c.Len())
	}
	if c.lineage.fault != "" || len(c.lineage.ids) != c.Len() {
		return fmt.Errorf("model: CPU prefix token lineage positions=%d, want %d", len(c.lineage.ids), c.Len())
	}
	for i, pos := range c.pos {
		if pos != i {
			return fmt.Errorf("model: CPU prefix position %d carries absolute position %d", i, pos)
		}
	}
	if err := validateCPUCacheGeometry(c); err != nil {
		return err
	}
	return nil
}

func validateCPUCacheGeometry(c *KVCache) error {
	layers := c.cfg.NumLayers
	if layers < 0 || len(c.K) != layers || len(c.Kraw) != layers || len(c.V) != layers {
		return fmt.Errorf("model: CPU prefix cache layer geometry K/Kraw/V=%d/%d/%d, want %d", len(c.K), len(c.Kraw), len(c.V), layers)
	}
	stride := c.kvStride()
	if stride < 0 || (c.Len() > 0 && stride == 0) || (stride > 0 && c.Len() > math.MaxInt/stride) {
		return errors.New("model: CPU prefix cache row geometry overflows")
	}
	want := c.Len() * stride
	precision := c.prec
	if precision == "" {
		precision = KVPrecisionFP32
	}
	if precision != KVPrecisionFP32 && precision != KVPrecisionQ8_0 {
		return fmt.Errorf("model: CPU prefix cache has unsupported KV precision %q", precision)
	}
	if precision == KVPrecisionQ8_0 && (len(c.kQ8) != layers || len(c.vQ8) != layers) {
		return errors.New("model: CPU prefix packed-cache layer geometry mismatch")
	}
	for layer := 0; layer < layers; layer++ {
		linear := c.cfg.IsQwen35Hybrid() && c.cfg.isLinearAttnLayer(layer)
		layerWant := want
		if linear {
			layerWant = 0
		}
		if len(c.Kraw[layer]) != layerWant {
			return fmt.Errorf("model: CPU prefix layer %d raw-K values=%d, want %d", layer, len(c.Kraw[layer]), layerWant)
		}
		if precision == KVPrecisionFP32 {
			if len(c.K[layer]) != layerWant || len(c.V[layer]) != layerWant {
				return fmt.Errorf("model: CPU prefix layer %d K/V values=%d/%d, want %d", layer, len(c.K[layer]), len(c.V[layer]), layerWant)
			}
			continue
		}
		if len(c.K[layer]) != 0 || len(c.V[layer]) != 0 {
			return fmt.Errorf("model: CPU prefix quantized layer %d carries unexpected f32 K/V", layer)
		}
		if err := validatePackedCPUCacheRow(c.kQ8[layer], layerWant, stride); err != nil {
			return fmt.Errorf("model: CPU prefix layer %d packed K: %w", layer, err)
		}
		if err := validatePackedCPUCacheRow(c.vQ8[layer], layerWant, stride); err != nil {
			return fmt.Errorf("model: CPU prefix layer %d packed V: %w", layer, err)
		}
	}
	if err := validateCPURecurrentCache(c); err != nil {
		return err
	}
	if c.glm != nil && (len(c.glm.K) != layers || len(c.glm.Kraw) != layers || len(c.glm.V) != layers || len(c.glm.IndexK) != layers || len(c.glm.IndexKraw) != layers) {
		return errors.New("model: CPU prefix GLM cache layer geometry mismatch")
	}
	if c.msa != nil && (len(c.msa.IndexK) != layers || len(c.msa.IndexKraw) != layers) {
		return errors.New("model: CPU prefix MiniMax cache layer geometry mismatch")
	}
	return nil
}

func validatePackedCPUCacheRow(row kvPackedRow, values, width int) error {
	if values == 0 {
		if row.width != width || len(row.codes) != 0 || len(row.scales) != 0 {
			return errors.New("empty row has non-empty or mismatched packing")
		}
		return nil
	}
	if row.width != width || len(row.codes) != values {
		return fmt.Errorf("width/codes=%d/%d, want %d/%d", row.width, len(row.codes), width, values)
	}
	rows := values / width
	groupsPerRow := (width + kvQ8_0GroupSize - 1) / kvQ8_0GroupSize
	if len(row.scales) != rows*groupsPerRow {
		return fmt.Errorf("scales=%d, want %d", len(row.scales), rows*groupsPerRow)
	}
	return nil
}

func validateCPURecurrentCache(c *KVCache) error {
	if !c.cfg.IsQwen35Hybrid() {
		if c.linear != nil {
			return errors.New("model: CPU prefix has recurrent cache for a non-hybrid configuration")
		}
		return nil
	}
	if c.linear == nil || len(c.linear.layers) != c.cfg.NumLayers {
		return errors.New("model: CPU prefix Qwen recurrent-cache layer geometry mismatch")
	}
	_, nV, kHd, vHd, _, _, convDim := c.cfg.linearAttnDims()
	for layer := range c.linear.layers {
		state := c.linear.layers[layer]
		if !c.cfg.isLinearAttnLayer(layer) {
			if len(state.conv) != 0 || len(state.recurrent) != 0 {
				return fmt.Errorf("model: CPU prefix Qwen recurrent state on full-attention layer %d", layer)
			}
			continue
		}
		if len(state.recurrent) != nV {
			return fmt.Errorf("model: CPU prefix Qwen recurrent heads=%d, want %d at layer %d", len(state.recurrent), nV, layer)
		}
		for _, row := range state.recurrent {
			if len(row) != kHd*vHd {
				return fmt.Errorf("model: CPU prefix Qwen recurrent row geometry mismatch at layer %d", layer)
			}
		}
		wantConvRows := min(c.Len(), max(0, c.cfg.LinearConvKernelDim-1))
		if len(state.conv) != wantConvRows {
			return fmt.Errorf("model: CPU prefix Qwen convolution rows=%d, want %d at layer %d", len(state.conv), wantConvRows, layer)
		}
		for _, row := range state.conv {
			if len(row) != convDim {
				return fmt.Errorf("model: CPU prefix Qwen convolution row geometry mismatch at layer %d", layer)
			}
		}
	}
	return nil
}
