//go:build darwin && arm64 && cgo

package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// AdmitDeviceKV sizes a device-resident KV triple for a walk that ends at context
// `tokens` and attaches it to the backend, so Qwen35MetalForwardSequence routes
// full-attention layers through FullAttentionDevice. One triple is shared by every
// full-attention layer (layer l's rows at l*layerStride). When the session already
// holds a host prefix (`base = s.Cache.Len() > 0`, an append), each layer's
// existing KRaw/KPost/V rows are uploaded into the triple first, so the device walk
// continues the same prefix the host walk would have re-uploaded per panel.
// Returns nil (decline, fail-open) for an unsupported session, an unallocatable
// geometry, or an already-attached walk, leaving host KV/state unmutated.
func (b *metalQwen35GDNSequenceBackend) AdmitDeviceKV(s *Session, tokens int) *metalgemm.DeviceKV {
	if b == nil || s == nil || s.M == nil || tokens <= 0 {
		return nil
	}
	if s.Backend != nil || !s.Q4K || !s.MetalQ4K || s.qwen35HAL == nil || !s.qwen35HAL.sequenceAccepted {
		return nil
	}
	cfg := s.M.Cfg
	if qwen35MetalForwardGeometryError(cfg) != nil {
		return nil
	}
	base := s.Cache.Len()
	total := base + tokens
	layers := 0
	for l := 0; l < cfg.NumLayers; l++ {
		if !cfg.isLinearAttnLayer(l) {
			layers++
		}
	}
	if layers == 0 {
		return nil // no full-attention layer: nothing to keep device-resident
	}
	kvWidth := cfg.NumKVHeads * cfg.HeadDim
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.deviceKV != nil {
		return nil // a walk is already in flight; never share the triple across walks
	}
	capacity := total
	if qwen35PersistentDecodeDeviceKV() {
		capacity = -1
		for l := 0; l < cfg.NumLayers; l++ {
			if cfg.isLinearAttnLayer(l) {
				continue
			}
			for _, plane := range [][]float32{s.Cache.Kraw[l], s.Cache.K[l], s.Cache.V[l]} {
				c := cap(plane) / kvWidth
				if capacity < 0 {
					capacity = c
				} else if c != capacity {
					return nil
				}
			}
		}
		if capacity < total {
			return nil
		}
	}
	kv := metalgemm.NewDeviceKV(layers, capacity, kvWidth)
	if kv == nil {
		return nil
	}
	// Seed an append's existing prefix into the device triple so the first panel's
	// attention sees it. Layer l occupies ordinal*layerStride rows on each side.
	if base > 0 {
		want := base * kvWidth
		stride := kv.LayerStride()
		ordinal := 0
		for l := 0; l < cfg.NumLayers; l++ {
			if cfg.isLinearAttnLayer(l) {
				continue
			}
			off := ordinal * stride
			ordinal++
			if len(s.Cache.Kraw[l]) < want || len(s.Cache.K[l]) < want || len(s.Cache.V[l]) < want {
				kv.Close()
				return nil // prefix shorter than the cache claims: decline, host walk
			}
			if err := kv.UploadRegion(0, off, s.Cache.Kraw[l][:want]); err != nil {
				kv.Close()
				return nil
			}
			if err := kv.UploadRegion(1, off, s.Cache.K[l][:want]); err != nil {
				kv.Close()
				return nil
			}
			if err := kv.UploadRegion(2, off, s.Cache.V[l][:want]); err != nil {
				kv.Close()
				return nil
			}
		}
	}
	b.deviceKV = kv
	b.deviceKVCache = s.Cache
	b.deviceKVRows = base
	b.deviceKVCapacity = capacity
	b.deviceKVLineageLen = len(s.Cache.lineage.ids)
	b.deviceKVLineage = append(b.deviceKVLineage[:0], s.Cache.lineage.ids...)
	if b.deviceKVLineageLen > 0 {
		b.deviceKVLineageTail = s.Cache.lineage.ids[b.deviceKVLineageLen-1]
	}
	b.deviceKVPersistent = qwen35PersistentDecodeDeviceKV()
	return kv
}

// DetachDeviceKV clears and returns the walk's device KV pair. It touches no host
// KV/state, so a caller that declines the device path after admitting can detach
// and free the pair without any host mutation.
func (b *metalQwen35GDNSequenceBackend) DetachDeviceKV() *metalgemm.DeviceKV {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	kv := b.deviceKV
	b.deviceKV = nil
	b.deviceKVCache = nil
	b.deviceKVRows, b.deviceKVCapacity, b.deviceKVLineageLen = 0, 0, 0
	b.deviceKVLineage = nil
	b.deviceKVPersistent = false
	return kv
}

func (b *metalQwen35GDNSequenceBackend) KeepDeviceKV(s *Session) bool {
	if b == nil || s == nil || !qwen35PersistentDecodeDeviceKV() {
		return false
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.deviceKV == nil || !b.deviceKVPersistent || b.deviceKVCache != s.Cache {
		return false
	}
	b.deviceKVRows = s.Cache.Len()
	b.deviceKVLineageLen = len(s.Cache.lineage.ids)
	b.deviceKVLineage = append(b.deviceKVLineage[:0], s.Cache.lineage.ids...)
	if b.deviceKVLineageLen > 0 {
		b.deviceKVLineageTail = s.Cache.lineage.ids[b.deviceKVLineageLen-1]
	}
	return true
}

func (b *metalQwen35GDNSequenceBackend) decodeDeviceKV(s *Session) *metalgemm.DeviceKV {
	if b == nil || s == nil || !qwen35PersistentDecodeDeviceKV() {
		return nil
	}
	b.mu.Lock()
	valid := b.deviceKV != nil && b.deviceKVPersistent && b.deviceKVCache == s.Cache &&
		b.deviceKVRows == s.Cache.Len() && b.deviceKVCapacity > s.Cache.Len() &&
		b.deviceKVLineageLen == len(s.Cache.lineage.ids) && b.deviceKVLineageLen == s.Cache.Len()
	if valid && b.deviceKVLineageLen > 0 {
		valid = b.deviceKVLineageTail == s.Cache.lineage.ids[b.deviceKVLineageLen-1]
	}
	if valid {
		for i := range b.deviceKVLineage {
			if b.deviceKVLineage[i] != s.Cache.lineage.ids[i] {
				valid = false
				break
			}
		}
	}
	if valid {
		kvWidth := s.M.Cfg.NumKVHeads * s.M.Cfg.HeadDim
		for l := 0; l < s.M.Cfg.NumLayers && valid; l++ {
			if s.M.Cfg.isLinearAttnLayer(l) {
				continue
			}
			want := b.deviceKVRows * kvWidth
			valid = len(s.Cache.Kraw[l]) == want && len(s.Cache.K[l]) == want && len(s.Cache.V[l]) == want
		}
	}
	if valid {
		kv := b.deviceKV
		b.mu.Unlock()
		return kv
	}
	kv := b.deviceKV
	b.deviceKV = nil
	b.deviceKVCache = nil
	b.deviceKVPersistent = false
	b.deviceKVLineage = nil
	b.mu.Unlock()
	if kv != nil {
		kv.Close()
	}
	return nil
}

func (b *metalQwen35GDNSequenceBackend) invalidateDeviceKV() {
	if kv := b.DetachDeviceKV(); kv != nil {
		kv.Close()
	}
}

// ReconcileDeviceKV downloads the device triple once and appends each
// full-attention layer's newly-written rows to the host cache, restoring the host
// cache contract the decode path depends on after a device-resident panel walk.
// `base` is the host prefix length the walk started from (s.Cache.Len() before the
// walk) and `rows` is how many rows the panels actually appended; the device rows
// [base, base+rows) are the ones absent from the host cache. Every layer's host
// rows are staged first and only committed once ALL layers validate, so a geometry
// mismatch leaves the host cache byte-identical (fail-closed).
func (b *metalQwen35GDNSequenceBackend) ReconcileDeviceKV(s *Session, base, rows int) error {
	if b == nil || s == nil || s.M == nil {
		return errors.New("metalgemm: device KV reconcile without a session")
	}
	b.mu.Lock()
	kv := b.deviceKV
	b.mu.Unlock()
	if kv == nil {
		return nil
	}
	cfg := s.M.Cfg
	kvWidth := cfg.NumKVHeads * cfg.HeadDim
	stride := kv.LayerStride()
	want := rows * kvWidth
	if kvWidth <= 0 || stride <= 0 || rows <= 0 || (base+rows)*kvWidth > stride {
		return fmt.Errorf("metalgemm: device KV reconcile geometry mismatch: base=%d rows=%d kvWidth=%d stride=%d", base, rows, kvWidth, stride)
	}
	// The device triple holds exactly the three host rows the walk appended: side 0
	// KRaw (pre-norm key), side 1 KPost (the attention prefix) and side 2 V. Layer l
	// occupies rows [l*stride, l*stride+stride); this walk wrote [base, base+rows)
	// of that slice, so stage those rows and commit only once every download passed.
	type layerRows struct {
		layer          int
		kraw, kpost, v []float32
	}
	planned := make([]layerRows, 0, cfg.NumLayers)
	ordinal := 0
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isLinearAttnLayer(l) {
			continue
		}
		off := ordinal*stride + base*kvWidth
		ordinal++
		kraw := make([]float32, want)
		kpost := make([]float32, want)
		v := make([]float32, want)
		if err := kv.DownloadRegion(0, off, kraw); err != nil {
			return err
		}
		if err := kv.DownloadRegion(1, off, kpost); err != nil {
			return err
		}
		if err := kv.DownloadRegion(2, off, v); err != nil {
			return err
		}
		planned = append(planned, layerRows{layer: l, kraw: kraw, kpost: kpost, v: v})
	}
	for _, row := range planned {
		s.Cache.Kraw[row.layer] = append(s.Cache.Kraw[row.layer], row.kraw...)
		s.Cache.K[row.layer] = append(s.Cache.K[row.layer], row.kpost...)
		s.Cache.V[row.layer] = append(s.Cache.V[row.layer], row.v...)
	}
	return nil
}
