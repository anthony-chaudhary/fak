package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// HostPrefixSnapshot is a complete PrefixSnapshot payload copied into ordinary
// process-owned host DRAM. It includes the host model cache, attention K/Kraw/V,
// positions, and every Qwen3.5/3.6 convolution/recurrent tensor needed to resume.
type HostPrefixSnapshot struct {
	cache   *KVCache
	kv      compute.KVHostSnapshot
	qwen35  *hostQwen35State
	backend compute.Backend
	tokens  int
}

type hostQwen35State struct {
	backend Qwen35GDNBackend
	layers  []hostQwen35LayerState
}

type hostQwen35LayerState struct {
	conv      hostTensor
	recurrent hostTensor
}

type hostTensor struct {
	shape []int
	data  []float32
}

// ResidencyBytes splits this live snapshot's payload by its physical owner. On
// a device backend, KV positions and the model cache remain host metadata while
// K/Kraw/V and Qwen recurrent tensors are device-resident.
func (p *PrefixSnapshot) ResidencyBytes() (host, device int64) {
	if p == nil || p.Cache == nil {
		return 0, 0
	}
	host = p.Cache.residentBytes()
	deviceBacked := p.Backend != nil && p.Backend.Caps().DeviceMemory
	if p.halKV != nil {
		bytes := p.halKV.ResidentBytes()
		if deviceBacked {
			posBytes := int64(len(p.halKV.Pos())) * hostPositionBytes
			if posBytes > bytes {
				posBytes = bytes
			}
			host += posBytes
			device += bytes - posBytes
		} else {
			host += bytes
		}
	}
	if p.qwen35 != nil {
		for i := range p.qwen35.layers {
			bytes := tensorResidentBytes(p.qwen35.layers[i].conv) +
				tensorResidentBytes(p.qwen35.layers[i].recurrent)
			if deviceBacked {
				device += bytes
			} else {
				host += bytes
			}
		}
	}
	return host, device
}

// CloneToHost copies the complete prefix into host DRAM without mutating the live
// snapshot. Callers can therefore stage first and free the device owner only after
// the host copy is confirmed, preserving the capacity adapter's fail-safe ordering.
func (p *PrefixSnapshot) CloneToHost() (out *HostPrefixSnapshot, err error) {
	if p == nil || p.Cache == nil || p.Backend == nil || p.halKV == nil {
		return nil, errors.New("model: prefix snapshot has no complete backend state to stage")
	}
	defer func() {
		if r := recover(); r != nil {
			if out != nil {
				out.Close()
			}
			out = nil
			err = fmt.Errorf("model: stage prefix snapshot to host: %v", r)
		}
	}()

	hostKV, err := compute.SnapshotKVToHost(p.halKV)
	if err != nil {
		return nil, fmt.Errorf("model: stage prefix attention KV to host: %w", err)
	}
	out = &HostPrefixSnapshot{
		cache:   p.Cache.Clone(),
		kv:      hostKV,
		backend: p.Backend,
		tokens:  p.Tokens,
	}
	if p.qwen35 != nil {
		out.qwen35 = &hostQwen35State{
			backend: p.qwen35.backend,
			layers:  make([]hostQwen35LayerState, len(p.qwen35.layers)),
		}
		for layer := range p.qwen35.layers {
			var copyErr error
			out.qwen35.layers[layer].conv, copyErr = tensorToHost(p.Backend, p.qwen35.layers[layer].conv)
			if copyErr != nil {
				out.Close()
				return nil, fmt.Errorf("model: stage Qwen3.5 layer %d convolution state: %w", layer, copyErr)
			}
			out.qwen35.layers[layer].recurrent, copyErr = tensorToHost(p.Backend, p.qwen35.layers[layer].recurrent)
			if copyErr != nil {
				out.Close()
				return nil, fmt.Errorf("model: stage Qwen3.5 layer %d recurrent state: %w", layer, copyErr)
			}
		}
	}
	return out, nil
}

func tensorToHost(be compute.Backend, t compute.Tensor) (hostTensor, error) {
	if t.Buf() == nil {
		return hostTensor{}, nil
	}
	if t.Dtype != compute.F32 {
		return hostTensor{}, fmt.Errorf("state dtype %s is not lossless F32", t.Dtype)
	}
	data := append([]float32(nil), be.Read(t)...)
	if len(data) != t.Numel() {
		return hostTensor{}, fmt.Errorf("read %d values, want %d", len(data), t.Numel())
	}
	return hostTensor{shape: append([]int(nil), t.Shape...), data: data}, nil
}

// Restore materializes an independently owned live PrefixSnapshot on the original
// backend. The host image remains intact for later L2 hits.
func (h *HostPrefixSnapshot) Restore() (out *PrefixSnapshot, err error) {
	if h == nil || h.cache == nil || h.backend == nil {
		return nil, errors.New("model: invalid host prefix snapshot restore")
	}
	defer func() {
		if r := recover(); r != nil {
			if out != nil {
				out.Close()
			}
			out = nil
			err = fmt.Errorf("model: restore prefix snapshot from host: %v", r)
		}
	}()

	kv, err := compute.RestoreKVFromHost(h.backend, h.kv)
	if err != nil {
		return nil, fmt.Errorf("model: restore prefix attention KV from host: %w", err)
	}
	out = &PrefixSnapshot{
		Cache:   h.cache.Clone(),
		halKV:   kv,
		Backend: h.backend,
		Tokens:  h.tokens,
	}
	if h.qwen35 != nil {
		out.qwen35 = &qwen35HALState{
			backend: h.qwen35.backend,
			layers:  make([]qwen35HALLayerState, len(h.qwen35.layers)),
		}
		for layer := range h.qwen35.layers {
			out.qwen35.layers[layer].conv = hostTensorRestore(
				h.backend, h.qwen35.layers[layer].conv,
				"qwen35-gdn-conv-host-restore layer "+itoa(layer),
			)
			out.qwen35.layers[layer].recurrent = hostTensorRestore(
				h.backend, h.qwen35.layers[layer].recurrent,
				"qwen35-gdn-recurrent-host-restore layer "+itoa(layer),
			)
		}
	}
	return out, nil
}

func hostTensorRestore(be compute.Backend, t hostTensor, site string) compute.Tensor {
	if len(t.data) == 0 {
		return compute.Tensor{}
	}
	return uploadHostF32Class(be, t.shape, t.data, compute.MemoryKVCache, site)
}

// ResidentBytes reports the complete host payload, excluding Go slice/header overhead.
func (h *HostPrefixSnapshot) ResidentBytes() int64 {
	if h == nil || h.cache == nil {
		return 0
	}
	bytes := h.cache.residentBytes() + h.kv.ResidentBytes()
	if h.qwen35 != nil {
		for i := range h.qwen35.layers {
			bytes += int64(len(h.qwen35.layers[i].conv.data)+len(h.qwen35.layers[i].recurrent.data)) *
				int64(compute.F32.Bytes())
		}
	}
	return bytes
}

// TransferBytes reports D2H/H2D payload bytes for one stage/restore. Host-only
// cache metadata and positions are excluded because they never cross the device bus.
func (h *HostPrefixSnapshot) TransferBytes() int64 {
	if h == nil || h.backend == nil || !h.backend.Caps().DeviceMemory {
		return 0
	}
	bytes := h.kv.TransferBytes()
	if h.qwen35 != nil {
		for i := range h.qwen35.layers {
			bytes += int64(len(h.qwen35.layers[i].conv.data)+len(h.qwen35.layers[i].recurrent.data)) *
				int64(compute.F32.Bytes())
		}
	}
	return bytes
}

// Tokens is the logical prefix length represented by the image.
func (h *HostPrefixSnapshot) Tokens() int {
	if h == nil {
		return 0
	}
	return h.tokens
}

// Close releases the host image to Go's allocator. It is idempotent.
func (h *HostPrefixSnapshot) Close() {
	if h == nil {
		return
	}
	h.cache = nil
	h.kv = compute.KVHostSnapshot{}
	h.qwen35 = nil
	h.backend = nil
	h.tokens = 0
}
