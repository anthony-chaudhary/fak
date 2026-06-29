package model

import (
	"os"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// newHALKVStore selects the KV store for Backend-backed sessions. The default remains the
// backend's existing contiguous store; FAK_PAGED_KV=1 opts the CPU-reference HAL path into
// the block-paged allocator so the native path can prove paged gather/Evict without moving
// the optimized direct []float32 session off its stable default.
func newHALKVStore(be compute.Backend, cfg Config) compute.KVStore {
	if !pagedHALKVEnabled() {
		return be.NewKV(halKVConfig(cfg))
	}
	if be.Name() != "cpu-ref" {
		panic("model: FAK_PAGED_KV is currently supported only on the cpu-ref HAL path")
	}
	if cfg.isGLMMoeDsa() {
		panic("model: FAK_PAGED_KV does not yet support GLM-DSA's separate index/state cache")
	}
	return newPagedHALKV(be, cfg, envIntMin("FAK_PAGED_KV_BLOCK_TOKENS", 1, 16))
}

func pagedHALKVEnabled() bool {
	switch os.Getenv("FAK_PAGED_KV") {
	case "1", "true", "TRUE", "True", "on", "ON":
		return true
	default:
		return false
	}
}

func halKVConfig(cfg Config) compute.KVConfig {
	return compute.KVConfig{
		NumLayers:  cfg.NumLayers,
		NumKVHeads: cfg.NumKVHeads,
		HeadDim:    cfg.HeadDim,
		RopeTheta:  cfg.RopeTheta,
	}
}

type pagedHALKV struct {
	be   compute.Backend
	cfg  Config
	kcfg compute.KVConfig
	pool *PagedKVPool
	seq  *PagedKV
	pos  []int
}

func newPagedHALKV(be compute.Backend, cfg Config, blockTokens int) *pagedHALKV {
	pool := NewPagedKVPoolWithRaw(cfg, blockTokens)
	return &pagedHALKV{
		be:   be,
		cfg:  cfg,
		kcfg: halKVConfig(cfg),
		pool: pool,
		seq:  pool.NewSequence(),
	}
}

func (k *pagedHALKV) KVConfig() compute.KVConfig { return k.kcfg }

func (k *pagedHALKV) AppendKV(layer int, kRaw, kRoPE, v compute.Tensor, pos int) {
	raw := k.hostRow("kRaw", kRaw)
	rope := k.hostRow("kRoPE", kRoPE)
	val := k.hostRow("v", v)
	k.seq.AppendLayerRaw(layer, rope, raw, val)
	if layer == 0 {
		k.pos = append(k.pos, pos)
	}
}

func (k *pagedHALKV) hostRow(name string, t compute.Tensor) []float32 {
	row := k.be.Read(t)
	if row == nil {
		panic("model: paged HAL KV requires host-readable " + name + " tensor")
	}
	if len(row) != k.kcfg.NumKVHeads*k.kcfg.HeadDim {
		panic("model: paged HAL KV row has wrong width")
	}
	return append([]float32(nil), row...)
}

func (k *pagedHALKV) Len() int { return len(k.pos) }

func (k *pagedHALKV) KeysView(layer int) compute.Tensor {
	return compute.NewF32(k.be, []int{k.seq.Len(), k.kcfg.NumKVHeads * k.kcfg.HeadDim}, k.seq.GatherK(layer))
}

func (k *pagedHALKV) ValuesView(layer int) compute.Tensor {
	return compute.NewF32(k.be, []int{k.seq.Len(), k.kcfg.NumKVHeads * k.kcfg.HeadDim}, k.seq.GatherV(layer))
}

func (k *pagedHALKV) Pos() []int { return append([]int(nil), k.pos...) }

func (k *pagedHALKV) Evict(from, n int) int {
	removed := k.seq.Evict(from, n, k.cfg)
	if removed == 0 {
		return 0
	}
	end := from + removed
	k.pos = append(k.pos[:from], k.pos[end:]...)
	for i := range k.pos {
		k.pos[i] = i
	}
	return removed
}

func (k *pagedHALKV) Clone() compute.KVStore {
	return &pagedHALKV{
		be:   k.be,
		cfg:  k.cfg,
		kcfg: k.kcfg,
		pool: k.pool,
		seq:  k.seq.Clone(),
		pos:  append([]int(nil), k.pos...),
	}
}

func (k *pagedHALKV) Free() {
	if k.seq != nil {
		k.seq.Free()
	}
	k.pos = nil
}
