package ggufload

// expert_checkpoint_source.go — the internal/ggufload half of R5 (#5616, epic #5606,
// docs/MOE-ACTIVATED-OFFLOAD-PLAN.md): describe a GGUF's fused routed-expert slabs precisely
// enough that model.ExpertCheckpointTier can fault ONE expert out of one, without reading a byte
// of payload to do it.
//
// What was missing. The model side of R5 shipped a tier that faults a single expert out of a fused
// `blk.L.ffn_*_exps.weight` slab and hands it to the bounded routed-expert ring, but nothing
// constructed one: the only producer of per-expert bytes in the loader is
// splitGLMMoeDsaExpertsRawQuant (gguf_glm_tensors.go), which materializes the WHOLE [E,out,in]
// slab in host RAM first and then copies E per-expert segments out of it. On a GLM-5.2-shaped
// checkpoint that is the entire expert bulk — the bytes the ladder exists to not hold — so a tier
// built downstream of it would bound nothing.
//
// What this adds. FusedExpertTensors reads the GGUF tensor DIRECTORY (names, dims, types, file
// offsets — all parsed at open) and turns it into model.FusedExpertTensor descriptors. The
// directory already carries everything the tier's stride math needs, so the descriptors cost zero
// payload IO and the slab stays on disk until a router picks an expert out of it. That is the
// property TestFusedExpertDescriptorsReadNoPayload pins, because it is the only thing that makes
// this rung different from the eager split.
//
// What it declines, and why declining is the safe direction. A descriptor is emitted only for a
// tensor the tier can serve EXACTLY as the resident path would have:
//
//   - the arch must be one whose routed experts are batched at all (archUsesGGUFBatchedMoEExperts);
//   - the quant must be one the tier stages (Q4_K/Q5_K/Q6_K — checkpointExpertQuant). Q8_0, Q4_0,
//     IQ3_XXS, IQ4_XS and Q2_0 are residentable but have no checkpoint staging yet, so they keep
//     the unchanged eager split;
//   - the reduction dim must be whole 256-weight super-blocks, the same gate
//     splitGLMMoeDsaExpertsRawQuant applies before it will split raw bytes at all — an unaligned
//     row falls to the f32 dequant-split there, and must fall to it here too;
//   - the name must be resident-eligible (model.ResidentKQuantEligible), the same predicate that
//     decides whether the eager path may hold those bytes raw.
//
// Every decline leaves that tensor on the path it takes today. That asymmetry is deliberate: a
// missing descriptor costs host RAM, while a WRONG one would feed misaligned bytes to a GEMM and
// produce plausible garbage.
//
// The borrow contract. The descriptors carry offsets into the shard files THIS WeightSource holds
// open, and the tier reads through the source's own readers — it does not open its own handles and
// does not own them. So the source must outlive the tier: after ws.Close() every fault fails, and
// over a streamed checkpoint the expert is resident nowhere else, so that is a dead model rather
// than a slow one. The path-taking loader entry points close the source before returning, which is
// why LoadModelQ4KProfileOptions REFUSES WithStreamedExperts rather than handing back a model whose
// experts are unreadable.

import (
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// FusedExpertShard is the fused routed-expert slabs that live in ONE checkpoint file, with the
// reader that serves them. A split checkpoint yields one of these per shard file, because a
// tensor's FileOffset is relative to the shard holding it, not to the merged view.
type FusedExpertShard struct {
	Reader io.ReaderAt
	Size   int64
	Fused  []model.FusedExpertTensor
	// Data is the page-cache-visible mapped region backing this shard when the
	// retained reader is a FAK_GGUF_MMAP memory map (gguf_mmap.go) and the
	// FAK_EXPERT_PAGECACHE opt-in is on (expert_pagecache.go, issue #1302); nil
	// on the default os.Open path and on every platform without an mmap impl,
	// including Windows. It lets the model tier adopt a zero-copy read-through
	// path over the mapped file, and its nil-ness preserves the historical
	// ReadAt path byte-for-byte.
	Data []byte
}

// checkpointExpertQuant maps a GGUF tensor type onto the representation the checkpoint tier stages,
// ok=false for one it has no staging for. This is deliberately NARROWER than
// residentExpertBlockGeometry: that predicate answers "can these raw bytes be held resident", this
// one answers "can one expert's raw bytes be STAGED into the ring", and it tracks the
// model.ExpertCheckpointQuant members one-for-one.
//
// Every kind here stages its super-blocks VERBATIM through an internal/compute host constructor:
// Q2_K/Q4_K/Q5_K/Q6_K always had one, and Q3_K gained compute.NewQ3K in fak#13149. Admitting
// TensorQ3_K here is what lets the exact DeepSeek-V4.1 Q2_K artifact (Q2_K gate/up, Q3_K down)
// reach the streamed forward instead of being refused as partially unstageable (fak#13144).
func checkpointExpertQuant(t TensorType) (model.ExpertCheckpointQuant, bool) {
	switch t {
	case TensorQ2_K:
		return model.ExpertCheckpointQ2K, true
	case TensorQ3_K:
		return model.ExpertCheckpointQ3K, true
	case TensorQ4_K:
		return model.ExpertCheckpointQ4K, true
	case TensorQ5_K:
		return model.ExpertCheckpointQ5K, true
	case TensorQ6_K:
		return model.ExpertCheckpointQ6K, true
	}
	return 0, false
}

// stageableRoutedExpertSlab is the tier's per-slab admission rule, single-sourced so
// FusedExpertTensors and UnstageableRoutedExpertSlabs cannot disagree about which slabs the tier can
// serve. It reports the staging quant and the parsed [E,out,in] shape on admission; ok=false means
// this slab takes the eager path instead. An error means the directory is malformed.
//
// The three declines, in the order the eager path applies them:
//   - the quant must have a compute tensor kind (checkpointExpertQuant) - internal/compute's
//     verbatim host constructor for each, Q3_K included since fak#13149;
//   - the reduction dim must be whole 256-weight super-blocks, the same gate
//     splitGLMMoeDsaExpertsRawQuant applies before it will split raw bytes at all;
//   - the name must be resident-eligible, the same predicate that decides whether the eager path
//     may hold those bytes raw.
func (s *WeightSource) stageableRoutedExpertSlab(cfg model.Config, info TensorInfo) (quant model.ExpertCheckpointQuant, experts, rows, cols int, ok bool, err error) {
	quant, ok = checkpointExpertQuant(info.Type)
	if !ok {
		return 0, 0, 0, 0, false, nil // residentable perhaps, but not stageable one expert at a time
	}
	shape, serr := modelShapeFromGGUFDims(info.Name, info.Dims)
	if serr != nil {
		return 0, 0, 0, 0, false, serr
	}
	experts, rows, cols, perr := parseGLMMoeDsaExpertShape(shape)
	if perr != nil {
		return 0, 0, 0, 0, false, perr
	}
	if cols%qkK != 0 {
		return 0, 0, 0, 0, false, nil
	}
	layer, proj, _ := glmMoeDsaBatchedExpert(info.Name)
	if !model.ResidentKQuantEligible(cfg, batchedExpertCanonicalName(cfg.ModelType, layer, 0, proj)) {
		return 0, 0, 0, 0, false, nil
	}
	return quant, experts, rows, cols, true, nil
}

// UnstageableRoutedExpertSlabs names every batched routed-expert slab this checkpoint carries that
// the R5 tier CANNOT stage, in directory order. It performs NO payload IO, exactly like
// FusedExpertTensors, and consults the same admission rule so the two cannot drift.
//
// It exists because a PARTIAL decline is not the same as an empty one. When a checkpoint has some
// stageable expert slabs and some it cannot stage, the tier still builds (so the caller's
// WithStreamedExperts request is not refused outright), but the declined slabs fall off the
// streamed set and are eager-dequantized to f32 - materializing the very expert bulk the caller
// asked to bound. Naming them lets the loader refuse that half-measure instead of silently paying
// the bytes (fak#13144).
//
// It returns an error only for a malformed directory, matching FusedExpertTensors: a 3-D expert
// slab whose dims do not parse is corrupt, not a decline.
func (s *WeightSource) UnstageableRoutedExpertSlabs() ([]string, error) {
	if s == nil || s.File == nil {
		return nil, nil
	}
	cfg, err := s.File.Config()
	if err != nil {
		return nil, err
	}
	if !archUsesGGUFBatchedMoEExperts(cfg.ModelType) {
		return nil, nil
	}
	var names []string
	for _, info := range s.File.Tensors {
		if _, _, ok := glmMoeDsaBatchedExpert(info.Name); !ok {
			continue
		}
		_, _, _, _, stageable, serr := s.stageableRoutedExpertSlab(cfg, info)
		if serr != nil {
			return nil, serr
		}
		if !stageable {
			names = append(names, info.Name)
		}
	}
	return names, nil
}

// FusedExpertTensors describes every batched routed-expert slab this checkpoint carries that the
// R5 tier can serve, grouped by the shard file that holds it. It performs NO payload IO: every
// field comes from the tensor directory parsed at open.
//
// An empty result is the normal answer for a dense checkpoint, for an arch that does not batch its
// experts, and for a MoE checkpoint whose expert quant this tier cannot stage — in all three the
// caller keeps the unchanged eager path. An error means the directory itself is malformed (a
// non-3-D expert slab, a dimension that overflows int, an expert tensor with no shard reader):
// that is a corrupt checkpoint, not a decline, and reporting it here is cheaper than discovering it
// mid-decode.
func (s *WeightSource) FusedExpertTensors() ([]FusedExpertShard, error) {
	if s == nil || s.File == nil {
		return nil, nil
	}
	cfg, err := s.File.Config()
	if err != nil {
		return nil, err
	}
	if !archUsesGGUFBatchedMoEExperts(cfg.ModelType) {
		return nil, nil
	}
	var shards []FusedExpertShard
	// Reader identity groups the descriptors. Both retained reader kinds are pointers (*os.File on
	// the default path, *mmapReaderAt under FAK_GGUF_MMAP — gguf_mmap.go), so they are comparable
	// map keys; a future non-pointer reader would need a shard index carried alongside instead.
	at := make(map[io.ReaderAt]int)
	// dataSet tracks, per shard group, the mapped region its FIRST resident tensor resolved to, so
	// tensors that share a reader must also share the map (a mismatch is a malformed directory, not
	// a decline). nil on the default os.Open path; only FAK_GGUF_MMAP shards carry a region.
	dataSet := make(map[io.ReaderAt][]byte)
	for i, info := range s.File.Tensors {
		layer, proj, ok := glmMoeDsaBatchedExpert(info.Name)
		if !ok {
			continue
		}
		quant, experts, rows, cols, stageable, perr := s.stageableRoutedExpertSlab(cfg, info)
		if perr != nil {
			return nil, perr
		}
		if !stageable {
			continue
		}
		r, size := s.r, s.size
		// Only carry a mapped region when the page-cache opt-in is on (#1302): mapping a shard is a
		// load-wide storage decision, while adopting the map on the expert path is the behavior change,
		// so it needs its own gate. Gate off (the default) leaves data nil and every consumer on the
		// historical ReadAt path byte-for-byte.
		var data []byte
		if i < len(s.readerFor) && s.readerFor[i] != nil {
			r, size = s.readerFor[i], s.sizeFor[i]
		}
		if expertPageCacheEnabled() {
			data = s.data
			if i < len(s.dataFor) && s.dataFor[i] != nil {
				data = s.dataFor[i]
			}
		}
		if r == nil {
			return nil, fmt.Errorf("gguf: batched expert tensor %s has no shard reader", info.Name)
		}
		idx, seen := at[r]
		if !seen {
			idx = len(shards)
			at[r] = idx
			dataSet[r] = data
			shards = append(shards, FusedExpertShard{Reader: r, Size: size, Data: data})
		} else if dataSet[r] != nil && data != nil && !sameBytes(dataSet[r], data) {
			return nil, fmt.Errorf("gguf: batched expert tensors sharing a shard reader resolved to different mapped regions")
		}
		shards[idx].Fused = append(shards[idx].Fused, model.FusedExpertTensor{
			Name:    info.Name,
			Layer:   layer,
			Proj:    proj,
			Arch:    cfg.ModelType,
			Quant:   quant,
			Offset:  info.FileOffset,
			Experts: experts,
			Rows:    rows,
			Cols:    cols,
		})
	}
	return shards, nil
}

// buildExpertCheckpointTier assembles a tier over already-described shards. hostBytes is the tier's
// HOST retention budget, and 0 — the default — means stream-through: every fault is read, handed to
// the ring and dropped, which is the placement that makes a checkpoint larger than host RAM
// servable at all. It returns nil (no error) when there is nothing to serve, so callers can treat
// "this checkpoint has no streamable experts" as a plain absence.
func buildExpertCheckpointTier(shards []FusedExpertShard, hostBytes int64) (*model.ExpertCheckpointTier, error) {
	if len(shards) == 0 {
		return nil, nil
	}
	tier := model.NewExpertCheckpointTier(hostBytes)
	for _, sh := range shards {
		// AddShardData adopts the shard's mapped region when it carries one (#1302): a warm expert
		// fault is then served zero-copy instead of re-read from the device. A shard with no mapping
		// (the default os.Open path, and every platform without an mmap impl) passes nil and takes the
		// historical ReadAt path byte-for-byte.
		data, _ := pageCacheAdoption(sh, expertPageCacheEnabled())
		if err := tier.AddShardData(sh.Reader, sh.Size, data, sh.Fused); err != nil {
			return nil, err
		}
	}
	return tier, nil
}

// ExpertCheckpointTier builds an R5 checkpoint tier over THIS source's shard readers, or nil when
// the checkpoint carries no fused expert slab the tier can serve.
//
// The tier borrows the readers (see the file header): this source must outlive it, and every fault
// fails after s.Close(). Callers that hand the tier to a model via Model.SetExpertCheckpoint own
// that lifetime.
func (s *WeightSource) ExpertCheckpointTier(hostBytes int64) (*model.ExpertCheckpointTier, error) {
	shards, err := s.FusedExpertTensors()
	if err != nil {
		return nil, err
	}
	return buildExpertCheckpointTier(shards, hostBytes)
}
