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
}

// checkpointExpertQuant maps a GGUF tensor type onto the representation the checkpoint tier stages,
// ok=false for one it has no staging for. This is deliberately NARROWER than
// residentExpertBlockGeometry: that predicate answers "can these raw bytes be held resident", this
// one answers "can one expert's raw bytes be uploaded straight into the ring", and the second set
// is the three k-quants compute.NewQ4K/NewQ5K/NewQ6K accept. Widening it is a matter of teaching
// model.ExpertCheckpointQuant the extra kinds, not of relaxing anything here.
func checkpointExpertQuant(t TensorType) (model.ExpertCheckpointQuant, bool) {
	switch t {
	case TensorQ4_K:
		return model.ExpertCheckpointQ4K, true
	case TensorQ5_K:
		return model.ExpertCheckpointQ5K, true
	case TensorQ6_K:
		return model.ExpertCheckpointQ6K, true
	}
	return 0, false
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
	for i, info := range s.File.Tensors {
		layer, proj, ok := glmMoeDsaBatchedExpert(info.Name)
		if !ok {
			continue
		}
		quant, ok := checkpointExpertQuant(info.Type)
		if !ok {
			continue // residentable perhaps, but not stageable one expert at a time — eager path
		}
		shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
		if err != nil {
			return nil, err
		}
		experts, rows, cols, err := parseGLMMoeDsaExpertShape(shape)
		if err != nil {
			return nil, err
		}
		// The same reduction-dim gate splitGLMMoeDsaExpertsRawQuant applies: a resident raw-quant
		// row must be a whole number of super-blocks, because the GEMV dequantizes blocks ALONG
		// each row. Unaligned means the eager path dequant-splits this slab to f32, and so must we.
		if cols%qkK != 0 {
			continue
		}
		// The eager path decides residency on the FIRST expert's canonical name; ask the same
		// question of the same name, so a tensor the resident store would refuse is not quietly
		// admitted through the tier instead.
		if !model.ResidentKQuantEligible(cfg, fmt.Sprintf("model.layers.%d.mlp.experts.0.%s.weight", layer, proj)) {
			continue
		}
		r, size := s.r, s.size
		if i < len(s.readerFor) && s.readerFor[i] != nil {
			r, size = s.readerFor[i], s.sizeFor[i]
		}
		if r == nil {
			return nil, fmt.Errorf("gguf: batched expert tensor %s has no shard reader", info.Name)
		}
		idx, seen := at[r]
		if !seen {
			idx = len(shards)
			at[r] = idx
			shards = append(shards, FusedExpertShard{Reader: r, Size: size})
		}
		shards[idx].Fused = append(shards[idx].Fused, model.FusedExpertTensor{
			Name:    info.Name,
			Layer:   layer,
			Proj:    proj,
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
		if err := tier.AddShard(sh.Reader, sh.Size, sh.Fused); err != nil {
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
