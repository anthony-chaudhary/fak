package model

import "fmt"

// tensor_parallel_vocab.go — the VOCAB-PARALLEL tensor-parallel subsystem, ADAPTED from
// vLLM vllm/model_executor/layers/vocab_parallel_embedding.py @4ba787ec9959e566d3b587a796dff07673324da5
// (tag v0.33.0rc3, Apache-2.0; see third_party/vllm-vocab-parallel/NOTICE). tensor_parallel.go
// ported the FFN block and tensor_parallel_attn.go the attention block of the Megatron
// decomposition; this file ports the THIRD building block — the two LARGEST projections in
// every checkpoint, the token-embedding table and the LM head, sharded across the VOCABULARY
// dimension. For SmolLM2-135M the lm_head alone (49152x576) is ~10x every other projection
// combined, so vocab-parallel sharding is the lever that actually removes the largest tensor
// from a rank's footprint.
//
// THE UPSTREAM DECOMPOSITION (ported mechanisms):
//
//   - VocabParallelEmbedding.forward: the table is sharded into contiguous vocab bands
//     [V_0 | V_1 | ...]. Rank r looks up ONLY ids inside its band (upstream
//     get_masked_input_and_mask: out-of-band ids are masked and looked up as row 0, then
//     masked_fill_(input_mask, 0) zeroes them), and the per-rank rows are AllReduce-summed
//     (tensor_model_parallel_all_reduce). Each token id has EXACTLY ONE owning rank, so the
//     reduce adds zeros to that rank's row: the sum is the true embedding row EXACTLY, at any
//     rank count — this block is bit-exact vs the monolith lookup, unlike the row-parallel
//     matmuls whose AllReduce reassociates a real sum.
//
//   - ParallelLMHead (+ the sampler's logits gather in vllm logits_processor.py): the [V,H]
//     output projection is sharded along V; rank r computes logits[v] = fdot(W[v], x) for its
//     band only, and the bands are AllGather-concatenated in rank order
//     (tensor_model_parallel_all_gather). Each logit is computed by exactly one rank by the
//     SAME fdot over the SAME weight rows as the monolith, and upstream's
//     get_sharded_to_full_mapping — the post-gather reindex that undoes vocab padding and
//     LoRA-added rows — is the IDENTITY here (fak's GGUF path has neither), so the rank-order
//     concat IS token-id order. Bit-exact vs the monolith tail, at any rank count.
//
// HONEST DIVERGENCES from upstream (each is a fak adaptation, not an omission):
//
//   - Upstream pads the vocabulary to a multiple of DEFAULT_VOCAB_PADDING_SIZE (64) so every
//     rank owns the same shard width (divide), then reindexes padding out after the gather.
//     fak reuses its validated TPPlan tiling (near-even contiguous bands, remainder on the
//     first shards, no empty shard), so no padding exists and the masked-input arithmetic
//     degenerates to a single band test.
//   - Upstream supports LoRA-added embeddings (the added_vocab_* shard indices); fak's
//     serving path has no LoRA-added vocabulary rows, so those indices are absent.
//   - fak's sharded reads slice RAW f32 rows (the same matRows the monolith runs), so the
//     fail-closed contract matches the rest of the TP files: the embed table must be
//     f32-resident (a Q2K packed-embedding store has no f32 rows to shard) and the head must
//     be f32-resident with no prism input transform and no LoRA delta (those seams live in
//     residentMatRows, which the per-band matmul deliberately bypasses). Quant-aware
//     vocab sharding is a later lever.
//
// The gate chain this file joins (same grammar as tensor_parallel_forward.go):
//
//	ForwardTP(VocabRanks>1)  ==(max|Δ|=0)   ForwardTP(VocabRanks<=1)   [this file's tests]
//	ForwardTP(ranks=1)       ==(max|Δ|=0)   Forward                    [tensor_parallel_forward_test.go]
//	Forward                  ==(HF oracle)  HuggingFace                [oracle_test.go]
//
// so the vocab-parallel tail inherits the HF oracle transitively, and at ranks=1 the WHOLE
// wired path stays bit-identical to Forward.

// maskedVocabShardInput ports upstream get_masked_input_and_mask in the no-LoRA / no-padding
// form the GGUF path always has: upstream computes org_vocab_mask = (id >= start) & (id < end),
// shifts in-band ids by the band offset (valid_offset = org_vocab_start_index), and maps
// out-of-band ids to 0 with input_mask = ~vocab_mask. Here the band IS the rank's TPShard
// [Lo,Hi), so: in-band id -> local index id-Lo; out-of-band id -> local index 0 (upstream's
// masked_input) with mask[i] = true (upstream's input_mask, zeroed by masked_fill_ before the
// reduce). The caller validates ids in [0,V) BEFORE this helper, so "out of band" never means
// "out of range".
func maskedVocabShardInput(s TPShard, ids []int) (local []int, mask []bool) {
	local = make([]int, len(ids))
	mask = make([]bool, len(ids))
	for i, id := range ids {
		if id >= s.Lo && id < s.Hi {
			local[i] = id - s.Lo
		} else {
			local[i] = 0   // upstream masked_input: out-of-band ids look up the shard's row 0...
			mask[i] = true // ...and are zeroed by masked_fill_ before the reduce
		}
	}
	return local, mask
}

// tpVocabPlan builds the vocabulary-parallel plan over the model's vocab size: the same
// validated near-even contiguous tiling the FFN/attention plans use, over Dim = VocabSize.
// ranks <= 1 clamps to the 1-shard plan (the monolith tail).
func tpVocabPlan(cfg Config, ranks int) (TPPlan, error) {
	if ranks < 0 {
		return TPPlan{}, fmt.Errorf("model: vocab-parallel ranks = %d, want >= 0", ranks)
	}
	return NewTPPlan(cfg.VocabSize, clampRanks(ranks, cfg.VocabSize))
}

// requireTPVocabResidentF32 fails closed unless the token-embedding table is unpacked
// resident f32 — the only form the vocab-parallel bands can slice. The Q2K packed-embedding
// store has no f32 rows (embedRows refuses it too), so the vocab-parallel path refuses it
// with a named error instead of the monolith's panic.
func (m *Model) requireTPVocabResidentF32() error {
	if m.Q2KEmbedding != nil {
		return fmt.Errorf("model: vocab-parallel embedding needs the unpacked f32 table (the Q2K packed-embedding store has no f32 rows to shard)")
	}
	if !m.has("model.embed_tokens.weight") {
		return fmt.Errorf("model: vocab-parallel embedding: model.embed_tokens.weight is not resident f32")
	}
	return nil
}

// requireTPHeadResidentF32 fails closed unless the LM head is a plain resident f32 tensor:
// the sharded band matmul slices raw f32 rows and bypasses residentMatRows, so a
// quant-resident head (read from the Q8/Q4/k-quant store by the monolith tail), a prism
// input transform, or an active LoRA delta on the head would make the sharded read a
// DIFFERENT function than the monolith's — each is refused with the seam that makes it so.
func (m *Model) requireTPHeadResidentF32() error {
	name := m.headName()
	if !m.has(name) {
		if m.hasWeight(name) {
			return fmt.Errorf("model: vocab-parallel head %s is quant-resident; the sharded band matmul slices raw f32 rows (quant-aware head sharding is a later lever)", name)
		}
		return fmt.Errorf("model: vocab-parallel head %s has no resident tensor", name)
	}
	if m.prism != nil && m.prism.weightWidth[name] != 0 {
		return fmt.Errorf("model: vocab-parallel head %s has a prism input transform; the sharded band matmul bypasses residentMatRows (prism-aware head sharding is a later lever)", name)
	}
	if m.lora != nil && len(m.lora.activeFor(name)) > 0 {
		return fmt.Errorf("model: vocab-parallel head %s has an active LoRA delta; the sharded band matmul bypasses residentMatRows' loraApply (LoRA-aware head sharding is a later lever)", name)
	}
	return nil
}

// tpEmbedShardPartials computes each rank's [seq][hidden] masked-lookup rows for the
// vocab-parallel embedding: upstream VocabParallelEmbedding.forward up to (and including)
// the masked_fill_. Rank r owns the table's contiguous band rows embed[Lo*H : Hi*H]; an
// in-band id contributes its true row, an out-of-band id contributes the zeroed row-0
// lookup (upstream masked_input -> F.embedding -> masked_fill_(input_mask, 0)). The
// AllReduce that follows (tpEmbedLayer) is where the full row materializes.
func (m *Model) tpEmbedShardPartials(ids []int, plan TPPlan) ([][][]float32, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	if err := m.requireTPVocabResidentF32(); err != nil {
		return nil, err
	}
	cfg := m.Cfg
	H := cfg.HiddenSize
	for _, id := range ids {
		if id < 0 || id >= cfg.VocabSize {
			return nil, fmt.Errorf("model: vocab-parallel embedding token id %d outside [0,%d)", id, cfg.VocabSize)
		}
	}
	embed := m.embedRows()
	seq := len(ids)
	rankPartials := make([][][]float32, len(plan.Shards))
	for r, s := range plan.Shards {
		lo, w := s.Lo, s.Width()
		rows := embed[lo*H : (lo+w)*H] // the rank's contiguous band of the table (zero-copy view)
		local, mask := maskedVocabShardInput(s, ids)
		part := make([][]float32, seq)
		for t := 0; t < seq; t++ {
			row := make([]float32, H)
			copy(row, rows[local[t]*H:(local[t]+1)*H])
			if mask[t] {
				for i := range row {
					row[i] = 0 // upstream masked_fill_(input_mask.unsqueeze(-1), 0)
				}
			}
			part[t] = row
		}
		rankPartials[r] = part
	}
	return rankPartials, nil
}

// tpEmbedLayer is the vocab-parallel embedding: the per-rank masked lookups are AllReduce-summed
// per position (upstream tensor_model_parallel_all_reduce) and the arch embed scale applies ONCE
// per position AFTER the reduce — the same single-scale instruction stream as the monolith
// embedBand, whose scale would otherwise be applied per rank and re-scaled by the sum. Only the
// owning rank contributes a nonzero row, so the reduce is exact at ANY rank count: this block is
// bit-exact vs embedBand (max|Δ|=0), the vocab-parallel analog of the column-parallel guarantee.
// A nil collective defaults to LocalCollective.
func (m *Model) tpEmbedLayer(ids []int, plan TPPlan, coll Collective) ([][]float32, error) {
	parts, err := m.tpEmbedShardPartials(ids, plan)
	if err != nil {
		return nil, err
	}
	if coll == nil {
		coll = LocalCollective{}
	}
	H := m.Cfg.HiddenSize
	out := make([][]float32, len(ids))
	for t := range ids {
		col := make([][]float32, len(parts))
		for r := range parts {
			col[r] = parts[r][t]
		}
		row, err := coll.AllReduceSum(col)
		if err != nil {
			return nil, fmt.Errorf("model: vocab-parallel embedding reduce: %w", err)
		}
		if len(row) != H {
			return nil, fmt.Errorf("model: vocab-parallel embedding reduce produced %d elements, want hidden %d", len(row), H)
		}
		scaleEmbedInPlace(row, m.Cfg)
		out[t] = row
	}
	return out, nil
}

// tpHeadBandPartials computes each rank's logits band over the final-normed hidden states xf:
// per position, logits[v] = fdot(W[v], xf[t]) for v in the rank's vocab band — the SAME fdot
// the monolith tail (logitsFromHidden -> residentMatRows -> matRows over the f32-resident head)
// computes for those rows, so each band is bit-exact vs the matching slice of the monolith
// logits. plan.Dim must equal the vocab size; the head must pass requireTPHeadResidentF32.
func (m *Model) tpHeadBandPartials(xf [][]float32, plan TPPlan) ([][][]float32, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	if plan.Dim != m.Cfg.VocabSize {
		return nil, fmt.Errorf("model: vocab-parallel head plan.Dim = %d, want vocab = %d", plan.Dim, m.Cfg.VocabSize)
	}
	if err := m.requireTPHeadResidentF32(); err != nil {
		return nil, err
	}
	H := m.Cfg.HiddenSize
	head := m.tensor(m.headName())
	seq := len(xf)
	rankPartials := make([][][]float32, len(plan.Shards))
	for r, s := range plan.Shards {
		band := shardWeightRows(head, H, s.Lo, s.Hi) // the rank's zero-copy [w,H] row band
		w := s.Width()
		part := make([][]float32, seq)
		for t := 0; t < seq; t++ {
			if len(xf[t]) != H {
				return nil, fmt.Errorf("model: vocab-parallel head position %d hidden length = %d, want %d", t, len(xf[t]), H)
			}
			part[t] = matRows(band, xf[t], w, H)
		}
		rankPartials[r] = part
	}
	return rankPartials, nil
}

// tpHeadLogits is the vocab-parallel forward TAIL (the ParallelLMHead analog of fillSeqLogits):
// the final norm runs per position (elementwise, unsharded — identical on every rank), the head
// matmul is sharded over the vocab plan, the bands are AllGather-concatenated per position in
// rank order (upstream tensor_model_parallel_all_gather; get_sharded_to_full_mapping is the
// identity here because fak has no vocab padding and no LoRA-added rows, so the concat order IS
// token-id order), and the arch logit scale applies ONCE after the gather — the same
// instruction stream as logitsFromHidden with only the head matmul split across ranks.
// Bit-exact vs the monolith tail at ANY rank count (each logit is one rank's fdot over the
// same rows). A nil collective defaults to LocalCollective.
func (m *Model) tpHeadLogits(x [][]float32, plan TPPlan, coll Collective) ([][]float32, error) {
	if coll == nil {
		coll = LocalCollective{}
	}
	seq := len(x)
	xf := make([][]float32, seq)
	for t := range x {
		xf[t] = m.finalNorm(x[t])
	}
	parts, err := m.tpHeadBandPartials(xf, plan)
	if err != nil {
		return nil, err
	}
	out := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		col := make([][]float32, len(parts))
		for r := range parts {
			col[r] = parts[r][t]
		}
		logits, err := coll.AllGather(col, plan)
		if err != nil {
			return nil, fmt.Errorf("model: vocab-parallel head gather: %w", err)
		}
		if len(logits) != m.Cfg.VocabSize {
			return nil, fmt.Errorf("model: vocab-parallel head gather produced %d logits, want vocab %d", len(logits), m.Cfg.VocabSize)
		}
		logitScaleInPlace(logits, m.Cfg)
		out[t] = logits
	}
	return out, nil
}
