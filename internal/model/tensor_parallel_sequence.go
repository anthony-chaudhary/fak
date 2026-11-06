package model

import "fmt"

// tensor_parallel_sequence.go — MEGATRON SEQUENCE PARALLELISM (SP), ADAPTED from
// Megatron-LM megatron/core/tensor_parallel/mappings.py (the sequence-parallel region
// mappings: gather_from_sequence_parallel_region and reduce_scatter_to_sequence_parallel_
// region, with copy_to_tensor_model_parallel_region / scatter_to_tensor_model_parallel_
// region as their single-rank degenerations). tensor_parallel.go ported the FFN block,
// tensor_parallel_attn.go the attention block, and tensor_parallel_vocab.go the vocab
// block of the intra-layer decomposition; this file ports the OUTER lever they leave out:
// sharding the ACTIVATION along the SEQUENCE dimension. With SP armed, the residual stream
// lives SHARDED ALONG THE SEQUENCE outside the tensor-parallel cores: rank r owns the
// contiguous band [Lo,Hi) of positions. Entering a TP core (attention/FFN) the full
// sequence is re-materialized with an AllGather over the sequence bands; leaving a
// row-parallel core, instead of broadcasting the AllReduce result to every rank, the
// per-rank partials are ReduceScatter-ed back to sequence bands. Norms are per-position
// elementwise over hidden, so each rank norms only its band — identical values to the
// monolith at those positions.
//
// THE BIT-EXACTNESS ARGUMENT (what the gate pins): SP changes only WHERE data lives,
// never the arithmetic or the reduce order.
//
//   - ENTERING a core: the per-rank normed bands are AllGather-concatenated in rank order
//     into the full sequence. Each position belongs to exactly one band, and the band's
//     value at that position is the SAME per-position norm the monolith computes there, so
//     the concat IS the full normed sequence, byte-for-byte.
//   - INSIDE a core: the EXISTING tpAttnLayer / tpFFNLayer run UNCHANGED over that full
//     sequence — same weights, same RoPE (over the full gathered sequence), same per-rank
//     partials, same single AllReduceSum per position, bias added once after the reduce.
//     Identical input bytes, deterministic code path, identical output bytes.
//   - LEAVING a core: the ReduceScatter-by-composition below. The AllReduceSum half is the
//     reduction the sub-layer already performs (tpReduceWithBias); the scatter half slices
//     the reduced result into contiguous sequence bands. By the identity pinned in
//     internal/compute/collective.go — ReduceScatter reuses the IDENTICAL rank-order sum as
//     AllReduceSum and slices it — this composition is byte-identical to a real
//     ReduceScatter.
//   - RESIDUAL: the add is position-local; each rank adds its band of sub-layer output onto
//     its own band of the hidden state, on operands that are identical to the full path's
//     at those positions.
//
// Hence ForwardTP(SequenceRanks=P) == ForwardTP(SequenceRanks off) at max|Δ|=0 — full
// logits AND every hidden snapshot — at every P, for every supported topology. The gate
// chain this file joins (same grammar as tensor_parallel_forward.go / _vocab.go):
//
//	ForwardTP(SequenceRanks=P) ==(max|Δ|=0)  ForwardTP(SequenceRanks off)  [this file's tests]
//	ForwardTP(ranks=1)         ==(max|Δ|=0)  Forward                      [tensor_parallel_forward_test.go]
//	Forward                    ==(HF oracle) HuggingFace                   [oracle_test.go]
//
// HONEST DIVERGENCES from upstream (each is a fak adaptation, not an omission):
//
//   - COMPOSITION INSTEAD OF INTERFACE for the ReduceScatter: model.Collective exposes only
//     AllGather/AllReduceSum (the two-method NCCL swap-in seam), so the SP scatter is built
//     by composing the existing rank-order AllReduceSum with a band slice rather than adding
//     a third interface method. That composition is the conformance target a device
//     ReduceScatter must reproduce, and collective_bridge.go already maps onto
//     compute.CollectiveBackend, which HAS a real ReduceScatter for the device swap-in —
//     the bytes exchanged here are exactly what it would produce.
//   - NEAR-EVEN BANDS instead of NCCL's equal N/P shards: real NCCL requires
//     sendcount % nranks == 0; fak reuses its validated TPPlan tiling (near-even contiguous
//     bands, remainder on the first shards, no empty shard) over [0,seq), so seq need not
//     divide P.
//   - WITNESS MATERIALIZATION of act.Hidden: the snapshots stay FULL-sequence [seq*H] per
//     layer — each is materialized by concatenating the per-rank bands in rank order. A real
//     deployment keeps the bands resident across layers; the memory win is a deployment
//     property, not observable in the single-box witness.
//   - EMBEDDING AND TAIL stay unsharded along the sequence: the embedding is per-position
//     independent (monolithic embedBand, or the vocab-parallel tpEmbedLayer when
//     VocabRanks>1), and the tail (final norm + logits) runs on the AllGather-ed final full
//     sequence through the EXISTING tail unchanged — SP composes with AttnRanks / FFNRanks /
//     VocabRanks.
//
// SCOPE — fail-closed: SP supports exactly the three sequential placements
// composeSeqSublayer lowers (PreNorm / PostNorm / SandwichNorm); any other topology (the
// ParallelResidual shared-norm two-branch graph) is refused with a named error.

// spElementPlan lifts the sequence-dimension plan into the flattened [seq*H] element space:
// sequence shard [Lo,Hi) becomes element shard [Lo*H, Hi*H). Every position contributes
// exactly H contiguous elements, so the validated near-even contiguous sequence tiling maps
// to a valid element tiling (Validate passes: same contiguity, scaled bounds), and the
// Collective's width check (parts[r] length == shard width) lines up with the flattened
// bands.
func spElementPlan(sp TPPlan, H int) TPPlan {
	el := TPPlan{Dim: sp.Dim * H, Shards: make([]TPShard, len(sp.Shards))}
	for r, s := range sp.Shards {
		el.Shards[r] = TPShard{Rank: s.Rank, Lo: s.Lo * H, Hi: s.Hi * H}
	}
	return el
}

// spGatherBands ports upstream gather_from_sequence_parallel_region: the per-rank sequence
// bands ([width][H] each) are AllGather-concatenated in rank order into the full [seq][H]
// sequence. Each band is flattened to width*H elements and the element-space plan
// (spElementPlan) drives the Collective — whose rank-order concat is upstream's
// tensor_model_parallel_all_gather. Bit-exact: the concatenation of identical per-position
// values IS the full sequence. A nil collective defaults to LocalCollective.
func spGatherBands(bands [][][]float32, sp TPPlan, H int, coll Collective) ([][]float32, error) {
	if err := sp.Validate(); err != nil {
		return nil, err
	}
	if len(bands) != len(sp.Shards) {
		return nil, fmt.Errorf("model: sequence-parallel gather got %d bands, plan has %d shards", len(bands), len(sp.Shards))
	}
	if coll == nil {
		coll = LocalCollective{}
	}
	parts := make([][]float32, len(bands))
	for r, s := range sp.Shards {
		flat := flatten(bands[r])
		if len(flat) != s.Width()*H {
			return nil, fmt.Errorf("model: sequence-parallel gather band %d has %d elements, want %d (width %d x hidden %d)", r, len(flat), s.Width()*H, s.Width(), H)
		}
		parts[r] = flat
	}
	full, err := coll.AllGather(parts, spElementPlan(sp, H))
	if err != nil {
		return nil, fmt.Errorf("model: sequence-parallel gather: %w", err)
	}
	if len(full) != sp.Dim*H {
		return nil, fmt.Errorf("model: sequence-parallel gather produced %d elements, want %d (seq %d x hidden %d)", len(full), sp.Dim*H, sp.Dim, H)
	}
	out := make([][]float32, sp.Dim)
	for t := range out {
		out[t] = full[t*H : (t+1)*H]
	}
	return out, nil
}

// spSliceBands is the "scatter" half of the ReduceScatter-by-composition: it slices the
// full [seq][H] reduced sub-layer result into the per-rank contiguous sequence bands
// [Lo,Hi) — rank r keeps its band. Composed with the sub-layer's AllReduceSum (the "reduce"
// half, performed inside tpAttnLayer / tpFFNLayer via tpReduceWithBias, bias added once
// after the reduce), this reproduces upstream reduce_scatter_to_sequence_parallel_region
// byte-for-byte by the identity pinned in internal/compute/collective.go: ReduceScatter
// reuses the IDENTICAL rank-order sum as AllReduceSum and slices it. The only divergence is
// that fak slices into the TPPlan's near-even bands (remainder on the first shards) rather
// than NCCL's equal N/P shards. Rows are copied so each band owns its storage outright.
func spSliceBands(full [][]float32, sp TPPlan) [][][]float32 {
	bands := make([][][]float32, len(sp.Shards))
	for r, s := range sp.Shards {
		w := s.Width()
		bands[r] = make([][]float32, w)
		for t := 0; t < w; t++ {
			row := make([]float32, len(full[s.Lo+t]))
			copy(row, full[s.Lo+t])
			bands[r][t] = row
		}
	}
	return bands
}

// spPreNorm normalizes each position of a band with the pre-norm weights (fresh rows):
// per-position elementwise over hidden, so the band result equals the monolith's normSeq at
// exactly those positions, bit-for-bit.
func spPreNorm(band [][]float32, n normWeights, eps float32, cfg Config) [][]float32 {
	out := make([][]float32, len(band))
	for t := range band {
		out[t] = normCfg(band[t], n.pre, n.preBias, eps, cfg)
	}
	return out
}

// spPostNorm is the post-norm twin (n.post / n.postBias) — the band form of addPostNormed's
// normalization step, applied where the topology places the post-norm.
func spPostNorm(band [][]float32, n normWeights, eps float32, cfg Config) [][]float32 {
	out := make([][]float32, len(band))
	for t := range band {
		out[t] = normCfg(band[t], n.post, n.postBias, eps, cfg)
	}
	return out
}

// spTopologySupported fails closed on any block topology the band-composed loop does not
// implement: sequence parallelism supports exactly the three sequential placements
// composeSeqSublayer lowers (PreNorm / PostNorm / SandwichNorm). ParallelResidual (shared
// norm, two branches both reading the ORIGINAL residual before their deltas are summed) is
// a different graph the band loop does not lower, and is refused with a named error rather
// than mis-served.
func spTopologySupported(t BlockTopology) error {
	switch t {
	case PreNorm, PostNorm, SandwichNorm:
		return nil
	}
	return fmt.Errorf("model: sequence-parallel ForwardTP supports PreNorm/PostNorm/SandwichNorm only (got %s; the shared-norm two-branch residual is a later lever)", t)
}

// spSublayer applies ONE residual sub-layer under sequence parallelism for topology t,
// mutating the per-rank hidden-state bands in place (the position-local residual). The six
// steps mirror composeSeqSublayer's placement with the SP dataflow around the body:
//
//  1. band-local input transform — the pre-norm where the topology places it (per-position
//     elementwise, so band-local == monolith at those positions); PostNorm reads the RAW
//     residual stream, band-local;
//  2. AllGather the bands in rank order into the full sequence entering the TP core;
//  3. the EXISTING sharded sub-layer body (tpAttnLayer / tpFFNLayer, unchanged) over that
//     full sequence — per-TP-rank partials, one AllReduceSum per position, bias once;
//  4. ReduceScatter-by-composition: slice the reduced result back into sequence bands;
//  5. the post-norm where the topology places it, per band (elementwise);
//  6. each rank adds its band of sub-layer output onto its own hidden-state band.
func (m *Model) spSublayer(t BlockTopology, spPlan TPPlan, bands [][][]float32, n normWeights, eps float32, cfg Config, coll Collective, body func(xn [][]float32) ([][]float32, error)) error {
	if err := spTopologySupported(t); err != nil {
		return err
	}
	H := cfg.HiddenSize
	in := make([][][]float32, len(bands))
	switch t {
	case PostNorm:
		for r := range bands {
			in[r] = bands[r] // the sub-layer reads the RAW residual stream, band-local
		}
	default: // PreNorm, SandwichNorm
		for r := range bands {
			in[r] = spPreNorm(bands[r], n, eps, cfg)
		}
	}
	full, err := spGatherBands(in, spPlan, H, coll)
	if err != nil {
		return err
	}
	outFull, err := body(full)
	if err != nil {
		return err
	}
	out := spSliceBands(outFull, spPlan)
	if t == PostNorm || t == SandwichNorm {
		for r := range out {
			out[r] = spPostNorm(out[r], n, eps, cfg)
		}
	}
	for r := range bands {
		for p := range bands[r] {
			for i := range bands[r][p] {
				bands[r][p][i] += out[r][p][i]
			}
		}
	}
	return nil
}

// forwardTPSP runs the sequence-parallel layer loop over the live model: the residual
// stream is held as per-rank sequence bands; each TP sub-layer is ENTERED by AllGather-ing
// the (normed) bands into the full sequence and running the EXISTING sharded sub-layer
// (tpAttnLayer / tpFFNLayer, unchanged), and LEFT by ReduceScatter-ing the reduced result
// back to bands by composition (the sub-layer's AllReduceSum, then the band slice). Norms
// are per-position elementwise, so each rank norms only its band; residuals are
// position-local; RoPE/attention run over the full gathered sequence. It appends the
// full-sequence hidden snapshot per layer (materialized by concatenating the per-rank bands
// in rank order — the witness materialization; a deployment keeps the bands resident) and
// returns the final full sequence for the tail.
func (m *Model) forwardTPSP(x0 [][]float32, spPlan, attnPlan, ffnPlan TPPlan, coll Collective, act *Activations) ([][]float32, error) {
	cfg := m.Cfg
	H := cfg.HiddenSize
	eps := float32(cfg.RMSNormEps)
	seq := len(x0)

	bands := spSliceBands(x0, spPlan)

	var subErr error
	for l := 0; l < cfg.NumLayers; l++ {
		rp := newRopeForLayer(cfg, l, seq)
		attnBody := func(xn [][]float32) ([][]float32, error) {
			return m.tpAttnLayer(l, xn, rp, attnPlan, coll)
		}
		mlpBody := func(xn [][]float32) ([][]float32, error) {
			return m.tpFFNLayer(l, xn, ffnPlan, coll)
		}
		if err := m.spSublayer(cfg.BlockTopology, spPlan, bands, m.attentionNorms(l), eps, cfg, coll, attnBody); err != nil {
			subErr = fmt.Errorf("model: ForwardTP sequence-parallel layer %d attention: %w", l, err)
			break
		}
		if err := m.spSublayer(cfg.BlockTopology, spPlan, bands, m.mlpNorms(l), eps, cfg, coll, mlpBody); err != nil {
			subErr = fmt.Errorf("model: ForwardTP sequence-parallel layer %d mlp: %w", l, err)
			break
		}
		full, err := spGatherBands(bands, spPlan, H, coll)
		if err != nil {
			subErr = fmt.Errorf("model: ForwardTP sequence-parallel layer %d snapshot: %w", l, err)
			break
		}
		act.Hidden = append(act.Hidden, flatten(full))
	}
	if subErr != nil {
		return nil, subErr
	}
	// The tail runs on the AllGather-ed final full sequence (existing tail, unchanged).
	return spGatherBands(bands, spPlan, H, coll)
}
