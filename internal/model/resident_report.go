package model

import "strings"

// resident_report.go — observability for the resident hybrid Q4_K model: tallies which
// weights landed in which resident store (raw Q2_0/Q4_K vs Q8_0 vs f32) and the bytes each
// contributes, then derives the per-decode-token bandwidth stream. This is the small,
// 27B-free way to SEE the load's memory shape + the decode-bandwidth win (and predict
// tok/s) — run it right after LoadModelQ4K, before any generation. cmd/q4kdiag prints it.

// ResidentReport is a point-in-time tally of a loaded Model's resident weight stores.
type ResidentReport struct {
	Q2Tensors  int   `json:"q2_tensors"` // matmul weights held as raw PQ2_0 blocks
	Q2Bytes    int64 `json:"q2_bytes"`
	Q2Params   int64 `json:"q2_params"`
	Q4KTensors int   `json:"q4k_tensors"` // matmul weights held as raw Q4_K blocks
	Q4KBytes   int64 `json:"q4k_bytes"`   // their resident bytes (the q4_k_m majority)
	Q4KParams  int64 `json:"q4k_params"`  // weight elements those bytes encode
	Q8Tensors  int   `json:"q8_tensors"`  // matmul weights held as Q8_0 (normalize-sensitive + Q6_K)
	Q8Bytes    int64 `json:"q8_bytes"`    // their resident bytes
	Q8Params   int64 `json:"q8_params"`
	// KQuantTensors are MoE experts held as raw non-Q4_K GGUF quant blocks (the mixed-quant bulk
	// that no longer pays the f32 round-trip; quant_kquant.go).
	KQuantTensors int   `json:"kquant_tensors"`
	KQuantBytes   int64 `json:"kquant_bytes"`
	KQuantParams  int64 `json:"kquant_params"`
	// Q2KEmbedTensors is the packed Q2_K embedding table (TICKET-15).
	Q2KEmbedTensors int   `json:"q2k_embed_tensors"`
	Q2KEmbedBytes   int64 `json:"q2k_embed_bytes"`
	Q2KEmbedParams  int64 `json:"q2k_embed_params"`
	// PQ2Embed* is the packed GGUF group-128 ternary embedding table.
	PQ2EmbedTensors int   `json:"pq2_embed_tensors"`
	PQ2EmbedBytes   int64 `json:"pq2_embed_bytes"`
	PQ2EmbedParams  int64 `json:"pq2_embed_params"`
	// Q4KEmbed* is the packed Q4_K embedding table. It remains separate from
	// Q4K* because row-gather embeddings are excluded from decode weight traffic.
	Q4KEmbedTensors int   `json:"q4k_embed_tensors"`
	Q4KEmbedBytes   int64 `json:"q4k_embed_bytes"`
	Q4KEmbedParams  int64 `json:"q4k_embed_params"`
	// Q6KEmbed* is a tied Q6_K token table stored ONCE (fak#13567): the same bytes serve the
	// row gather and the LM head, so they are counted here and NOT again under KQuant*.
	// Because the tied head streams the whole table every token, these bytes ARE part of
	// DecodeBytesPerToken (unlike the gather-only packed embeds above).
	Q6KEmbedTensors int   `json:"q6k_embed_tensors"`
	Q6KEmbedBytes   int64 `json:"q6k_embed_bytes"`
	Q6KEmbedParams  int64 `json:"q6k_embed_params"`
	// TiedEmbedF32Bytes / TiedHeadQ8Bytes isolate the legacy tied two-copy layout (an f32
	// gather table plus a native-Q8 head for model.embed_tokens.weight). They are subsets of
	// F32Bytes / Q8Bytes, reported so a regression to that layout is visible as non-zero.
	TiedEmbedF32Bytes int64 `json:"tied_embed_f32_bytes"`
	TiedHeadQ8Bytes   int64 `json:"tied_head_q8_bytes"`
	// LMHead is the head route (LMHeadRoute): e.g. metal-q6k, cpu-q6k, cpu-q8.
	LMHead     string `json:"lm_head"`
	F32Tensors int    `json:"f32_tensors"` // small f32 manifest tensors (norms, embed, biases)
	F32Bytes   int64  `json:"f32_bytes"`   // their resident bytes

	TotalResidentBytes int64 `json:"total_resident_bytes"` // q2 + q4k + q8 + kquant + packed embeds + f32
	// DecodeBytesPerToken is the weight-byte stream one batch=1 decode step walks: every
	// matmul weight (q2 + q4k + q8 + kquant) is read once per generated token, so this is the
	// bandwidth-bound number that sets the decode tok/s ceiling (tok/s ≈ memBW / this).
	// Embedding is a row-gather (hidden·4 B), not a full stream, so it is excluded; norms
	// are negligible. f32 here is small-tensor-only and not on the matmul stream.
	DecodeBytesPerToken int64 `json:"decode_bytes_per_token"`
	// DecodeGBPerToken is DecodeBytesPerToken in GiB for a human-readable ceiling.
	DecodeGiBPerToken float64 `json:"decode_gib_per_token"`
}

// ResidentReport tallies the model's resident weight stores. It walks the q2w/q4kw/q8w maps +
// the f32 manifest once (O(tensor count), no allocation on the hot path) — cheap to run
// after load. For a q4_k_m Qwen3.6 the expectation is a dominant Q4K majority (MLP +
// v/o_proj + lm_head where the GGUF used Q4_K) and a small Q8 minority (the normalize-
// sensitive q/k + linear-attention projections, plus any Q6_K tensors), which is exactly
// the split that makes decode-bandwidth competitive with llama.cpp's q4_k_m.
func (m *Model) ResidentReport() *ResidentReport {
	r := m.residentStoreReport()
	r.LMHead = m.LMHeadRoute()
	return r
}

// residentStoreReport is ResidentReport without the LM-head route. The route probes the Metal
// handle tables under metalQ4KMu, so the Metal admission paths that already hold that lock
// (and only need byte totals) must use this form.
func (m *Model) residentStoreReport() *ResidentReport {
	r := &ResidentReport{}
	for _, qt := range m.q2w {
		r.Q2Tensors++
		r.Q2Bytes += int64(len(qt.raw)) + int64(len(qt.q)) + int64(len(qt.d))*4
		r.Q2Params += int64(qt.out) * int64(qt.in)
	}
	for _, qt := range m.q4kw {
		r.Q4KTensors++
		bytes := len(qt.raw)
		if qt.lazy != nil {
			bytes = qt.lazy.Bytes
		}
		// A prepared mapped view is CPU-resident even while its lazy metadata
		// remains available for offset-aware Metal uploads. Bare descriptors
		// have no raw bytes to charge.
		r.Q4KBytes += int64(len(qt.raw))
		// Params count logical weights even when the checkpoint payload is lazy.
		r.Q4KParams += int64(bytes / q4kBlockBytes * qkK)
	}
	tiedQ6K := m.tiedQ6KHead()
	// q8w is filled lazily by decode (quantizeOnDemand, under q8Mu): never iterate it unlocked.
	q8Mu.RLock()
	defer q8Mu.RUnlock()
	for name, qt := range m.q8w {
		if name == tiedEmbeddingName && m.Cfg.TieWordEmbeddings {
			r.TiedHeadQ8Bytes = int64(len(qt.q)) + int64(len(qt.d))*4
		}
		r.Q8Tensors++
		// q8Tensor resident bytes: out*in int8 codes + out*nblk f32 scales.
		r.Q8Bytes += int64(len(qt.q)) + int64(len(qt.d))*4
		r.Q8Params += int64(qt.out) * int64(qt.in)
	}
	for _, qt := range m.kqw {
		if qt == tiedQ6K {
			continue // counted once as Q6KEmbed*
		}
		r.KQuantTensors++
		rawBytes := len(qt.residentRawSnapshot())
		r.KQuantBytes += int64(rawBytes)
		r.KQuantParams += int64(rawBytes / qt.kind.blockBytes() * qt.kind.blockWeights())
	}
	if m.Q2KEmbedding != nil {
		switch m.Q2KEmbedding.Format() {
		case "Q4_K":
			r.Q4KEmbedTensors = 1
			r.Q4KEmbedBytes = int64(m.Q2KEmbedding.Bytes())
			r.Q4KEmbedParams = int64(m.Q2KEmbedding.Vocab()) * int64(m.Q2KEmbedding.Hidden())
		case "Q6_K":
			r.Q6KEmbedTensors = 1
			r.Q6KEmbedBytes = int64(m.Q2KEmbedding.Bytes())
			r.Q6KEmbedParams = int64(m.Q2KEmbedding.Vocab()) * int64(m.Q2KEmbedding.Hidden())
		case "PQ2_0":
			r.PQ2EmbedTensors = 1
			r.PQ2EmbedBytes = int64(m.Q2KEmbedding.Bytes())
			r.PQ2EmbedParams = int64(m.Q2KEmbedding.Vocab()) * int64(m.Q2KEmbedding.Hidden())
		default:
			r.Q2KEmbedTensors = 1
			r.Q2KEmbedBytes = int64(m.Q2KEmbedding.Bytes())
			r.Q2KEmbedParams = int64(m.Q2KEmbedding.Vocab()) * int64(m.Q2KEmbedding.Hidden())
		}
	}
	for name, meta := range m.manifest {
		if name == tiedEmbeddingName && m.Cfg.TieWordEmbeddings {
			r.TiedEmbedF32Bytes = int64(meta.Nbytes)
		}
		r.F32Tensors++
		r.F32Bytes += int64(meta.Nbytes)
	}
	r.TotalResidentBytes = r.Q2Bytes + r.Q4KBytes + r.Q8Bytes + r.KQuantBytes + r.Q2KEmbedBytes + r.PQ2EmbedBytes + r.Q4KEmbedBytes + r.Q6KEmbedBytes + r.F32Bytes
	// The matmul weights read per decode token = all of q2w + q4kw + q8w + kqw (the LM head is in
	// one of them; every projection + MLP weight streams once). This is the decode bandwidth.
	// Embedding is a row-gather (hidden·4 B), not a full stream, so it is excluded; norms
	// are negligible. f32 here is small-tensor-only and not on the matmul stream.
	r.DecodeBytesPerToken = r.Q2Bytes + r.Q4KBytes + r.Q8Bytes + r.KQuantBytes + r.Q6KEmbedBytes
	r.DecodeGiBPerToken = float64(r.DecodeBytesPerToken) / (1 << 30)
	return r
}

// isRoutedExpertTensor reports whether a weight name is one of the per-layer ROUTED experts
// (model.layers.<L>.mlp.experts.<e>.<proj>.weight) — the only weights expert parallelism shards
// across ranks. The always-on GLM shared expert (mlp.shared_experts.* / mlp.shared_expert.*) is
// REPLICATED on every rank (it fires every token), so it deliberately does NOT match: the segment
// after ".mlp." is "shared_experts", not "experts", so ".mlp.experts." is not a substring of it.
// The native non-MLA DeepSeek-V4.1 forward names its routed experts
// model.layers.<L>.ffn.experts.<e>.{w1,w3,w2}.weight (ffn, not mlp), so this
// predicate must also match ".ffn.experts." or MoEResidentWeightBytes would count
// the whole V4.1 routed-expert bulk as REPLICATED — breaking the expert-parallel
// per-rank fit plan and the expert-spill budget sizing for the very artifact the
// resident Q2_K spine exists to serve (fak#13271).
func isRoutedExpertTensor(name string) bool {
	return strings.Contains(name, ".mlp.experts.") || strings.Contains(name, ".ffn.experts.")
}

// MoEResidentWeightBytes partitions the model's RESIDENT weight bytes into the routed-expert bytes
// (the only weights expert parallelism shards across ranks — model.layers.<L>.mlp.experts.<e>.*)
// and the replicated remainder (dense FFN + attention + router + embeddings + the always-on shared
// expert — held on EVERY rank). It walks the SAME resident stores ResidentReport tallies
// (q2w / q4kw / q8w / kqw / the f32 manifest), so it is quant-correct BY CONSTRUCTION — every tensor is
// counted at its actual resident size in whatever store holds it, never an f32 estimate of a
// quantized weight — and replicated+expert equals ResidentReport().TotalResidentBytes (the test
// pins this). It partitions purely by NAME (isRoutedExpertTensor).
//
// It is the loaded-model input to compute.ExpertParallelPerRankPlan: replicated stays per-rank
// fixed while the expert term shards ~1/ranks, so a serve can pre-check whether `--expert-parallel N`
// actually fits each GPU before the multi-minute weight load. ok is false when nothing is resident
// (an empty/unloaded model), so a caller fails OPEN (skips the fit pre-check) rather than refusing on
// a zero footprint.
func (m *Model) MoEResidentWeightBytes() (replicated, expert int64, ok bool) {
	add := func(name string, bytes int64) {
		if bytes <= 0 {
			return
		}
		if isRoutedExpertTensor(name) {
			expert += bytes
		} else {
			replicated += bytes
		}
	}
	for name, qt := range m.q4kw {
		add(name, int64(len(qt.raw)))
	}
	for name, qt := range m.q2w {
		add(name, int64(len(qt.raw))+int64(len(qt.q))+int64(len(qt.d))*4)
	}
	q8Mu.RLock()
	for name, qt := range m.q8w {
		add(name, int64(len(qt.q))+int64(len(qt.d))*4)
	}
	q8Mu.RUnlock()
	tiedQ6K := m.tiedQ6KHead()
	for name, qt := range m.kqw {
		if qt == tiedQ6K {
			continue // shared with Q2KEmbedding below; count once
		}
		add(name, int64(len(qt.residentRawSnapshot())))
	}
	if m.Q2KEmbedding != nil {
		add("model.embed_tokens.weight", int64(m.Q2KEmbedding.Bytes()))
	}
	for name, meta := range m.manifest {
		add(name, int64(meta.Nbytes))
	}
	return replicated, expert, replicated+expert > 0
}

// DecodeTokSCeiling estimates the bandwidth-bound decode ceiling at a given machine memory
// bandwidth (GB/s): memBWGBps / decode-GB-per-token. It is a CEILING (perfect bandwidth
// utilization), not a measured speed — the real number sits below it by whatever the kernel
// leaves on the table. Useful to predict whether a load can reach a target bar (e.g. the
// 7.29 tok/s q4_k_m bar) before spending a 27B run.
func (r *ResidentReport) DecodeTokSCeiling(memBWGBps float64) float64 {
	if r.DecodeGiBPerToken <= 0 || memBWGBps <= 0 {
		return 0
	}
	// decodeGiBPerToken is in GiB (2^30); memBW in GB/s (10^9). Convert GiB→GB.
	return memBWGBps / (r.DecodeGiBPerToken * 1.073741824)
}

// FormatResidentReport renders a one-line human-readable summary for tool stderr: the
// resident split + the decode-bandwidth stream. Print after LoadModelQ4K to SEE the load's
// memory shape and the predicted decode ceiling without running generation.
func FormatResidentReport(r *ResidentReport) string {
	mib := func(b int64) float64 { return float64(b) / (1 << 20) }
	embedStr := ""
	if r.Q2KEmbedTensors > 0 {
		embedStr = "  Q2_K_embed=" + itoa(r.Q2KEmbedTensors) + "/" + fmtFloat(mib(r.Q2KEmbedBytes)) + "MiB"
	} else if r.PQ2EmbedTensors > 0 {
		embedStr = "  PQ2_0_embed=" + itoa(r.PQ2EmbedTensors) + "/" + fmtFloat(mib(r.PQ2EmbedBytes)) + "MiB"
	} else if r.Q4KEmbedTensors > 0 {
		embedStr = "  Q4_K_embed=" + itoa(r.Q4KEmbedTensors) + "/" + fmtFloat(mib(r.Q4KEmbedBytes)) + "MiB"
	} else if r.Q6KEmbedTensors > 0 {
		embedStr = "  Q6_K_tied_embed=" + itoa(r.Q6KEmbedTensors) + "/" + fmtFloat(mib(r.Q6KEmbedBytes)) + "MiB"
	}
	if r.LMHead != "" {
		embedStr += "  lm_head=" + r.LMHead
	}
	return "resident: Q2_0=" + itoa(r.Q2Tensors) + " tensors/" + fmtFloat(mib(r.Q2Bytes)) + "MiB" +
		"  Q4_K=" + itoa(r.Q4KTensors) + " tensors/" + fmtFloat(mib(r.Q4KBytes)) + "MiB" +
		"  rawExpertQuant=" + itoa(r.KQuantTensors) + "/" + fmtFloat(mib(r.KQuantBytes)) + "MiB" +
		"  Q8=" + itoa(r.Q8Tensors) + "/" + fmtFloat(mib(r.Q8Bytes)) + "MiB" +
		embedStr +
		"  f32=" + itoa(r.F32Tensors) + "/" + fmtFloat(mib(r.F32Bytes)) + "MiB" +
		"  total=" + fmtFloat(mib(r.TotalResidentBytes)) + "MiB" +
		"  decode=" + fmtFloat(r.DecodeGiBPerToken) + "GiB/tok"
}

// fmtFloat is a tiny strconv-free formatter (avoids pulling strconv into this file and
// keeps the report self-contained). 2 decimal places.
func fmtFloat(f float64) string {
	whole := int(f)
	frac := int((f - float64(whole)) * 100)
	if frac < 0 {
		frac = -frac
	}
	return itoa(whole) + "." + fracDigits(frac)
}

func fracDigits(n int) string {
	if n < 10 {
		return "0" + itoa(n)
	}
	return itoa(n)
}
