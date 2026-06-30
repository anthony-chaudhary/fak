package model

import "strings"

// resident_report.go — observability for the resident hybrid Q4_K model: tallies which
// weights landed in which resident store (raw Q4_K vs Q8_0 vs f32) and the bytes each
// contributes, then derives the per-decode-token bandwidth stream. This is the small,
// 27B-free way to SEE the load's memory shape + the decode-bandwidth win (and predict
// tok/s) — run it right after LoadModelQ4K, before any generation. cmd/q4kdiag prints it.

// ResidentReport is a point-in-time tally of a loaded Model's resident weight stores.
type ResidentReport struct {
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
	F32Tensors    int   `json:"f32_tensors"` // small f32 manifest tensors (norms, embed, biases)
	F32Bytes      int64 `json:"f32_bytes"`   // their resident bytes

	TotalResidentBytes int64 `json:"total_resident_bytes"` // q4k + q8 + f32
	// DecodeBytesPerToken is the weight-byte stream one batch=1 decode step walks: every
	// matmul weight (q4k + q8) is read exactly once per generated token, so this is the
	// bandwidth-bound number that sets the decode tok/s ceiling (tok/s ≈ memBW / this).
	// Embedding is a row-gather (hidden·4 B), not a full stream, so it is excluded; norms
	// are negligible. f32 here is small-tensor-only and not on the matmul stream.
	DecodeBytesPerToken int64 `json:"decode_bytes_per_token"`
	// DecodeGBPerToken is DecodeBytesPerToken in GiB for a human-readable ceiling.
	DecodeGiBPerToken float64 `json:"decode_gib_per_token"`
}

// ResidentReport tallies the model's resident weight stores. It walks the q4kw/q8w maps +
// the f32 manifest once (O(tensor count), no allocation on the hot path) — cheap to run
// after load. For a q4_k_m Qwen3.6 the expectation is a dominant Q4K majority (MLP +
// v/o_proj + lm_head where the GGUF used Q4_K) and a small Q8 minority (the normalize-
// sensitive q/k + linear-attention projections, plus any Q6_K tensors), which is exactly
// the split that makes decode-bandwidth competitive with llama.cpp's q4_k_m.
func (m *Model) ResidentReport() *ResidentReport {
	r := &ResidentReport{}
	for _, qt := range m.q4kw {
		r.Q4KTensors++
		r.Q4KBytes += int64(len(qt.raw))
		// Each 144-byte Q4_K super-block encodes 256 weights.
		r.Q4KParams += int64(len(qt.raw) / q4kBlockBytes * qkK)
	}
	for _, qt := range m.q8w {
		r.Q8Tensors++
		// q8Tensor resident bytes: out*in int8 codes + out*nblk f32 scales.
		r.Q8Bytes += int64(len(qt.q)) + int64(len(qt.d))*4
		r.Q8Params += int64(qt.out) * int64(qt.in)
	}
	for _, qt := range m.kqw {
		r.KQuantTensors++
		r.KQuantBytes += int64(len(qt.raw))
		r.KQuantParams += int64(len(qt.raw) / qt.kind.blockBytes() * qt.kind.blockWeights())
	}
	for _, meta := range m.manifest {
		r.F32Tensors++
		r.F32Bytes += int64(meta.Nbytes)
	}
	r.TotalResidentBytes = r.Q4KBytes + r.Q8Bytes + r.KQuantBytes + r.F32Bytes
	// The matmul weights read per decode token = all of q4kw + q8w + kqw (the LM head is in
	// one of them; every projection + MLP weight streams once). This is the decode bandwidth.
	r.DecodeBytesPerToken = r.Q4KBytes + r.Q8Bytes + r.KQuantBytes
	r.DecodeGiBPerToken = float64(r.DecodeBytesPerToken) / (1 << 30)
	return r
}

// isRoutedExpertTensor reports whether a weight name is one of the per-layer ROUTED experts
// (model.layers.<L>.mlp.experts.<e>.<proj>.weight) — the only weights expert parallelism shards
// across ranks. The always-on GLM shared expert (mlp.shared_experts.* / mlp.shared_expert.*) is
// REPLICATED on every rank (it fires every token), so it deliberately does NOT match: the segment
// after ".mlp." is "shared_experts", not "experts", so ".mlp.experts." is not a substring of it.
func isRoutedExpertTensor(name string) bool {
	return strings.Contains(name, ".mlp.experts.")
}

// MoEResidentWeightBytes partitions the model's RESIDENT weight bytes into the routed-expert bytes
// (the only weights expert parallelism shards across ranks — model.layers.<L>.mlp.experts.<e>.*)
// and the replicated remainder (dense FFN + attention + router + embeddings + the always-on shared
// expert — held on EVERY rank). It walks the SAME resident stores ResidentReport tallies
// (q4kw / q8w / kqw / the f32 manifest), so it is quant-correct BY CONSTRUCTION — every tensor is
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
	for name, qt := range m.q8w {
		add(name, int64(len(qt.q))+int64(len(qt.d))*4)
	}
	for name, qt := range m.kqw {
		add(name, int64(len(qt.raw)))
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
	return "resident: Q4_K=" + itoa(r.Q4KTensors) + " tensors/" + fmtFloat(mib(r.Q4KBytes)) + "MiB" +
		"  rawExpertQuant=" + itoa(r.KQuantTensors) + "/" + fmtFloat(mib(r.KQuantBytes)) + "MiB" +
		"  Q8=" + itoa(r.Q8Tensors) + "/" + fmtFloat(mib(r.Q8Bytes)) + "MiB" +
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
