package model

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
	F32Tensors int   `json:"f32_tensors"` // small f32 manifest tensors (norms, embed, biases)
	F32Bytes   int64 `json:"f32_bytes"`   // their resident bytes

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
	for _, meta := range m.manifest {
		r.F32Tensors++
		r.F32Bytes += int64(meta.Nbytes)
	}
	r.TotalResidentBytes = r.Q4KBytes + r.Q8Bytes + r.F32Bytes
	// The matmul weights read per decode token = all of q4kw + q8w (the LM head is in one
	// of them; every projection + MLP weight streams once). This is the decode bandwidth.
	r.DecodeBytesPerToken = r.Q4KBytes + r.Q8Bytes
	r.DecodeGiBPerToken = float64(r.DecodeBytesPerToken) / (1 << 30)
	return r
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
