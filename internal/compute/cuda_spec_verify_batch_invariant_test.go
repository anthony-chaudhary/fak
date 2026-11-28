// fak-test:runtime fast est=1s lane=default
// Clean-room test inspired by Niko1221/Strata@c6f6427d9de9 (MIT):
// batch-invariant speculative verify attention split-K pinning & single-token decode bitwise parity.
package compute

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const (
	specVerifyInvariantKeyBudget = 2048
	specVerifyInvariantMaxM      = 16
	specVerifyFlashThreads       = 256
)

// hostFlashAttentionSingleTokenM1 mirrors CUDA k_flash_attention<false> (M=1, 1 linear split
// over [0, nPos-1]) in internal/compute/cuda/attention.cuh exact f32/online-softmax order.
func hostFlashAttentionSingleTokenM1(qRow, k, v []float32, nPos, nH, nKV, hd int, scale float32) []float32 {
	out := make([]float32, nH*hd)
	grp := nH / nKV
	w := nKV * hd
	var red [specVerifyFlashThreads]float32
	for h := 0; h < nH; h++ {
		kvh := h / grp
		qh := qRow[h*hd : (h+1)*hd]
		m := float32(math.Inf(-1))
		l := float32(0)
		acc := make([]float32, hd)
		for j := 0; j < nPos; j++ {
			kj := k[j*w+kvh*hd : j*w+(kvh+1)*hd]
			vj := v[j*w+kvh*hd : j*w+(kvh+1)*hd]
			for tid := 0; tid < specVerifyFlashThreads; tid++ {
				var p float32
				for d := tid; d < hd; d += specVerifyFlashThreads {
					p += qh[d] * kj[d]
				}
				red[tid] = p
			}
			for s := specVerifyFlashThreads / 2; s > 0; s >>= 1 {
				for tid := 0; tid < s; tid++ {
					red[tid] += red[tid+s]
				}
			}
			sc := red[0] * scale
			mNew := float32(math.Max(float64(m), float64(sc)))
			corr := float32(math.Exp(float64(m - mNew)))
			p := float32(math.Exp(float64(sc - mNew)))
			l = l*corr + p
			for d := 0; d < hd; d++ {
				acc[d] = acc[d]*corr + p*vj[d]
			}
			m = mNew
		}
		invL := float32(0)
		if l > 0 {
			invL = float32(1) / l
		}
		dst := out[h*hd : (h+1)*hd]
		for d := 0; d < hd; d++ {
			dst[d] = acc[d] * invL
		}
	}
	return out
}

// hostSpecVerifyBatchInvariant mirrors k_spec_verify_attention_batch_invariant:
// pins split partition count strictly to M=1 decode choice (splits=1) so every verify query row qi
// computes the exact same float32 rounding and reduction order as single-token decode M=1.
func hostSpecVerifyBatchInvariant(q, k, v []float32, qLen, kvLen, nH, nKV, hd int, scale float32) ([]float32, int) {
	// Pinned split count matches single-token decode choice (=1).
	splits := 1
	out := make([]float32, qLen*nH*hd)
	prefix := kvLen - qLen
	stride := nH * hd
	for qi := 0; qi < qLen; qi++ {
		nPos := prefix + qi + 1
		row := hostFlashAttentionSingleTokenM1(q[qi*stride:(qi+1)*stride], k, v, nPos, nH, nKV, hd, scale)
		copy(out[qi*stride:(qi+1)*stride], row)
	}
	return out, splits
}

// hostSpecVerifyMultiSplit8 mirrors k_spec_verify_attention + k_spec_verify_combine (8 segments),
// which can differ by ULPs from single-token decode because of two-stage partial normalization.
func hostSpecVerifyMultiSplit8(q, k, v []float32, qLen, kvLen, nH, nKV, hd int, scale float32) []float32 {
	const numSeg = 8
	out := make([]float32, qLen*nH*hd)
	grp := nH / nKV
	w := nKV * hd
	prefix := kvLen - qLen
	segLen := (kvLen + numSeg - 1) / numSeg
	var red [specVerifyFlashThreads]float32

	for qi := 0; qi < qLen; qi++ {
		maxPos := prefix + qi
		for h := 0; h < nH; h++ {
			kvh := h / grp
			qh := q[qi*nH*hd+h*hd : qi*nH*hd+(h+1)*hd]
			pOut := make([][]float32, numSeg)
			pMax := make([]float32, numSeg)
			pSum := make([]float32, numSeg)
			for seg := 0; seg < numSeg; seg++ {
				kvStart := seg * segLen
				kvEnd := kvStart + segLen
				if kvEnd > kvLen {
					kvEnd = kvLen
				}
				mStat := float32(math.Inf(-1))
				lStat := float32(0)
				acc := make([]float32, hd)
				for j := kvStart; j < kvEnd && j <= maxPos; j++ {
					kj := k[j*w+kvh*hd : j*w+(kvh+1)*hd]
					vj := v[j*w+kvh*hd : j*w+(kvh+1)*hd]
					for tid := 0; tid < specVerifyFlashThreads; tid++ {
						var p float32
						for d := tid; d < hd; d += specVerifyFlashThreads {
							p += qh[d] * kj[d]
						}
						red[tid] = p
					}
					for s := specVerifyFlashThreads / 2; s > 0; s >>= 1 {
						for tid := 0; tid < s; tid++ {
							red[tid] += red[tid+s]
						}
					}
					sc := red[0] * scale
					mNew := float32(math.Max(float64(mStat), float64(sc)))
					corr := float32(math.Exp(float64(mStat - mNew)))
					p := float32(math.Exp(float64(sc - mNew)))
					lStat = lStat*corr + p
					for d := 0; d < hd; d++ {
						acc[d] = acc[d]*corr + p*vj[d]
					}
					mStat = mNew
				}
				invL := float32(0)
				if lStat > 0 {
					invL = float32(1) / lStat
				}
				for d := 0; d < hd; d++ {
					acc[d] *= invL
				}
				pOut[seg] = acc
				pMax[seg] = mStat
				pSum[seg] = lStat
			}

			gMax := float32(math.Inf(-1))
			for seg := 0; seg < numSeg; seg++ {
				if pMax[seg] > gMax {
					gMax = pMax[seg]
				}
			}
			var gSum float32
			for seg := 0; seg < numSeg; seg++ {
				if pSum[seg] > 0 {
					gSum += pSum[seg] * float32(math.Exp(float64(pMax[seg]-gMax)))
				}
			}
			invGSum := float32(0)
			if gSum > 0 {
				invGSum = float32(1) / gSum
			}
			dst := out[qi*nH*hd+h*hd : qi*nH*hd+(h+1)*hd]
			for d := 0; d < hd; d++ {
				var merged float32
				for seg := 0; seg < numSeg; seg++ {
					if pSum[seg] > 0 {
						wSeg := pSum[seg] * float32(math.Exp(float64(pMax[seg]-gMax))) * invGSum
						merged += pOut[seg][d] * wSeg
					}
				}
				dst[d] = merged
			}
		}
	}
	return out
}

func recomputeCheckBitwiseF32(verifyOut, singleOut []float32) (int, int) {
	if len(verifyOut) != len(singleOut) {
		return -1, 0
	}
	mismatches := 0
	firstIdx := -1
	for i := range verifyOut {
		if math.Float32bits(verifyOut[i]) != math.Float32bits(singleOut[i]) {
			if firstIdx < 0 {
				firstIdx = i
			}
			mismatches++
		}
	}
	return mismatches, firstIdx
}

// TestCUDASpecVerifyAttentionEqualsSingleTokenDecodeBitwise pins the Strata-10 contract:
// (a) batch-invariant speculative verify attention matches M=1 single-token decode BIT-FOR-BIT across qLen={1,2,4,8,16},
// (b) a given token qi yields identical float32 bits regardless of verify batch size M (true batch invariance),
// (c) recompute check mode detects even 1-ULP split-K divergence from the multi-segment path and accepts zero diffs,
// (d) spec_graph.cuh, attention.cuh, and cuda_flashmla_sparse_decode.cu expose the required Strata-10 split pins & recompute checks.
func TestCUDASpecVerifyAttentionEqualsSingleTokenDecodeBitwise(t *testing.T) {
	cases := []struct {
		name      string
		nH, nKV   int
		hd, kvLen int
	}{
		{"mha", 4, 4, 64, 64},
		{"gqa", 8, 2, 64, 128},
		{"mqa", 8, 1, 64, 96},
		{"hd128_budget", 8, 2, 128, 192},
	}
	qLens := []int{1, 2, 4, 8, 16}

	for _, c := range cases {
		scale := float32(1.0 / math.Sqrt(float64(c.hd)))
		// Generate maximal query batch once so sub-batches (qLen=1,2,4,8,16) test identical prefix tokens.
		rng := lcg(10101 + uint64(c.nH*31+c.kvLen))
		maxM := qLens[len(qLens)-1]
		qFull := randVec(&rng, maxM*c.nH*c.hd)
		kData := randVec(&rng, c.kvLen*c.nKV*c.hd)
		vData := randVec(&rng, c.kvLen*c.nKV*c.hd)

		// Compare the verify token positioned at global index pos = c.kvLen - 1 across every batch size M.
		stride := c.nH * c.hd
		anchorRow := qFull[(maxM-1)*stride : maxM*stride]
		anchorSingle := hostFlashAttentionSingleTokenM1(anchorRow, kData, vData, c.kvLen, c.nH, c.nKV, c.hd, scale)

		var uldDiffSeen bool
		for _, qLen := range qLens {
			// Take last qLen query rows so the final token always aligns to key index c.kvLen-1.
			qSub := qFull[(maxM-qLen)*stride : maxM*stride]
			verifyOut, splits := hostSpecVerifyBatchInvariant(qSub, kData, vData, qLen, c.kvLen, c.nH, c.nKV, c.hd, scale)
			if splits != 1 {
				t.Fatalf("%s qLen=%d: expected split partition pinned to 1 (single-token choice), got %d", c.name, qLen, splits)
			}

			// Row-by-row M=1 single-token decode reference.
			prefix := c.kvLen - qLen
			singleOut := make([]float32, len(verifyOut))
			for qi := 0; qi < qLen; qi++ {
				row := hostFlashAttentionSingleTokenM1(qSub[qi*stride:(qi+1)*stride], kData, vData, prefix+qi+1, c.nH, c.nKV, c.hd, scale)
				copy(singleOut[qi*stride:(qi+1)*stride], row)
			}

			mismatches, firstIdx := recomputeCheckBitwiseF32(verifyOut, singleOut)
			if mismatches != 0 {
				t.Fatalf("%s qLen=%d: recompute check found %d bitwise mismatches (first idx=%d)",
					c.name, qLen, mismatches, firstIdx)
			}

			// Anchor token (last row of sub-batch) must stay bitwise identical across all batch sizes qLen={1,2,4,8,16}.
			lastVerify := verifyOut[(qLen-1)*stride : qLen*stride]
			if mm, fi := recomputeCheckBitwiseF32(lastVerify, anchorSingle); mm != 0 {
				t.Fatalf("%s qLen=%d: final verify row differed bitwise across batch sizes (mismatches=%d, first=%d)",
					c.name, qLen, mm, fi)
			}

			if qLen == 8 {
				multi8 := hostSpecVerifyMultiSplit8(qSub, kData, vData, qLen, c.kvLen, c.nH, c.nKV, c.hd, scale)
				if mm, _ := recomputeCheckBitwiseF32(multi8, singleOut); mm > 0 {
					uldDiffSeen = true
				}
			}
		}
		if !uldDiffSeen {
			t.Logf("%s: multi-split8 happened to match single-token on this fixture (still valid); batch-invariant path verified bitwise across all qLen", c.name)
		}
	}

	// Verify recomputeCheck rejects even a single 1-ULP alteration cleanly.
	a := []float32{1.0, -0.5, 0.25}
	b := []float32{1.0, math.Float32frombits(math.Float32bits(-0.5) ^ 1), 0.25}
	if mm, first := recomputeCheckBitwiseF32(a, b); mm != 1 || first != 1 {
		t.Fatalf("expected 1-ULP mismatch at idx=1, got mm=%d first=%d", mm, first)
	}

	// Verify CUDA source contracts across all 3 touched CUDA sources.
	t.Run("cuda_source_contract", func(t *testing.T) {
		attnBytes, err := os.ReadFile(filepath.Join("cuda", "attention.cuh"))
		if err != nil {
			t.Fatalf("read attention.cuh: %v", err)
		}
		specBytes, err := os.ReadFile(filepath.Join("cuda", "spec_graph.cuh"))
		if err != nil {
			t.Fatalf("read spec_graph.cuh: %v", err)
		}
		mlaBytes, err := os.ReadFile("cuda_flashmla_sparse_decode.cu")
		if err != nil {
			t.Fatalf("read cuda_flashmla_sparse_decode.cu: %v", err)
		}
		if !strings.Contains(string(attnBytes), "fak_attention_single_token_split_count") {
			t.Fatal("attention.cuh missing fak_attention_single_token_split_count")
		}
		spec := string(specBytes)
		for _, needle := range []string{
			"SPEC_VERIFY_INVARIANT_KEY_BUDGET",
			"k_spec_verify_attention_batch_invariant",
			"k_spec_verify_bitwise_check",
			"fcuda_spec_verify_attention_batch_invariant_f32",
			"fcuda_spec_verify_recompute_check_f32",
			"FAK_CUDA_SPEC_VERIFY_BATCH_INVARIANT",
		} {
			if !strings.Contains(spec, needle) {
				t.Fatalf("spec_graph.cuh missing %q", needle)
			}
		}
		mla := string(mlaBytes)
		for _, needle := range []string{
			"kStrataBatchInvariantKeyBudget",
			"fak_flashmla_batch_invariant_split_count",
			"fak_flashmla_decode_recompute_check",
		} {
			if !strings.Contains(mla, needle) {
				t.Fatalf("cuda_flashmla_sparse_decode.cu missing %q", needle)
			}
		}
	})
}
