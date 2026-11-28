// fak-test:runtime fast est=1s lane=default
package compute

import (
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
)

const (
	pagedPrefillBlockSize = 16
	pagedPrefillVecX      = 4
	pagedPrefillThreads   = 256
)

// hostPagedKOffset mirrors fak_vllm_prefill::paged_k_elem_offset:
// K_cache [blocks, nKV, hd/4, block_size, 4].
func hostPagedKOffset(blockIdx, blockOff, kvh, d, nKV, hd, blockSize int) int {
	hdVec := hd / pagedPrefillVecX
	dVec := d / pagedPrefillVecX
	dSub := d % pagedPrefillVecX
	return ((((blockIdx*nKV+kvh)*hdVec+dVec)*blockSize+blockOff)*pagedPrefillVecX + dSub)
}

// hostPagedVOffset mirrors fak_vllm_prefill::paged_v_elem_offset:
// V_cache [blocks, nKV, hd, block_size].
func hostPagedVOffset(blockIdx, blockOff, kvh, d, nKV, hd, blockSize int) int {
	return (((blockIdx*nKV+kvh)*hd+d)*blockSize + blockOff)
}

// hostWritePagedKV inserts logical contiguous K/V [positions, nKV, hd] into physical paged arenas.
func hostWritePagedKV(kContig, vContig []float32, kCache, vCache []float32, blockTable []int, startPos, count, nKV, hd, blockSize int) {
	kvWidth := nKV * hd
	for t := 0; t < count; t++ {
		pos := startPos + t
		phys := blockTable[pos/blockSize]
		off := pos % blockSize
		for kvh := 0; kvh < nKV; kvh++ {
			base := t*kvWidth + kvh*hd
			for d := 0; d < hd; d++ {
				kCache[hostPagedKOffset(phys, off, kvh, d, nKV, hd, blockSize)] = kContig[base+d]
				vCache[hostPagedVOffset(phys, off, kvh, d, nKV, hd, blockSize)] = vContig[base+d]
			}
		}
	}
}

// hostContiguousPromptPanelOracle reproduces k_prompt_causal_attention_panel's exact
// online-softmax arithmetic and reduction order (grouped by FLASH_THREADS=256 partial sums).
func hostContiguousPromptPanelOracle(q, k, v []float32, tokens, prefix, nH, nKV, hd int, scale float32) []float32 {
	out := make([]float32, tokens*nH*hd)
	grp := nH / nKV
	kvWidth := nKV * hd
	var red [pagedPrefillThreads]float32
	for token := 0; token < tokens; token++ {
		nPos := prefix + token + 1
		for h := 0; h < nH; h++ {
			kvh := h / grp
			qh := q[token*nH*hd+h*hd : token*nH*hd+(h+1)*hd]
			m := float32(math.Inf(-1))
			l := float32(0)
			acc := make([]float32, hd)
			for j := 0; j < nPos; j++ {
				kj := k[j*kvWidth+kvh*hd : j*kvWidth+(kvh+1)*hd]
				vj := v[j*kvWidth+kvh*hd : j*kvWidth+(kvh+1)*hd]
				for tid := 0; tid < pagedPrefillThreads; tid++ {
					var p float32
					for d := tid; d < hd; d += pagedPrefillThreads {
						p += qh[d] * kj[d]
					}
					red[tid] = p
				}
				for s := pagedPrefillThreads / 2; s > 0; s >>= 1 {
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
			dst := out[token*nH*hd+h*hd : token*nH*hd+(h+1)*hd]
			for d := 0; d < hd; d++ {
				dst[d] = acc[d] * invL
			}
		}
	}
	return out
}

// hostPagedChunkedPrefillOracle reproduces k_paged_chunked_prefill_attention over paged K/V prefix
// plus optional contiguous fresh chunk without allocating a concatenated temporary KV buffer.
func hostPagedChunkedPrefillOracle(
	q, kCache, vCache []float32,
	blockTable []int,
	kFresh, vFresh []float32,
	tokens, prefix, nH, nKV, hd, blockSize int, scale float32,
) ([]float32, error) {
	if tokens <= 0 || prefix < 0 || nH <= 0 || nKV <= 0 || nH%nKV != 0 || hd <= 0 || hd%pagedPrefillVecX != 0 || blockSize != pagedPrefillBlockSize {
		return nil, fmt.Errorf("invalid paged chunked prefill shape tokens=%d prefix=%d nH=%d nKV=%d hd=%d block=%d",
			tokens, prefix, nH, nKV, hd, blockSize)
	}
	useFresh := kFresh != nil && vFresh != nil
	if (kFresh == nil) != (vFresh == nil) {
		return nil, fmt.Errorf("kFresh and vFresh must both be nil or non-nil")
	}
	pagedPositions := prefix
	if !useFresh {
		pagedPositions = prefix + tokens
	}
	if pagedPositions > 0 {
		needBlocks := (pagedPositions + blockSize - 1) / blockSize
		if len(blockTable) < needBlocks {
			return nil, fmt.Errorf("blockTable too short: len=%d < need=%d", len(blockTable), needBlocks)
		}
		seen := make(map[int]bool, needBlocks)
		for i := 0; i < needBlocks; i++ {
			if blockTable[i] < 0 {
				return nil, fmt.Errorf("negative physical block %d", blockTable[i])
			}
			if seen[blockTable[i]] {
				return nil, fmt.Errorf("duplicate physical block %d", blockTable[i])
			}
			seen[blockTable[i]] = true
		}
	}

	out := make([]float32, tokens*nH*hd)
	grp := nH / nKV
	kvWidth := nKV * hd
	var red [pagedPrefillThreads]float32
	for token := 0; token < tokens; token++ {
		nPos := prefix + token + 1
		for h := 0; h < nH; h++ {
			kvh := h / grp
			qh := q[token*nH*hd+h*hd : token*nH*hd+(h+1)*hd]
			m := float32(math.Inf(-1))
			l := float32(0)
			acc := make([]float32, hd)
			for j := 0; j < nPos; j++ {
				fromFresh := useFresh && j >= prefix
				var phys, off, freshBase int
				if fromFresh {
					freshBase = (j-prefix)*kvWidth + kvh*hd
				} else {
					phys = blockTable[j/blockSize]
					off = j % blockSize
				}
				for tid := 0; tid < pagedPrefillThreads; tid++ {
					var p float32
					for d := tid; d < hd; d += pagedPrefillThreads {
						var kv float32
						if fromFresh {
							kv = kFresh[freshBase+d]
						} else {
							kv = kCache[hostPagedKOffset(phys, off, kvh, d, nKV, hd, blockSize)]
						}
						p += qh[d] * kv
					}
					red[tid] = p
				}
				for s := pagedPrefillThreads / 2; s > 0; s >>= 1 {
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
					var vv float32
					if fromFresh {
						vv = vFresh[freshBase+d]
					} else {
						vv = vCache[hostPagedVOffset(phys, off, kvh, d, nKV, hd, blockSize)]
					}
					acc[d] = acc[d]*corr + p*vv
				}
				m = mNew
			}
			invL := float32(0)
			if l > 0 {
				invL = float32(1) / l
			}
			dst := out[token*nH*hd+h*hd : token*nH*hd+(h+1)*hd]
			for d := 0; d < hd; d++ {
				dst[d] = acc[d] * invL
			}
		}
	}
	return out, nil
}

// TestPagedChunkedPrefillMatchesContiguousPanel pins the vLLM paged chunked prefill contract:
// (a) single-block paged prefix + fresh chunk == contiguous panel bitwise,
// (b) N-block non-contiguous page-crossing window == contiguous panel bitwise,
// (c) empty prefix (cold start, prefix=0) == contiguous panel bitwise,
// (d) all-paged (fresh==nil) + invalid blockTable refusal + CUDA source-contract checks.
func TestPagedChunkedPrefillMatchesContiguousPanel(t *testing.T) {
	cases := []struct {
		name       string
		prefix     int
		tokens     int
		nH, nKV    int
		hd         int
		numBlocks  int
		blockTable []int
		freshNil   bool
	}{
		{
			name:       "single_block_prefix",
			prefix:     10,
			tokens:     4,
			nH:         4,
			nKV:        2,
			hd:         64,
			numBlocks:  8,
			blockTable: []int{5},
		},
		{
			name:       "n_block_page_crossing_window",
			prefix:     37, // crosses blocks at 16, 32 and partial 37 -> 3 prefix blocks
			tokens:     19, // total causal span up to 56 positions -> crosses block 48 if all-paged
			nH:         8,
			nKV:        2,
			hd:         64,
			numBlocks:  16,
			blockTable: []int{11, 3, 14, 1},
		},
		{
			name:       "empty_prefix_cold_start",
			prefix:     0,
			tokens:     11,
			nH:         8,
			nKV:        4,
			hd:         64,
			numBlocks:  0,
			blockTable: nil,
		},
		{
			name:       "all_paged_prefix_and_chunk",
			prefix:     29,
			tokens:     7,
			nH:         8,
			nKV:        2,
			hd:         128,
			numBlocks:  10,
			blockTable: []int{9, 2, 6},
			freshNil:   true,
		},
	}

	for idx, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rng := lcg(8080 + uint64(idx*37))
			totalKV := tc.prefix + tc.tokens
			scale := float32(1.0 / math.Sqrt(float64(tc.hd)))
			qData := randVec(&rng, tc.tokens*tc.nH*tc.hd)
			kAll := randVec(&rng, totalKV*tc.nKV*tc.hd)
			vAll := randVec(&rng, totalKV*tc.nKV*tc.hd)

			var kCache, vCache []float32
			if tc.numBlocks > 0 {
				kCache = make([]float32, tc.numBlocks*tc.nKV*tc.hd*pagedPrefillBlockSize)
				vCache = make([]float32, tc.numBlocks*tc.nKV*tc.hd*pagedPrefillBlockSize)
				insertCount := tc.prefix
				if tc.freshNil {
					insertCount = totalKV
				}
				hostWritePagedKV(kAll[:insertCount*tc.nKV*tc.hd], vAll[:insertCount*tc.nKV*tc.hd],
					kCache, vCache, tc.blockTable, 0, insertCount, tc.nKV, tc.hd, pagedPrefillBlockSize)
			}

			var kFresh, vFresh []float32
			if !tc.freshNil {
				kFresh = kAll[tc.prefix*tc.nKV*tc.hd:]
				vFresh = vAll[tc.prefix*tc.nKV*tc.hd:]
			}

			got, err := hostPagedChunkedPrefillOracle(
				qData, kCache, vCache, tc.blockTable, kFresh, vFresh,
				tc.tokens, tc.prefix, tc.nH, tc.nKV, tc.hd, pagedPrefillBlockSize, scale,
			)
			if err != nil {
				t.Fatalf("hostPagedChunkedPrefillOracle: %v", err)
			}
			want := hostContiguousPromptPanelOracle(qData, kAll, vAll, tc.tokens, tc.prefix, tc.nH, tc.nKV, tc.hd, scale)

			if len(got) != len(want) {
				t.Fatalf("len mismatch got=%d want=%d", len(got), len(want))
			}
			for i := range want {
				gb := math.Float32bits(got[i])
				wb := math.Float32bits(want[i])
				if gb != wb {
					t.Fatalf("bitwise mismatch at [%d]: got=0x%08x (%g) want=0x%08x (%g)",
						i, gb, got[i], wb, want[i])
				}
			}
		})
	}

	t.Run("rejects_duplicate_and_short_block_tables", func(t *testing.T) {
		q := make([]float32, 4*4*64)
		kCache := make([]float32, 4*2*64*16)
		vCache := make([]float32, 4*2*64*16)
		kf := make([]float32, 4*2*64)
		vf := make([]float32, 4*2*64)
		if _, err := hostPagedChunkedPrefillOracle(q, kCache, vCache, []int{1, 1}, kf, vf, 4, 20, 4, 2, 64, 16, 0.125); err == nil {
			t.Fatal("expected error for duplicate physical blocks in blockTable")
		}
		if _, err := hostPagedChunkedPrefillOracle(q, kCache, vCache, []int{1}, kf, vf, 4, 20, 4, 2, 64, 16, 0.125); err == nil {
			t.Fatal("expected error for short blockTable covering < prefix")
		}
	})

	t.Run("cuda_source_contract", func(t *testing.T) {
		prefillSrc, err := os.ReadFile("prefill_paged.cu")
		if err != nil {
			t.Fatalf("read prefill_paged.cu: %v", err)
		}
		kernelsSrc, err := os.ReadFile("cuda_kernels.cu")
		if err != nil {
			t.Fatalf("read cuda_kernels.cu: %v", err)
		}
		src := string(prefillSrc)
		if !strings.Contains(string(kernelsSrc), `#include "prefill_paged.cu"`) {
			t.Fatal(`cuda_kernels.cu missing #include "prefill_paged.cu"`)
		}
		for _, needle := range []string{
			"k_paged_chunked_prefill_attention",
			"fcuda_paged_chunked_prefill_attention_f32",
			`FCUDA_TS("k_paged_chunked_prefill_attention"`,
		} {
			if !strings.Contains(src, needle) {
				t.Fatalf("prefill_paged.cu missing %q", needle)
			}
		}
		if strings.Contains(src, "float m = -INFINITY, l = 0.f;") {
			t.Fatal("prefill_paged.cu must not reuse combined finite-reduction sentinel line")
		}
	})
}
