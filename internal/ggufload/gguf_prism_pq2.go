package ggufload

// PQ2_0 uses PrismML's group-128 ternary block from ggml-common.h at
// adfffbe41b2cabcd51fff326ab045662265062bb. The rows below are permuted as
// complete blocks, so no codes or f16 scales are changed. The permutations are
// the packed equivalents of normalizeCanonicalTensorData.
//
// PrismML-Eng/llama.cpp original notice:
//
// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 PrismML
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import (
	"fmt"
	"math"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// prismPQ2ResidentEligible recognizes the Bonsai matrices whose GGUF-to-model
// normalization is either identity or a lossless whole-row permutation. The
// grouped GDN V flag is necessary for ssm_out: its columns are already in the
// order produced by fak's GDN core.
func prismPQ2ResidentEligible(cfg model.Config, canon string, grouped bool) bool {
	if !cfg.IsQwen35Hybrid() || cfg.IsMoE() {
		return false
	}
	resolved, keep := model.QuantSourceTensorName(cfg, canon)
	if !keep || !model.IsQuantWeight(resolved) {
		return false
	}
	if strings.HasSuffix(resolved, ".linear_attn.out_proj.weight") {
		return grouped
	}
	if strings.HasSuffix(resolved, ".linear_attn.in_proj_qkv.weight") ||
		strings.HasSuffix(resolved, ".linear_attn.in_proj_z.weight") ||
		strings.HasSuffix(resolved, ".self_attn.q_proj.weight") ||
		strings.HasSuffix(resolved, ".self_attn.k_proj.weight") {
		return true
	}
	return model.ResidentKQuantEligible(cfg, canon)
}

// normalizePrismPQ2Rows applies the existing Qwen35 row-layout transforms to
// raw PQ2 bytes. A raw row is independently quantized, so moving entire rows
// gives the same decoded tensor as normalizing F32, without requantization.
func normalizePrismPQ2Rows(canon string, shape []int, raw []byte, cfg model.Config) ([]byte, error) {
	resolved, keep := model.QuantSourceTensorName(cfg, canon)
	if !keep {
		return nil, fmt.Errorf("gguf: PQ2_0 tensor %s was skipped by model source mapping", canon)
	}
	if len(shape) != 2 || shape[0] <= 0 || shape[1] <= 0 || shape[1]%128 != 0 {
		return nil, fmt.Errorf("gguf: PQ2_0 tensor %s cannot reorder shape %v", canon, shape)
	}
	rowBytes := shape[1] / 128 * blockPQ2_0Bytes
	if rowBytes <= 0 || shape[0] > math.MaxInt/rowBytes || len(raw) != shape[0]*rowBytes {
		return nil, fmt.Errorf("gguf: PQ2_0 tensor %s has %d bytes for shape %v", canon, len(raw), shape)
	}
	var sourceRow func(dst int) int
	switch {
	case strings.HasSuffix(resolved, ".linear_attn.in_proj_qkv.weight"):
		nK, nV, span := cfg.LinearNumKeyHeads, cfg.LinearNumValueHeads, cfg.LinearValueHeadDim
		keyDim := nK * cfg.LinearKeyHeadDim
		if nK <= 0 || nV <= 0 || nV%nK != 0 || span <= 0 || cfg.LinearKeyHeadDim != span || shape[0] != 2*keyDim+nV*span {
			return nil, fmt.Errorf("gguf: PQ2_0 tensor %s has incompatible GDN QKV shape %v", canon, shape)
		}
		ratio, vOff := nV/nK, 2*keyDim
		sourceRow = func(dst int) int {
			if dst < vOff {
				return dst
			}
			v := dst - vOff
			head, within := v/span, v%span
			return vOff + ((head%ratio)*nK+head/ratio)*span + within
		}
	case strings.HasSuffix(resolved, ".linear_attn.in_proj_z.weight"):
		nK, nV, span := cfg.LinearNumKeyHeads, cfg.LinearNumValueHeads, cfg.LinearValueHeadDim
		if nK <= 0 || nV <= 0 || nV%nK != 0 || span <= 0 || shape[0] != nV*span {
			return nil, fmt.Errorf("gguf: PQ2_0 tensor %s has incompatible GDN gate shape %v", canon, shape)
		}
		ratio := nV / nK
		sourceRow = func(dst int) int {
			head, within := dst/span, dst%span
			return ((head%ratio)*nK+head/ratio)*span + within
		}
	case strings.HasSuffix(resolved, ".self_attn.q_proj.weight"):
		hd, heads := cfg.HeadDim, cfg.NumHeads
		if hd <= 0 || hd%2 != 0 || heads <= 0 || shape[0] != heads*2*hd {
			return nil, fmt.Errorf("gguf: PQ2_0 tensor %s has incompatible gated Q shape %v", canon, shape)
		}
		sourceRow = func(dst int) int {
			head, within := dst/(2*hd), dst%(2*hd)
			if within >= hd { // the gate half is not rotary permuted
				return dst
			}
			return head*2*hd + (within%(hd/2))*2 + within/(hd/2)
		}
	case strings.HasSuffix(resolved, ".self_attn.k_proj.weight"):
		hd, heads := cfg.HeadDim, cfg.NumKVHeads
		if hd <= 0 || hd%2 != 0 || heads <= 0 || shape[0] != heads*hd {
			return nil, fmt.Errorf("gguf: PQ2_0 tensor %s has incompatible K shape %v", canon, shape)
		}
		sourceRow = func(dst int) int {
			head, within := dst/hd, dst%hd
			return head*hd + (within%(hd/2))*2 + within/(hd/2)
		}
	default:
		return raw, nil
	}
	dst := make([]byte, len(raw))
	for row := 0; row < shape[0]; row++ {
		src := sourceRow(row)
		copy(dst[row*rowBytes:(row+1)*rowBytes], raw[src*rowBytes:(src+1)*rowBytes])
	}
	return dst, nil
}
