// Ported from github.com/FasterDecoding/SnapKV (snapkv/monkeypatch/snapkv_utils.py@e216ddc84c5bd210378cbdbbba12ba02102aa640), Apache-2.0.
// SPDX-License-Identifier: Apache-2.0

package vcachecal

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// PoolingMode selects the 1D smoothing applied to per-head importance before top-k.
type PoolingMode int

const (
	// PoolAvg is 1-D average pooling with zero padding counted in the divisor (count_include_pad).
	PoolAvg PoolingMode = iota
	// PoolMax is 1-D max pooling; padding never wins the max.
	PoolMax
)

// PooledTopKConfig parameterizes PooledPerHeadTopK. Budget is the total kept length
// (prefix picks plus the observation window); Window is the trailing observation
// window that is always kept; KernelSize is the pooling width (stride 1, padding
// KernelSize/2).
type PooledTopKConfig struct {
	Window     int
	Budget     int
	KernelSize int
	Pooling    PoolingMode
}

// PooledPerHeadTopK selects, per head, which sequence positions survive compression.
// scores[h][i] is head h's importance of prefix position i (the attention mass the
// observation window puts on it); every head must cover the same prefix length L, and
// the full sequence is L+Window long with the window at positions L..L+Window-1.
//
// When the sequence is shorter than Budget nothing is compressed and every position is
// kept. Otherwise each head's scores are pooled and the Budget-Window highest positions
// are kept alongside the whole window. Kept indices are returned ascending per head;
// equal pooled scores prefer the lower position, and NaN ranks above any number.
func PooledPerHeadTopK(scores [][]float64, cfg PooledTopKConfig) ([][]int, error) {
	if cfg.Window < 1 {
		return nil, errors.New("vcachecal: pooled top-k window must be >= 1")
	}
	if cfg.Budget-cfg.Window <= 0 {
		return nil, errors.New("vcachecal: pooled top-k budget must exceed window")
	}
	// An even kernel with padding k/2 yields L+1 pooled values, one past the prefix.
	if cfg.KernelSize < 1 || cfg.KernelSize%2 == 0 {
		return nil, fmt.Errorf("vcachecal: pooled top-k kernel size %d must be odd and >= 1", cfg.KernelSize)
	}
	if cfg.Pooling != PoolAvg && cfg.Pooling != PoolMax {
		return nil, fmt.Errorf("vcachecal: unsupported pooling mode %d", cfg.Pooling)
	}
	if len(scores) == 0 {
		return nil, nil
	}
	prefix := len(scores[0])
	for h, s := range scores {
		if len(s) != prefix {
			return nil, fmt.Errorf("vcachecal: head %d has %d prefix scores, want %d", h, len(s), prefix)
		}
	}
	total := prefix + cfg.Window
	keep := cfg.Budget - cfg.Window
	out := make([][]int, len(scores))
	for h, s := range scores {
		if total < cfg.Budget {
			out[h] = seqRange(0, total)
			continue
		}
		pooled := smooth1d(s, cfg.KernelSize, cfg.Pooling)
		order := seqRange(0, prefix)
		sort.SliceStable(order, func(a, b int) bool {
			return rankAbove(pooled[order[a]], pooled[order[b]])
		})
		kept := append([]int(nil), order[:keep]...)
		sort.Ints(kept)
		out[h] = append(kept, seqRange(prefix, total)...)
	}
	return out, nil
}

func smooth1d(s []float64, k int, mode PoolingMode) []float64 {
	pad := k / 2
	out := make([]float64, len(s))
	for i := range s {
		sum, best, nan := 0.0, math.Inf(-1), false
		for j := i - pad; j <= i+pad; j++ {
			if j < 0 || j >= len(s) {
				continue
			}
			v := s[j]
			if math.IsNaN(v) {
				nan = true
			}
			sum += v
			if v > best {
				best = v
			}
		}
		switch {
		case nan:
			out[i] = math.NaN()
		case mode == PoolMax:
			out[i] = best
		default:
			out[i] = sum / float64(k)
		}
	}
	return out
}

// rankAbove orders like torch.topk, where NaN is the largest value.
func rankAbove(a, b float64) bool {
	if math.IsNaN(a) {
		return !math.IsNaN(b)
	}
	if math.IsNaN(b) {
		return false
	}
	return a > b
}

func seqRange(lo, hi int) []int {
	r := make([]int, 0, hi-lo)
	for i := lo; i < hi; i++ {
		r = append(r, i)
	}
	return r
}
