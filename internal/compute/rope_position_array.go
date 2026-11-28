// Clean-room implementation inspired by vllm-project/vllm@ae065918640a198a56f6f17eb45e251c3229fe41 (Apache-2.0):
// persistent [max_position, rot_dim] RoPE cos/sin table cache and batched position-id array rotation.
package compute

import (
	"errors"
	"fmt"
	"math"
)

// RopeCosSinCache stores a precomputed persistent [MaxPosition, RotDim] RoPE frequency
// table with layout per row pos: cos[0..RotDim/2-1] followed by sin[0..RotDim/2-1].
type RopeCosSinCache struct {
	MaxPosition int
	RotDim      int
	Theta       float64
	Table       []float32 // Flat [MaxPosition * RotDim] buffer.
}

// NewRopeCosSinCache builds the persistent [maxPosition, rotDim] cos/sin table using
// exact float64 trigonometric phase evaluation converted to float32 matching cpuref.ropeRow.
func NewRopeCosSinCache(maxPosition, rotDim int, theta float64) (*RopeCosSinCache, error) {
	if maxPosition <= 0 {
		return nil, fmt.Errorf("rope cache: maxPosition must be > 0, got %d", maxPosition)
	}
	if rotDim <= 0 || rotDim%2 != 0 {
		return nil, fmt.Errorf("rope cache: rotDim must be positive and even, got %d", rotDim)
	}
	if math.IsNaN(theta) || math.IsInf(theta, 0) || theta <= 0 {
		return nil, fmt.Errorf("rope cache: theta must be positive finite, got %g", theta)
	}
	half := rotDim / 2
	invFreq := make([]float64, half)
	for j := 0; j < half; j++ {
		invFreq[j] = 1.0 / math.Pow(theta, float64(2*j)/float64(rotDim))
	}

	table := make([]float32, maxPosition*rotDim)
	for pos := 0; pos < maxPosition; pos++ {
		row := table[pos*rotDim : (pos+1)*rotDim]
		for j := 0; j < half; j++ {
			a := float64(pos) * invFreq[j]
			row[j] = float32(math.Cos(a))
			row[half+j] = float32(math.Sin(a))
		}
	}

	return &RopeCosSinCache{
		MaxPosition: maxPosition,
		RotDim:      rotDim,
		Theta:       theta,
		Table:       table,
	}, nil
}

// Row returns the cached (cos, sin) views of length RotDim/2 for position pos.
func (c *RopeCosSinCache) Row(pos int) (cos, sin []float32, err error) {
	if c == nil {
		return nil, nil, errors.New("rope cache: nil cache")
	}
	if pos < 0 || pos >= c.MaxPosition {
		return nil, nil, fmt.Errorf("rope cache: position %d out of range [0, %d)", pos, c.MaxPosition)
	}
	half := c.RotDim / 2
	row := c.Table[pos*c.RotDim : (pos+1)*c.RotDim]
	return row[:half], row[half:], nil
}

// RopePositionConfig configures batched position-array rotary application over [tokens, nHeads, headDim].
type RopePositionConfig struct {
	Tokens        int
	NumHeads      int
	HeadDim       int
	RopeDimOffset int  // Starting coordinate within each head receiving RoPE (supports partial RoPE).
	IsNeox        bool // true: split-half (j, j+half) matching cpuref; false: interleaved (2j, 2j+1).
	Inverse       bool // true: apply inverse rotation R(-theta).
}

// ApplyRopePositionArray returns a newly allocated tensor with position-id RoPE applied across
// all tokens and heads according to cfg without mutating x or the cache.
func ApplyRopePositionArray(x []float32, positions []int32, cache *RopeCosSinCache, cfg RopePositionConfig) ([]float32, error) {
	out := append([]float32(nil), x...)
	if err := ApplyRopePositionArrayInPlace(out, positions, cache, cfg); err != nil {
		return nil, err
	}
	return out, nil
}

// ApplyRopePositionArrayInPlace applies rotary embeddings in-place using positions[tokens]
// indexed into cache. Matches cpuref.applyRope float32 rounding order bitwise for forward NeoX.
func ApplyRopePositionArrayInPlace(x []float32, positions []int32, cache *RopeCosSinCache, cfg RopePositionConfig) error {
	if cache == nil {
		return errors.New("rope position array: nil cache")
	}
	if cfg.Tokens <= 0 || cfg.NumHeads <= 0 || cfg.HeadDim <= 0 {
		return fmt.Errorf("rope position array: invalid dims tokens=%d heads=%d headDim=%d",
			cfg.Tokens, cfg.NumHeads, cfg.HeadDim)
	}
	if cfg.RopeDimOffset < 0 || cfg.RopeDimOffset+cache.RotDim > cfg.HeadDim {
		return fmt.Errorf("rope position array: slice [%d, %d) exceeds headDim=%d",
			cfg.RopeDimOffset, cfg.RopeDimOffset+cache.RotDim, cfg.HeadDim)
	}
	if len(positions) != cfg.Tokens {
		return fmt.Errorf("rope position array: positions len %d != tokens %d", len(positions), cfg.Tokens)
	}
	wantElems := cfg.Tokens * cfg.NumHeads * cfg.HeadDim
	if len(x) != wantElems {
		return fmt.Errorf("rope position array: x len %d != %d", len(x), wantElems)
	}

	half := cache.RotDim / 2
	for tok := 0; tok < cfg.Tokens; tok++ {
		cos, sin, err := cache.Row(int(positions[tok]))
		if err != nil {
			return fmt.Errorf("rope position array token %d: %w", tok, err)
		}
		for h := 0; h < cfg.NumHeads; h++ {
			base := (tok*cfg.NumHeads+h)*cfg.HeadDim + cfg.RopeDimOffset
			hv := x[base : base+cache.RotDim]
			for j := 0; j < half; j++ {
				c := cos[j]
				s := sin[j]
				if cfg.Inverse {
					s = -s
				}
				if cfg.IsNeox {
					a, b := hv[j], hv[j+half]
					hv[j] = float32(a*c) - float32(b*s)
					hv[j+half] = float32(b*c) + float32(a*s)
				} else {
					a, b := hv[2*j], hv[2*j+1]
					hv[2*j] = float32(a*c) - float32(b*s)
					hv[2*j+1] = float32(b*c) + float32(a*s)
				}
			}
		}
	}
	return nil
}
