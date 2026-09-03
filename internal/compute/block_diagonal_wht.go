package compute

import (
	"fmt"
	"math"
)

// ErrBlockDiagonalWHT documents invalid dimensions or non-power-of-two block sizes
// for block-diagonal Walsh-Hadamard Transform operations.
type ErrBlockDiagonalWHT struct {
	Message   string
	BlockSize int
	Length    int
}

func (e ErrBlockDiagonalWHT) Error() string {
	if e.Message != "" {
		return "compute: " + e.Message
	}
	if e.BlockSize != 0 || e.Length != 0 {
		return fmt.Sprintf("compute: invalid dimensions or non-power-of-two block size: blockSize=%d length=%d", e.BlockSize, e.Length)
	}
	return "compute: invalid dimensions or non-power-of-two block size"
}

func (e ErrBlockDiagonalWHT) Is(target error) bool {
	switch target.(type) {
	case ErrBlockDiagonalWHT, *ErrBlockDiagonalWHT:
		return true
	default:
		return false
	}
}

// isPowerOfTwo reports whether n is a positive power of two (1, 2, 4, 8, ...).
func isPowerOfTwo(n int) bool {
	return n > 0 && n&(n-1) == 0
}

// walshHadamardButterfly applies the fast Walsh-Hadamard butterfly in-place
// in natural order over a slice whose length is a power of two.
func walshHadamardButterfly(v []float32) {
	n := len(v)
	for span := 1; span < n; span *= 2 {
		for base := 0; base < n; base += span * 2 {
			for j := base; j < base+span; j++ {
				a := v[j]
				b := v[j+span]
				v[j] = a + b
				v[j+span] = a - b
			}
		}
	}
}

// BlockDiagonalWHT applies the block-diagonal Walsh-Hadamard Transform in-place.
// It partitions vec into contiguous chunks of blockSize, applies the fast
// butterfly to each chunk, and scales each chunk by 1/sqrt(blockSize) to preserve
// isometry/orthogonality and ensure involution (H^2 = I).
//
// In partial-rotary architectures (e.g. Qwen, MiniMax), applying a full WHT across
// the entire head dimension mixes the RoPE-rotated prefix with the unrotated suffix.
// Using a block-diagonal transform aligned to the rotary dimension preserves prefix
// isolation while smoothing outliers within each block.
func BlockDiagonalWHT(vec []float32, blockSize int) error {
	if blockSize <= 0 {
		return ErrBlockDiagonalWHT{
			Message:   fmt.Sprintf("block size must be positive, got %d", blockSize),
			BlockSize: blockSize,
			Length:    len(vec),
		}
	}
	if !isPowerOfTwo(blockSize) {
		return ErrBlockDiagonalWHT{
			Message:   fmt.Sprintf("block size must be a power of two, got %d", blockSize),
			BlockSize: blockSize,
			Length:    len(vec),
		}
	}
	if len(vec)%blockSize != 0 {
		return ErrBlockDiagonalWHT{
			Message:   fmt.Sprintf("vector length %d must be divisible by block size %d", len(vec), blockSize),
			BlockSize: blockSize,
			Length:    len(vec),
		}
	}
	if len(vec) == 0 {
		return nil
	}

	scale := float32(1.0 / math.Sqrt(float64(blockSize)))
	for base := 0; base < len(vec); base += blockSize {
		chunk := vec[base : base+blockSize]
		walshHadamardButterfly(chunk)
		for j := range chunk {
			chunk[j] *= scale
		}
	}
	return nil
}

// BlockDiagonalWHTRotate is the forward block-diagonal Walsh-Hadamard transform alias.
func BlockDiagonalWHTRotate(vec []float32, blockSize int) error {
	return BlockDiagonalWHT(vec, blockSize)
}

// BlockDiagonalWHTInverse is the inverse block-diagonal Walsh-Hadamard transform alias.
// Because the normalized Walsh-Hadamard transform is involutory (H^2 = I), the inverse
// is identical to the forward transform.
func BlockDiagonalWHTInverse(vec []float32, blockSize int) error {
	return BlockDiagonalWHT(vec, blockSize)
}

// BlockDiagonalWHTBatch applies the block-diagonal Walsh-Hadamard transform to a
// contiguous batch of vectors of length headDim.
func BlockDiagonalWHTBatch(data []float32, headDim, blockSize int) error {
	if headDim <= 0 {
		return ErrBlockDiagonalWHT{
			Message:   fmt.Sprintf("head dimension must be positive, got %d", headDim),
			BlockSize: blockSize,
			Length:    len(data),
		}
	}
	if len(data)%headDim != 0 {
		return ErrBlockDiagonalWHT{
			Message:   fmt.Sprintf("data length %d must be divisible by headDim %d", len(data), headDim),
			BlockSize: blockSize,
			Length:    len(data),
		}
	}
	if blockSize <= 0 {
		return ErrBlockDiagonalWHT{
			Message:   fmt.Sprintf("block size must be positive, got %d", blockSize),
			BlockSize: blockSize,
			Length:    len(data),
		}
	}
	if !isPowerOfTwo(blockSize) {
		return ErrBlockDiagonalWHT{
			Message:   fmt.Sprintf("block size must be a power of two, got %d", blockSize),
			BlockSize: blockSize,
			Length:    len(data),
		}
	}
	if headDim%blockSize != 0 {
		return ErrBlockDiagonalWHT{
			Message:   fmt.Sprintf("head dimension %d must be divisible by block size %d", headDim, blockSize),
			BlockSize: blockSize,
			Length:    len(data),
		}
	}
	if len(data) == 0 {
		return nil
	}

	for base := 0; base < len(data); base += headDim {
		vec := data[base : base+headDim]
		if err := BlockDiagonalWHT(vec, blockSize); err != nil {
			return err
		}
	}
	return nil
}

// DeriveRotaryBlockSize determines the block size for partial-rotary architectures.
// If rotaryDim <= 0 or rotaryDim >= headDim, returns headDim (if power of 2).
// Otherwise checks rotaryDim is power of 2 and divides headDim, returning rotaryDim.
func DeriveRotaryBlockSize(headDim, rotaryDim int) (int, error) {
	if rotaryDim <= 0 || rotaryDim >= headDim {
		if !isPowerOfTwo(headDim) {
			return 0, ErrBlockDiagonalWHT{
				Message:   fmt.Sprintf("head dimension %d is not a power of two", headDim),
				BlockSize: headDim,
				Length:    headDim,
			}
		}
		return headDim, nil
	}

	if !isPowerOfTwo(rotaryDim) {
		return 0, ErrBlockDiagonalWHT{
			Message:   fmt.Sprintf("rotary dimension %d is not a power of two", rotaryDim),
			BlockSize: rotaryDim,
			Length:    headDim,
		}
	}
	if headDim%rotaryDim != 0 {
		return 0, ErrBlockDiagonalWHT{
			Message:   fmt.Sprintf("head dimension %d must be divisible by rotary dimension %d", headDim, rotaryDim),
			BlockSize: rotaryDim,
			Length:    headDim,
		}
	}
	return rotaryDim, nil
}
