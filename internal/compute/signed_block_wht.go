package compute

import "fmt"

// PrismWHTBlock is the block size of the Prism Bonsai-2 signed block Walsh-Hadamard
// rotation. The rotation is block-diagonal over contiguous blocks of this size and
// scales each block by 1/sqrt(PrismWHTBlock) (1/32 here).
const PrismWHTBlock = 1024

// ErrSignedBlockWHT documents an invalid signed block Walsh-Hadamard configuration, or
// an operand whose length is not one of the registered widths. Callers may match it with
// errors.Is.
type ErrSignedBlockWHT struct {
	Message string
	Field   string
	Value   int
}

func (e ErrSignedBlockWHT) Error() string {
	if e.Message != "" {
		return "compute: " + e.Message
	}
	return fmt.Sprintf("compute: invalid signed block WHT %s=%d", e.Field, e.Value)
}

func (e ErrSignedBlockWHT) Is(target error) bool {
	switch target.(type) {
	case ErrSignedBlockWHT, *ErrSignedBlockWHT:
		return true
	default:
		return false
	}
}

// SignedBlockWHT is a width-keyed signed block Walsh-Hadamard rotation. Each registered
// width owns a ±1 sign vector of exactly that length; Forward and Inverse select it by
// the operand length. The transform is block-diagonal with one shared block size, so it
// reuses BlockDiagonalWHT's butterfly and scaling — there is no second butterfly.
//
// Forward applies the sign BEFORE the transform and Inverse applies it AFTER, matching
// the Prism contract: forward y = WHT_block(x ⊙ s), inverse x = s ⊙ WHT_block(y). This is
// deliberately NOT the unsigned involutory aliases (BlockDiagonalWHTInverse /
// HadamardInverse): once explicit signs are present, fwd(fwd(x)) != x and the unsigned
// inverse does not undo the signed forward.
type SignedBlockWHT struct {
	block        int
	signsByWidth map[int][]float32
}

// NewSignedBlockWHT builds the width-keyed rotation from a parsed (widths, values) sign
// pair. The Nth width consumes the next width entries of values, as upstream does: the
// sign vector does not carry its own width, so the widths partition the flat value slice
// in order. It refuses a non-power-of-two block, empty/non-positive/duplicate widths, a
// width that is not a multiple of the block, a sign value other than ±1, and a value
// count that does not sum to the declared widths.
func NewSignedBlockWHT(block int, widths, values []int) (*SignedBlockWHT, error) {
	if block <= 0 || !isPowerOfTwo(block) {
		return nil, ErrSignedBlockWHT{
			Message: fmt.Sprintf("block size must be a positive power of two, got %d", block),
			Field:   "block",
			Value:   block,
		}
	}
	if len(widths) == 0 {
		return nil, ErrSignedBlockWHT{Message: "at least one sign width is required", Field: "widths"}
	}
	signs := make(map[int][]float32, len(widths))
	seen := make(map[int]struct{}, len(widths))
	offset := 0
	for idx, width := range widths {
		if width <= 0 {
			return nil, ErrSignedBlockWHT{
				Message: fmt.Sprintf("width %d must be positive, got %d", idx, width),
				Field:   "width",
				Value:   width,
			}
		}
		if _, dup := seen[width]; dup {
			return nil, ErrSignedBlockWHT{
				Message: fmt.Sprintf("duplicate width %d is not allowed; signs are keyed by width", width),
				Field:   "width",
				Value:   width,
			}
		}
		seen[width] = struct{}{}
		if width%block != 0 {
			return nil, ErrSignedBlockWHT{
				Message: fmt.Sprintf("width %d must be a multiple of block %d", width, block),
				Field:   "width",
				Value:   width,
			}
		}
		// Bound the remaining values before addition or allocation. A valid width
		// may still overflow offset+width when the declared widths are malformed.
		if offset < 0 || offset > len(values) || width > len(values)-offset {
			return nil, ErrSignedBlockWHT{
				Message: fmt.Sprintf("sign values are shorter than the declared widths: width %d at offset %d exceeds %d values", width, offset, len(values)),
				Field:   "sign",
				Value:   len(values),
			}
		}
		vec := make([]float32, width)
		for i := 0; i < width; i++ {
			v := values[offset+i]
			if v != 1 && v != -1 {
				return nil, ErrSignedBlockWHT{
					Message: fmt.Sprintf("sign value %d at offset %d must be +1 or -1", v, offset+i),
					Field:   "sign",
					Value:   v,
				}
			}
			vec[i] = float32(v)
		}
		signs[width] = vec
		offset += width
	}
	if offset != len(values) {
		return nil, ErrSignedBlockWHT{
			Message: fmt.Sprintf("sum of widths %d does not match %d sign values", offset, len(values)),
			Field:   "sign",
			Value:   len(values),
		}
	}
	return &SignedBlockWHT{block: block, signsByWidth: signs}, nil
}

// Block returns the rotation's block size.
func (w *SignedBlockWHT) Block() int {
	if w == nil {
		return 0
	}
	return w.block
}

// signFor returns the sign vector registered for vec's length, or a typed refusal when
// that length is not one of the widths the wrapper was built from.
func (w *SignedBlockWHT) signFor(vec []float32) ([]float32, error) {
	if w == nil {
		return nil, ErrSignedBlockWHT{Message: "nil signed block WHT"}
	}
	s, ok := w.signsByWidth[len(vec)]
	if !ok {
		return nil, ErrSignedBlockWHT{
			Message: fmt.Sprintf("operand length %d is not a registered width", len(vec)),
			Field:   "length",
			Value:   len(vec),
		}
	}
	return s, nil
}

// Forward rotates vec in place as y = WHT_block(vec ⊙ s): the sign is applied before the
// transform. It returns a typed error when len(vec) is not a registered width or the
// block transform refuses the operand.
func (w *SignedBlockWHT) Forward(vec []float32) error {
	s, err := w.signFor(vec)
	if err != nil {
		return err
	}
	for i := range vec {
		vec[i] *= s[i]
	}
	return BlockDiagonalWHT(vec, w.block)
}

// Inverse un-rotates vec in place as x = s ⊙ WHT_block(vec): the transform is applied
// before the sign. This is the signed inverse of Forward for the same width.
func (w *SignedBlockWHT) Inverse(vec []float32) error {
	s, err := w.signFor(vec)
	if err != nil {
		return err
	}
	if err := BlockDiagonalWHT(vec, w.block); err != nil {
		return err
	}
	for i := range vec {
		vec[i] *= s[i]
	}
	return nil
}
