package compute

import (
	"errors"
	"fmt"
	"math"
)

// V41CompressorRingBackend keeps one DeepSeek V4.1 compressor group resident
// on the device, so decode stops staging projected rows through the host. A
// ring is a pair of caller-owned F32 [ratio,width] tensors (kv, scores).
//
//   - V41CompressorRingAlloc returns a fresh ring whose rows are unspecified
//     until stored.
//   - V41CompressorRingStore records kvRow/scoreRow (F32, width elements) into
//     ring row slot. Without emit it returns a zero Tensor and does not wait.
//     With emit the same dispatch pools every ring row with exactly the
//     V41CompressorPool arithmetic into a fresh caller-owned F32 [2,width]
//     tensor whose checked Read completes the work: row 0 is the pooled row,
//     row 1 the per-dimension status DecodeV41CompressorRing reads.
//
// Capability absence permits the caller's host-staged path; a selected failure
// must propagate without host replay.
type V41CompressorRingBackend interface {
	SupportsV41CompressorRing() bool
	V41CompressorRingAlloc(ratio, width int) (kv, scores Tensor, err error)
	V41CompressorRingStore(kv, scores, kvRow, scoreRow Tensor, slot, ratio, width int, emit bool) (Tensor, error)
}

// V41CompressorRingProducerBackend owns the connected same-stream producers
// before enqueue; selected failures must not fall back to separate MatMuls.
type V41CompressorRingProducerBackend interface {
	V41CompressorRingProjectStore(kv, scores, wkv, wscore, x Tensor, slot, ratio, width int, emit bool) (Tensor, error)
}

var (
	// ErrV41CompressorRingGeometry marks a ring request refused before any
	// device work.
	ErrV41CompressorRingGeometry = errors.New("compute: invalid V4.1 compressor ring geometry")
	// ErrV41CompressorRingNonFinite marks a group whose kv or score rows hold a
	// non-finite value; the pooled row must not be published.
	ErrV41CompressorRingNonFinite = errors.New("compute: V4.1 compressor ring group is non-finite")
	// ErrV41CompressorRingStatus marks an emitted status word outside the closed
	// status set, which withholds the pooled row.
	ErrV41CompressorRingStatus = errors.New("compute: V4.1 compressor ring status is malformed")
)

// V41CompressorRingStatus bits, one word per latent dimension.
const (
	V41CompressorRingKVNonFinite    = 1
	V41CompressorRingScoreNonFinite = 2
)

// V41CompressorRingNonFiniteError attributes the first non-finite dimension.
type V41CompressorRingNonFiniteError struct {
	Dim    int
	Status int
}

func (e *V41CompressorRingNonFiniteError) Error() string {
	return fmt.Sprintf("compute: V4.1 compressor ring dimension %d status %d", e.Dim, e.Status)
}

func (e *V41CompressorRingNonFiniteError) Unwrap() error { return ErrV41CompressorRingNonFinite }

// ValidateV41CompressorRingAlloc checks a ring geometry. Ratio one has no gate
// and therefore no ring.
func ValidateV41CompressorRingAlloc(ratio, width int) error {
	// Divide only after positive bounded ratio validation. Multiplication here
	// can wrap even uint64 on 64-bit hosts; bytes must also fit the host int.
	if ratio < 2 || ratio > V41CompressorPoolMaxRatio || width <= 0 ||
		width > math.MaxInt32/ratio || width > math.MaxInt/(ratio*4) {
		return fmt.Errorf("%w: ratio=%d width=%d", ErrV41CompressorRingGeometry, ratio, width)
	}
	return nil
}

// ValidateV41CompressorRingStore checks one store request's host-visible shape.
func ValidateV41CompressorRingStore(kv, scores, kvRow, scoreRow Tensor, slot, ratio, width int) error {
	if err := ValidateV41CompressorRingAlloc(ratio, width); err != nil {
		return err
	}
	if slot < 0 || slot >= ratio {
		return fmt.Errorf("%w: slot %d outside ratio %d", ErrV41CompressorRingGeometry, slot, ratio)
	}
	for _, operand := range []struct {
		name string
		t    Tensor
		n    int
	}{{"kv ring", kv, ratio * width}, {"score ring", scores, ratio * width}, {"kv row", kvRow, width}, {"score row", scoreRow, width}} {
		if operand.t.Dtype != F32 || operand.t.Layout != RowMajor || operand.t.Quant != nil || operand.t.Numel() != operand.n {
			return fmt.Errorf("%w: %s must be an unquantized row-major F32 tensor of %d elements", ErrV41CompressorRingGeometry, operand.name, operand.n)
		}
	}
	if kv.Buf() != nil && kv.Buf() == scores.Buf() {
		return fmt.Errorf("%w: kv and score rings alias", ErrV41CompressorRingGeometry)
	}
	return nil
}

// DecodeV41CompressorRing splits an emitted [2,width] readback into the pooled
// row and refuses any non-zero status before the row can be published.
func DecodeV41CompressorRing(values []float32, width int) ([]float32, error) {
	if width <= 0 || len(values) != 2*width {
		return nil, fmt.Errorf("%w: emitted %d values for width %d", ErrV41CompressorRingStatus, len(values), width)
	}
	if err := v41CompressorRingStatusErr(values[width:]); err != nil {
		return nil, err
	}
	return append([]float32(nil), values[:width]...), nil
}

// v41CompressorRingStatusErr applies the closed status rule to a bare status
// row, one word per latent dimension.
func v41CompressorRingStatusErr(words []float32) error {
	for d, word := range words {
		switch word {
		case 0:
		case V41CompressorRingKVNonFinite, V41CompressorRingScoreNonFinite, V41CompressorRingKVNonFinite | V41CompressorRingScoreNonFinite:
			return &V41CompressorRingNonFiniteError{Dim: d, Status: int(word)}
		default:
			return fmt.Errorf("%w: dimension %d status bits %#08x", ErrV41CompressorRingStatus, d, math.Float32bits(word))
		}
	}
	return nil
}

func vulkanV41CompressorRingStatusError(status int, reason string) *BackendError {
	err := vulkanV41IndexerScoreStatusError(status, reason)
	err.Site = "V41CompressorRing"
	err.Message = fmt.Sprintf("dedicated compressor ring failed closed (code %d): %s", status, reason)
	return err
}
