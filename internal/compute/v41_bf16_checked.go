package compute

import (
	"encoding/binary"
	"fmt"
	"math"
)

// V41BF16CheckedBackend is a finite-only publication alongside the unchanged
// raw, all-bit V41BF16BoundaryBackend operation. It returns a fresh, disjoint
// F32 [N] tensor only after device validation of every original and rounded
// word. Rounding remains unsigned wrapping RNE followed by widening to F32.
// Either nonfinite input or overflow to BF16 infinity is an arithmetic error.
//
// The two site IDs identify real final-norm publications. This synchronous
// prerequisite does not support an active batch or establish resident chaining.
// Capability absence permits the existing path before selection; a selected
// failure never permits host replay. Ordinary Vulkan does not implement it.
type V41BF16CheckedBackend interface {
	V41BF16CheckedAvailable() bool
	V41BF16Checked(input Tensor, site uint32) (Tensor, error)
}

const (
	V41BF16CheckedFinalNormInput  uint32 = 1
	V41BF16CheckedFinalNormOutput uint32 = 2

	v41BF16CheckedStatusMagic uint32 = 0x42463143
	v41BF16CheckedStatusBytes        = 16
	v41BF16CheckedPushBytes          = 8
)

// V41BF16CheckedArithmeticError identifies the lowest invalid element within
// an issued publication site. No offending-value payload is inferred from the
// status record; output bytes are unspecified and withheld on every failure.
type V41BF16CheckedArithmeticError struct {
	Site  uint32
	Index int
}

func (e *V41BF16CheckedArithmeticError) Error() string {
	return fmt.Sprintf("compute: checked V4.1 BF16 %s has nonfinite input or rounded overflow at element %d", v41BF16CheckedSiteName(e.Site), e.Index)
}

// V41BF16CheckedProtocolError is malformed status, distinct from an attributed
// arithmetic fault or a transport failure that prevents status consumption.
type V41BF16CheckedProtocolError struct{ Message string }

func (e *V41BF16CheckedProtocolError) Error() string {
	return "compute: checked V4.1 BF16 status protocol: " + e.Message
}

func v41BF16CheckedSiteName(site uint32) string {
	switch site {
	case V41BF16CheckedFinalNormInput:
		return "final-norm input"
	case V41BF16CheckedFinalNormOutput:
		return "final-norm output"
	default:
		return fmt.Sprintf("unknown site %d", site)
	}
}

func validateV41BF16Checked(input Tensor, site uint32) (int, error) {
	if site != V41BF16CheckedFinalNormInput && site != V41BF16CheckedFinalNormOutput {
		return 0, fmt.Errorf("compute: checked V4.1 BF16 requires a known final-norm site")
	}
	// Keep the raw metadata/byte bounds, including N <= INT32_MAX. No device
	// contents are read here: the checked shader independently validates bits.
	return validateV41BF16Boundary(input)
}

// decodeV41BF16CheckedStatus consumes the exact little-endian 16-byte record:
// magic, site, lowest invalid zero-based index (UINT32_MAX success), count.
// Freshness is supplied by checked all-ones initialization and ordering on
// EVERY dispatch; a decoder alone cannot detect an identical stale success.
func decodeV41BF16CheckedStatus(raw []byte, count int, site uint32) error {
	bad := func(message string) error { return &V41BF16CheckedProtocolError{Message: message} }
	if count <= 0 || count > math.MaxInt32 || count > math.MaxInt/4 ||
		(site != V41BF16CheckedFinalNormInput && site != V41BF16CheckedFinalNormOutput) || len(raw) != v41BF16CheckedStatusBytes {
		return bad("invalid issued geometry or status size")
	}
	magic := binary.LittleEndian.Uint32(raw[0:4])
	observedSite := binary.LittleEndian.Uint32(raw[4:8])
	index := binary.LittleEndian.Uint32(raw[8:12])
	observedCount := binary.LittleEndian.Uint32(raw[12:16])
	if magic != v41BF16CheckedStatusMagic || observedSite != site || observedCount != uint32(count) {
		return bad("magic, site or count does not match the issued operation")
	}
	if index == math.MaxUint32 {
		return nil
	}
	if index >= uint32(count) {
		return bad("invalid element index exceeds issued count")
	}
	return &V41BF16CheckedArithmeticError{Site: site, Index: int(index)}
}
