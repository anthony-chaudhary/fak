package compute

import (
	"encoding/binary"
	"fmt"
	"math"
)

// V41F32FiniteBackend validates raw F32 bits without rounding or changing input.
// Capability absence allows an ordinary path before selection. Once selected,
// errors must not cause host replay. The caller retains input until its checked
// execution owner completes; V41ReadDeferred consumes the returned record.
// This optional primitive does not select any model composition by itself.
type V41F32FiniteBackend interface {
	SupportsV41F32Finite(count int) bool
	V41F32FiniteDeferred(input Tensor, site uint32) (V41DeferredStatus, error)
}

const (
	V41F32FiniteRoutedSum    uint32 = 1
	V41F32FiniteFinalNormRaw uint32 = 2

	v41F32FiniteMagic       uint32 = 0x46333243
	v41F32FiniteStatusBytes        = 16
)

func v41F32FiniteSite(site uint32) bool {
	return site == V41F32FiniteRoutedSum || site == V41F32FiniteFinalNormRaw
}

// V41F32FiniteArithmeticError attributes the first nonfinite raw F32 element.
// A finite value which would overflow BF16 rounding is not an error here.
type V41F32FiniteArithmeticError struct {
	Site  uint32
	Index int
}

func (e *V41F32FiniteArithmeticError) Error() string {
	return fmt.Sprintf("compute: V4.1 raw F32 site %d has nonfinite element %d", e.Site, e.Index)
}

// V41F32FiniteProtocolError withholds malformed or uninitialized status.
type V41F32FiniteProtocolError struct{ Message string }

func (e *V41F32FiniteProtocolError) Error() string {
	return "compute: V4.1 raw F32 status protocol: " + e.Message
}

func validateV41F32Finite(input Tensor, site uint32) (int, error) {
	if !v41F32FiniteSite(site) || input.Dtype != F32 || input.Layout != RowMajor ||
		input.Quant != nil || len(input.Shape) != 1 || input.Shape[0] <= 0 ||
		input.Shape[0] > math.MaxInt32 || input.Shape[0] > math.MaxInt/4 {
		return 0, fmt.Errorf("raw F32 validation requires an unquantized row and known site")
	}
	return input.Shape[0] * 4, nil
}

func decodeV41F32FiniteStatus(raw []byte, count int, site uint32) error {
	bad := func(reason string) error { return &V41F32FiniteProtocolError{Message: reason} }
	if !v41F32FiniteSite(site) || count <= 0 || count > math.MaxInt32 ||
		count > math.MaxInt/4 || len(raw) != v41F32FiniteStatusBytes {
		return bad("invalid issued geometry or status size")
	}
	magic := binary.LittleEndian.Uint32(raw[0:4])
	observedSite := binary.LittleEndian.Uint32(raw[4:8])
	index := binary.LittleEndian.Uint32(raw[8:12])
	observedCount := binary.LittleEndian.Uint32(raw[12:16])
	if magic != v41F32FiniteMagic || observedSite != site || observedCount != uint32(count) {
		return bad("magic, site or count does not match the issued operation")
	}
	if index == math.MaxUint32 {
		return nil
	}
	if index >= uint32(count) {
		return bad("invalid element index exceeds issued count")
	}
	return &V41F32FiniteArithmeticError{Site: site, Index: int(index)}
}
