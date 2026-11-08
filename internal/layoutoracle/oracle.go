// Package layoutoracle is an independent geometry/stride oracle for tensor
// descriptors. Given a declared descriptor (what the admission plan expects) and
// an actual descriptor (what the artifact's real header carries), it computes
// byte geometry from first principles and refuses mismatches before dispatch.
//
// The oracle exists because total-byte agreement does not imply layout
// agreement. Layout-mismatch incident ledger row LML-02 (model-bringup lesson
// MBL-027) witnessed published native-safetensors inventory shapes applied
// unchanged to converted GGUF: mHC [4H,24] flattened vs grouped and grouped
// attn_output_a carry the SAME element count but different axis partitionings,
// so a naive bytes = prod(shape)*dtype comparison accepts those pairs while the
// forward consumes a wrong sub-matrix. This package refuses them. Per the
// TensorLayoutPlan.ParityCommand doctrine, parity evidence must come from an
// independent oracle; a candidate-produced golden is not an independent oracle.
//
// Closed policy (fail-closed, documented here rather than inline):
//   - Negative strides are refused with ErrInvalidDescriptor. With non-negative
//     strides the minimum offset over the strided box is always 0, so the span
//     is maxOffset+1 where maxOffset sums (Shape[i]-1)*Strides[i].
//   - Zero-size tensors (any dim == 0) are defined and occupy 0 bytes.
//   - Rank-0 descriptors (no dims) are one element: ExpectedBytes returns
//     DTypeSize.
//   - Huge dims that would overflow int64 byte arithmetic error with
//     ErrOverflow instead of wrapping.
//   - Groups partitions the LEADING axis: Groups=1 is flat, Groups=G>1 requires
//     rank >= 1 and Shape[0] % G == 0. Two descriptors with identical
//     shape/strides/dtype but different Groups describe different logical
//     layouts (group-major storage vs flat storage) and refuse even though their
//     total bytes match - that is the LML-02 defect shape.
//   - Nil Strides means row-major contiguous; explicit strides may be any
//     non-negative values, so noncontiguous views are first-class inputs.
//
// Callers pin behavior with errors.Is against the exported sentinels
// (ErrGeometryMismatch, ErrGroupingMismatch, ErrInvalidDescriptor,
// ErrOverflow), never against rendered text.
package layoutoracle

import (
	"errors"
	"fmt"
	"math"
)

// Sentinel refusals. errors.Is-pinnable; concrete types carry structured
// fields naming the disagreeing dimension/stride or rejected field.
var (
	// ErrGeometryMismatch pins a structural disagreement between two valid
	// descriptors: rank, per-axis shape, per-axis element stride, dtype size,
	// or computed byte span.
	ErrGeometryMismatch = errors.New("layoutoracle: geometry mismatch")
	// ErrGroupingMismatch pins a leading-axis partitioning disagreement: the
	// descriptors agree on every byte-level property but declare different
	// group counts (flat vs grouped storage of the same element set).
	ErrGroupingMismatch = errors.New("layoutoracle: grouping mismatch")
	// ErrInvalidDescriptor pins a descriptor that violates the closed policy
	// (negative dim/stride, non-positive dtype size, inconsistent group
	// partitioning, stride/rank length mismatch).
	ErrInvalidDescriptor = errors.New("layoutoracle: invalid descriptor")
	// ErrOverflow pins byte-geometry arithmetic that exceeds int64.
	ErrOverflow = errors.New("layoutoracle: byte geometry overflows int64")
)

// Descriptor is one side of a geometry check: the declared expectation or the
// artifact's actual header geometry.
type Descriptor struct {
	// Shape is the outermost-first logical extent. nil/empty is a rank-0
	// scalar (one element). Every entry must be >= 0.
	Shape []int
	// Strides are ELEMENT strides (not bytes). nil means row-major
	// contiguous; otherwise len(Strides) must equal len(Shape) and every entry
	// must be >= 0. A zero stride marks a padded/broadcast axis that does not
	// extend the span.
	Strides []int
	// DTypeSize is the bytes-per-element, >= 1.
	DTypeSize int
	// Groups is the leading-axis partition: 1 = flat, G > 1 = G blocks of
	// Shape[0]/G along axis 0. Must be >= 1; when > 1 it must divide Shape[0].
	Groups int
}

// Mismatch is the typed refusal returned by CheckGeometry for two VALID
// descriptors that disagree. Kind is one of "rank", "shape", "stride",
// "dtype", "span", "grouping"; Dim is the axis index, -1 for whole-tensor
// kinds. Error() renders these fields, but callers must pin behavior with
// errors.Is against the sentinel Unwrap returns.
type Mismatch struct {
	Kind     string
	Dim      int
	Declared int64
	Actual   int64
}

func (m *Mismatch) Error() string {
	where := "whole tensor"
	if m.Dim >= 0 {
		where = fmt.Sprintf("dim %d", m.Dim)
	}
	return fmt.Sprintf("%s (%s): declared=%d actual=%d",
		m.sentinel(), where, m.Declared, m.Actual)
}

func (m *Mismatch) sentinel() error {
	if m.Kind == "grouping" {
		return ErrGroupingMismatch
	}
	return ErrGeometryMismatch
}

// Unwrap exposes the pinnable sentinel.
func (m *Mismatch) Unwrap() error { return m.sentinel() }

// InvalidDescriptor is the typed refusal for a descriptor that violates the
// closed policy. Which is "declared" or "actual"; Field is one of "shape",
// "strides", "dtype_size", "groups"; Index is the offending axis (-1 for
// whole-field); Value is the offending numeric value.
type InvalidDescriptor struct {
	Which string
	Field string
	Index int
	Value int64
}

func (e *InvalidDescriptor) Error() string {
	where := e.Field
	if e.Index >= 0 {
		where = fmt.Sprintf("%s[%d]", e.Field, e.Index)
	}
	return fmt.Sprintf("%s: %s %s has value %d", ErrInvalidDescriptor, e.Which, where, e.Value)
}

// Unwrap exposes the pinnable sentinel.
func (e *InvalidDescriptor) Unwrap() error { return ErrInvalidDescriptor }

// validate enforces the closed policy on one descriptor. which labels the side
// in refusals ("declared" or "actual").
func validate(d Descriptor, which string) error {
	if d.DTypeSize < 1 {
		return &InvalidDescriptor{Which: which, Field: "dtype_size", Index: -1, Value: int64(d.DTypeSize)}
	}
	if len(d.Strides) != 0 && len(d.Strides) != len(d.Shape) {
		return &InvalidDescriptor{Which: which, Field: "strides", Index: -1, Value: int64(len(d.Strides))}
	}
	for i, s := range d.Shape {
		if s < 0 {
			return &InvalidDescriptor{Which: which, Field: "shape", Index: i, Value: int64(s)}
		}
	}
	for i, s := range d.Strides {
		if s < 0 {
			return &InvalidDescriptor{Which: which, Field: "strides", Index: i, Value: int64(s)}
		}
	}
	if d.Groups < 1 {
		return &InvalidDescriptor{Which: which, Field: "groups", Index: -1, Value: int64(d.Groups)}
	}
	if d.Groups > 1 && (len(d.Shape) == 0 || d.Shape[0]%d.Groups != 0) {
		return &InvalidDescriptor{Which: which, Field: "groups", Index: 0, Value: int64(d.Groups)}
	}
	return nil
}

// effectiveStrides returns the element strides, deriving row-major contiguous
// strides when d.Strides is nil. Overflow-checked; zero-extent suffixes yield
// zero strides without spurious overflow.
func effectiveStrides(d Descriptor) ([]int, error) {
	if d.Strides != nil {
		out := make([]int, len(d.Strides))
		copy(out, d.Strides)
		return out, nil
	}
	n := len(d.Shape)
	s := make([]int, n)
	var acc int64 = 1
	zeroSuffix := false
	for i := n - 1; i >= 0; i-- {
		if zeroSuffix {
			continue // s[i] stays 0
		}
		s[i] = int(acc)
		if d.Shape[i] > 0 {
			m, err := mulChecked(acc, int64(d.Shape[i]))
			if err != nil {
				return nil, err
			}
			acc = m
		} else {
			zeroSuffix = true
		}
	}
	return s, nil
}

// ExpectedBytes computes the byte geometry of d from first principles: the
// min/max offset over the strided box times the dtype size. With the closed
// policy (non-negative strides) the minimum offset is 0, so the element span is
// maxOffset+1. Zero-size tensors return 0; rank-0 returns DTypeSize. Overflow
// errors instead of wrapping.
func ExpectedBytes(d Descriptor) (int64, error) {
	if err := validate(d, "descriptor"); err != nil {
		return 0, err
	}
	for _, s := range d.Shape {
		if s == 0 {
			return 0, nil
		}
	}
	strides, err := effectiveStrides(d)
	if err != nil {
		return 0, err
	}
	var maxOff int64
	for i, s := range d.Shape {
		if s <= 1 {
			continue
		}
		term, err := mulChecked(int64(s-1), int64(strides[i]))
		if err != nil {
			return 0, err
		}
		maxOff, err = addChecked(maxOff, term)
		if err != nil {
			return 0, err
		}
	}
	span, err := addChecked(maxOff, 1)
	if err != nil {
		return 0, err
	}
	return mulChecked(span, int64(d.DTypeSize))
}

// CheckGeometry refuses a declared-vs-actual descriptor pair whose byte
// geometry or logical grouping disagrees, BEFORE dispatch. Both sides are
// validated first (invalid side -> *InvalidDescriptor naming the side); then,
// in order: rank, per-axis shape, per-axis element stride, dtype size, byte
// span, and leading-axis grouping. The first disagreement is reported as a
// *Mismatch naming the axis (Dim) and both values; grouping disagreements
// unwrap to ErrGroupingMismatch, all others to ErrGeometryMismatch. Identical
// geometry (including nil-vs-explicit row-major strides) passes.
func CheckGeometry(declared, actual Descriptor) error {
	if err := validate(declared, "declared"); err != nil {
		return err
	}
	if err := validate(actual, "actual"); err != nil {
		return err
	}
	if len(declared.Shape) != len(actual.Shape) {
		return &Mismatch{Kind: "rank", Dim: -1, Declared: int64(len(declared.Shape)), Actual: int64(len(actual.Shape))}
	}
	for i := range declared.Shape {
		if declared.Shape[i] != actual.Shape[i] {
			return &Mismatch{Kind: "shape", Dim: i, Declared: int64(declared.Shape[i]), Actual: int64(actual.Shape[i])}
		}
	}
	dStrides, err := effectiveStrides(declared)
	if err != nil {
		return err
	}
	aStrides, err := effectiveStrides(actual)
	if err != nil {
		return err
	}
	for i := range dStrides {
		if dStrides[i] != aStrides[i] {
			return &Mismatch{Kind: "stride", Dim: i, Declared: int64(dStrides[i]), Actual: int64(aStrides[i])}
		}
	}
	if declared.DTypeSize != actual.DTypeSize {
		return &Mismatch{Kind: "dtype", Dim: -1, Declared: int64(declared.DTypeSize), Actual: int64(actual.DTypeSize)}
	}
	dBytes, err := ExpectedBytes(declared)
	if err != nil {
		return err
	}
	aBytes, err := ExpectedBytes(actual)
	if err != nil {
		return err
	}
	if dBytes != aBytes {
		return &Mismatch{Kind: "span", Dim: -1, Declared: dBytes, Actual: aBytes}
	}
	if declared.Groups != actual.Groups {
		return &Mismatch{Kind: "grouping", Dim: 0, Declared: int64(declared.Groups), Actual: int64(actual.Groups)}
	}
	return nil
}

// mulChecked multiplies without overflowing int64.
func mulChecked(a, b int64) (int64, error) {
	if a == 0 || b == 0 {
		return 0, nil
	}
	if a > math.MaxInt64/b || a < math.MinInt64/b {
		return 0, ErrOverflow
	}
	return a * b, nil
}

// addChecked adds without overflowing int64.
func addChecked(a, b int64) (int64, error) {
	if b > 0 && a > math.MaxInt64-b {
		return 0, ErrOverflow
	}
	if b < 0 && a < math.MinInt64-b {
		return 0, ErrOverflow
	}
	return a + b, nil
}
