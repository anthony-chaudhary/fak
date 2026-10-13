package model

import (
	"fmt"
	"math"
	"sort"
	"unsafe"
)

// NativeManifestSpanError identifies a violated native f32 descriptor invariant.
// Tensor is empty only when the backing weights buffer itself is misaligned.
// Descriptor admission does not establish the model's graph or config validity.
type NativeManifestSpanError struct {
	Tensor    string
	Invariant string
	Detail    string
}

func (e *NativeManifestSpanError) Error() string {
	return fmt.Sprintf("model: native manifest tensor %q violates %s: %s", e.Tensor, e.Invariant, e.Detail)
}

// TensorName identifies the descriptor refused by native manifest admission.
func (e *NativeManifestSpanError) TensorName() string { return e.Tensor }

// FailedInvariant names the descriptor requirement that was not satisfied.
func (e *NativeManifestSpanError) FailedInvariant() string { return e.Invariant }

// validateNativeManifest admits source descriptors before architecture transforms
// derive aliases, subviews, or non-f32 stores. Its overlap rule is a native-input
// policy: positive spans may alias exactly, but may not overlap otherwise. Empty
// spans, adjacent spans, gaps, and unreferenced trailing bytes are allowed.
func validateNativeManifest(man map[string]tensorMeta, raw []byte) error {
	if len(raw) > 0 && uintptr(unsafe.Pointer(&raw[0]))%unsafe.Alignof(float32(0)) != 0 {
		return &NativeManifestSpanError{
			Invariant: "raw_alignment", Detail: "weights.f32 base is not aligned for float32",
		}
	}
	names := make([]string, 0, len(man))
	for name := range man {
		names = append(names, name)
	}
	sort.Strings(names)
	type interval struct {
		name       string
		start, end int
	}
	spans := make([]interval, 0, len(man))
	for _, name := range names {
		meta := man[name]
		refuse := func(invariant, detail string) error {
			return &NativeManifestSpanError{Tensor: name, Invariant: invariant, Detail: detail}
		}
		if meta.Dtype != "f32" && meta.Dtype != "F32" {
			return refuse("dtype", fmt.Sprintf("expected f32 or F32, got %q", meta.Dtype))
		}
		if meta.Shape == nil {
			return refuse("shape_missing", "shape must be explicit; [] denotes a scalar")
		}
		// Scan all dimensions before multiplying: a later zero annihilates a
		// large positive prefix, but never excuses a negative dimension.
		empty := false
		for _, dim := range meta.Shape {
			if dim < 0 {
				return refuse("shape_negative", fmt.Sprintf("negative dimension %d", dim))
			}
			empty = empty || dim == 0
		}
		elems := 1 // the explicit zero-rank shape is one scalar
		if empty {
			elems = 0
		} else {
			for _, dim := range meta.Shape {
				if elems > (math.MaxInt/4)/dim {
					return refuse("shape_overflow", "f32 shape byte count exceeds int")
				}
				elems *= dim
			}
		}
		if meta.Offset < 0 {
			return refuse("offset_negative", fmt.Sprintf("negative offset %d", meta.Offset))
		}
		if meta.Nbytes < 0 {
			return refuse("size_negative", fmt.Sprintf("negative byte count %d", meta.Nbytes))
		}
		if meta.Offset%4 != 0 {
			return refuse("offset_alignment", fmt.Sprintf("offset %d is not four-byte aligned", meta.Offset))
		}
		if meta.Nbytes%4 != 0 {
			return refuse("size_alignment", fmt.Sprintf("byte count %d is not four-byte aligned", meta.Nbytes))
		}
		// Subtraction is safe only after proving the nonnegative offset fits.
		// Do not compute an artifact-controlled offset+size before this check.
		if meta.Offset > len(raw) || meta.Nbytes > len(raw)-meta.Offset {
			return refuse("bounds", fmt.Sprintf("offset %d and byte count %d exceed %d weight bytes", meta.Offset, meta.Nbytes, len(raw)))
		}
		if elems != meta.Nbytes/4 {
			return refuse("shape_size", fmt.Sprintf("shape has %d elements but byte count %d holds %d", elems, meta.Nbytes, meta.Nbytes/4))
		}
		if meta.Nbytes > 0 {
			spans = append(spans, interval{name: name, start: meta.Offset, end: meta.Offset + meta.Nbytes})
		}
	}
	sort.Slice(spans, func(i, j int) bool {
		if spans[i].start != spans[j].start {
			return spans[i].start < spans[j].start
		}
		if spans[i].end != spans[j].end {
			return spans[i].end < spans[j].end
		}
		return spans[i].name < spans[j].name
	})
	for i := 1; i < len(spans); i++ {
		prev, span := spans[i-1], spans[i]
		if span.start < prev.end && (span.start != prev.start || span.end != prev.end) {
			return &NativeManifestSpanError{
				Tensor: span.name, Invariant: "overlap",
				Detail: fmt.Sprintf("[%d,%d) overlaps tensor %q [%d,%d); only exact positive-span aliases are allowed", span.start, span.end, prev.name, prev.start, prev.end),
			}
		}
	}
	return nil
}
