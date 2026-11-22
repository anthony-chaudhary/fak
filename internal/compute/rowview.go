package compute

import "strconv"

// RowViewError is the typed refusal of RowView: the backend, dtype or row range
// cannot be aliased without misreading the parent's storage.
type RowViewError struct {
	Backend string
	Dtype   Dtype
	Reason  string
}

func (e *RowViewError) Error() string {
	return "compute: RowView of " + e.Dtype.String() + " on " + e.Backend + ": " + e.Reason
}

// rowViewBuffer is implemented by storage that can alias a row range of itself.
type rowViewBuffer interface {
	rowView(w Tensor, rowStart, rows int) (Tensor, error)
}

// RowView returns a non-owning [rows, in] view of rows [rowStart, rowStart+rows)
// of the row-major weight w, sharing w's storage. MatMul on the view equals
// MatMul on a standalone tensor of those rows. Free on the view releases
// nothing; w must outlive every view of it.
func RowView(be Backend, w Tensor, rowStart, rows int) (Tensor, error) {
	name := "<nil>"
	if be != nil {
		name = be.Name()
	}
	refuse := func(reason string) (Tensor, error) {
		return Tensor{}, &RowViewError{Backend: name, Dtype: w.Dtype, Reason: reason}
	}
	if len(w.Shape) != 2 || w.Layout != RowMajor || w.Shape[0] <= 0 || w.Shape[1] <= 0 {
		return refuse("needs a positive rank-two row-major weight")
	}
	if rowStart < 0 || rows <= 0 || rowStart > w.Shape[0]-rows {
		return refuse("rows [" + strconv.Itoa(rowStart) + ", " + strconv.Itoa(rowStart+rows) +
			") outside [0, " + strconv.Itoa(w.Shape[0]) + ")")
	}
	rv, ok := w.buf.(rowViewBuffer)
	if !ok {
		return refuse("storage has no row views")
	}
	v, err := rv.rowView(w, rowStart, rows)
	if e, ok := err.(*RowViewError); ok && e.Backend == "" {
		e.Backend = name
	}
	return v, err
}

// rowViewGeometry reports the payload bytes per weight row and, for formats
// with a separate scale plane (Q8_0 f32 scales, NVFP4 E4M3 scales), the scale
// bytes per row. reason is non-empty for a dtype or width without row views.
func rowViewGeometry(w Tensor) (rowBytes, scaleRowBytes int, reason string) {
	in := w.Shape[1]
	super := func(blockBytes int) (int, int, string) {
		if in%q4kSuper != 0 {
			return 0, 0, "needs in divisible by 256"
		}
		return in / q4kSuper * blockBytes, 0, ""
	}
	switch w.Dtype {
	case F32:
		return in * F32.Bytes(), 0, ""
	case Q8_0:
		if w.Quant == nil || w.Quant.Block <= 0 || in%w.Quant.Block != 0 {
			return 0, 0, "needs a Q8_0 block that divides in"
		}
		return in, in / w.Quant.Block * F32.Bytes(), ""
	case Q3_K:
		return super(q3kSuperBlock)
	case Q4_K:
		return super(q4kSuperBlock)
	case Q5_K:
		return super(176)
	case Q6_K:
		return super(210)
	case NVFP4:
		if in%NVFP4BlockSize != 0 {
			return 0, 0, "needs in divisible by 16"
		}
		return in / 2, in / NVFP4BlockSize, ""
	}
	if f, ok := rawIQFormats[w.Dtype]; ok {
		return super(f.blockBytes)
	}
	return 0, 0, "dtype has no row-view geometry"
}

// rowViewQuant narrows a per-row scale vector (host Q8_0 scales) to the view's
// rows and keeps a single tensor-wide scale (NVFP4) as is.
func rowViewQuant(q *QuantSpec, out, rowStart, rows int) *QuantSpec {
	if q == nil {
		return nil
	}
	v := *q
	if n := len(q.Scale); n > 1 && n%out == 0 {
		per := n / out
		v.Scale = q.Scale[rowStart*per : (rowStart+rows)*per : (rowStart+rows)*per]
	}
	return &v
}

func (h *hostBuf) rowView(w Tensor, rowStart, rows int) (Tensor, error) {
	rowBytes, scaleRowBytes, reason := rowViewGeometry(w)
	if reason != "" {
		return Tensor{}, &RowViewError{Dtype: w.Dtype, Reason: reason}
	}
	out, in := w.Shape[0], w.Shape[1]
	shape := []int{rows, in}
	q := rowViewQuant(w.Quant, out, rowStart, rows)
	switch {
	case w.Dtype == F32:
		if len(h.f32) != out*in {
			return Tensor{}, &RowViewError{Dtype: w.Dtype, Reason: "host payload does not match the shape"}
		}
		lo, hi := rowStart*in, (rowStart+rows)*in
		return makeTensor(w.be, F32, RowMajor, shape, q, &hostBuf{f32: h.f32[lo:hi:hi]}), nil
	case w.Dtype == NVFP4:
		packed, scales := h.i8, h.nvfp4Scales
		if scales == nil {
			if len(h.i8) != out*(rowBytes+scaleRowBytes) {
				return Tensor{}, &RowViewError{Dtype: w.Dtype, Reason: "host payload does not match the shape"}
			}
			packed, scales = h.i8[:out*rowBytes], h.i8[out*rowBytes:]
		}
		if !h.nvfp4 || len(packed) != out*rowBytes || len(scales) != out*scaleRowBytes {
			return Tensor{}, &RowViewError{Dtype: w.Dtype, Reason: "host payload does not match the shape"}
		}
		plo, phi := rowStart*rowBytes, (rowStart+rows)*rowBytes
		slo, shi := rowStart*scaleRowBytes, (rowStart+rows)*scaleRowBytes
		return makeTensor(w.be, NVFP4, RowMajor, shape, q,
			&hostBuf{i8: packed[plo:phi:phi], nvfp4Scales: scales[slo:shi:shi], nvfp4: true}), nil
	default:
		if len(h.i8) != out*rowBytes {
			return Tensor{}, &RowViewError{Dtype: w.Dtype, Reason: "host payload does not match the shape"}
		}
		if w.Dtype == Q8_0 && (q == nil || len(q.Scale) != rows*scaleRowBytes/F32.Bytes()) {
			return Tensor{}, &RowViewError{Dtype: w.Dtype, Reason: "host Q8_0 scales do not match the shape"}
		}
		lo, hi := rowStart*rowBytes, (rowStart+rows)*rowBytes
		return makeTensor(w.be, w.Dtype, RowMajor, shape, q, &hostBuf{i8: h.i8[lo:hi:hi]}), nil
	}
}
