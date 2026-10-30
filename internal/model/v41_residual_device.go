package model

import (
	"errors"
	"slices"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// errV41DeviceStreams reports a malformed or foreign device stream panel.
var errV41DeviceStreams = errors.New("model: invalid V4.1 device-resident stream panel")

// v41DeviceStreams is the device-resident mHC residual stream panel of one
// forward: one F32 row-major tensor [rows, 4, hidden] on the session backend.
// Row r holds token r's four hyper-connection streams laid end to end, so the
// flattened [rows, 4*hidden] view is exactly the mHC projection input. Every
// publication into the panel is BF16-rounded, so its values are
// BF16-representable.
//
// The panel owns its tensor. Tensor and Flat lend it until the next Replace or
// Free. A device op that produces successor streams (the mHC post-mix, the
// Engram gate/add) returns a fresh tensor and hands it to Replace, which takes
// ownership and frees the predecessor; the op never writes the lent tensor in
// place.
type v41DeviceStreams struct {
	be     compute.Backend
	t      compute.Tensor
	rows   int
	hidden int
}

// newV41DeviceStreams takes ownership of t, which must be a live F32 row-major
// [rows, 4, hidden] tensor on be. On error the caller still owns t.
func newV41DeviceStreams(be compute.Backend, t compute.Tensor) (*v41DeviceStreams, error) {
	if !v41DeviceStreamsShapeOK(be, t) {
		return nil, errV41DeviceStreams
	}
	return &v41DeviceStreams{be: be, t: t, rows: t.Shape[0], hidden: t.Shape[2]}, nil
}

func v41DeviceStreamsShapeOK(be compute.Backend, t compute.Tensor) bool {
	return be != nil && t.Buf() != nil && t.Backend() == be && t.Dtype == compute.F32 && t.Layout == compute.RowMajor &&
		t.Quant == nil && len(t.Shape) == 3 && t.Shape[0] > 0 && t.Shape[1] == compute.V41MHCStreams && t.Shape[2] > 0
}

// Tensor lends the [rows, 4, hidden] panel.
func (d *v41DeviceStreams) Tensor() compute.Tensor { return d.t }

// Flat lends the same buffer viewed as [rows, 4*hidden], the mHC projection input.
func (d *v41DeviceStreams) Flat() compute.Tensor {
	view := d.t
	view.Shape = []int{d.rows, compute.V41MHCStreams * d.hidden}
	return view
}

func (d *v41DeviceStreams) Rows() int   { return d.rows }
func (d *v41DeviceStreams) Hidden() int { return d.hidden }

// Replace takes ownership of next, a fresh panel of the same geometry on the same
// backend, and frees the predecessor. On error the caller still owns next and
// the panel is unchanged.
func (d *v41DeviceStreams) Replace(next compute.Tensor) error {
	if d == nil || d.t.Buf() == nil || !v41DeviceStreamsShapeOK(d.be, next) || !slices.Equal(next.Shape, d.t.Shape) || next.Buf() == d.t.Buf() {
		return errV41DeviceStreams
	}
	old := d.t
	d.t = next
	d.be.Free(old)
	return nil
}

// ReadHost returns an owned host copy [rows][4][hidden]. It is the host fence
// for a caller that still has a host-side consumer of the streams.
func (d *v41DeviceStreams) ReadHost() ([][][]float32, error) {
	if d == nil || d.t.Buf() == nil {
		return nil, errV41DeviceStreams
	}
	values := d.be.Read(d.t)
	if len(values) != d.rows*compute.V41MHCStreams*d.hidden {
		return nil, errV41DeviceStreams
	}
	flat := append([]float32(nil), values...)
	out := make([][][]float32, d.rows)
	for r := range out {
		out[r] = make([][]float32, compute.V41MHCStreams)
		for h := range out[r] {
			base := (r*compute.V41MHCStreams + h) * d.hidden
			out[r][h] = flat[base : base+d.hidden : base+d.hidden]
		}
	}
	return out, nil
}

// Free releases the panel; later calls are no-ops.
func (d *v41DeviceStreams) Free() {
	if d == nil || d.t.Buf() == nil {
		return
	}
	owned := d.t
	d.t = compute.Tensor{}
	d.be.Free(owned)
}
