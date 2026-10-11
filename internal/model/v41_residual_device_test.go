package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41StreamsFreeRecorder counts frees so ownership hand-offs are observable.
type v41StreamsFreeRecorder struct {
	compute.Backend
	freed map[compute.Buffer]int
}

func (b *v41StreamsFreeRecorder) Free(t compute.Tensor) {
	b.freed[t.Buf()]++
	b.Backend.Free(t)
}

// fak-test:runtime fast est=10ms lane=default
func TestV41DeviceStreamsOwnership(t *testing.T) {
	be := &v41StreamsFreeRecorder{Backend: compute.Default(), freed: map[compute.Buffer]int{}}
	const rows, hidden = 2, 3
	values := make([]float32, rows*4*hidden)
	for i := range values {
		values[i] = float32(i)
	}
	panel := compute.NewF32(be, []int{rows, 4, hidden}, values)
	for _, bad := range []compute.Tensor{
		compute.NewF32(be, []int{rows, 3, hidden}, make([]float32, rows*3*hidden)),
		compute.NewF32(be, []int{rows * 4 * hidden}, values),
		compute.NewF32(compute.Default(), []int{rows, 4, hidden}, values),
	} {
		if _, err := newV41DeviceStreams(be, bad); !errors.Is(err, errV41DeviceStreams) {
			t.Fatalf("accepted malformed or foreign panel %v", bad.Shape)
		}
	}
	d, err := newV41DeviceStreams(be, panel)
	if err != nil || d.Rows() != rows || d.Hidden() != hidden || d.Tensor().Buf() != panel.Buf() {
		t.Fatalf("panel=%+v err=%v", d, err)
	}
	if flat := d.Flat(); flat.Buf() != panel.Buf() || !reflect.DeepEqual(flat.Shape, []int{rows, 4 * hidden}) || !reflect.DeepEqual(d.Tensor().Shape, []int{rows, 4, hidden}) {
		t.Fatal("flat view is not the same buffer as [rows, 4*hidden], or it reshaped the panel")
	}
	host, err := d.ReadHost()
	if err != nil || len(host) != rows || host[1][2][1] != float32((1*4+2)*hidden+1) {
		t.Fatalf("host copy=%v err=%v", host, err)
	}
	host[0][0][0] = 99
	if be.Read(d.Tensor())[0] == 99 {
		t.Fatal("ReadHost lent backend storage instead of an owned copy")
	}
	if err := d.Replace(panel); err == nil || be.freed[panel.Buf()] != 0 {
		t.Fatal("Replace accepted the live panel itself")
	}
	if err := d.Replace(compute.NewF32(be, []int{rows + 1, 4, hidden}, make([]float32, (rows+1)*4*hidden))); err == nil {
		t.Fatal("Replace accepted a different geometry")
	}
	next := compute.NewF32(be, []int{rows, 4, hidden}, make([]float32, rows*4*hidden))
	if err := d.Replace(next); err != nil || d.Tensor().Buf() != next.Buf() || be.freed[panel.Buf()] != 1 || be.freed[next.Buf()] != 0 {
		t.Fatalf("Replace did not hand off ownership exactly once: err=%v freed=%v", err, be.freed)
	}
	d.Free()
	d.Free()
	if be.freed[next.Buf()] != 1 {
		t.Fatalf("Free released the successor %d times, want once", be.freed[next.Buf()])
	}
	if _, err := d.ReadHost(); !errors.Is(err, errV41DeviceStreams) {
		t.Fatal("ReadHost succeeded after Free")
	}
}
