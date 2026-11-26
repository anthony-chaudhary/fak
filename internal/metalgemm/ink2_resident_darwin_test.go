//go:build darwin && arm64 && cgo

package metalgemm

import "testing"

// MakeResident binds more buffers than one touch dispatch takes, skips absent
// ones, and leaves every byte as it was.
func TestInk2MakeResidentLeavesBuffersUnchanged(t *testing.T) {
	d := ink2TestDevice(t)
	bufs := []Ink2Buf{{}}
	for i := range 2*ink2ResidentTouchArgs + 3 {
		b, err := d.Alloc(64)
		if err != nil {
			t.Fatal(err)
		}
		defer d.Free(b)
		v := make([]float32, 64)
		for j := range v {
			v[j] = float32(i*64 + j)
		}
		if err := d.Write(Ink2Ref{Buf: b}, v); err != nil {
			t.Fatal(err)
		}
		bufs = append(bufs, b)
	}
	before := d.Stats().Dispatches
	if err := d.MakeResident(bufs); err != nil {
		t.Fatal(err)
	}
	if got := d.Stats().Dispatches - before; got != 3 {
		t.Fatalf("touch dispatches = %d, want 3", got)
	}
	for i, b := range bufs[1:] {
		got, err := d.Read(Ink2Ref{Buf: b}, 64)
		if err != nil {
			t.Fatal(err)
		}
		for j, x := range got {
			if x != float32(i*64+j) {
				t.Fatalf("buffer %d[%d] = %v after MakeResident", i, j, x)
			}
		}
	}
	if err := d.MakeResident([]Ink2Buf{{}}); err != nil {
		t.Fatal(err)
	}
}
