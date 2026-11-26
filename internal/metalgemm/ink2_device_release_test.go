//go:build darwin && arm64 && cgo

package metalgemm

import "testing"

// fak-test: runtime=2s requires=metal
func TestInk2FreedBuffersReturnDeviceMemoryAfterHostAccess(t *testing.T) {
	d := ink2TestDevice(t)
	const floats, rounds = 16 << 20, 8 // 64 MiB a round
	before := gdnCurrentAllocatedBytes()
	for range rounds {
		b, err := d.Alloc(floats)
		if err != nil {
			t.Fatal(err)
		}
		ref := Ink2Ref{Buf: b}
		if err := d.Write(ref, []float32{1}); err != nil {
			t.Fatal(err)
		}
		if _, err := d.Read(ref, 1); err != nil {
			t.Fatal(err)
		}
		d.Free(b)
	}
	if grew := int64(gdnCurrentAllocatedBytes()) - int64(before); grew > 4*floats {
		t.Fatalf("%d MiB stayed allocated after %d rounds of a 64 MiB buffer written, read and freed", grew>>20, rounds)
	}
}
