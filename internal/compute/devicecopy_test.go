package compute

import "testing"

func TestReadDeviceFloats(t *testing.T) {
	got := readDeviceFloats(2, func(dst []float32) { dst[0], dst[1] = 1, 2 })
	if len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("readDeviceFloats = %v", got)
	}
	called := false
	if got := readDeviceFloats(0, func([]float32) { called = true }); len(got) != 0 || called {
		t.Fatalf("empty read = %v, called=%v", got, called)
	}
}
