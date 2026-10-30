package compute

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestV41WKVScalePacking(t *testing.T) {
	raw := make([]byte, 16*160)
	for i := range raw {
		raw[i] = byte(i % 255)
	}
	got, err := packV41WKVScales(raw)
	if err != nil {
		t.Fatal(err)
	}
	for row := 0; row < 512; row++ {
		for k := 0; k < 160; k++ {
			if got[((k/4)*512+row)*4+k%4] != raw[(row/32)*160+k] {
				t.Fatal("MN-major scale mismatch")
			}
		}
	}
	raw[0] = 255
	if _, err := packV41WKVScales(raw); err == nil {
		t.Fatal("accepted NaN scale")
	}
}
