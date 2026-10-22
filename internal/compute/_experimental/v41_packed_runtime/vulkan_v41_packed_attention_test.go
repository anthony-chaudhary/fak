//go:build ignore

// SOURCE-ONLY: integration sketch for package compute, not a standalone package.
// Original platform constraint: vulkan && (windows || linux) && cgo

package compute

import "testing"

// fak-test:runtime fast est=5ms lane=default
// Device-free authored regression. Compilation/execution not yet authorized.
func TestV41PackedAttentionRejectsInitializedZeroScales(t *testing.T) {
	record := make([]byte, 288)
	if v41PackedScaleRecordValid(record) {
		t.Fatal("zero-filled capacity became attention eligible")
	}
	for i := 256; i < 288; i++ {
		record[i] = 1
	}
	for code := 0; code < 256; code++ {
		record[287] = byte(code)
		if got, want := v41PackedScaleRecordValid(record), code > 0 && code < 127; got != want {
			t.Fatalf("scale %02x: got %v want %v", code, got, want)
		}
	}
	if v41PackedScaleRecordValid(record[:287]) {
		t.Fatal("accepted partial row")
	}
}
