package model

import (
	"github.com/anthony-chaudhary/fak/internal/kvbudget"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestKVCacheShapeV4FlashCompressedGolden(t *testing.T) {
	_, cfg := readDeepSeekV4FlashConfig(t)
	shape := cfg.KVCacheShape()
	if shape.Kind != kvbudget.MLA {
		t.Fatalf("shape kind=%v, want MLA shared row", shape.Kind)
	}
	for _, tc := range []struct {
		tokens int
		bytes  int64
	}{{8192, 123994112}, {32768, 462159872}, {32769, 462254592}, {131072, 1814822912}, {1048576, 14439677952}} {
		if got := shape.KVBytesPerStream(tc.tokens, kvbudget.F32); got != float64(tc.bytes) {
			t.Errorf("tokens=%d shape bytes=%g want=%d", tc.tokens, got, tc.bytes)
		}
	}
}
