package model

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/kvbudget"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestContextSizeV4FlashCompressedGoldenAndGate(t *testing.T) {
	_, cfg := readDeepSeekV4FlashConfig(t)
	t.Setenv("FAK_HYBRID_KV", "0")
	off := cfg.ContextSizeConfigWithPrecision(compute.KVPrecisionF32)
	if got := off.KVBytes(32768); got != 32768*264192 {
		t.Fatalf("gate-off bytes=%d, want unchanged uniform=%d", got, 32768*264192)
	}
	t.Setenv("FAK_HYBRID_KV", "1")
	on := cfg.ContextSizeConfigWithPrecision(compute.KVPrecisionF32)
	if len(on.SessionState) == 0 {
		t.Fatal("compressor in-flight buffer uncharged")
	}
	for _, tc := range []struct {
		tokens int
		bytes  int64
	}{{8192, 123994112}, {32768, 462159872}, {32769, 462254592}, {131072, 1814822912}, {1048576, 14439677952}} {
		if got := on.KVBytes(tc.tokens); got != tc.bytes {
			t.Errorf("tokens=%d compute bytes=%d want=%d", tc.tokens, got, tc.bytes)
		}
		if got := cfg.KVCacheShape().KVBytesPerStream(tc.tokens, kvbudget.F32); got != float64(on.KVBytes(tc.tokens)) {
			t.Errorf("tokens=%d shape=%g differs compute=%d", tc.tokens, got, on.KVBytes(tc.tokens))
		}
	}
	if on.KVBytes(1048576) == 33816576 {
		t.Fatal("compression schedule charged window-only")
	}
	t.Setenv("FAK_HYBRID_KV", "0")
	if again := cfg.ContextSizeConfigWithPrecision(compute.KVPrecisionF32); !reflect.DeepEqual(off, again) {
		t.Fatal("gate-off config changed after gated projection")
	}
}

// fak-test:runtime fast est=1ms
func TestContextSizeV41CompressionNeverWindowOnly(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	t.Setenv("FAK_HYBRID_KV", "0")
	off := cfg.ContextSizeConfigWithPrecision(compute.KVPrecisionF32)
	t.Setenv("FAK_HYBRID_KV", "1")
	on := cfg.ContextSizeConfigWithPrecision(compute.KVPrecisionF32)
	if got, want := on.KVBytes(32768), off.KVBytes(32768); got != want {
		t.Fatalf("V4.1 unsupported compressed projection discounted uniform: got=%d want=%d", got, want)
	}
	if got, want := cfg.KVCacheShape().KVBytesPerStream(32768, kvbudget.F32), cfg.KVCacheShape().KVBytesPerStream(1048576, kvbudget.F32); got == want {
		t.Fatal("V4.1 shape collapsed schedule to bounded window-only")
	}
}
