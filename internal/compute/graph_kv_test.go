package compute

import (
	"errors"
	"testing"
)

// qwen38GraphKVConfig is the Qwen3.8-27B device-KV geometry: 4 KV heads of 256 dims, and
// rows = 16 compacted full-attention layers (or 64 when every model layer is charged).
func qwen38GraphKVConfig(rows int) KVConfig {
	return KVConfig{NumLayers: rows, NumKVHeads: 4, HeadDim: 256, GraphCapture: true}
}

// fak-test:runtime fast est=1s
func TestGraphKVCapacityPreallocBytesChargeWrittenRows(t *testing.T) {
	const positions = 32768
	const gib = int64(1) << 30
	if got := GraphKVPreallocBytes(qwen38GraphKVConfig(16), positions); got != 6*gib {
		t.Fatalf("16 written rows at %d positions = %d bytes, want %d (6 GiB)", positions, got, 6*gib)
	}
	if got := GraphKVPreallocBytes(qwen38GraphKVConfig(64), positions); got != 24*gib {
		t.Fatalf("64 rows at %d positions = %d bytes, want %d (24 GiB)", positions, got, 24*gib)
	}
	if got := GraphKVPreallocBytes(qwen38GraphKVConfig(16), 1); got != 192*1024 {
		t.Fatalf("one position over 16 rows = %d bytes, want 196608 (192 KiB)", got)
	}
	if got := GraphKVPreallocBytes(KVConfig{}, positions); got != 0 {
		t.Fatalf("empty geometry = %d bytes, want 0", got)
	}
}

// fak-test:runtime fast est=1s
func TestGraphKVCapacityPreallocatesOnlyEligibleSessionsThatFit(t *testing.T) {
	const positions = 32768
	const need = int64(6) << 30
	eligible := qwen38GraphKVConfig(16)
	ineligible := eligible
	ineligible.GraphCapture = false
	for _, tc := range []struct {
		name    string
		graphOn bool
		cfg     KVConfig
		free    int64
		want    bool
		reason  string
	}{
		{"graph off", false, eligible, FreeUnknown, false, GraphKVGraphOff},
		{"not capture eligible", true, ineligible, FreeUnknown, false, GraphKVNotEligible},
		{"empty geometry", true, KVConfig{GraphCapture: true}, FreeUnknown, false, GraphKVEmptyGeometry},
		{"does not fit", true, eligible, need - 1, false, GraphKVDoesNotFit},
		{"exact fit", true, eligible, need, true, GraphKVPreallocated},
		{"free unknown fails open", true, eligible, FreeUnknown, true, GraphKVPreallocated},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := GraphKVPreallocFor(tc.graphOn, tc.cfg, positions, tc.free)
			if got.Preallocate != tc.want || got.Reason != tc.reason || got.Positions != positions {
				t.Fatalf("prealloc = %+v, want preallocate=%v reason=%q positions=%d", got, tc.want, tc.reason, positions)
			}
			if tc.cfg.NumLayers == 16 && got.Bytes != need {
				t.Fatalf("prealloc bytes = %d, want %d", got.Bytes, need)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestGraphKVCapacityGrowInsideCaptureIsTyped(t *testing.T) {
	var err error = graphKVGrowError("kv-key-grow layer 0")
	var op *CUDAOpError
	if !errors.As(err, &op) || op.Op != "AppendKV" || op.Site != GraphKVGrowSite || op.Class != MemoryKVCache {
		t.Fatalf("grow refusal = %#v, want CUDAOpError op=AppendKV site=%s class=kv", err, GraphKVGrowSite)
	}
}
