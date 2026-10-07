package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// fak-test:runtime fast est=5ms lane=default
func TestNativeDecodeSummaryCarriesCacheTierAndRestore(t *testing.T) {
	cases := []struct {
		name               string
		res                inKernelGenerateResult
		wantTier, wantRest string
	}{
		{"device hit", inKernelGenerateResult{matched: 40, cacheable: 40, sourceTier: radixkv.SnapshotTierDeviceL1}, "device_l1", NativeCacheRestoreHit},
		{"host l2 hit", inKernelGenerateResult{matched: 40, cacheable: 48, sourceTier: radixkv.SnapshotTierHostL2}, "host_dram_l2", NativeCacheRestoreHit},
		{"remote l3 hit", inKernelGenerateResult{matched: 8, cacheable: 8, sourceTier: radixkv.SnapshotTierRemoteL3}, "remote_http_l3", NativeCacheRestoreHit},
		{"matched but unserved", inKernelGenerateResult{matched: 0, cacheable: 32, sourceTier: radixkv.SnapshotTierMiss}, NativeCacheTierNone, NativeCacheRestoreUnserved},
		{"cold miss", inKernelGenerateResult{}, NativeCacheTierNone, NativeCacheRestoreMiss},
		{"hit with unresolved tier stays unlabeled", inKernelGenerateResult{matched: 4, cacheable: 4}, "", NativeCacheRestoreHit},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sum := newNativeDecodeSummary(tc.res)
			if sum.CacheTier != tc.wantTier || sum.CacheRestore != tc.wantRest {
				t.Fatalf("tier/restore = %q/%q, want %q/%q", sum.CacheTier, sum.CacheRestore, tc.wantTier, tc.wantRest)
			}
		})
	}
}
