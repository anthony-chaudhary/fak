package kvbudget

import (
	"errors"
	"testing"
)

// TestKVBudgetGraphMemoryReserve pins the CUDA-graph pool reservation term in
// the KV-block derivation (fak oss-port-vllmtop-cudagraph-mem-profile-20261103,
// adapting vLLM profile_cudagraph_memory + determine_available_memory @
// ae065918640a198a56f6f17eb45e251c3229fe41):
//   - (a) injected probe measurements (first=100MiB, delta=2MiB x 8 buckets)
//     yield the exact reserved bytes and the reduced block count;
//   - (b) zero-bucket / NONE-mode input reserves 0 and leaves the block count
//     unchanged;
//   - (c) negative free-after-reservation fails closed with the too-small
//     error, not a negative block count.
//
// Software-only arithmetic: no GPU or throughput claim.
func TestKVBudgetGraphMemoryReserve(t *testing.T) {
	const MiB = uint64(1 << 20)

	t.Run("probe extrapolation", func(t *testing.T) {
		cases := []struct {
			name    string
			deltas  []uint64
			buckets int
			want    uint64
		}{
			// first capture pays the pool baseline; each later bucket adds the
			// second sample: 100MiB + 7 x 2MiB = 114MiB.
			{name: "first 100MiB delta 2MiB over 8 buckets", deltas: []uint64{100 * MiB, 2 * MiB}, buckets: 8, want: 114 * MiB},
			// Second sample under the floor is charged at MinPerGraphBytes:
			// 100MiB + 3 x 1MiB = 103MiB.
			{name: "per-bucket floored at MinPerGraphBytes", deltas: []uint64{100 * MiB, 512 * 1024}, buckets: 4, want: 103 * MiB},
			// One sample has no per-bucket cost: the first capture only.
			{name: "single sample pays first only", deltas: []uint64{100 * MiB}, buckets: 8, want: 100 * MiB},
			// Only the first two samples drive the estimate; the rest are noise.
			{name: "samples beyond second are ignored", deltas: []uint64{100 * MiB, 2 * MiB, 5 * MiB, 9 * MiB}, buckets: 8, want: 114 * MiB},
			// A one-bucket ladder captures nothing past the first.
			{name: "one-bucket ladder pays first only", deltas: []uint64{100 * MiB, 2 * MiB}, buckets: 1, want: 100 * MiB},
			// Zero-bucket probe (nothing captured) reserves 0.
			{name: "zero-bucket probe reserves 0", deltas: nil, buckets: 8, want: 0},
			// NONE-mode ladder (no buckets to capture) reserves 0 even with a
			// recorded sample.
			{name: "zero totalBuckets reserves 0", deltas: []uint64{100 * MiB, 2 * MiB}, buckets: 0, want: 0},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				got := GraphMemProbe{Deltas: tc.deltas}.ExtrapolatedPoolFootprint(tc.buckets)
				if got != tc.want {
					t.Fatalf("ExtrapolatedPoolFootprint(%d buckets) = %d, want %d", tc.buckets, got, tc.want)
				}
			})
		}
	})

	t.Run("reserved bytes reduce the block count", func(t *testing.T) {
		probe := GraphMemProbe{Deltas: []uint64{100 * MiB, 2 * MiB}}
		reserve := probe.ExtrapolatedPoolFootprint(8)
		if reserve != 114*MiB {
			t.Fatalf("reserve = %d, want %d (114MiB)", reserve, 114*MiB)
		}
		// Baseline without the graph term: 1GiB / 1MiB = 1024 blocks.
		base, err := DeriveMaxBlocks(GiB, 0, 1.0, MiB)
		if err != nil {
			t.Fatalf("DeriveMaxBlocks err = %v, want nil", err)
		}
		if base != 1024 {
			t.Fatalf("baseline blocks = %d, want 1024", base)
		}
		// Charging the 114MiB pool against the 1GiB budget leaves 910 blocks.
		got, err := DeriveMaxBlocksWithGraphReserve(GiB, 0, reserve, 1.0, MiB)
		if err != nil {
			t.Fatalf("DeriveMaxBlocksWithGraphReserve err = %v, want nil", err)
		}
		if got != 910 {
			t.Fatalf("blocks = %d, want 910 (1024 - 114)", got)
		}
	})

	t.Run("zero reserve leaves the block count unchanged", func(t *testing.T) {
		// NONE-mode / zero-bucket probe reserves 0.
		if got := (GraphMemProbe{}).ExtrapolatedPoolFootprint(8); got != 0 {
			t.Fatalf("empty probe footprint = %d, want 0", got)
		}
		// A zero reserve must reduce to DeriveMaxBlocks exactly, cell for cell.
		cases := []struct {
			free, nonKV uint64
			util        float64
			bpb         uint64
			want        int
		}{
			{GiB, 0, 1.0, MiB, 1024},
			{80 * GiB, 20 * GiB, 0.9, 2 * MiB, 26624},
			{1000, 100, 0.5, 300, 1},
		}
		for _, tc := range cases {
			base, err := DeriveMaxBlocks(tc.free, tc.nonKV, tc.util, tc.bpb)
			if err != nil {
				t.Fatalf("DeriveMaxBlocks err = %v, want nil", err)
			}
			if base != tc.want {
				t.Fatalf("test bug: DeriveMaxBlocks baseline = %d, want %d", base, tc.want)
			}
			got, err := DeriveMaxBlocksWithGraphReserve(tc.free, tc.nonKV, 0, tc.util, tc.bpb)
			if err != nil {
				t.Fatalf("zero-reserve err = %v, want nil", err)
			}
			if got != tc.want {
				t.Fatalf("zero-reserve blocks = %d, want %d (unchanged)", got, tc.want)
			}
		}
	})

	t.Run("negative free after reservation fails closed", func(t *testing.T) {
		cases := []struct {
			name      string
			free, nkv uint64
			reserve   uint64
			util      float64
			bpb       uint64
			wantErr   error
		}{
			// 100MiB free minus a 101MiB reserve is negative: refuse, do not
			// return a wrapped/negative block count.
			{name: "reserve alone exceeds free", free: 100 * MiB, nkv: 0, reserve: 101 * MiB, util: 1.0, bpb: MiB, wantErr: ErrNonKVExceedsBudget},
			// 1GiB free, 900MiB non-KV + 200MiB reserve = 1100MiB > 1024MiB.
			{name: "non-KV plus reserve exceeds utilized", free: GiB, nkv: 900 * MiB, reserve: 200 * MiB, util: 1.0, bpb: MiB, wantErr: ErrNonKVExceedsBudget},
			// The utilization guard still fires before the subtraction.
			{name: "zero utilization", free: GiB, nkv: 0, reserve: MiB, util: 0, bpb: MiB, wantErr: ErrUtilizationOutOfRange},
			// The zero-block-size guard still fires before the subtraction.
			{name: "zero block size", free: GiB, nkv: 0, reserve: MiB, util: 1.0, bpb: 0, wantErr: ErrZeroBlockSize},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				got, err := DeriveMaxBlocksWithGraphReserve(tc.free, tc.nkv, tc.reserve, tc.util, tc.bpb)
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("err = %v, want %v", err, tc.wantErr)
				}
				if got != 0 {
					t.Fatalf("blocks = %d, want 0 on refusal", got)
				}
			})
		}
	})

	t.Run("composition with utilization below one", func(t *testing.T) {
		// floor(80GiB x 0.9) = 72GiB; 72GiB - 20GiB non-KV - 4GiB reserve =
		// 48GiB; 48GiB / 2MiB = 24576 blocks.
		got, err := DeriveMaxBlocksWithGraphReserve(80*GiB, 20*GiB, 4*GiB, 0.9, 2*MiB)
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}
		if got != 24576 {
			t.Fatalf("blocks = %d, want 24576", got)
		}
	})
}
