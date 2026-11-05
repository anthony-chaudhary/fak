package compute

import (
	"testing"
)

// TestCudaPagedPrefillSelect pins oss-port-flashinfer-fp8-prefill-6393 (PR #3321):
// the arch-gated FP8 paged-prefill branch of the CUDA block-table paging path.
// INSPIRE-ONLY port of the principle behind flashinfer-ai/flashinfer's SM120 FMHA
// paged prefill (Apache-2.0): an FP8 paged path exists only where the arch admits
// it; every other arch keeps the generic path. No kernel is added; this witnesses
// host-side selection and byte accounting only. [SW-VERIFIED]
func TestCudaPagedPrefillSelect(t *testing.T) {
	const blockSize = 16
	const stride = 64 // elements per token per layer

	t.Run("gate", func(t *testing.T) {
		cases := []struct {
			major, minor int
			want         CUDAPagedPrefillPath
		}{
			{12, 0, CUDAPagedPrefillFP8},     // sm_120 — the admitted FP8 arch
			{12, 1, CUDAPagedPrefillGeneric}, // sm_121 — same major, unadmitted minor
			{10, 3, CUDAPagedPrefillGeneric}, // sm_103
			{10, 0, CUDAPagedPrefillGeneric}, // sm_100
			{9, 0, CUDAPagedPrefillGeneric},  // sm_90
			{8, 9, CUDAPagedPrefillGeneric},  // sm_89
			{8, 0, CUDAPagedPrefillGeneric},  // sm_80
			{0, 0, CUDAPagedPrefillGeneric},  // unknown capability fails closed to generic
		}
		for _, c := range cases {
			if got := SelectCUDAPagedPrefillPath(c.major, c.minor); got != c.want {
				t.Errorf("SelectCUDAPagedPrefillPath(%d,%d) = %q, want %q", c.major, c.minor, got, c.want)
			}
		}
	})

	t.Run("gated-arch-fp8-pages-match-generic-oracle", func(t *testing.T) {
		generic := NewCUDABlockTable(blockSize, stride)
		fp8 := NewCUDABlockTableForArch(blockSize, stride, 12, 0)
		if got := fp8.PagedPrefillPath(); got != CUDAPagedPrefillFP8 {
			t.Fatalf("sm_120 table path = %q, want %q", got, CUDAPagedPrefillFP8)
		}
		if got := generic.PagedPrefillPath(); got != CUDAPagedPrefillGeneric {
			t.Fatalf("default table path = %q, want %q", got, CUDAPagedPrefillGeneric)
		}

		// Same paging sequence on both paths: appends that cross page boundaries.
		seq := []int{7, 16, 40, 1}
		for i, n := range seq {
			gAlloc := generic.AppendLogicalTokens(n)
			fAlloc := fp8.AppendLogicalTokens(n)
			if len(gAlloc) != len(fAlloc) {
				t.Fatalf("append %d (%d tokens): allocated %d blocks on generic vs %d on fp8", i, n, len(gAlloc), len(fAlloc))
			}
		}

		// Oracle: identical block topology — same count, IDs, per-block tokens, strides.
		gBlocks, fBlocks := generic.Blocks(), fp8.Blocks()
		if len(gBlocks) != len(fBlocks) {
			t.Fatalf("block count = %d on fp8, want %d (generic oracle)", len(fBlocks), len(gBlocks))
		}
		for i := range gBlocks {
			if fBlocks[i].ID != gBlocks[i].ID {
				t.Fatalf("block %d ID = %d on fp8, want %d (generic oracle)", i, fBlocks[i].ID, gBlocks[i].ID)
			}
			if fBlocks[i].Tokens != gBlocks[i].Tokens {
				t.Fatalf("block %d tokens = %d on fp8, want %d (generic oracle)", i, fBlocks[i].Tokens, gBlocks[i].Tokens)
			}
			if fBlocks[i].Stride != gBlocks[i].Stride {
				t.Fatalf("block %d stride = %d on fp8, want %d", i, fBlocks[i].Stride, gBlocks[i].Stride)
			}
		}
		if fp8.TokenCount() != generic.TokenCount() {
			t.Fatalf("token count = %d on fp8, want %d (generic oracle)", fp8.TokenCount(), generic.TokenCount())
		}

		// Byte accounting differs exactly by dtype width: FP8 pages at 1 byte/elem,
		// generic at 4 — same topology, so each block's bytes are exactly 1/4.
		for i := range gBlocks {
			if gBlocks[i].Bytes != 4*fBlocks[i].Bytes {
				t.Fatalf("block %d bytes = %d on fp8, want %d (generic %d at 4x width)", i, fBlocks[i].Bytes, gBlocks[i].Bytes/4, gBlocks[i].Bytes)
			}
		}
		if generic.TotalBytes() != 4*fp8.TotalBytes() {
			t.Fatalf("total bytes = %d on fp8, want %d", fp8.TotalBytes(), generic.TotalBytes()/4)
		}

		// Sharing semantics survive the gate: a clone stays FP8, shares zero-copy,
		// and a divergent append allocates at FP8 width.
		child := fp8.Clone()
		if got := child.PagedPrefillPath(); got != CUDAPagedPrefillFP8 {
			t.Fatalf("clone path = %q, want %q", got, CUDAPagedPrefillFP8)
		}
		if child.SharedBytes() != fp8.TotalBytes() {
			t.Fatalf("clone shared bytes = %d, want %d (zero-copy sharing)", child.SharedBytes(), fp8.TotalBytes())
		}
		if child.UniqueBytes() != 0 {
			t.Fatalf("clone unique bytes = %d, want 0", child.UniqueBytes())
		}
		child.AppendLogicalTokens(16)
		if child.UniqueBytes() != int64(16*stride) {
			t.Fatalf("clone unique bytes after divergent append = %d, want %d (16 tokens x %d elems x 1 byte)", child.UniqueBytes(), 16*stride, stride)
		}
	})

	t.Run("ungated-arch-keeps-generic-bit-identical", func(t *testing.T) {
		hopper := NewCUDABlockTableForArch(blockSize, stride, 9, 0)
		if got := hopper.PagedPrefillPath(); got != CUDAPagedPrefillGeneric {
			t.Fatalf("sm_90 table path = %q, want %q", got, CUDAPagedPrefillGeneric)
		}
		legacy := NewCUDABlockTable(blockSize, stride)
		for _, n := range []int{7, 16, 40, 1} {
			hAlloc := hopper.AppendLogicalTokens(n)
			lAlloc := legacy.AppendLogicalTokens(n)
			if len(hAlloc) != len(lAlloc) {
				t.Fatalf("allocated %d blocks on sm_90 table vs %d on legacy", len(hAlloc), len(lAlloc))
			}
		}
		if hopper.TotalBytes() != legacy.TotalBytes() {
			t.Fatalf("total bytes = %d on sm_90 table, want %d (bit-identical generic accounting)", hopper.TotalBytes(), legacy.TotalBytes())
		}
		// And the legacy F32 width itself: 4 bytes per element.
		if hopper.TotalBytes() != int64(hopper.TokenCount()*stride*4) {
			t.Fatalf("total bytes = %d, want %d (%d tokens x %d elems x 4 bytes)", hopper.TotalBytes(), int64(hopper.TokenCount()*stride*4), hopper.TokenCount(), stride)
		}
	})
}
