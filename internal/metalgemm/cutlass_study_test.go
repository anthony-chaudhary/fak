package metalgemm

import (
	"strings"
	"testing"
)

// TestCutlassStudy asserts the pinned CUTLASS template-study model: the tile
// scheduler shape pins, the collective mainloop template, the NVFP4
// tensor-core template, and fail-closed refusal on unsupported tile configs.
// It is a template study only — no kernel port, no device speed claim.
func TestCutlassStudy(t *testing.T) {
	t.Run("tile scheduler shape pins", func(t *testing.T) {
		model := CutlassStudyModel()
		if model.Repo != "NVIDIA/cutlass" {
			t.Fatalf("Repo = %q, want NVIDIA/cutlass", model.Repo)
		}
		if model.Pin != "b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd" {
			t.Fatalf("Pin = %q, want the pinned commit", model.Pin)
		}
		if len(model.Schedulers) != 4 {
			t.Fatalf("Schedulers = %d, want 4 pinned kinds", len(model.Schedulers))
		}
		// Each pinned scheduler must carry a real upstream path:line and a
		// distinct kind, and the lookup must round-trip.
		seen := make(map[TileSchedulerKind]bool)
		for _, s := range model.Schedulers {
			if seen[s.Kind] {
				t.Fatalf("duplicate scheduler kind %v", s.Kind)
			}
			seen[s.Kind] = true
			if s.UpstreamFile.Path == "" || s.UpstreamFile.Line <= 0 {
				t.Fatalf("scheduler %v has empty upstream pin %+v", s.Kind, s.UpstreamFile)
			}
			if !strings.HasPrefix(s.UpstreamFile.Path, "include/cutlass/gemm/kernel/") {
				t.Fatalf("scheduler %v upstream path %q is not under the gemm kernel tree", s.Kind, s.UpstreamFile.Path)
			}
			got, ok := PinnedSchedulerFor(s.Kind)
			if !ok {
				t.Fatalf("PinnedSchedulerFor(%v) = false, want true", s.Kind)
			}
			if got.UpstreamFile != s.UpstreamFile {
				t.Fatalf("PinnedSchedulerFor(%v) upstream = %+v, want %+v", s.Kind, got.UpstreamFile, s.UpstreamFile)
			}
		}
		// Spot-check the two load-bearing scheduler pins against the study.
		clc, ok := PinnedSchedulerFor(TileSchedulerDynamicPersistentCLC)
		if !ok {
			t.Fatal("CLC scheduler not pinned")
		}
		if clc.UpstreamFile.Path != "include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp" || clc.UpstreamFile.Line != 59 {
			t.Fatalf("CLC pin = %+v, want sm100_tile_scheduler.hpp:59", clc.UpstreamFile)
		}
		if _, ok := PinnedSchedulerFor(TileSchedulerKind(99)); ok {
			t.Fatal("PinnedSchedulerFor(unknown) = true, want false")
		}
	})

	t.Run("collective mainloop template", func(t *testing.T) {
		sm90, ok := PinnedMainloopFor("sm90")
		if !ok {
			t.Fatal("sm90 mainloop not pinned")
		}
		if sm90.Pipeline != "PipelineTmaAsync" || sm90.AccumulatorEngine != "smem" {
			t.Fatalf("sm90 mainloop = %s/%s, want PipelineTmaAsync/smem", sm90.Pipeline, sm90.AccumulatorEngine)
		}
		wantML := []string{"prefetch_tma_descriptors", "load_init", "load", "load_tail", "mma", "mma_tail"}
		if !equalStrings(sm90.MainloopContract, wantML) {
			t.Fatalf("sm90 mainloop contract = %v, want %v", sm90.MainloopContract, wantML)
		}
		wantEp := []string{"load", "store", "store_tail"}
		if !equalStrings(sm90.EpilogueContract, wantEp) {
			t.Fatalf("sm90 epilogue contract = %v, want %v", sm90.EpilogueContract, wantEp)
		}

		sm100, ok := PinnedMainloopFor("sm100")
		if !ok {
			t.Fatal("sm100 mainloop not pinned")
		}
		if sm100.Pipeline != "PipelineUmmaAsync" || sm100.AccumulatorEngine != "tmem" {
			t.Fatalf("sm100 mainloop = %s/%s, want PipelineUmmaAsync/tmem", sm100.Pipeline, sm100.AccumulatorEngine)
		}
		// The sm100 warp role set is the load-bearing shape: five roles, in
		// the pinned order.
		wantRoles := []MainloopWarpRole{
			MainloopWarpRoleMMA,
			MainloopWarpRoleSched,
			MainloopWarpRoleMainloopLoad,
			MainloopWarpRoleEpilogueLoad,
			MainloopWarpRoleEpilogue,
		}
		if len(sm100.WarpRoles) != len(wantRoles) {
			t.Fatalf("sm100 warp roles = %v, want %v", sm100.WarpRoles, wantRoles)
		}
		for i, want := range wantRoles {
			if sm100.WarpRoles[i] != want {
				t.Fatalf("sm100 warp role %d = %v, want %v", i, sm100.WarpRoles[i], want)
			}
		}
		if !equalStrings(sm100.MainloopContract, wantML) {
			t.Fatalf("sm100 mainloop contract = %v, want %v", sm100.MainloopContract, wantML)
		}
		if _, ok := PinnedMainloopFor("sm80"); ok {
			t.Fatal("PinnedMainloopFor(sm80) = true, want false")
		}
	})

	t.Run("NVFP4 tensor-core template", func(t *testing.T) {
		tmpl := CutlassStudyModel().NVFP4
		if tmpl.Arch != "sm103a" {
			t.Fatalf("NVFP4 arch = %q, want sm103a (NVFP4 is not the sm100a gate)", tmpl.Arch)
		}
		if tmpl.SFVecSize != 16 || tmpl.SFFormat != "e4m3" {
			t.Fatalf("NVFP4 SF = %d/%s, want 16/e4m3", tmpl.SFVecSize, tmpl.SFFormat)
		}
		if tmpl.SFChunkMN != 128 || tmpl.SFChunkSF != 4 || tmpl.SFPerTMEMWord != 4 {
			t.Fatalf("NVFP4 SF chunk = %d/%d/%d, want 128/4/4", tmpl.SFChunkMN, tmpl.SFChunkSF, tmpl.SFPerTMEMWord)
		}
		if tmpl.TMEMRows != 128 || tmpl.TMEMColumns != 512 {
			t.Fatalf("NVFP4 TMEM = %dx%d, want 128x512", tmpl.TMEMRows, tmpl.TMEMColumns)
		}
		if !equalInts(tmpl.TilesM, []int{128, 256}) || tmpl.TilesN != 256 || tmpl.TilesK != 768 {
			t.Fatalf("NVFP4 tiles = M%v N%d K%d, want M[128 256] N256 K768", tmpl.TilesM, tmpl.TilesN, tmpl.TilesK)
		}
		// The template must pin the blockscaled layout and the sm103 example.
		foundLayout, foundExample := false, false
		for _, f := range tmpl.UpstreamFiles {
			if f.Path == "include/cutlass/detail/sm100_blockscaled_layout.hpp" && f.Line == 48 {
				foundLayout = true
			}
			if strings.HasPrefix(f.Path, "examples/89_sm103_fp4_ultra_gemm/") {
				foundExample = true
			}
		}
		if !foundLayout {
			t.Fatal("NVFP4 template missing the blockscaled layout pin sm100_blockscaled_layout.hpp:48")
		}
		if !foundExample {
			t.Fatal("NVFP4 template missing the sm103 fp4 ultra GEMM example pin")
		}
	})

	t.Run("upstream file record is complete and de-duplicated", func(t *testing.T) {
		files := CutlassStudyFiles()
		if len(files) < 10 {
			t.Fatalf("CutlassStudyFiles = %d files, want at least 10 distinct upstream pins", len(files))
		}
		seen := make(map[string]bool)
		for _, f := range files {
			key := f.Path + ":" + itoa(f.Line)
			if seen[key] {
				t.Fatalf("CutlassStudyFiles has duplicate pin %s", key)
			}
			seen[key] = true
			if f.Path == "" || f.Line <= 0 || f.Role == "" {
				t.Fatalf("CutlassStudyFiles has an incomplete pin %+v", f)
			}
		}
	})

	t.Run("fail-closed refusal on unsupported tile config", func(t *testing.T) {
		// A config inside the envelope must pass.
		if err := (NVFP4TileConfig{Arch: "sm103a", M: 128, N: 256, K: 768, SFVecSize: 16}).Validate(); err != nil {
			t.Fatalf("valid NVFP4 tile config refused: %v", err)
		}
		if err := (NVFP4TileConfig{Arch: "sm103a", M: 256, N: 128, K: 1536, SFVecSize: 16}).Validate(); err != nil {
			t.Fatalf("valid 2SM NVFP4 tile config refused: %v", err)
		}

		cases := []struct {
			name string
			cfg  NVFP4TileConfig
			want string // substring of the refusal reason
		}{
			{
				name: "sm100a is not the NVFP4 gate",
				cfg:  NVFP4TileConfig{Arch: "sm100a", M: 128, N: 256, K: 768, SFVecSize: 16},
				want: "arch sm100a",
			},
			{
				name: "MX SFVecSize 32 is a different family",
				cfg:  NVFP4TileConfig{Arch: "sm103a", M: 128, N: 256, K: 768, SFVecSize: 32},
				want: "SFVecSize 32",
			},
			{
				name: "M outside the pinned tile set",
				cfg:  NVFP4TileConfig{Arch: "sm103a", M: 64, N: 256, K: 768, SFVecSize: 16},
				want: "M 64",
			},
			{
				name: "N not a multiple of 32",
				cfg:  NVFP4TileConfig{Arch: "sm103a", M: 128, N: 100, K: 768, SFVecSize: 16},
				want: "N 100",
			},
			{
				name: "N above the pinned 256",
				cfg:  NVFP4TileConfig{Arch: "sm103a", M: 128, N: 512, K: 768, SFVecSize: 16},
				want: "N 512",
			},
			{
				name: "K not a multiple of 64",
				cfg:  NVFP4TileConfig{Arch: "sm103a", M: 128, N: 256, K: 100, SFVecSize: 16},
				want: "K 100",
			},
			{
				name: "zero K",
				cfg:  NVFP4TileConfig{Arch: "sm103a", M: 128, N: 256, K: 0, SFVecSize: 16},
				want: "K 0",
			},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				err := tc.cfg.Validate()
				if err == nil {
					t.Fatalf("config %+v accepted, want fail-closed refusal", tc.cfg)
				}
				if !IsCutlassStudyRefusal(err) {
					t.Fatalf("Validate returned %T, want CutlassStudyRefusal", err)
				}
				if !strings.Contains(err.Error(), tc.want) {
					t.Fatalf("refusal %q does not name %q", err.Error(), tc.want)
				}
			})
		}
	})
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
