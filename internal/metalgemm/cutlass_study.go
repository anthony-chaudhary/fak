package metalgemm

// cutlass_study.go is an INSPIRE-ONLY, host-side template-study model of the
// NVIDIA/cutlass tiling, collective-mainloop, and NVFP4 tensor-core kernel
// templates that fak's physical W4A16 decoder item and the NVFP4 device-queue
// family assume. It pins the exact upstream files (path:line at a fixed commit)
// that a native CUDA kernel would adapt, and it records the template shapes
// (tile-scheduler kinds, warp roles, mainloop/epilogue contracts, scale-factor
// chunk layouts, TMEM capacity, MMA tile shapes) as load-bearing Go data.
//
// This is a template study only: it ports no kernel, makes no device speed
// claim, touches no GPU, and uses no network or external dependency. Its value
// is that the shape of the upstream design is asserted by a host test
// (TestCutlassStudy) and fail-closed-validated for tile configs, so a future
// CUDA port cannot silently drift from the pinned upstream reference.
//
// Provenance: lifted from the deep study
// docs/research/STUDY-NVIDIA-CUTLASS-2026-11-06.md (fak-private), pinned at
// NVIDIA-EULA and INSPIRE-ONLY, so nothing here is copied from it.

// CutlassStudyPin is one pinned upstream reference: a file path and line in the
// pinned NVIDIA/cutlass tree that a native CUDA kernel would adapt.
type CutlassStudyPin struct {
	// Path is the repo-relative upstream path (e.g. "include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp").
	Path string
	// Line is the 1-based line number at the pinned commit.
	Line int
	// Role is the short mechanism this pin names (e.g. "CLC query pipeline").
	Role string
}

// TileSchedulerKind is the class of CUTLASS tile scheduler the study pins.
type TileSchedulerKind int

const (
	// TileSchedulerStaticPersistent is the sm90 static persistent scheduler:
	// raster order + L2 swizzle, no hardware launch control.
	TileSchedulerStaticPersistent TileSchedulerKind = iota
	// TileSchedulerDynamicPersistentCLC is the sm100 dynamic persistent
	// scheduler driven by the hardware Cluster Launch Control (CLC) query
	// pipeline.
	TileSchedulerDynamicPersistentCLC
	// TileSchedulerStreamK is the sm90 stream-K split scheduler with a
	// striped-reduce fixup and L2-warm split ordering.
	TileSchedulerStreamK
	// TileSchedulerGroup is the sm90 grouped-GEMM scheduler that caches two
	// problem shapes in shared memory.
	TileSchedulerGroup
)

// String names the scheduler the way a receipt or log would print it.
func (k TileSchedulerKind) String() string {
	switch k {
	case TileSchedulerStaticPersistent:
		return "static-persistent"
	case TileSchedulerDynamicPersistentCLC:
		return "dynamic-persistent-clc"
	case TileSchedulerStreamK:
		return "stream-k"
	case TileSchedulerGroup:
		return "group"
	default:
		return "unknown"
	}
}

// TileSchedulerPin pins one tile-scheduler template to its upstream file.
type TileSchedulerPin struct {
	Kind         TileSchedulerKind
	Arch         string // "sm90" or "sm100"
	Mechanism    string
	UpstreamFile CutlassStudyPin
}

// MainloopWarpRole is one warp role in the sm100 warp-specialized GEMM kernel.
type MainloopWarpRole int

const (
	// MainloopWarpRoleMMA issues the tcgen05 UMMA sequence from one elected thread.
	MainloopWarpRoleMMA MainloopWarpRole = iota
	// MainloopWarpRoleSched drives the tile scheduler.
	MainloopWarpRoleSched
	// MainloopWarpRoleMainloopLoad issues the TMA loads for A/B (and SFs).
	MainloopWarpRoleMainloopLoad
	// MainloopWarpRoleEpilogueLoad loads the epilogue inputs.
	MainloopWarpRoleEpilogueLoad
	// MainloopWarpRoleEpilogue runs the EVT epilogue and stores D.
	MainloopWarpRoleEpilogue
)

// String names the warp role.
func (r MainloopWarpRole) String() string {
	switch r {
	case MainloopWarpRoleMMA:
		return "MMA"
	case MainloopWarpRoleSched:
		return "Sched"
	case MainloopWarpRoleMainloopLoad:
		return "MainloopLoad"
	case MainloopWarpRoleEpilogueLoad:
		return "EpilogueLoad"
	case MainloopWarpRoleEpilogue:
		return "Epilogue"
	default:
		return "unknown"
	}
}

// CollectiveMainloopPin pins the collective mainloop + kernel shell template.
// The five reusable seams are: Arguments, static to_underlying_arguments
// lowering, SharedStorage union, thread-count policy, and scheduler selection
// via TileSchedulerSelector. The mainloop contract is
// prefetch_tma_descriptors/load_init/load/load_tail/mma/mma_tail; the epilogue
// contract is load/store/store_tail.
type CollectiveMainloopPin struct {
	Arch              string
	Pipeline          string // "PipelineTmaAsync" (sm90) or "PipelineUmmaAsync" (sm100)
	AccumulatorEngine string // "smem" (sm90 wgmma) or "tmem" (sm100 tcgen05)
	WarpRoles         []MainloopWarpRole
	MainloopContract  []string
	EpilogueContract  []string
	UpstreamFiles     []CutlassStudyPin
}

// NVFP4TensorCoreTemplate pins the sm103 blockscaled NVFP4 tensor-core kernel
// template: the scale-factor chunk layout, the TMEM capacity, and the MMA tile
// shapes. NVFP4 (SFVecSize 16, e4m3 scale factors) is gated to sm103a; the MX
// formats (SFVecSize 32, ue8m0) share the family but are a distinct gate.
type NVFP4TensorCoreTemplate struct {
	Arch          string // "sm103a"
	SFVecSize     int    // 16 weights per e4m3 scale factor
	SFFormat      string // "e4m3"
	SFChunkMN     int    // Blk_MN: 128
	SFChunkSF     int    // Blk_SF: 4 scale factors per 32-bit TMEM word
	SFPerTMEMWord int    // 4 (K-major atom packing)
	TMEMRows      int    // 128
	TMEMColumns   int    // 512
	TilesM        []int  // 1SM: 128, 2SM: 256
	TilesN        int    // 256
	TilesK        int    // 768
	UpstreamFiles []CutlassStudyPin
}

// W4A16DecoderTemplate pins the mixed-input W4A16 (int4/E2M1 weight x fp16
// activation) decode template: raw-packed TMA loads with dequant as an explicit
// consumer-warp step fused into the mainloop via a prmt/lop3 LUT. No extra
// pass, no gmem round-trip.
type W4A16DecoderTemplate struct {
	Arch             string
	DequantMechanism string
	UpstreamFiles    []CutlassStudyPin
}

// CutlassStudy is the assembled template-study model: the pinned upstream
// commit plus every template shape a native CUDA port would adapt.
type CutlassStudy struct {
	Repo       string
	Pin        string
	License    string
	Schedulers []TileSchedulerPin
	Mainloops  []CollectiveMainloopPin
	NVFP4      NVFP4TensorCoreTemplate
	W4A16      W4A16DecoderTemplate
}

// cutlassStudyPin is the pinned upstream commit for every reference in this
// model.
const cutlassStudyPin = "b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd"

// CutlassStudyModel returns the pinned template-study model. It is a pure
// literal: the shapes are the load-bearing datum, asserted by TestCutlassStudy.
func CutlassStudyModel() CutlassStudy {
	return CutlassStudy{
		Repo:    "NVIDIA/cutlass",
		Pin:     cutlassStudyPin,
		License: "BSD-3-Clause (include/ tree); python/CuTeDSL is NVIDIA-EULA, INSPIRE-ONLY",
		Schedulers: []TileSchedulerPin{
			{
				Kind:      TileSchedulerStaticPersistent,
				Arch:      "sm90",
				Mechanism: "static persistent: raster order + L2 swizzle",
				UpstreamFile: CutlassStudyPin{
					Path: "include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp",
					Line: 68,
					Role: "static persistent scheduler",
				},
			},
			{
				Kind:      TileSchedulerDynamicPersistentCLC,
				Arch:      "sm100",
				Mechanism: "dynamic persistent via hardware Cluster Launch Control",
				UpstreamFile: CutlassStudyPin{
					Path: "include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp",
					Line: 59,
					Role: "CLC dynamic persistent scheduler",
				},
			},
			{
				Kind:      TileSchedulerStreamK,
				Arch:      "sm90",
				Mechanism: "stream-K split + striped-reduce fixup, L2-warm split ordering",
				UpstreamFile: CutlassStudyPin{
					Path: "include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp",
					Line: 408,
					Role: "stream-K split",
				},
			},
			{
				Kind:      TileSchedulerGroup,
				Arch:      "sm90",
				Mechanism: "grouped-GEMM scheduler caching two problem shapes in smem",
				UpstreamFile: CutlassStudyPin{
					Path: "include/cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp",
					Line: 140,
					Role: "group scheduler smem shape cache",
				},
			},
		},
		Mainloops: []CollectiveMainloopPin{
			{
				Arch:              "sm90",
				Pipeline:          "PipelineTmaAsync",
				AccumulatorEngine: "smem",
				WarpRoles:         []MainloopWarpRole{}, // sm90 uses warp-groups, not the sm100 role set
				MainloopContract: []string{
					"prefetch_tma_descriptors",
					"load_init",
					"load",
					"load_tail",
					"mma",
					"mma_tail",
				},
				EpilogueContract: []string{"load", "store", "store_tail"},
				UpstreamFiles: []CutlassStudyPin{
					{Path: "include/cutlass/gemm/device/gemm_universal_adapter.h", Line: 122, Role: "GemmUniversalAdapter"},
					{Path: "include/cutlass/device_kernel.h", Line: 112, Role: "device_kernel<Operator> launch"},
					{Path: "include/cutlass/pipeline/sm90_pipeline.hpp", Line: 289, Role: "PipelineTmaAsync<Stages>"},
					{Path: "include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp", Line: 146, Role: "Arguments"},
					{Path: "include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp", Line: 114, Role: "TileSchedulerSelector"},
				},
			},
			{
				Arch:              "sm100",
				Pipeline:          "PipelineUmmaAsync",
				AccumulatorEngine: "tmem",
				WarpRoles: []MainloopWarpRole{
					MainloopWarpRoleMMA,
					MainloopWarpRoleSched,
					MainloopWarpRoleMainloopLoad,
					MainloopWarpRoleEpilogueLoad,
					MainloopWarpRoleEpilogue,
				},
				MainloopContract: []string{
					"prefetch_tma_descriptors",
					"load_init",
					"load",
					"load_tail",
					"mma",
					"mma_tail",
				},
				EpilogueContract: []string{"load", "store", "store_tail"},
				UpstreamFiles: []CutlassStudyPin{
					{Path: "include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp", Line: 676, Role: "sm100 warp-specialized mainloop"},
					{Path: "include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp", Line: 169, Role: "PipelineUmmaAsync accumulator handoff"},
					{Path: "include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp", Line: 234, Role: "warp role set"},
					{Path: "include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp", Line: 791, Role: "TMEM dealloc handshake"},
				},
			},
		},
		NVFP4: NVFP4TensorCoreTemplate{
			Arch:          "sm103a",
			SFVecSize:     16,
			SFFormat:      "e4m3",
			SFChunkMN:     128,
			SFChunkSF:     4,
			SFPerTMEMWord: 4,
			TMEMRows:      128,
			TMEMColumns:   512,
			TilesM:        []int{128, 256},
			TilesN:        256,
			TilesK:        768,
			UpstreamFiles: []CutlassStudyPin{
				{Path: "include/cutlass/detail/sm100_blockscaled_layout.hpp", Line: 48, Role: "SF chunk layout Blk_MN=128 Blk_SF=4"},
				{Path: "include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp", Line: 1018, Role: "smem->TMEM UTCCP"},
				{Path: "include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp", Line: 1029, Role: "blockscaled UMMA issue"},
				{Path: "include/cute/arch/tmem_capacity_sm100.hpp", Line: 37, Role: "TMEM 128x512 capacity"},
				{Path: "examples/89_sm103_fp4_ultra_gemm/89_sm103_fp4_ultra_gemm.cu", Line: 141, Role: "sm103 NVFP4 ultra GEMM example"},
			},
		},
		W4A16: W4A16DecoderTemplate{
			Arch:             "sm90",
			DequantMechanism: "raw-packed TMA load + consumer-warp dequant via prmt/lop3 LUT fused in-mainloop",
			UpstreamFiles: []CutlassStudyPin{
				{Path: "include/cutlass/detail/collective/mixed_input_utils.hpp", Line: 832, Role: "mixed-input dequant consumer step"},
				{Path: "include/cutlass/detail/collective/mixed_input_utils.hpp", Line: 794, Role: "int4/E2M1 prmt/lop3 LUT"},
			},
		},
	}
}

// PinnedSchedulerFor returns the pinned tile-scheduler template of the given
// kind, or false if the kind is unknown. It is the lookup a CUDA port uses to
// find the exact upstream file to adapt.
func PinnedSchedulerFor(kind TileSchedulerKind) (TileSchedulerPin, bool) {
	for _, s := range CutlassStudyModel().Schedulers {
		if s.Kind == kind {
			return s, true
		}
	}
	return TileSchedulerPin{}, false
}

// PinnedMainloopFor returns the pinned collective mainloop template for the
// given arch ("sm90" or "sm100"), or false if the arch is unknown.
func PinnedMainloopFor(arch string) (CollectiveMainloopPin, bool) {
	for _, m := range CutlassStudyModel().Mainloops {
		if m.Arch == arch {
			return m, true
		}
	}
	return CollectiveMainloopPin{}, false
}

// CutlassStudyFiles returns the full, de-duplicated list of upstream files a
// native CUDA kernel would adapt, in a stable order. It is the "exact upstream
// files" record the ticket asks for.
func CutlassStudyFiles() []CutlassStudyPin {
	seen := make(map[string]bool)
	var out []CutlassStudyPin
	add := func(p CutlassStudyPin) {
		key := p.Path + ":" + itoa(p.Line)
		if !seen[key] {
			seen[key] = true
			out = append(out, p)
		}
	}
	model := CutlassStudyModel()
	for _, s := range model.Schedulers {
		add(s.UpstreamFile)
	}
	for _, m := range model.Mainloops {
		for _, f := range m.UpstreamFiles {
			add(f)
		}
	}
	for _, f := range model.NVFP4.UpstreamFiles {
		add(f)
	}
	for _, f := range model.W4A16.UpstreamFiles {
		add(f)
	}
	return out
}

// itoa is a tiny int->string helper so CutlassStudyFiles stays stdlib-only and
// avoids importing strconv for a single use.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}

// NVFP4TileConfig is a candidate NVFP4 GEMM tile a native CUDA kernel would
// instantiate. Validate fail-closes it against the pinned sm103a template.
type NVFP4TileConfig struct {
	Arch      string
	M, N, K   int
	SFVecSize int
}

// CutlassStudyRefusal is the fail-closed error returned when an NVFP4 tile
// config is outside the pinned sm103a template envelope.
type CutlassStudyRefusal struct {
	Reason string
}

// Error implements error.
func (e CutlassStudyRefusal) Error() string { return "cutlass study: " + e.Reason }

// IsCutlassStudyRefusal reports whether err is a CutlassStudyRefusal.
func IsCutlassStudyRefusal(err error) bool {
	_, ok := err.(CutlassStudyRefusal)
	return ok
}

// Validate fail-closes an NVFP4 tile config against the pinned sm103a
// template. It returns a CutlassStudyRefusal naming the first violated
// constraint, or nil if the config is inside the envelope. The constraints are
// the arch gate (NVFP4 is sm103a, not sm100a), the M tile set (1SM=128,
// 2SM=256), the N multiple-of-32 bound, the K multiple-of-64 bound (4 SFs per
// 32-bit TMEM word x SFVecSize 16 = 64 K elements per word), and the SFVecSize
// (16 for NVFP4; 32 is the MX family and is refused here).
func (c NVFP4TileConfig) Validate() error {
	tmpl := CutlassStudyModel().NVFP4
	if c.Arch != tmpl.Arch {
		return CutlassStudyRefusal{Reason: "arch " + c.Arch + " is not the NVFP4 gate " + tmpl.Arch}
	}
	if c.SFVecSize != tmpl.SFVecSize {
		return CutlassStudyRefusal{Reason: "SFVecSize " + itoa(c.SFVecSize) + " is not NVFP4 " + itoa(tmpl.SFVecSize) + " (32 is the MX family)"}
	}
	mOK := false
	for _, m := range tmpl.TilesM {
		if c.M == m {
			mOK = true
			break
		}
	}
	if !mOK {
		return CutlassStudyRefusal{Reason: "M " + itoa(c.M) + " is not in the pinned tile set " + intSlice(tmpl.TilesM)}
	}
	if c.N <= 0 || c.N%32 != 0 || c.N > tmpl.TilesN {
		return CutlassStudyRefusal{Reason: "N " + itoa(c.N) + " must be a positive multiple of 32 and <= " + itoa(tmpl.TilesN)}
	}
	if c.K <= 0 || c.K%64 != 0 {
		return CutlassStudyRefusal{Reason: "K " + itoa(c.K) + " must be a positive multiple of 64 (4 SFs x SFVecSize 16 per 32-bit TMEM word)"}
	}
	return nil
}

// intSlice renders a []int the way a refusal message wants it ("[128 256]").
func intSlice(v []int) string {
	if len(v) == 0 {
		return "[]"
	}
	out := "["
	for i, n := range v {
		if i > 0 {
			out += " "
		}
		out += itoa(n)
	}
	return out + "]"
}
