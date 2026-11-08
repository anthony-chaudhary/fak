// Package hwspecs is the public, importable inventory of cited hardware
// specifications for the accelerator classes the fleet uses or rents: one row
// per device (H100, H200, B200, B300, MI300X, MI355X, RTX PRO Blackwell,
// GeForce, A100, L4, Strix Halo, ...), each carrying the physical detail a
// serving planner needs from first principles — memory capacity and type,
// memory bandwidth, dense low-precision tensor compute, board power, scale-up
// bandwidth, and the CUDA compute capability or AMD ISA that identifies it —
// with the source it was read from and the access date.
//
// Specs are facts read from vendor product pages and datasheets on the access
// date recorded in each row's Source; nothing in this package is a
// measurement. Compute rates are DENSE (no 2:4 structured sparsity); where a
// vendor publishes only a sparse figure the row says so and carries dense =
// sparse ÷ 2. A zero rate or power field means "not cited", never zero
// hardware capability.
//
// Prices are deliberately NOT here: dollar figures stay in fak-private
// (platform/finops hw-prices tables, cmd/fak-tco), which joins this catalog by
// Key to attach cited price rows. This package is the single physical-spec
// source both repositories read.
package hwspecs

import (
	"sort"
	"strings"
	"sync"
)

// Schema is the frozen wire version of the catalog JSON. Changing any field
// tag or meaning below is a contract change that requires a new schema value.
const Schema = "fak.hwspecs.v1"

// DeviceSpec is one accelerator's cited physical detail.
type DeviceSpec struct {
	// Key is the stable catalog key, shared with fak-private cmd/fak-tco
	// (which joins cited price rows by it).
	Key string `json:"key"`
	// Name is the canonical display name.
	Name string `json:"name"`
	// Vendor is the closed vendor vocabulary: nvidia or amd.
	Vendor string `json:"vendor"`
	// Architecture is the vendor marketing generation (Hopper, Blackwell,
	// Blackwell Ultra, Ampere, Ada, CDNA3, CDNA4, RDNA 3.5).
	Architecture string `json:"architecture"`
	// GPUNames are the device-name spellings telemetry (nvidia-smi, rocm-smi,
	// vulkan, benchmark receipts) has reported for this device; MatchGPUName
	// resolves them.
	GPUNames []string `json:"gpu_names,omitempty"`
	// Aliases are short human spellings accepted by Lookup beyond the key and
	// GPUNames.
	Aliases []string `json:"aliases,omitempty"`
	// CUDAComputeMajor/Minor identify the CUDA compute capability (sm_XY);
	// both zero means not applicable (AMD) or not cited.
	CUDAComputeMajor int `json:"cuda_compute_major,omitempty"`
	CUDAComputeMinor int `json:"cuda_compute_minor,omitempty"`
	// GPUArch is the AMD ISA (gfx942, gfx950, gfx1151, gfx1102); empty on
	// NVIDIA rows.
	GPUArch string `json:"gpu_arch,omitempty"`
	// VRAMGiB is the memory capacity as the vendor lists it (e.g. H200 141),
	// carried as GiB per the fak-tco VRAMGiB convention; vendor "GB" listings
	// are treated as GiB. For the unified Strix Halo box this is the
	// configurable system memory.
	VRAMGiB float64 `json:"vram_gib"`
	// MemoryType is the cited memory technology (HBM3, HBM3E, GDDR7, GDDR6,
	// LPDDR5X).
	MemoryType string `json:"memory_type"`
	// BandwidthGBps is the cited peak memory bandwidth.
	BandwidthGBps float64 `json:"bandwidth_gbps"`
	// FP8TFLOPSDense, FP16TFLOPSDense and FP4TFLOPSDense are dense tensor
	// rates at the inference precisions; 0 means not cited. The roofline
	// prefill prefers FP8, then FP16.
	FP8TFLOPSDense  float64 `json:"fp8_tflops_dense,omitempty"`
	FP16TFLOPSDense float64 `json:"fp16_tflops_dense,omitempty"`
	FP4TFLOPSDense  float64 `json:"fp4_tflops_dense,omitempty"`
	// PowerWatts is the cited board TDP/TBP (or TGP ceiling on laptop parts);
	// 0 means not cited.
	PowerWatts float64 `json:"power_watts,omitempty"`
	// NVLinkGBps is the cited per-GPU scale-up bandwidth (NVIDIA NVLink or
	// AMD Infinity Fabric aggregate); 0 means not cited / not applicable.
	NVLinkGBps float64 `json:"nvlink_gbps,omitempty"`
	// Source cites where the numbers were read and when. Every row carries
	// one; a row without a citation is a catalog bug.
	Source string `json:"source"`
}

// catalogDate is the access date cited by every row; a row refreshed on a
// later date carries its own date in Source instead.
const catalogDate = "accessed 2026-11-07"

// catalog is the cited inventory. Dense figures derived from a cited sparse
// figure say so in the row's Source.
func catalog() []DeviceSpec {
	return []DeviceSpec{
		{
			Key: "a100-80gb", Name: "NVIDIA A100 80GB SXM4", Vendor: "nvidia", Architecture: "Ampere",
			GPUNames:         []string{"NVIDIA A100 80GB SXM4", "A100 80GB SXM4", "NVIDIA A100 80GB PCIe", "NVIDIA A100 80GB"},
			Aliases:          []string{"a100 80gb", "a100"},
			CUDAComputeMajor: 8, CUDAComputeMinor: 0,
			VRAMGiB: 80, MemoryType: "HBM2e", BandwidthGBps: 2039,
			FP16TFLOPSDense: 312, PowerWatts: 400, NVLinkGBps: 600,
			Source: "nvidia.com A100 datasheet (r4): FP16/BF16 Tensor 312 TFLOPS dense (624 with sparsity), 80GB HBM2e, 2039 GB/s (SXM), 400 W, NVLink 600 GB/s; " + catalogDate,
		},
		{
			Key: "a100-40gb", Name: "NVIDIA A100 40GB", Vendor: "nvidia", Architecture: "Ampere",
			GPUNames:         []string{"NVIDIA A100 40GB", "NVIDIA A100 40GB (Ampere)", "A100 40GB PCIe", "A100 40GB SXM"},
			Aliases:          []string{"a100 40gb"},
			CUDAComputeMajor: 8, CUDAComputeMinor: 0,
			VRAMGiB: 40, MemoryType: "HBM2", BandwidthGBps: 1555,
			FP16TFLOPSDense: 312, PowerWatts: 400, NVLinkGBps: 600,
			Source: "nvidia.com A100 datasheet (r4): FP16/BF16 Tensor 312 TFLOPS dense (624 with sparsity), 40GB HBM2, 1555 GB/s, 400 W, NVLink 600 GB/s; " + catalogDate,
		},
		{
			Key: "a800-80gb", Name: "NVIDIA A800 80GB", Vendor: "nvidia", Architecture: "Ampere",
			GPUNames:         []string{"NVIDIA A800 GPU", "NVIDIA A800 80GB PCIe", "NVIDIA A800 SXM"},
			Aliases:          []string{"a800"},
			CUDAComputeMajor: 8, CUDAComputeMinor: 0,
			VRAMGiB: 80, MemoryType: "HBM2e", BandwidthGBps: 2039,
			FP16TFLOPSDense: 312, PowerWatts: 400,
			Source: "NVIDIA A800 product page (China-market A100 variant, same GA100 specs: 80GB HBM2e, 2039 GB/s, 400 W, FP16 tensor dense 312 TFLOPS); " + catalogDate,
		},
		{
			Key: "a10g", Name: "NVIDIA A10G", Vendor: "nvidia", Architecture: "Ampere",
			GPUNames:         []string{"NVIDIA A10G", "NVIDIA A10G 24GB"},
			Aliases:          []string{"a10g"},
			CUDAComputeMajor: 8, CUDAComputeMinor: 6,
			VRAMGiB: 24, MemoryType: "GDDR6", BandwidthGBps: 600,
			FP16TFLOPSDense: 62.5, PowerWatts: 300,
			Source: "NVIDIA A10G product page (FP16/BF16 Tensor 125 TFLOPS with sparsity -> dense 62.5; 24GB GDDR6, 600 GB/s, 300 W); " + catalogDate,
		},
		{
			Key: "l4", Name: "NVIDIA L4", Vendor: "nvidia", Architecture: "Ada",
			GPUNames:         []string{"NVIDIA L4", "NVIDIA L4 (Ada)"},
			CUDAComputeMajor: 8, CUDAComputeMinor: 9,
			VRAMGiB: 24, MemoryType: "GDDR6", BandwidthGBps: 300,
			FP8TFLOPSDense: 242, FP16TFLOPSDense: 121, PowerWatts: 72,
			Source: "nvidia.com/en-us/data-center/l4 datasheet: FP8 Tensor 242 TFLOPS, FP16 121 TFLOPS, 24GB GDDR6, 300 GB/s, 72 W; " + catalogDate,
		},
		{
			Key: "rtx4070-laptop", Name: "NVIDIA GeForce RTX 4070 Laptop GPU", Vendor: "nvidia", Architecture: "Ada",
			GPUNames:         []string{"NVIDIA GeForce RTX 4070 Laptop GPU"},
			Aliases:          []string{"rtx 4070 laptop", "4070 laptop"},
			CUDAComputeMajor: 8, CUDAComputeMinor: 9,
			VRAMGiB: 8, MemoryType: "GDDR6", BandwidthGBps: 272,
			PowerWatts: 100,
			Source:     "nvidia.com GeForce RTX 4070 Laptop GPU page: 8GB GDDR6, 192-bit, 272 GB/s, TGP up to 100 W (35-100 W configurable); tensor rates not cited on the page; " + catalogDate,
		},
		{
			Key: "rtx5090", Name: "NVIDIA GeForce RTX 5090", Vendor: "nvidia", Architecture: "Blackwell",
			GPUNames:         []string{"NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 5090 32GB", "NVIDIA GeForce RTX 5090 GPU"},
			Aliases:          []string{"rtx 5090", "5090"},
			CUDAComputeMajor: 12, CUDAComputeMinor: 0,
			VRAMGiB: 32, MemoryType: "GDDR7", BandwidthGBps: 1792,
			FP8TFLOPSDense: 838.4, FP16TFLOPSDense: 419.2, FP4TFLOPSDense: 1676.8, PowerWatts: 575,
			Source: "nvidia.com/en-us/geforce/rtx-5090 (AI TOPS 3352 = FP4 tensor sparse; dense = sparse / 2 per bit-depth scaling: FP4 1676, FP8 838.4, FP16 419.2 TFLOPS); 32GB GDDR7, 512-bit, 1792 GB/s, 575 W; " + catalogDate,
		},
		{
			Key: "rtx5090-laptop", Name: "NVIDIA GeForce RTX 5090 Laptop GPU", Vendor: "nvidia", Architecture: "Blackwell",
			GPUNames:         []string{"NVIDIA GeForce RTX 5090 Laptop GPU", "NVIDIA GeForce RTX 5090 Laptop GPU 24 GB"},
			Aliases:          []string{"rtx 5090 laptop", "5090 laptop"},
			CUDAComputeMajor: 12, CUDAComputeMinor: 0,
			VRAMGiB: 24, MemoryType: "GDDR7", BandwidthGBps: 896,
			PowerWatts: 150,
			Source:     "nvidia.com GeForce RTX 5090 Laptop GPU page: 24GB GDDR7, 256-bit, 896 GB/s, TGP up to 150 W; tensor rates not cited on the page; " + catalogDate,
		},
		{
			Key: "rtx6000pro", Name: "NVIDIA RTX PRO 6000 Blackwell (workstation)", Vendor: "nvidia", Architecture: "Blackwell",
			GPUNames:         []string{"NVIDIA RTX PRO 6000 Blackwell", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "RTX PRO 6000 Blackwell Workstation Edition"},
			Aliases:          []string{"rtx pro 6000", "rtx6000pro"},
			CUDAComputeMajor: 12, CUDAComputeMinor: 0,
			VRAMGiB: 96, MemoryType: "GDDR7", BandwidthGBps: 1792,
			FP8TFLOPSDense: 1007.6, FP16TFLOPSDense: 503.8, FP4TFLOPSDense: 2015.2, PowerWatts: 600,
			Source: "NVIDIA RTX Blackwell PRO GPU architecture whitepaper v1.0 (GB202: Peak FP8 Tensor 1007.6 dense / 2015.2 sparse, FP16 503.8/1007.6, FP4 2015.2/4030.4 TFLOPS); 96GB GDDR7 ECC, 512-bit, 1792 GB/s, 600 W; " + catalogDate,
		},
		{
			Key: "rtxpro5000", Name: "NVIDIA RTX PRO 5000 Blackwell", Vendor: "nvidia", Architecture: "Blackwell",
			GPUNames:         []string{"NVIDIA RTX PRO 5000 Blackwell 48GB", "NVIDIA RTX PRO 5000 Blackwell"},
			CUDAComputeMajor: 12, CUDAComputeMinor: 0,
			VRAMGiB: 48, MemoryType: "GDDR7", BandwidthGBps: 1344,
			PowerWatts: 300,
			Source:     "nvidia.com RTX PRO 5000 Blackwell page + published partner spec tables (48GB GDDR7 ECC, 384-bit, 1344 GB/s, 300 W); tensor rates not cited here; " + catalogDate,
		},
		{
			Key: "rtxpro4500", Name: "NVIDIA RTX PRO 4500 Blackwell", Vendor: "nvidia", Architecture: "Blackwell",
			GPUNames:         []string{"NVIDIA RTX PRO 4500 Blackwell 32GB", "NVIDIA RTX PRO 4500 Blackwell"},
			CUDAComputeMajor: 12, CUDAComputeMinor: 0,
			VRAMGiB: 32, MemoryType: "GDDR7", BandwidthGBps: 896,
			PowerWatts: 200,
			Source:     "nvidia.com RTX PRO 4500 Blackwell page + published partner spec tables (32GB GDDR7 ECC, 256-bit, 896 GB/s, 200 W); tensor rates not cited here; " + catalogDate,
		},
		{
			Key: "rtxpro4000", Name: "NVIDIA RTX PRO 4000 Blackwell", Vendor: "nvidia", Architecture: "Blackwell",
			GPUNames:         []string{"NVIDIA RTX PRO 4000 Blackwell 24GB", "NVIDIA RTX PRO 4000 Blackwell"},
			CUDAComputeMajor: 12, CUDAComputeMinor: 0,
			VRAMGiB: 24, MemoryType: "GDDR7", BandwidthGBps: 672,
			PowerWatts: 140,
			Source:     "nvidia.com RTX PRO 4000 Blackwell page + published partner spec tables (24GB GDDR7 ECC, 192-bit, 672 GB/s, 140 W); tensor rates not cited here; " + catalogDate,
		},
		{
			Key: "h100", Name: "NVIDIA H100 SXM", Vendor: "nvidia", Architecture: "Hopper",
			GPUNames:         []string{"NVIDIA H100 SXM", "NVIDIA H100 80GB HBM3", "H100 SXM5", "NVIDIA H100 80GB"},
			Aliases:          []string{"h100", "h100 sxm"},
			CUDAComputeMajor: 9, CUDAComputeMinor: 0,
			VRAMGiB: 80, MemoryType: "HBM3", BandwidthGBps: 3352,
			FP8TFLOPSDense: 1978.9, FP16TFLOPSDense: 989.4, PowerWatts: 700, NVLinkGBps: 900,
			Source: "nvidia.com/en-us/data-center/h100 product table (H100 SXM: FP8 3,958 / BF16 1,979 TFLOPS with sparsity -> dense / 2: 1978.9 / 989.4); 80GB HBM3, 3.35 TB/s, 700 W, NVLink 900 GB/s; " + catalogDate,
		},
		{
			Key: "h100-nvl", Name: "NVIDIA H100 NVL (PCIe)", Vendor: "nvidia", Architecture: "Hopper",
			GPUNames:         []string{"NVIDIA H100 NVL", "H100 NVL"},
			Aliases:          []string{"h100 nvl"},
			CUDAComputeMajor: 9, CUDAComputeMinor: 0,
			VRAMGiB: 94, MemoryType: "HBM3", BandwidthGBps: 3900,
			FP8TFLOPSDense: 1670.5, FP16TFLOPSDense: 835.5, PowerWatts: 400,
			Source: "nvidia.com/en-us/data-center/h100 H100 NVL column (FP8 3,341 / BF16 1,671 TFLOPS with sparsity -> dense / 2); 94GB HBM3, 3.9 TB/s, 350-400 W configurable; " + catalogDate,
		},
		{
			Key: "h100-pcie", Name: "NVIDIA H100 PCIe", Vendor: "nvidia", Architecture: "Hopper",
			GPUNames:         []string{"NVIDIA H100 PCIe", "NVIDIA H100 80GB PCIe", "H100 PCIe"},
			Aliases:          []string{"h100 pcie"},
			CUDAComputeMajor: 9, CUDAComputeMinor: 0,
			VRAMGiB: 80, MemoryType: "HBM3", BandwidthGBps: 2039,
			FP8TFLOPSDense: 1513, FP16TFLOPSDense: 756.5, PowerWatts: 350,
			Source: "NVIDIA H100 PCIe datasheet (FP8 3,026 / FP16 1,513 TFLOPS with sparsity -> dense / 2); 80GB HBM3, 2.0 TB/s, 350 W; " + catalogDate,
		},
		{
			Key: "h200", Name: "NVIDIA H200 SXM", Vendor: "nvidia", Architecture: "Hopper",
			GPUNames:         []string{"NVIDIA H200 SXM", "H200 SXM5", "NVIDIA H200 143GB"},
			Aliases:          []string{"h200", "h200 sxm"},
			CUDAComputeMajor: 9, CUDAComputeMinor: 0,
			VRAMGiB: 141, MemoryType: "HBM3E", BandwidthGBps: 4800,
			FP8TFLOPSDense: 1978.9, FP16TFLOPSDense: 989.4, PowerWatts: 700, NVLinkGBps: 900,
			Source: "nvidia.com/en-us/data-center/h200 (H200 SXM: FP8 3,958 / BF16 1,979 TFLOPS with sparsity -> dense / 2; 141GB HBM3e, 4.8 TB/s, up to 700 W, NVLink 900 GB/s); " + catalogDate,
		},
		{
			Key: "h200-nvl", Name: "NVIDIA H200 NVL (PCIe)", Vendor: "nvidia", Architecture: "Hopper",
			GPUNames:         []string{"NVIDIA H200 NVL 141GB PCIe", "NVIDIA H200 NVL", "H200 NVL"},
			Aliases:          []string{"h200 nvl"},
			CUDAComputeMajor: 9, CUDAComputeMinor: 0,
			VRAMGiB: 141, MemoryType: "HBM3E", BandwidthGBps: 4800,
			FP8TFLOPSDense: 1670.5, FP16TFLOPSDense: 835.5, PowerWatts: 600,
			Source: "nvidia.com/en-us/data-center/h200 H200 NVL column (FP8 3,341 / BF16 1,671 TFLOPS with sparsity -> dense / 2); 141GB HBM3e, 4.8 TB/s, up to 600 W; " + catalogDate,
		},
		{
			Key: "b200", Name: "NVIDIA B200 SXM", Vendor: "nvidia", Architecture: "Blackwell",
			GPUNames:         []string{"NVIDIA B200 SXM", "B200 SXM6", "NVIDIA B200"},
			Aliases:          []string{"b200", "b200 sxm"},
			CUDAComputeMajor: 10, CUDAComputeMinor: 0,
			VRAMGiB: 180, MemoryType: "HBM3E", BandwidthGBps: 8000,
			FP8TFLOPSDense: 4500, FP16TFLOPSDense: 2250, FP4TFLOPSDense: 9000, PowerWatts: 1000, NVLinkGBps: 1800,
			Source: "nvidia.com/en-us/data-center/hgx HGX B200 node table (FP8/FP6 72 PFLOPS sparse, FP16/BF16 36 PFLOPS sparse, FP4 108 dense | 144 sparse per 8-GPU node -> per-GPU dense / 8: 4500 / 2250 / 9000); 180GB HBM3E, 8 TB/s, up to 1,000 W, NVLink 5 1.8 TB/s; SemiAnalysis InferenceX chip table (B200 SXM dense FP8 4,500, BF16 2,250); " + catalogDate,
		},
		{
			Key: "b300", Name: "NVIDIA B300 SXM (Blackwell Ultra)", Vendor: "nvidia", Architecture: "Blackwell Ultra",
			GPUNames:         []string{"NVIDIA B300 SXM", "NVIDIA B300", "B300"},
			Aliases:          []string{"b300", "b300 sxm", "blackwell ultra"},
			CUDAComputeMajor: 10, CUDAComputeMinor: 3,
			VRAMGiB: 288, MemoryType: "HBM3E", BandwidthGBps: 8000,
			FP8TFLOPSDense: 4500, FP16TFLOPSDense: 2250, FP4TFLOPSDense: 13500, PowerWatts: 1400, NVLinkGBps: 1800,
			Source: "nvidia.com/en-us/data-center/hgx HGX B300 node table (FP8/FP6 72 PFLOPS sparse, FP16/BF16 36 PFLOPS sparse, FP4 144 sparse | 108 dense per 8-GPU node -> per-GPU dense / 8: 4500 / 2250 / 13500); 288GB HBM3E, 8 TB/s, TDP up to 1,400 W per the Blackwell Ultra datasheet (HGX OEM configs commonly 1,100 W), NVLink 5 1.8 TB/s; SemiAnalysis InferenceX chip table (B300 SXM dense FP8 4,500, BF16 2,250, FP4 13,500); " + catalogDate,
		},
		{
			Key: "mi300x", Name: "AMD Instinct MI300X", Vendor: "amd", Architecture: "CDNA3",
			GPUNames: []string{"AMD Instinct MI300X", "MI300X OAM"},
			Aliases:  []string{"mi300x"},
			GPUArch:  "gfx942",
			VRAMGiB:  192, MemoryType: "HBM3", BandwidthGBps: 5300,
			FP8TFLOPSDense: 2614.9, FP16TFLOPSDense: 1307.4, PowerWatts: 750, NVLinkGBps: 448,
			Source: "amd.com Instinct MI300X page + datasheet (FP8 2.61 PFLOPs dense / 5.22 with sparsity, FP16 1.3 PFLOPs dense; 192GB HBM3, 5.3 TB/s, 750 W TBP, gfx942; scale-up Infinity Fabric 448 GB/s); " + catalogDate,
		},
		{
			Key: "mi355x", Name: "AMD Instinct MI355X", Vendor: "amd", Architecture: "CDNA4",
			GPUNames: []string{"AMD Instinct MI355X", "MI355X OAM"},
			Aliases:  []string{"mi355x"},
			GPUArch:  "gfx950",
			VRAMGiB:  288, MemoryType: "HBM3E", BandwidthGBps: 8000,
			FP8TFLOPSDense: 5033.2, FP16TFLOPSDense: 2516.6, FP4TFLOPSDense: 10066.3, PowerWatts: 1400, NVLinkGBps: 538,
			Source: "amd.com Instinct MI355X page + brochure (OCP-FP8 5.0332 PFLOPs dense / 10.0664 with sparsity, FP16/BF16 2.5166 PFLOPs dense, MXFP4 10.0663 PFLOPs; 288GB HBM3E, 8 TB/s, 1400 W TBP, gfx950; scale-up Infinity Fabric 538 GB/s); " + catalogDate,
		},
		{
			Key: "rx7600", Name: "AMD Radeon RX 7600", Vendor: "amd", Architecture: "RDNA 3",
			GPUNames: []string{"AMD Radeon RX 7600", "AMD Radeon RX 7600 (RADV GFX1102)"},
			Aliases:  []string{"rx 7600"},
			GPUArch:  "gfx1102",
			VRAMGiB:  8, MemoryType: "GDDR6", BandwidthGBps: 288,
			PowerWatts: 165,
			Source:     "amd.com Radeon RX 7600 page (8GB GDDR6, 128-bit, 288 GB/s, 165 W board power, gfx1102); matrix rates not cited here; " + catalogDate,
		},
		{
			Key: "strix-halo", Name: "AMD Ryzen AI Max+ 395 (Strix Halo)", Vendor: "amd", Architecture: "RDNA 3.5",
			GPUNames: []string{"AMD Radeon 8060S Graphics", "AMD Ryzen AI Max+ 395"},
			Aliases:  []string{"strix halo", "strix-halo", "ryzen ai max+ 395"},
			GPUArch:  "gfx1151",
			VRAMGiB:  128, MemoryType: "LPDDR5X", BandwidthGBps: 273.056,
			FP16TFLOPSDense: 59.4, PowerWatts: 135,
			Source: "LPDDR5X-8533 theoretical 273.056 GB/s and WMMA BF16 peak 59.4 TFLOPS per fak platform/strix roofline constants; 135 W whole-box legacy paid-path draw; unified 128 GB (aligned with fak-private cmd/fak-tco hw_specs.go); " + catalogDate,
		},
	}
}

// normalize folds a card spelling the way catalog keys are spelled: lowercase,
// every non-alphanumeric run dropped.
func normalize(v string) string {
	v = strings.ToLower(v)
	var b strings.Builder
	for _, r := range v {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '+' {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// aliasEntry is one normalized spelling mapped to its catalog key.
type aliasEntry struct {
	normalized string
	key        string
}

var (
	aliasOnce  sync.Once
	aliasIndex []aliasEntry // sorted by normalized length descending
	specIndex  map[string]DeviceSpec
)

func buildIndex() {
	aliasOnce.Do(func() {
		specIndex = map[string]DeviceSpec{}
		seen := map[string]bool{}
		for _, sp := range catalog() {
			if seen[sp.Key] {
				continue
			}
			seen[sp.Key] = true
			specIndex[normalize(sp.Key)] = sp
			normKey := normalize(sp.Key)
			add := func(spellings []string) {
				for _, s := range spellings {
					n := normalize(s)
					if n == "" {
						continue
					}
					aliasIndex = append(aliasIndex, aliasEntry{normalized: n, key: normKey})
				}
			}
			add([]string{sp.Key, sp.Name})
			add(sp.GPUNames)
			add(sp.Aliases)
		}
		sort.Slice(aliasIndex, func(i, j int) bool {
			if len(aliasIndex[i].normalized) != len(aliasIndex[j].normalized) {
				return len(aliasIndex[i].normalized) > len(aliasIndex[j].normalized)
			}
			if aliasIndex[i].normalized != aliasIndex[j].normalized {
				return aliasIndex[i].normalized < aliasIndex[j].normalized
			}
			return aliasIndex[i].key < aliasIndex[j].key
		})
	})
}

// All returns the catalog rows in catalog order. The slice is a copy; callers
// cannot mutate the catalog through it.
func All() []DeviceSpec {
	out := make([]DeviceSpec, 0, len(catalog()))
	seen := map[string]bool{}
	for _, sp := range catalog() {
		if seen[sp.Key] {
			continue
		}
		seen[sp.Key] = true
		out = append(out, sp)
	}
	return out
}

// Keys lists the catalog keys in catalog order.
func Keys() []string {
	out := make([]string, 0, len(catalog()))
	seen := map[string]bool{}
	for _, sp := range catalog() {
		if seen[sp.Key] {
			continue
		}
		seen[sp.Key] = true
		out = append(out, sp.Key)
	}
	return out
}

// ByKey resolves an exact catalog key (any spelling of it) to its row.
func ByKey(key string) (DeviceSpec, bool) {
	buildIndex()
	sp, ok := specIndex[normalize(key)]
	return sp, ok
}

// Lookup resolves any human spelling of a card — a catalog key, an alias, or a
// telemetry device name — to its row. Exact spellings win; otherwise the
// longest alias contained in the normalized query wins (so "NVIDIA H200 NVL
// 141GB PCIe" resolves to h200-nvl, not h200). Ok is false for unknown cards.
func Lookup(query string) (DeviceSpec, bool) {
	buildIndex()
	q := normalize(query)
	if q == "" {
		return DeviceSpec{}, false
	}
	if sp, ok := specIndex[q]; ok {
		return sp, true
	}
	for _, e := range aliasIndex {
		if e.normalized == q {
			if sp, ok := specIndex[e.key]; ok {
				return sp, true
			}
		}
	}
	for _, e := range aliasIndex {
		if strings.Contains(q, e.normalized) {
			if sp, ok := specIndex[e.key]; ok {
				return sp, true
			}
		}
	}
	return DeviceSpec{}, false
}

// MatchGPUName resolves a live telemetry device name (nvidia-smi, rocm-smi,
// vulkan, a benchmark receipt) to its catalog row. It is Lookup with a name
// that documents the caller's intent.
func MatchGPUName(name string) (DeviceSpec, bool) {
	return Lookup(name)
}
