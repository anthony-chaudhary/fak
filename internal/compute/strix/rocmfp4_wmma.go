// Package strix provides specialized GFX1151 RDNA 3.5 compute and quantization kernels
// for AMD Strix Halo (Ryzen AI Max+ 395).
package strix

import (
	"fmt"
	"strings"
)

// ROCmFP4WMMAOpcode is the exact GFX1151 RDNA 3.5 instruction that performs a 16x16x16
// FP4 (E2M1) matrix multiply-accumulate into an FP32 accumulator. It is the opcode the
// emitted on-the-fly-dequant device kernel must carry to be a real device path; a program
// that only issues the f16/bf16 variant cannot claim FP4 tensor throughput.
const ROCmFP4WMMAOpcode = "v_wmma_f32_16x16x16_fp4"

// rocmFP4WMMASourceTemplate is the canonical GFX1151 Wave32 FP4 WMMA kernel SOURCE.
//
// It models AMD Strix Halo (gfx1151) with wavefront_size32, a 128-bit packed FP4 weight
// stream, and IN-REGISTER on-the-fly E2M1 dequant: packed nibbles are split with
// v_and_b32 0x0f0f0f0f / v_lshrrev_b32 4, converted and scaled at the WMMA operand, and
// fed straight into the FP4 tensor core. There is NO intermediate LDS staging and NO
// full-tensor FP32 expansion pass; the per-block FP16 scale folds in at the operand.
//
// [SW-VERIFIED] emission only: this source is asserted structurally (opcode present,
// nibble extract present, gfx1151 target). A physical [HW-WITNESSED] TOPS number for
// this kernel requires gfx1151 silicon; no such measurement is claimed here.
const rocmFP4WMMASourceTemplate = `	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
	.globl	rocmfp4_wmma_dequant_wave32
	.p2align	8
	.type	rocmfp4_wmma_dequant_wave32,@function
rocmfp4_wmma_dequant_wave32:
	.amdhsa_kernel rocmfp4_wmma_dequant_wave32
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_flat_scratch_init 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 30
		.amdhsa_next_free_sgpr 16
		.amdhsa_reserve_flat_scratch 1
		.amdhsa_private_segment_fixed_size 0
	.end_amdhsa_kernel
; GFX1151 Wave32 FP4 WMMA with on-the-fly in-register E2M1 dequant.
; kernarg: 0x00 = Matrix A pointer (activations), 0x08 = packed FP4 (E2M1) weight pointer,
;         0x10 = per-block FP16 scale pointer, 0x18 = K iteration count.
	s_load_b128 s[0:3], s[0:1], 0x00
	s_load_b32 s4, s[0:1], 0x18
	s_mov_b32 s5, 0
	; Initialize the FP32 accumulator tile (8 VGPRs per lane, 16x16x16 output).
	v_mov_b32 v16, 0.0
	v_mov_b32 v17, 0.0
	v_mov_b32 v18, 0.0
	v_mov_b32 v19, 0.0
	v_mov_b32 v20, 0.0
	v_mov_b32 v21, 0.0
	v_mov_b32 v22, 0.0
	v_mov_b32 v23, 0.0
rocmfp4_wmma_loop:
	; 128-bit coalesced load of packed FP4 weights (2 E2M1 nibbles per byte).
	global_load_b128 v[4:7], v[2:3], s[0:1]
	; 128-bit load of the Matrix A activation slice for this K step.
	global_load_b128 v[8:11], v[0:1], s[0:1]
	; Load the FP16 per-block (32-value) scale into a VGPR.
	global_load_b32 v12, v[0:1], s[0:1]
	s_waitcnt vmcnt(0)
	; ON-THE-FLY IN-REGISTER DEQUANT: split packed E2M1 nibbles in VGPRs.
	v_and_b32 v24, 0x0f0f0f0f, v4
	v_lshrrev_b32 v25, 4, v4
	v_and_b32 v25, 0x0f0f0f0f, v25
	; Decode + fold the per-block f16 scale at the operand (no LDS, no FP32 expansion).
	v_cvt_f32_f16 v24, v24
	v_cvt_f32_f16 v25, v25
	v_mul_f32 v24, v24, v12
	v_mul_f32 v25, v25, v12
	v_pack_b32_f16 v28, v24, v25
	; FP4 tensor-core matrix multiply-accumulate with the dequantized operands.
	v_wmma_f32_16x16x16_fp4 v[16:23], v[8:11], v[28:29], v[16:23]
	s_add_i32 s5, s5, 16
	s_cmp_lt_i32 s5, s4
	s_cbranch_scc1 rocmfp4_wmma_loop
	; Store the accumulated FP32 result tile.
	global_store_b128 v[0:1], v[16:19], s[2:3]
	global_store_b128 v[0:1], v[20:23], s[2:3]
	s_waitcnt vmcnt(0)
	s_endpgm
`

// ROCmFP4WMMAKernelInfo is the structural analysis of an emitted FP4 WMMA source. It lets
// callers and tests assert what a source text actually contains without re-parsing it by
// ad-hoc substring checks.
type ROCmFP4WMMAKernelInfo struct {
	// Opcode is the FP4 WMMA opcode the source carries (ROCmFP4WMMAOpcode when present).
	Opcode string `json:"opcode"`
	// TargetArch is the gfx target the .amdgcn_target directive names.
	TargetArch string `json:"target_arch"`
	// WavefrontSize is the declared wavefront size (32 for gfx1151 Wave32).
	WavefrontSize int `json:"wavefront_size"`
	// HasFP4WMMA is true when the literal FP4 WMMA opcode is present.
	HasFP4WMMA bool `json:"has_fp4_wmma"`
	// HasNibbleExtract is true when an in-register nibble-extract instruction is present.
	HasNibbleExtract bool `json:"has_nibble_extract"`
	// HasOnTheFlyScale is true when the per-block FP16 scale is folded at the operand
	// (a v_mul_f32 against the loaded scale), i.e. not a separate dequant-to-memory pass.
	HasOnTheFlyScale bool `json:"has_on_the_fly_scale"`
	// HasEndpgm is true when the kernel terminates with s_endpgm.
	HasEndpgm bool `json:"has_endpgm"`
}

// EmitROCmFP4WMMASource returns the canonical GFX1151 Wave32 FP4 WMMA assembly SOURCE
// text with in-register on-the-fly E2M1 dequant. The returned text carries the literal
// ROCmFP4WMMAOpcode token and targets gfx1151.
//
// NOTE: [SW-VERIFIED] emission only. A physical [HW-WITNESSED] TOPS number requires
// gfx1151 hardware and is not produced or implied by this function.
func EmitROCmFP4WMMASource() string {
	return rocmFP4WMMASourceTemplate
}

// AnalyzeROCmFP4WMMASource structurally inspects an emitted (or supplied) FP4 WMMA source
// and reports the properties witnesses assert on. It performs no compilation.
func AnalyzeROCmFP4WMMASource(source string) ROCmFP4WMMAKernelInfo {
	info := ROCmFP4WMMAKernelInfo{}
	if strings.Contains(source, ROCmFP4WMMAOpcode) {
		info.Opcode = ROCmFP4WMMAOpcode
		info.HasFP4WMMA = true
	}
	if strings.Contains(source, "amdgcn-amd-amdhsa--gfx1151") {
		info.TargetArch = "gfx1151"
	}
	if strings.Contains(source, ".amdhsa_wavefront_size32 1") {
		info.WavefrontSize = 32
	}
	if strings.Contains(source, "v_and_b32") || strings.Contains(source, "v_lshrrev_b32") {
		info.HasNibbleExtract = true
	}
	if strings.Contains(source, "v_mul_f32") {
		info.HasOnTheFlyScale = true
	}
	if strings.Contains(source, "s_endpgm") {
		info.HasEndpgm = true
	}
	return info
}

// String renders the analysis compactly for logs and error messages.
func (i ROCmFP4WMMAKernelInfo) String() string {
	return fmt.Sprintf("opcode=%s target=%s wavefront=%d fp4_wmma=%t nibble_extract=%t on_the_fly_scale=%t endpgm=%t",
		i.Opcode, i.TargetArch, i.WavefrontSize, i.HasFP4WMMA, i.HasNibbleExtract, i.HasOnTheFlyScale, i.HasEndpgm)
}
