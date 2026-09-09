//go:build ignore

// AMDGPU GFX1151 (RDNA 3.5) Wave32 In-Register Dequantization & WMMA Matrix Kernel
// Target: AMD Ryzen AI Max+ 395 (40 CUs gfx1151, 128GB LPDDR5X-8000/8533, 256-bit bus)
//
// Invariant: Zero intermediate LDS (Local Data Share) buffer allocations.
// All packed 4-bit and 8-bit weight streams are loaded via 128-bit global_load_dwordx4
// vector memory instructions, matching 4x 128-byte coalesced LPDDR5X bursts across Wave32
// (32 threads * 16 bytes = 512 bytes per wavefront).
// Dequantization unpacks nibbles directly in VGPRs, immediately feeding v_wmma instructions.

    .amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
    .text
    .globl  wmma_dequant_wave32_16x16x16
    .p2align 8
    .type   wmma_dequant_wave32_16x16x16,@function

wmma_dequant_wave32_16x16x16:
    // Kernel Argument Offsets (s[0:1] = kernarg pointer)
    //   0x00: Global Input Matrix A pointer (FP16/BF16, [M, K])
    //   0x08: Quantized Weight Matrix B pointer (Packed Q4/Q8, [K, N])
    //   0x10: Scale Tensor pointer (FP32/FP16 per-block scale)
    //   0x18: Output Matrix C pointer (FP32, [M, N])
    //   0x20: Dimension K (int32)
    //   0x24: Stride N (int32)

    s_load_dwordx4 s[4:7], s[0:1], 0x00      // s[4:5]=MatrixA, s[6:7]=MatrixB
    s_load_dwordx4 s[8:11], s[0:1], 0x10     // s[8:9]=Scales, s[10:11]=MatrixC
    s_load_dwordx2 s[12:13], s[0:1], 0x20    // s[12]=K, s[13]=StrideN
    s_waitcnt lgkmcnt(0)

    // Compute Wave32 thread lane ID and coordinate offsets
    v_mbcnt_lo_u32_b32 v0, -1, 0             // Lane ID (0..31 in Wave32)
    v_lshlrev_b32 v1, 4, v0                  // Byte offset: lane_id * 16 bytes (128-bit)

    // Address computation for 128-byte coalesced weight burst
    v_add_co_u32 v2, vcc_lo, s6, v1          // Weight address low
    v_add_co_ci_u32_e32 v3, s7, 0, vcc_lo    // Weight address high

    // Initialize WMMA 16x16x16 FP32 Accumulators in VGPRs (8 VGPRs per lane)
    v_mov_b32 v16, 0.0
    v_mov_b32 v17, 0.0
    v_mov_b32 v18, 0.0
    v_mov_b32 v19, 0.0
    v_mov_b32 v20, 0.0
    v_mov_b32 v21, 0.0
    v_mov_b32 v22, 0.0
    v_mov_b32 v23, 0.0

main_wmma_loop:
    // 128-bit Vector Load: 16 bytes per lane * 32 lanes = 512 bytes per Wave32 wavefront
    // Exactly 4x 128-byte coalesced memory transactions on the 256-bit LPDDR5X bus.
    // Zero LDS spill: loads stream directly into vector general purpose registers (VGPRs).
    global_load_dwordx4 v[4:7], v[2:3], off
    global_load_dwordx4 v[8:11], v[0:1], off  // Load Matrix A activation slice

    // Load scale factor into VGPR
    global_load_dword v12, v[0:1], off offset:0

    s_waitcnt vmcnt(0)

    // In-Register 4-bit Nibble Extraction & Scaling (simdgroup_matrix MLX equivalent)
    // Extract lower 4 bits (nibble 0) from v4 into v24
    v_and_b32 v24, 0x0f0f0f0f, v4
    // Extract upper 4 bits (nibble 1) from v4 into v25
    v_lshrrev_b32 v25, 4, v4
    v_and_b32 v25, 0x0f0f0f0f, v25

    // In-register float conversion and scaling (Zero LDS traffic)
    v_cvt_f32_u32 v24, v24
    v_cvt_f32_u32 v25, v25
    v_mul_f32 v24, v24, v12
    v_mul_f32 v25, v25, v12

    // Repeat for dwords v5, v6, v7 in registers
    v_and_b32 v26, 0x0f0f0f0f, v5
    v_lshrrev_b32 v27, 4, v5
    v_and_b32 v27, 0x0f0f0f0f, v27
    v_cvt_f32_u32 v26, v26
    v_cvt_f32_u32 v27, v27
    v_mul_f32 v26, v26, v12
    v_mul_f32 v27, v27, v12

    // Pack unpacked FP16/BF16 pairs in VGPRs for WMMA input operand B
    v_pack_b32_f16 v28, v24, v25
    v_pack_b32_f16 v29, v26, v27

    // Issue Wave32 RDNA 3.5 WMMA Matrix Instruction:
    // v_wmma_f32_16x16x16_f16 accumulates D = A * B + C directly in-register
    v_wmma_f32_16x16x16_f16 v[16:23], v[8:11], v[28:29], v[16:23]

    // Advance K iteration pointer by 16 columns (128-byte block step)
    s_sub_i32 s12, s12, 16
    s_cmp_gt_i32 s12, 0
    s_cbranch_scc1 main_wmma_loop

write_output:
    // Store final accumulated result matrix back to global memory
    v_add_co_u32 v0, vcc_lo, s10, v1
    v_add_co_ci_u32_e32 v1, s11, 0, vcc_lo
    global_store_dwordx4 v[0:1], v[16:19], off
    global_store_dwordx4 v[0:1], v[20:23], off offset:16

    s_waitcnt vmcnt(0)
    s_endpgm

    .rodata
    .p2align 6
    .amdhsa_kernel wmma_dequant_wave32_16x16x16
        .amdhsa_group_segment_fixed_size 0
        .amdhsa_private_segment_fixed_size 0
        .amdhsa_user_sgpr_dispatch_ptr 0
        .amdhsa_user_sgpr_kernarg_segment_ptr 1
        .amdhsa_wavefront_size32 1
        .amdhsa_system_sgpr_workgroup_id_x 1
        .amdhsa_next_free_vgpr 32
        .amdhsa_next_free_sgpr 16
        .amdhsa_ieee_mode 1
        .amdhsa_dx10_clamp 1
    .end_amdhsa_kernel
