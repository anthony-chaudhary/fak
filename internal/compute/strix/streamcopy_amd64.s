//go:build amd64

#include "textflag.h"

// func streamCopyAVX512Kernel(dst, src unsafe.Pointer, chunks int64)
// Executes 64-byte aligned non-temporal streaming copy:
//   VMOVNTDQA (streaming load from write-combined memory)
//   VMOVNTDQ  (non-temporal streaming store, bypassing CPU cache)
// Loop is 8x unrolled (512 bytes per iteration) when chunks >= 8,
// 4x unrolled (256 bytes) when chunks >= 4, and 1x (64 bytes) for remainder.
// Ends with SFENCE and VZEROUPPER.
TEXT ·streamCopyAVX512Kernel(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ chunks+16(FP), CX

	TESTQ CX, CX
	JLE done_kernel

loop8_kernel:
	CMPQ CX, $8
	JL loop4_kernel

	VMOVNTDQA 0(SI), Z0
	VMOVNTDQA 64(SI), Z1
	VMOVNTDQA 128(SI), Z2
	VMOVNTDQA 192(SI), Z3
	VMOVNTDQA 256(SI), Z4
	VMOVNTDQA 320(SI), Z5
	VMOVNTDQA 384(SI), Z6
	VMOVNTDQA 448(SI), Z7

	VMOVNTDQ Z0, 0(DI)
	VMOVNTDQ Z1, 64(DI)
	VMOVNTDQ Z2, 128(DI)
	VMOVNTDQ Z3, 192(DI)
	VMOVNTDQ Z4, 256(DI)
	VMOVNTDQ Z5, 320(DI)
	VMOVNTDQ Z6, 384(DI)
	VMOVNTDQ Z7, 448(DI)

	ADDQ $512, SI
	ADDQ $512, DI
	SUBQ $8, CX
	JMP loop8_kernel

loop4_kernel:
	CMPQ CX, $4
	JL loop1_kernel

	VMOVNTDQA 0(SI), Z0
	VMOVNTDQA 64(SI), Z1
	VMOVNTDQA 128(SI), Z2
	VMOVNTDQA 192(SI), Z3

	VMOVNTDQ Z0, 0(DI)
	VMOVNTDQ Z1, 64(DI)
	VMOVNTDQ Z2, 128(DI)
	VMOVNTDQ Z3, 192(DI)

	ADDQ $256, SI
	ADDQ $256, DI
	SUBQ $4, CX
	JMP loop4_kernel

loop1_kernel:
	TESTQ CX, CX
	JLE done_kernel

	VMOVNTDQA 0(SI), Z0
	VMOVNTDQ Z0, 0(DI)

	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JMP loop1_kernel

done_kernel:
	SFENCE
	VZEROUPPER
	RET

// func streamCopyAVX512StoreOnly(dst, src unsafe.Pointer, chunks int64)
// Executes 64-byte aligned non-temporal streaming store to destination with
// unaligned vector loads from source:
//   VMOVUPS   (unaligned 512-bit vector load)
//   VMOVNTDQ  (64-byte aligned non-temporal streaming store)
// Ends with SFENCE and VZEROUPPER.
TEXT ·streamCopyAVX512StoreOnly(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ chunks+16(FP), CX

	TESTQ CX, CX
	JLE done_store

loop8_store:
	CMPQ CX, $8
	JL loop4_store

	VMOVUPS 0(SI), Z0
	VMOVUPS 64(SI), Z1
	VMOVUPS 128(SI), Z2
	VMOVUPS 192(SI), Z3
	VMOVUPS 256(SI), Z4
	VMOVUPS 320(SI), Z5
	VMOVUPS 384(SI), Z6
	VMOVUPS 448(SI), Z7

	VMOVNTDQ Z0, 0(DI)
	VMOVNTDQ Z1, 64(DI)
	VMOVNTDQ Z2, 128(DI)
	VMOVNTDQ Z3, 192(DI)
	VMOVNTDQ Z4, 256(DI)
	VMOVNTDQ Z5, 320(DI)
	VMOVNTDQ Z6, 384(DI)
	VMOVNTDQ Z7, 448(DI)

	ADDQ $512, SI
	ADDQ $512, DI
	SUBQ $8, CX
	JMP loop8_store

loop4_store:
	CMPQ CX, $4
	JL loop1_store

	VMOVUPS 0(SI), Z0
	VMOVUPS 64(SI), Z1
	VMOVUPS 128(SI), Z2
	VMOVUPS 192(SI), Z3

	VMOVNTDQ Z0, 0(DI)
	VMOVNTDQ Z1, 64(DI)
	VMOVNTDQ Z2, 128(DI)
	VMOVNTDQ Z3, 192(DI)

	ADDQ $256, SI
	ADDQ $256, DI
	SUBQ $4, CX
	JMP loop4_store

loop1_store:
	TESTQ CX, CX
	JLE done_store

	VMOVUPS 0(SI), Z0
	VMOVNTDQ Z0, 0(DI)

	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JMP loop1_store

done_store:
	SFENCE
	VZEROUPPER
	RET

// func streamCopyAVX512LoadOnly(dst, src unsafe.Pointer, chunks int64)
// Executes 64-byte aligned streaming loads from source (write-combining APU memory)
// with unaligned vector stores to destination:
//   VMOVNTDQA (64-byte aligned streaming load)
//   VMOVUPS   (unaligned 512-bit vector store)
// Ends with VZEROUPPER.
TEXT ·streamCopyAVX512LoadOnly(SB), NOSPLIT, $0-24
	MOVQ dst+0(FP), DI
	MOVQ src+8(FP), SI
	MOVQ chunks+16(FP), CX

	TESTQ CX, CX
	JLE done_load

loop8_load:
	CMPQ CX, $8
	JL loop4_load

	VMOVNTDQA 0(SI), Z0
	VMOVNTDQA 64(SI), Z1
	VMOVNTDQA 128(SI), Z2
	VMOVNTDQA 192(SI), Z3
	VMOVNTDQA 256(SI), Z4
	VMOVNTDQA 320(SI), Z5
	VMOVNTDQA 384(SI), Z6
	VMOVNTDQA 448(SI), Z7

	VMOVUPS Z0, 0(DI)
	VMOVUPS Z1, 64(DI)
	VMOVUPS Z2, 128(DI)
	VMOVUPS Z3, 192(DI)
	VMOVUPS Z4, 256(DI)
	VMOVUPS Z5, 320(DI)
	VMOVUPS Z6, 384(DI)
	VMOVUPS Z7, 448(DI)

	ADDQ $512, SI
	ADDQ $512, DI
	SUBQ $8, CX
	JMP loop8_load

loop4_load:
	CMPQ CX, $4
	JL loop1_load

	VMOVNTDQA 0(SI), Z0
	VMOVNTDQA 64(SI), Z1
	VMOVNTDQA 128(SI), Z2
	VMOVNTDQA 192(SI), Z3

	VMOVUPS Z0, 0(DI)
	VMOVUPS Z1, 64(DI)
	VMOVUPS Z2, 128(DI)
	VMOVUPS Z3, 192(DI)

	ADDQ $256, SI
	ADDQ $256, DI
	SUBQ $4, CX
	JMP loop4_load

loop1_load:
	TESTQ CX, CX
	JLE done_load

	VMOVNTDQA 0(SI), Z0
	VMOVUPS Z0, 0(DI)

	ADDQ $64, SI
	ADDQ $64, DI
	DECQ CX
	JMP loop1_load

done_load:
	VZEROUPPER
	RET

// func sfenceAsm()
// Executes an SFENCE instruction on amd64 to order stores and serialize
// write-combining buffers.
TEXT ·sfenceAsm(SB), NOSPLIT, $0-0
	SFENCE
	RET
