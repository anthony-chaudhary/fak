//go:build amd64

#include "textflag.h"

// func scanNextMatchAVX512_I32(tokens *int32, start int, n int, target int32) int
TEXT ·scanNextMatchAVX512_I32(SB), NOSPLIT, $0-40
	MOVQ tokens+0(FP), SI
	MOVQ start+8(FP), BX
	MOVQ n+16(FP), DX
	MOVL target+24(FP), CX

	// Bounds check
	TESTQ BX, BX
	JL    not_found_i32
	CMPQ  BX, DX
	JGE   not_found_i32

	VPBROADCASTD CX, Z0

	// Loop over 16-token (64-byte) blocks
	MOVQ DX, R9
	SUBQ $16, R9
	CMPQ BX, R9
	JG   scalar_tail_i32

loop16:
	VMOVDQU32 (SI)(BX*4), Z1
	VPCMPEQD  Z1, Z0, K1
	KMOVW     K1, AX
	TESTL     AX, AX
	JNZ       match_vector_i32

	ADDQ $16, BX
	CMPQ BX, R9
	JLE  loop16

scalar_tail_i32:
	CMPQ  BX, DX
	JGE   not_found_i32
	MOVL  (SI)(BX*4), R8
	CMPL  R8, CX
	JE    match_scalar_i32
	INCQ  BX
	JMP   scalar_tail_i32

match_vector_i32:
	TZCNTL AX, AX
	ADDQ   AX, BX
	MOVQ   BX, ret+32(FP)
	VZEROUPPER
	RET

match_scalar_i32:
	MOVQ BX, ret+32(FP)
	VZEROUPPER
	RET

not_found_i32:
	MOVQ $-1, ret+32(FP)
	VZEROUPPER
	RET

// func scanNextMatchAVX512_I64(tokens *int, start int, n int, target int) int
TEXT ·scanNextMatchAVX512_I64(SB), NOSPLIT, $0-40
	MOVQ tokens+0(FP), SI
	MOVQ start+8(FP), BX
	MOVQ n+16(FP), DX
	MOVQ target+24(FP), CX

	// Bounds check
	TESTQ BX, BX
	JL    not_found_i64
	CMPQ  BX, DX
	JGE   not_found_i64

	VPBROADCASTQ CX, Z0

	// Loop over 8-token (64-byte) blocks
	MOVQ DX, R9
	SUBQ $8, R9
	CMPQ BX, R9
	JG   scalar_tail_i64

loop8:
	VMOVDQU64 (SI)(BX*8), Z1
	VPCMPEQQ  Z1, Z0, K1
	KMOVB     K1, AX
	TESTL     AX, AX
	JNZ       match_vector_i64

	ADDQ $8, BX
	CMPQ BX, R9
	JLE  loop8

scalar_tail_i64:
	CMPQ  BX, DX
	JGE   not_found_i64
	MOVQ  (SI)(BX*8), R8
	CMPQ  R8, CX
	JE    match_scalar_i64
	INCQ  BX
	JMP   scalar_tail_i64

match_vector_i64:
	TZCNTL AX, AX
	ADDQ   AX, BX
	MOVQ   BX, ret+32(FP)
	VZEROUPPER
	RET

match_scalar_i64:
	MOVQ BX, ret+32(FP)
	VZEROUPPER
	RET

not_found_i64:
	MOVQ $-1, ret+32(FP)
	VZEROUPPER
	RET
