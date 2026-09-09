//go:build amd64

#include "textflag.h"

// func sfence()
// Executes an SFENCE instruction to drain and serialize CPU write-combining store buffers (WCBs).
TEXT ·sfence(SB), NOSPLIT, $0-0
	SFENCE
	RET

// func mfence()
// Executes an MFENCE instruction to serialize both load and store operations.
TEXT ·mfence(SB), NOSPLIT, $0-0
	MFENCE
	RET

// func clflushopt(addr uintptr)
// Flushes the cache line containing addr using CLFLUSHOPT (0x66 0x0F 0xAE /7).
TEXT ·clflushopt(SB), NOSPLIT, $0-8
	MOVQ addr+0(FP), AX
	BYTE $0x66; BYTE $0x0F; BYTE $0xAE; BYTE $0x38 // clflushopt (AX)
	RET
