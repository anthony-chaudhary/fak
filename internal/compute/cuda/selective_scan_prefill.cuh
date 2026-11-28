// selective_scan_prefill.cuh - structure-only contract for Mamba selective-scan chunked prefill.
//
// chunks, final-state extraction, MAX_DSTATE=256, sm_75+ (bf16 path sm_80+). No upstream code
// is copied.
//
// SKELETON ONLY: constants. No hardware run, no speed claim. The Go oracle in
// selective_scan_prefill_test.go reads these #defines.
#ifndef FAK_SELECTIVE_SCAN_PREFILL_CUH
#define FAK_SELECTIVE_SCAN_PREFILL_CUH

#define FAK_MAMBA_MAX_DSTATE 256
#define FAK_MAMBA_SM_MIN 75
#define FAK_MAMBA_BF16_SM_MIN 80
#define FAK_MAMBA_SCAN_OP_PREFIX 1

#endif  // FAK_SELECTIVE_SCAN_PREFILL_CUH
