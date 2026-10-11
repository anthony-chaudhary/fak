//go:build v41_tail_rope_exact_witness && vulkan && (windows || linux) && cgo

package compute

/*
#cgo v41_indexer_witness CFLAGS: -DFAK_V41_TAIL_ROPE_EXACT_CONFLICT
#cgo v41_bf16_boundary_witness CFLAGS: -DFAK_V41_TAIL_ROPE_EXACT_CONFLICT
#cgo v41_logit_scale_witness CFLAGS: -DFAK_V41_TAIL_ROPE_EXACT_CONFLICT
#ifdef FAK_V41_TAIL_ROPE_EXACT_CONFLICT
#error "v41_tail_rope_exact_witness cannot be combined with other V4.1 witness archives"
#endif
#define FAK_V41_TAIL_ROPE_EXACT_WITNESS
#include "vulkan_backend.h"
*/
import "C"

import "unsafe"

var _ V41TailRoPEExactBackend = (*vulkanBackend)(nil)

func v41TailRoPEExactModuleABI(code []byte) bool {
	if len(code) == 0 {
		return false
	}
	return C.fvk_v41_tail_rope_exact_module_abi(unsafe.Pointer(&code[0]), C.size_t(len(code))) != 0
}

// The separate archive binds the actual admitted tail module's three F32 modes
// and the selected device's matching properties. Rows still uses the existing
// checked arithmetic/dispatch. No tag or file digest alone qualifies hardware.
func (v *vulkanBackend) V41TailRoPEExactAvailable() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_v41_tail_rope_exact_available() != 0
}
