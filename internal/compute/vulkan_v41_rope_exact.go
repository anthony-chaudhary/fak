//go:build !v41_tail_rope_exact_witness && vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import "unsafe"

var _ V41TailRoPEExactBackend = (*vulkanBackend)(nil)

// v41TailRoPEExactModuleABI runs the shim's strict tail-RoPE proof on caller
// bytes. It needs no device and is no numerical or physical qualification.
func v41TailRoPEExactModuleABI(code []byte) bool {
	if len(code) == 0 {
		return false
	}
	return C.fvk_v41_tail_rope_strict_abi(unsafe.Pointer(&code[0]), C.size_t(len(code))) != 0
}

// V41TailRoPEExactAvailable reports the ordinary build's strict twin: loaded
// only when the selected device reports binary32 DenormPreserve,
// SignedZeroInfNanPreserve and RoundingModeRTE and the strict ABI proof admitted
// the exact module bytes. V41TailRoPERows runs on that pipeline while this holds.
func (v *vulkanBackend) V41TailRoPEExactAvailable() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_have_v41_tail_rope_exact() != 0
}
