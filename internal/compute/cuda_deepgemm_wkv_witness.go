//go:build cuda && cgo && cudaarchspecific && fak_cuda_deepgemm_wkv

package compute

/*
#include "cuda_deepgemm_wkv.h"
*/
import "C"
import "unsafe"

// V41FP8WKVDeviceFacts reads the actual instantiated kernel/device attributes;
// no environment value can substitute for the loaded image or physical SM count.
func (c *cudaBackend) V41FP8WKVDeviceFacts() (map[string]any, error) {
	if c == nil {
		return nil, ErrV41FP8WKVUnavailable
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var f C.fak_deepgemm_wkv_facts
	rc := C.fcuda_deepgemm_wkv_facts(&f)
	out := map[string]any{"device": int(f.device), "major": int(f.major), "minor": int(f.minor), "sms": int(f.sms), "compiled_sms": int(f.compiled_sms), "binary_version": int(f.binary_version), "runtime_version": int(f.runtime_version), "name": C.GoString(&f.name[0]), "uuid": C.GoBytes(unsafe.Pointer(&f.uuid[0]), 16)}
	return out, c.wkvStatus("facts", rc)
}
