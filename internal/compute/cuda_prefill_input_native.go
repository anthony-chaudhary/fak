//go:build cuda && cgo

package compute

/*
#include "cuda_backend.h"
*/
import "C"

import "unsafe"

var cudaPrefillInputNative = cudaPrefillInputOperations{
	admit: func() int { return int(C.fcuda_prefill_input_admit()) },
	acquire: func(host []float32, panel bool, dst *cudaBuf) int {
		var mode C.int
		if panel {
			mode = 1
		}
		var managed C.int
		status := int(C.fcuda_prefill_input_acquire(unsafe.Pointer(&host[0]), C.size_t(dst.n), mode, &dst.ptr, &managed))
		dst.managed = managed != 0
		return status
	},
	retire: func(b *cudaBuf) int { return int(C.fcuda_prefill_input_retire(b.ptr, C.size_t(b.n))) },
	failed: func(status int) { C.fcuda_prefill_input_failed(C.int(status)) },
}
