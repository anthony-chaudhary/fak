//go:build darwin && arm64 && cgo

package metalgemm

/*
#include <stdint.h>
typedef struct {
    uintptr_t command_buffer;
    int committed, completed_wait, host_readback, encoders;
    double gpu_milliseconds, wait_milliseconds;
    int timing_available;
} mg_execution_event;
typedef struct { mg_execution_event events[2]; int event_count; int submitted; } mg_issue8833_result;
int mg_issue8833_mixed_qkv(int, int, int, int, const signed char*, const float*, const float*,
                           int, int, int, int, float*, float*, float*, int, int, mg_issue8833_result*);
*/
import "C"

import "unsafe"

func executeMixedQKV(selector MixedQKVSelector, in MixedQKVInput) (out MixedQKVResult, err error) {
	id := nextMixedQKVCallID()
	out.CallID = id
	decline := func(detail string) (MixedQKVResult, error) {
		return out, &MixedQKVError{CallID: id, Stage: MixedQKVDeclined, Detail: detail}
	}
	if selector != MixedQKVControl && selector != MixedQKVCandidate || in.Q == nil || in.K == nil || in.V == nil ||
		in.Hidden != 4096 || len(in.XQ) < 4096 ||
		len(in.XD) < 128 || len(in.XF) < 4096 {
		return decline("unsupported geometry, family, or input")
	}
	out.Q, out.K, out.V = make([]float32, 8192), make([]float32, 1024), make([]float32, 1024)
	var native C.mg_issue8833_result
	rc := C.mg_issue8833_mixed_qkv(C.int(selector), C.int(in.Q.ID()), C.int(in.K.ID()), C.int(in.V.ID()),
		(*C.schar)(unsafe.Pointer(&in.XQ[0])), (*C.float)(unsafe.Pointer(&in.XD[0])),
		(*C.float)(unsafe.Pointer(&in.XF[0])), 4096, 8192, 1024, 1024,
		(*C.float)(unsafe.Pointer(&out.Q[0])), (*C.float)(unsafe.Pointer(&out.K[0])),
		(*C.float)(unsafe.Pointer(&out.V[0])), boolInt(in.injectSetup), boolInt(in.injectPost), &native)
	out.Submitted = native.submitted != 0
	for i := 0; i < int(native.event_count); i++ {
		e := native.events[i]
		event := ExecutionEvent{CommandBufferID: uint64(i + 1), Committed: e.committed != 0,
			CompletedWait: e.completed_wait != 0, HostReadback: e.host_readback != 0,
			Encoders: int(e.encoders), GPUMilliseconds: float64(e.gpu_milliseconds),
			WaitMilliseconds: float64(e.wait_milliseconds), TimingAvailable: e.timing_available != 0}
		if selector == MixedQKVControl {
			event.Operation = ExecutionOperation("mixed-qkv-control")
		} else {
			event.Operation = ExecutionOperation("mixed-qkv-candidate")
		}
		out.Observation.Events = append(out.Observation.Events, event)
		if in.Observer != nil {
			in.Observer.ObserveExecution(ScopedExecutionEvent{CallID: id, Event: event})
		}
	}
	if rc == 1 {
		return decline("native setup declined before encoding")
	}
	if rc != 0 {
		return out, &MixedQKVError{CallID: id, Stage: MixedQKVSubmitted, Detail: "native command buffer failed"}
	}
	return out, nil
}

func boolInt(v bool) C.int {
	if v {
		return 1
	}
	return 0
}
