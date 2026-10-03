//go:build darwin && arm64 && cgo

package metalgemm

/*
#cgo LDFLAGS: -framework IOKit
int mg_keepalive_begin(void);
void mg_keepalive_end(void);
int mg_keepalive_power(void);
void mg_keepalive_state(unsigned long long *submitted, int *inflight, int *failed, long long *code, char *domain, char *message, unsigned int *iterations, double *max_gpu_ms);
*/
import "C"

func init() {
	keepAliveNativeStart = func() bool { return Available() && C.mg_keepalive_begin() != 0 }
	keepAliveNativeStop = func() { C.mg_keepalive_end() }
	keepAliveNativePower = func() KeepAlivePowerState {
		bits := int(C.mg_keepalive_power())
		return KeepAlivePowerState{Known: bits&1 != 0, OnBattery: bits&2 != 0, LowPowerMode: bits&4 != 0}
	}
	keepAliveNativeState = func() KeepAliveReceipt {
		var submitted C.ulonglong
		var inflight, failed C.int
		var code C.longlong
		var iterations C.uint
		var maxGPU C.double
		var domain [96]C.char
		var message [192]C.char
		C.mg_keepalive_state(&submitted, &inflight, &failed, &code, &domain[0], &message[0], &iterations, &maxGPU)
		return KeepAliveReceipt{SubmittedBuffers: uint64(submitted), InFlight: int(inflight), Failed: failed != 0, FailureDomain: C.GoString(&domain[0]), FailureCode: int64(code), FailureMessage: C.GoString(&message[0]), Iterations: uint32(iterations), MaxCompletedGPUMilliseconds: float64(maxGPU)}
	}
}
