//go:build cuda && cgo

package compute

/*
int fak_v41_checked_stream_terminal(void);
*/
import "C"

// Every CUDA profile checks its matching native archive; no feature Go tag may
// silently select a no-op. A stale archive without the terminal symbol fails to
// link rather than running with a disabled safety bridge.
// Caller holds cudaMu. Existing peer terminal flags are non-atomic and must
// not be inspected from Ready(), telemetry, or other lock-free readers.
func cudaV41CheckedSubmit() {
	if C.fak_v41_checked_stream_terminal() != 0 {
		panic(&BackendError{Backend: "cuda", Site: "cudaBufForSubmit", Class: "cuda-terminal-stream", Message: "shared private stream is uncertain; process teardown required"})
	}
}
