package compute

// V41StepBatchBackend records one V4.1 decode step into a single command
// buffer window (#13820): uploads and dispatches record, and only a host
// readback or a status-owning synchronous kernel submits. End must be called
// after every successful Begin, including on error and panic paths.
type V41StepBatchBackend interface {
	BeginV41StepBatch() bool
	EndV41StepBatch() error
}

// V41StepBatchRecordingBackend optionally reports whether a healthy V4.1 step
// command buffer is currently recording. An open but suspended window is not
// recording. The snapshot neither acquires ownership nor identifies the session
// that opened the window. Callers relying on it must hold the matching live
// ForwardExecutionOwner scope through the dependent work, with no intervening
// operation or callback that could suspend, end, or fault recording.
type V41StepBatchRecordingBackend interface {
	V41StepBatchRecording() bool
}
