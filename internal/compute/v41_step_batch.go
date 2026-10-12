package compute

// V41StepBatchBackend records one V4.1 decode step into a single command
// buffer window (#13820): uploads and dispatches record, and only a host
// readback or a status-owning synchronous kernel submits. End must be called
// after every successful Begin, including on error and panic paths.
type V41StepBatchBackend interface {
	BeginV41StepBatch() bool
	EndV41StepBatch() error
}
