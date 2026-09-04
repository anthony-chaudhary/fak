// Package childproc provides robust child process execution, supervision, timeout handling,
// output stream buffering, and process group / tree termination using only the Go standard library.
//
// Invariant: all child process executions are managed under explicit context deadlines or cancellations.
// Guard: fail-closed exit code propagation: failures to spawn, aborts, timeouts, and unresolvable
// exit states are guaranteed to report a non-zero exit code (-1).
package childproc
