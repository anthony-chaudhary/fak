package gateway

import (
	"context"
	"errors"
	"io"
	"net"
	"syscall"
)

// upstream_transport_kind.go classifies a TRANSIENT upstream transport failure — the
// "transport" upstream-error kind. It is the signal `fak guard`'s supervisor reads to tell
// a transient WIRE crash (a mid-flight connection drop/reset, a truncated read, an I/O
// timeout — worth exactly one bounded relaunch) apart from a deterministic misconfiguration
// (refused/NXDOMAIN/TLS, which surfaces as UpstreamUnreachableError → "unreachable" and would
// only re-trip a relaunch) or a plain non-transport failure (the coarse "other" bucket).
//
// The gap this closes (#3514): the response-only upstream observer never sees a connection
// error (its RoundTrip callback fires only when resp != nil), so a wire crash reached the
// upstream-error fold classified as the opaque "other". Splitting a transient transport error
// into its own kind gives the supervisor a specific, queryable signal it can gate a retry on
// without loosening to a bare non-zero exit (which would re-mask a systematic crash).

// transientTransportError reports whether err is a transient upstream transport failure. It is
// deliberately CONSERVATIVE: it matches only the transient wire family and returns false for
// everything else, so a non-transport error (e.g. http.ErrHandlerTimeout, a plain string error)
// stays in the coarse "other" bucket and a deterministic dial failure stays "unreachable".
func transientTransportError(err error) bool {
	if err == nil {
		return false
	}
	// An explicitly aborted turn (client disconnect / context cancel) is NOT an upstream wire
	// crash — surfacing it as a transient transport error would drive a spurious relaunch.
	if errors.Is(err, context.Canceled) {
		return false
	}
	// A mid-flight connection reset or broken pipe: the upstream dropped an established
	// connection. errors.Is catches the BSD errno on Linux/macOS; a Windows reset (whose errno
	// != the BSD constant) arrives as a non-dial *net.OpError, caught by the last branch.
	if errors.Is(err, syscall.ECONNRESET) || errors.Is(err, syscall.EPIPE) {
		return true
	}
	// The connection dropped mid-body — a truncated read of the upstream response.
	if errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	// An I/O timeout (dial timeout, read/write deadline): transient packet loss or a slow
	// upstream, not a permanent misconfiguration.
	var ne net.Error
	if errors.As(err, &ne) && ne.Timeout() {
		return true
	}
	// A non-dial *net.OpError — a read/write on an already-established connection that failed
	// mid-flight (this is where a Windows reset lands). A DIAL OpError is excluded: a
	// deterministic dial failure is already "unreachable", and a dial TIMEOUT was caught above.
	var opErr *net.OpError
	if errors.As(err, &opErr) && opErr.Op != "dial" {
		return true
	}
	return false
}
