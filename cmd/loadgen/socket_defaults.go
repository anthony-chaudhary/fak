// Socket defaults for the loadgen HTTP client.
//
// DIRECT-PORT of aiperf's SocketDefaults (src/aiperf/transports/http_defaults.py
// @ ab610add4e8876fca223197596c728e17365f381, Apache-2.0) with its
// _HTTPSettings values (src/aiperf/common/environment.py). Every operative
// number below is the upstream default, named so it can be checked:
//
//	TCP_NODELAY=1 (Nagle off), SO_KEEPALIVE=1, SO_RCVBUF/SO_SNDBUF=10 MiB,
//	TCP_KEEPIDLE=60, TCP_KEEPINTVL=30, TCP_KEEPCNT=1,
//	TCP_QUICKACK=1, TCP_USER_TIMEOUT=30000 ms; on ENOBUFS halve the buffer
//	until it fits or reaches the 1024-byte floor.
//
// The per-OS policy is upstream's too, and it is the part that matters on this
// host: an explicit SO_SNDBUF/SO_RCVBUF is SKIPPED on Windows because it
// disables TCP Auto-Tuning, which then head-of-line-blocks large streams under
// concurrency (upstream observed aiohttp stalls of 9+ minutes at --concurrency
// 6). TCP_QUICKACK and TCP_USER_TIMEOUT are Linux-only (upstream guards them
// with hasattr), and the keepalive-timing knobs use the platform's own socket
// constant (Linux TCP_KEEPIDLE vs Darwin TCP_KEEPALIVE).
//
// The pure logic here (socketPlanFor, halveOnEnobufs) is platform-free and
// unit-tested on any GOOS; the socket-option writes live in per-OS files so
// every target cross-compiles.
package main

import (
	"errors"
	"net"
	"runtime"
	"syscall"
)

// Upstream SocketDefaults / _HTTPSettings values, in the units the OS wants.
const (
	socketDefaultRecvBuf     = 10 << 20 // SO_RCVBUF, 10 MiB
	socketDefaultSendBuf     = 10 << 20 // SO_SNDBUF, 10 MiB
	socketDefaultKeepIdle    = 60       // TCP_KEEPIDLE, seconds
	socketDefaultKeepIntvl   = 30       // TCP_KEEPINTVL, seconds
	socketDefaultKeepCnt     = 1        // TCP_KEEPCNT, probes
	socketDefaultUserTimeout = 30000    // TCP_USER_TIMEOUT, milliseconds
	socketMinBufferFallback  = 1024     // never halve below this
)

// socketPlan is the per-OS subset of the upstream defaults that is both
// meaningful and safe on that OS.
type socketPlan struct {
	// ExplicitBuffers sets SO_RCVBUF/SO_SNDBUF to the 10 MiB defaults. False on
	// Windows, where an explicit size disables TCP Auto-Tuning.
	ExplicitBuffers bool
	// KeepAliveTiming sets the TCP keepalive idle/intvl/cnt knobs (Linux/Darwin
	// carry these; Windows does not).
	KeepAliveTiming bool
	// QuickAck sets TCP_QUICKACK (Linux-only constant).
	QuickAck bool
	// UserTimeout sets TCP_USER_TIMEOUT (Linux-only constant).
	UserTimeout bool
}

// socketPlanFor returns the plan for one GOOS. It is the machine-readable form
// of upstream's IS_WINDOWS branch plus its hasattr(TCP_QUICKACK/USER_TIMEOUT)
// guards.
func socketPlanFor(goos string) socketPlan {
	switch goos {
	case "linux":
		return socketPlan{ExplicitBuffers: true, KeepAliveTiming: true, QuickAck: true, UserTimeout: true}
	case "darwin":
		return socketPlan{ExplicitBuffers: true, KeepAliveTiming: true}
	default:
		// Windows (auto-tuning) and every other OS: keepalive only.
		return socketPlan{}
	}
}

// halveOnEnobufs applies a socket buffer size and, when the OS rejects it with
// ENOBUFS, retries with the size halved until it is accepted or the floor is
// reached - the exact ladder of upstream _set_socket_buffer. set is the raw
// setsockopt (injected so the ladder is testable without a socket). It returns
// the size actually accepted; any non-ENOBUFS error, or an ENOBUFS that
// persists down to the floor, returns that error and no size. A size that fits
// on the first try returns it unchanged, matching upstream's no-log fast path.
func halveOnEnobufs(candidate, floor int, set func(int) error) (int, error) {
	for {
		err := set(candidate)
		if err == nil {
			return candidate, nil
		}
		if !errors.Is(err, syscall.ENOBUFS) {
			return 0, err
		}
		if candidate <= floor {
			return 0, err
		}
		next := candidate / 2
		if next < floor {
			next = floor
		}
		candidate = next
	}
}

// socketDefaultsDialer builds a net.Dialer whose Control hook applies
// socketPlanFor(runtime.GOOS) to every connection this client opens, so the
// tunings follow the connection rather than the process.
func socketDefaultsDialer() *net.Dialer {
	return &net.Dialer{Control: socketControl(runtime.GOOS)}
}

// socketControl builds the dialer Control for one GOOS. The returned function
// runs inside the dialer's syscall window, so it must not block; applySocketOptions
// is a handful of setsockopt calls.
func socketControl(goos string) func(network, address string, c syscall.RawConn) error {
	plan := socketPlanFor(goos)
	return func(_, _ string, c syscall.RawConn) error {
		var setErr error
		if err := c.Control(func(fd uintptr) { setErr = applySocketOptions(fd, plan) }); err != nil {
			return err
		}
		return setErr
	}
}
