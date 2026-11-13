//go:build linux

package main

import "syscall"

// applySocketOptions applies plan to a freshly created socket fd on Linux. This
// is the platform that motivates upstream's explicit sizing, so it is the only
// one that gets every knob: SO_KEEPALIVE, the 10 MiB SO_RCVBUF/SO_SNDBUF pair
// with the ENOBUFS halving ladder, the TCP keepalive timing trio
// (TCP_KEEPIDLE/INTVL/CNT), TCP_QUICKACK, and TCP_USER_TIMEOUT.
func applySocketOptions(fd uintptr, plan socketPlan) error {
	s := int(fd)
	if err := syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_KEEPALIVE, 1); err != nil {
		return err
	}
	if plan.KeepAliveTiming {
		if err := syscall.SetsockoptInt(s, syscall.IPPROTO_TCP, syscall.TCP_KEEPIDLE, socketDefaultKeepIdle); err != nil {
			return err
		}
		if err := syscall.SetsockoptInt(s, syscall.IPPROTO_TCP, syscall.TCP_KEEPINTVL, socketDefaultKeepIntvl); err != nil {
			return err
		}
		if err := syscall.SetsockoptInt(s, syscall.IPPROTO_TCP, syscall.TCP_KEEPCNT, socketDefaultKeepCnt); err != nil {
			return err
		}
	}
	if plan.ExplicitBuffers {
		if _, err := halveOnEnobufs(socketDefaultRecvBuf, socketMinBufferFallback, func(v int) error {
			return syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_RCVBUF, v)
		}); err != nil {
			return err
		}
		if _, err := halveOnEnobufs(socketDefaultSendBuf, socketMinBufferFallback, func(v int) error {
			return syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_SNDBUF, v)
		}); err != nil {
			return err
		}
	}
	if plan.QuickAck {
		// Best-effort: a kernel without the option is not a dial failure.
		_ = syscall.SetsockoptInt(s, syscall.IPPROTO_TCP, syscall.TCP_QUICKACK, 1)
	}
	if plan.UserTimeout {
		_ = syscall.SetsockoptInt(s, syscall.IPPROTO_TCP, tcpUserTimeoutOpt, socketDefaultUserTimeout)
	}
	return nil
}

// tcpUserTimeoutOpt is TCP_USER_TIMEOUT (18). Go's syscall package does not
// export it, so the Linux constant is named here; it matches linux/tcp.h.
const tcpUserTimeoutOpt = 18
