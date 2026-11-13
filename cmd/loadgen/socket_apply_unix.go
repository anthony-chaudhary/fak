//go:build unix && !linux

package main

import "syscall"

// applySocketOptions applies plan to a freshly created socket fd on a POSIX
// platform that is not Linux (Darwin, the BSDs). It covers the upstream options
// whose constants exist here: SO_KEEPALIVE and - when plan.ExplicitBuffers - the
// 10 MiB SO_RCVBUF/SO_SNDBUF pair with the ENOBUFS halving ladder. TCP_QUICKACK
// and TCP_USER_TIMEOUT are Linux-only constants (upstream guards them with
// hasattr), so plan.QuickAck/UserTimeout are false here and skipped.
func applySocketOptions(fd uintptr, plan socketPlan) error {
	s := int(fd)
	if err := syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_KEEPALIVE, 1); err != nil {
		return err
	}
	if !plan.ExplicitBuffers {
		return nil
	}
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
	return nil
}
