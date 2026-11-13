package main

// Witness tests for the aiperf-data-7 direct port: socket defaults
// (TCP_NODELAY/QUICKACK/USER_TIMEOUT, ENOBUFS-halving buffer fallback, Windows
// skip of explicit SO_SNDBUF/RCVBUF). The manifest's verify command is
// `go test ./cmd/loadgen -run TestSocketDefaults -count=1`.

import (
	"errors"
	"net"
	"syscall"
	"testing"
)

// TestSocketDefaults pins the platform-free logic: the per-OS plan matches
// upstream's IS_WINDOWS branch plus its hasattr guards, and the ENOBUFS halving
// ladder converges exactly as upstream _set_socket_buffer does.
// fak-test:runtime fast est=100ms
func TestSocketDefaults(t *testing.T) {
	t.Parallel()

	t.Run("plan matches upstream per-OS policy", func(t *testing.T) {
		linux := socketPlanFor("linux")
		if !linux.ExplicitBuffers || !linux.KeepAliveTiming || !linux.QuickAck || !linux.UserTimeout {
			t.Fatalf("linux plan=%+v want every knob on", linux)
		}
		darwin := socketPlanFor("darwin")
		if !darwin.ExplicitBuffers || !darwin.KeepAliveTiming {
			t.Fatalf("darwin plan=%+v want buffers+keepalive on", darwin)
		}
		if darwin.QuickAck || darwin.UserTimeout {
			t.Fatalf("darwin plan=%+v want linux-only knobs off", darwin)
		}
		// Windows (and any other OS): buffers OFF (auto-tuning) and no timing knobs.
		for _, goos := range []string{"windows", "plan9", "js", "freebsd"} {
			p := socketPlanFor(goos)
			if p.ExplicitBuffers || p.KeepAliveTiming || p.QuickAck || p.UserTimeout {
				t.Fatalf("%s plan=%+v want all-off (buffer sizing must not disable Auto-Tuning)", goos, p)
			}
		}
	})

	t.Run("upstream default values", func(t *testing.T) {
		if socketDefaultRecvBuf != 10<<20 || socketDefaultSendBuf != 10<<20 {
			t.Fatalf("buffers=%d/%d want 10 MiB each", socketDefaultRecvBuf, socketDefaultSendBuf)
		}
		if socketDefaultKeepIdle != 60 || socketDefaultKeepIntvl != 30 || socketDefaultKeepCnt != 1 {
			t.Fatalf("keepalive=%d/%d/%d want 60/30/1", socketDefaultKeepIdle, socketDefaultKeepIntvl, socketDefaultKeepCnt)
		}
		if socketDefaultUserTimeout != 30000 {
			t.Fatalf("user timeout=%d want 30000", socketDefaultUserTimeout)
		}
		if socketMinBufferFallback != 1024 {
			t.Fatalf("floor=%d want 1024", socketMinBufferFallback)
		}
	})

	t.Run("ENOBUFS halving ladder", func(t *testing.T) {
		// Accepts only at <= 2.5 MiB: from 10 MiB that is two halvings
		// (10 -> 5 -> 2.5 MiB), so set is called three times.
		var calls []int
		accepted, err := halveOnEnobufs(socketDefaultRecvBuf, socketMinBufferFallback, func(v int) error {
			calls = append(calls, v)
			if v > 5<<19 {
				return syscall.ENOBUFS
			}
			return nil
		})
		if err != nil {
			t.Fatalf("ladder err=%v want nil", err)
		}
		if accepted != 5<<19 {
			t.Fatalf("accepted=%d want %d", accepted, 5<<19)
		}
		want := []int{10 << 20, 5 << 20, 5 << 19}
		if len(calls) != len(want) {
			t.Fatalf("calls=%v want %v", calls, want)
		}
		for i := range want {
			if calls[i] != want[i] {
				t.Fatalf("calls=%v want %v", calls, want)
			}
		}
	})

	t.Run("first try success is unchanged", func(t *testing.T) {
		accepted, err := halveOnEnobufs(socketDefaultSendBuf, socketMinBufferFallback, func(int) error { return nil })
		if err != nil || accepted != socketDefaultSendBuf {
			t.Fatalf("accepted/err=%d/%v want %d/nil (no halving when it fits)", accepted, err, socketDefaultSendBuf)
		}
	})

	t.Run("floor: persistent ENOBUFS surfaces the error", func(t *testing.T) {
		var last int
		_, err := halveOnEnobufs(socketDefaultRecvBuf, socketMinBufferFallback, func(v int) error {
			last = v
			return syscall.ENOBUFS
		})
		if !errors.Is(err, syscall.ENOBUFS) {
			t.Fatalf("err=%v want ENOBUFS after exhausting the ladder", err)
		}
		if last != socketMinBufferFallback {
			t.Fatalf("last attempted=%d want the %d floor", last, socketMinBufferFallback)
		}
	})

	t.Run("non-ENOBUFS error is not retried", func(t *testing.T) {
		sentinel := errors.New("EPERM")
		calls := 0
		_, err := halveOnEnobufs(socketDefaultRecvBuf, socketMinBufferFallback, func(int) error {
			calls++
			return sentinel
		})
		if !errors.Is(err, sentinel) {
			t.Fatalf("err=%v want the sentinel (only ENOBUFS retries)", err)
		}
		if calls != 1 {
			t.Fatalf("calls=%d want 1", calls)
		}
	})
}

// TestSocketDefaultsDialerAppliesToConnections is the end-to-end witness: a
// dialer built by socketDefaultsDialer actually connects, so the Control hook is
// wired into a real socket path on this host rather than only unit-tested.
// fak-test:runtime fast est=1s
func TestSocketDefaultsDialerAppliesToConnections(t *testing.T) {
	t.Parallel()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	accepted := make(chan struct{}, 1)
	go func() {
		c, aerr := ln.Accept()
		if aerr == nil {
			_ = c.Close()
		}
		accepted <- struct{}{}
	}()

	d := socketDefaultsDialer()
	conn, err := d.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatalf("dial via socket-default dialer: %v", err)
	}
	// The Control hook ran inside Dial (it built this connection), so a
	// successful round-trip here proves the fd survived the socket-option
	// writes on this host.
	if tcp, ok := conn.(*net.TCPConn); ok {
		if nd, nerr := tcp.SyscallConn(); nerr == nil {
			if cerr := nd.Control(func(uintptr) {}); cerr != nil {
				t.Fatalf("post-dial Control: %v", cerr)
			}
		}
	}
	_ = conn.Close()
	<-accepted
}
