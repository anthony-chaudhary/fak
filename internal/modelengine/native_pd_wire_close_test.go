package modelengine

import (
	"errors"
	"io"
	"net"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

type pdWireReadObservedConn struct {
	net.Conn
	started chan struct{}
	once    sync.Once
}

type pdWirePausedAcceptListener struct {
	net.Listener
	accepted chan struct{}
	release  chan struct{}
}

func (l *pdWirePausedAcceptListener) Accept() (net.Conn, error) {
	conn, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	close(l.accepted)
	<-l.release
	return conn, nil
}

// fak-test:runtime fast est=1s lane=default
func TestPDWireCloseRejectsConnectionAcceptedDuringShutdown(t *testing.T) {
	server := NewPDWireServer(&NativePDCluster{m: &model.Model{}})
	if err := server.Listen("127.0.0.1:0"); err != nil {
		t.Fatal(err)
	}
	ln := &pdWirePausedAcceptListener{Listener: server.listener, accepted: make(chan struct{}), release: make(chan struct{})}
	server.listener = ln
	serveDone := make(chan error, 1)
	go func() { serveDone <- server.Serve() }()
	conn, err := net.DialTimeout("tcp", server.Addr().String(), time.Second)
	if err != nil {
		close(ln.release)
		_ = server.Close()
		t.Fatal(err)
	}
	defer conn.Close()
	select {
	case <-ln.accepted:
	case <-time.After(time.Second):
		close(ln.release)
		_ = conn.Close()
		_ = server.Close()
		t.Fatal("connection was not accepted")
	}
	// The kernel accepted the socket, but Serve has not registered it yet.
	// Closing here must prevent a new handler from starting after Wait ends.
	if err := server.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
	close(ln.release)
	select {
	case err := <-serveDone:
		if err != nil {
			t.Errorf("Serve: %v", err)
		}
	case <-time.After(time.Second):
		_ = conn.Close()
		t.Fatal("accept loop did not retire")
	}
	if err := conn.SetReadDeadline(time.Now().Add(500 * time.Millisecond)); err != nil {
		t.Fatal(err)
	}
	var b [1]byte
	_, readErr := conn.Read(b[:])
	_ = conn.Close()
	server.wg.Wait() // Serve has exited, so there can be no later Add.
	if !errors.Is(readErr, io.EOF) {
		t.Fatalf("accepted socket remained open after Close: %v", readErr)
	}
}

func (c *pdWireReadObservedConn) Read(p []byte) (int, error) {
	c.once.Do(func() { close(c.started) })
	return c.Conn.Read(p)
}

type pdWireReadObservedListener struct {
	net.Listener
	started chan struct{}
}

func (l *pdWireReadObservedListener) Accept() (net.Conn, error) {
	conn, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	return &pdWireReadObservedConn{Conn: conn, started: l.started}, nil
}

// fak-test:runtime fast est=1s lane=default
func TestPDWireCloseInterruptsIdleAcceptedConnection(t *testing.T) {
	// No header is sent, so this exercises the real TCP accept/read lifecycle
	// without model execution or a timer-based guess about when Accept ran.
	server := NewPDWireServer(&NativePDCluster{m: &model.Model{}})
	if err := server.Listen("127.0.0.1:0"); err != nil {
		t.Fatal(err)
	}
	readStarted := make(chan struct{})
	server.listener = &pdWireReadObservedListener{Listener: server.listener, started: readStarted}
	serveDone := make(chan error, 1)
	go func() { serveDone <- server.Serve() }()
	conn, err := net.DialTimeout("tcp", server.Addr().String(), time.Second)
	if err != nil {
		_ = server.Close()
		t.Fatal(err)
	}
	defer conn.Close()
	select {
	case <-readStarted:
	case <-time.After(time.Second):
		_ = conn.Close()
		_ = server.Close()
		t.Fatal("server did not start reading the accepted connection")
	}
	closeDone := make(chan error, 2)
	startClose := make(chan struct{})
	for range 2 {
		go func() {
			<-startClose
			closeDone <- server.Close()
		}()
	}
	close(startClose)
	for i := 0; i < 2; i++ {
		select {
		case err := <-closeDone:
			if err != nil {
				t.Errorf("Close: %v", err)
			}
		case <-time.After(500 * time.Millisecond):
			// Unblock the original implementation before failing, so the
			// negative control leaves no receiver or closer goroutine behind.
			_ = conn.Close()
			for ; i < 2; i++ {
				select {
				case <-closeDone:
				case <-time.After(time.Second):
					t.Fatal("Close stayed blocked after client cleanup")
				}
			}
			t.Fatal("Close waited for an idle client's header")
		}
	}
	select {
	case err := <-serveDone:
		if err != nil {
			t.Errorf("Serve: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("accept loop did not retire")
	}
	if err := server.Close(); err != nil {
		t.Errorf("repeated Close: %v", err)
	}
}
