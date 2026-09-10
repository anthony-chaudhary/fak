package main

import (
	"errors"
	"net"
	"net/http"
	"sync"
	"testing"
	"time"
)

func TestTurnkeyCloseRequestsNativeReleaseWhenHTTPServerCloseFails(t *testing.T) {
	base, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	closeErr := errors.New("listener close failed")
	listener := &closeErrorListener{
		Listener: base,
		err:      closeErr,
		entered:  make(chan struct{}),
	}
	httpServer := &http.Server{Handler: http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})}
	serveDone := make(chan error, 1)
	go func() { serveDone <- httpServer.Serve(listener) }()
	select {
	case <-listener.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("HTTP server did not enter Accept")
	}

	var modelClosed, admissionReleased bool
	server := &turnkeyServer{
		httpServer: httpServer,
		native: &turnkeyNativeResources{
			closeModel:       func() error { modelClosed = true; return nil },
			releaseAdmission: func() { admissionReleased = true },
		},
	}
	if err := server.Close(); !errors.Is(err, closeErr) {
		t.Fatalf("Close error = %v, want listener error", err)
	}
	if !modelClosed || !admissionReleased {
		t.Fatalf("native release after HTTP Close error = modelClosed=%v admissionReleased=%v, want true/true", modelClosed, admissionReleased)
	}
	select {
	case <-serveDone:
	case <-time.After(5 * time.Second):
		t.Fatal("HTTP Serve did not return after Close")
	}
}

type closeErrorListener struct {
	net.Listener
	err     error
	entered chan struct{}
	once    sync.Once
}

func (l *closeErrorListener) Accept() (net.Conn, error) {
	l.once.Do(func() { close(l.entered) })
	return l.Listener.Accept()
}

func (l *closeErrorListener) Close() error {
	_ = l.Listener.Close()
	return l.err
}
