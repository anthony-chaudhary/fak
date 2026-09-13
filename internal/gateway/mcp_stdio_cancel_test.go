package gateway

import (
	"context"
	"errors"
	"io"
	"testing"
	"time"
)

// TestServeStdioCanceledWhileBlocked pins the orphan-stdin false-negative: with
// stdin held open and silent, the old loop only checked ctx.Done() at the top and
// then blocked forever in readFrame. ServeStdio must observe cancellation and
// return ctx.Err() promptly instead of leaking past its parent's death.
func TestServeStdioCanceledWhileBlocked(t *testing.T) {
	srv := &Server{disableMCPDefer: true}
	pr, pw := io.Pipe()
	defer pw.Close()
	defer pr.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- srv.ServeStdio(ctx, pr, io.Discard)
	}()

	time.Sleep(100 * time.Millisecond)
	cancel()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("ServeStdio returned %v, want context.Canceled", err)
		}
	case <-time.After(1500 * time.Millisecond):
		t.Fatal("ServeStdio did not return within 1.5s after ctx cancel (blocking read swallowed cancellation)")
	}
}
