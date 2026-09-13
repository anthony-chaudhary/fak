package codelint

import (
	"context"
	"errors"
	"io"
	"testing"
	"time"
)

// TestLSPServerRunCanceledWhileBlocked pins the orphan-stdin false-negative: with
// stdin held open and silent, the old loop only checked ctx.Done() at the top and
// then blocked forever in readLSPFrame. Run must observe cancellation and return
// ctx.Err() promptly instead of leaking past its parent's death.
func TestLSPServerRunCanceledWhileBlocked(t *testing.T) {
	pr, pw := io.Pipe()
	defer pw.Close()
	defer pr.Close()

	srv := NewLSPServer(pr, io.Discard, nil)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- srv.Run(ctx)
	}()

	time.Sleep(100 * time.Millisecond)
	cancel()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Run returned %v, want context.Canceled", err)
		}
	case <-time.After(1500 * time.Millisecond):
		t.Fatal("Run did not return within 1.5s after ctx cancel (blocking read swallowed cancellation)")
	}
}
