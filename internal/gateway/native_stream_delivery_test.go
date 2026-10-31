package gateway

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Only the client is synthetic. Tests execute the real native delivery helper,
// CoalescingSender, stop buffer and existing liveChat writer/deadline controller.
type nativeDeliveryClient struct {
	header                                http.Header
	mu                                    sync.Mutex
	frames                                []string
	blocked, release, interrupted         chan struct{}
	blockOnce, releaseOnce, interruptOnce sync.Once
	unsupported, failText, failProgress   bool
}

func (w *nativeDeliveryClient) Header() http.Header { return w.header }
func (w *nativeDeliveryClient) WriteHeader(int)     {}
func (w *nativeDeliveryClient) Flush()              {}
func (w *nativeDeliveryClient) FlushError() error   { return nil }
func (w *nativeDeliveryClient) SetWriteDeadline(d time.Time) error {
	if w.unsupported {
		return http.ErrNotSupported
	}
	if !d.IsZero() && !d.After(time.Now()) {
		w.interruptOnce.Do(func() { close(w.interrupted) })
	}
	return nil
}
func (w *nativeDeliveryClient) unblock() { w.releaseOnce.Do(func() { close(w.release) }) }
func (w *nativeDeliveryClient) Write(p []byte) (int, error) {
	text := string(p)
	block := false
	if strings.HasPrefix(text, "text:") {
		w.blockOnce.Do(func() { block = true; close(w.blocked) })
	}
	if block {
		select {
		case <-w.release:
		case <-w.interrupted:
			return 0, io.ErrClosedPipe
		}
		if w.failText {
			return 0, io.ErrClosedPipe
		}
	}
	if w.failProgress && strings.HasPrefix(text, "progress:") {
		return 0, io.ErrClosedPipe
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.frames = append(w.frames, text)
	return len(p), nil
}
func awaitNativeDelivery(t *testing.T, ch <-chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(3 * time.Second):
		t.Fatal("native delivery ownership boundary not reached")
	}
}

func TestNativeStreamDeliveryLifecycle(t *testing.T) {
	for _, mode := range []string{"async", "sync", "cancel", "write_failure", "progress_failure", "generation_error"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			client := &nativeDeliveryClient{header: make(http.Header), blocked: make(chan struct{}), release: make(chan struct{}), interrupted: make(chan struct{}), unsupported: mode == "sync", failText: mode == "write_failure", failProgress: mode == "progress_failure"}
			writer := &liveChatResponseWriter{syncResponseWriter: newSyncResponseWriter(client), cancel: cancel}
			writer.control = newLiveChatWriteControl(ctx, client)
			var cancels atomic.Int32
			stop := func() { cancels.Add(1); cancel() }
			generated, done := make(chan struct{}), make(chan struct{})
			generationCause := errors.New("generation failure")
			var genErr, deliveryErr error
			go func() {
				defer close(done)
				defer writer.control.close()
				buffer := NewStopHoldbackBuffer([]string{"STOP"})
				write := func(s string) error { _, _ = io.WriteString(writer, s); writer.Flush(); return writer.err() }
				text := func(s string) error {
					if safe := buffer.Append(s); safe != "" {
						return write("text:" + safe)
					}
					return nil
				}
				generate := func(emit, progress func(string) error) error {
					if err := emit("hello "); err != nil {
						return err
					}
					select {
					case <-client.blocked:
					case <-ctx.Done():
						return ctx.Err()
					}
					if err := emit("world ST"); err != nil {
						return err
					}
					if err := progress("P"); err != nil {
						return err
					}
					if err := emit("OPhidden"); err != nil {
						return err
					}
					close(generated)
					if mode == "generation_error" {
						return generationCause
					}
					return nil
				}
				genErr, deliveryErr = runNativeStreamDelivery(ctx, writer.control != nil, stop, generate, text, func(s string) error { return write("progress:" + s) })
				// This fixture's terminal is a caller action after the actual helper
				// returns. The separate AST contract binds the real handler's trailer.
				if genErr == nil && deliveryErr == nil && writer.err() == nil {
					if tail := buffer.Flush(); tail != "" {
						_ = write("text:" + tail)
					}
					if writer.err() == nil {
						_ = write("terminal")
					}
				}
			}()
			t.Cleanup(func() { cancel(); client.unblock(); awaitNativeDelivery(t, done) })
			awaitNativeDelivery(t, client.blocked)
			select {
			case <-done:
				t.Fatal("returned before joining blocked client")
			default:
			}
			if mode == "sync" {
				select {
				case <-generated:
					t.Fatal("synchronous control produced past blocked write")
				default:
				}
				client.unblock()
				awaitNativeDelivery(t, generated)
			} else {
				awaitNativeDelivery(t, generated)
				if mode == "cancel" {
					cancel()
				} else {
					client.unblock()
				}
			}
			awaitNativeDelivery(t, done)
			client.mu.Lock()
			frames := append([]string(nil), client.frames...)
			client.mu.Unlock()
			success := mode == "async" || mode == "sync"
			if success {
				if genErr != nil || deliveryErr != nil {
					t.Fatalf("successful delivery: %v / %v", genErr, deliveryErr)
				}
				var text strings.Builder
				progressAt, terminalAt := -1, -1
				terminalCount := 0
				for i, f := range frames {
					if strings.HasPrefix(f, "text:") {
						text.WriteString(strings.TrimPrefix(f, "text:"))
					}
					if f == "progress:P" {
						progressAt = i
						if strings.Contains(text.String(), "ST") {
							t.Fatal("stop prefix leaked before progress")
						}
					}
					if f == "terminal" {
						terminalAt = i
						terminalCount++
					}
				}
				if text.String() != "hello world " || progressAt < 0 || terminalAt != len(frames)-1 || terminalAt < progressAt || terminalCount != 1 {
					t.Fatalf("content/order/terminal changed: %+v", frames)
				}
				if cancels.Load() != 0 {
					t.Fatalf("successful lifecycle canceled %d times", cancels.Load())
				}
			} else {
				for _, f := range frames {
					if f == "terminal" {
						t.Fatalf("failed lifecycle emitted terminal: %+v", frames)
					}
				}
				if mode == "generation_error" {
					if !errors.Is(genErr, generationCause) || deliveryErr != nil {
						t.Fatalf("generation cause changed: %v / %v", genErr, deliveryErr)
					}
				} else if deliveryErr == nil || cancels.Load() != 1 {
					t.Fatalf("failure not canceled/joined exactly once: %v / %v cancel=%d", genErr, deliveryErr, cancels.Load())
				}
			}
		})
	}
}

func TestNativeStreamDeliveryPanicCleanup(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	entered, done := make(chan struct{}), make(chan struct{})
	cause := &struct{}{}
	var recovered any
	var cancels atomic.Int32
	go func() {
		defer close(done)
		defer func() { recovered = recover() }()
		_, _ = runNativeStreamDelivery(ctx, true, func() { cancels.Add(1); cancel() }, func(text, progress func(string) error) error {
			if err := text("x"); err != nil {
				return err
			}
			<-entered
			panic(cause)
		}, func(string) error { close(entered); <-ctx.Done(); return ctx.Err() }, func(string) error { return nil })
	}()
	awaitNativeDelivery(t, done)
	if recovered != cause || cancels.Load() != 1 {
		t.Fatalf("panic/cancel changed: %v cancel=%d", recovered, cancels.Load())
	}
}

func TestNativeStreamDeliveryKeepsBothErrors(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	generationCause, deliveryCause := errors.New("generator"), errors.New("writer")
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	defer once.Do(func() { close(release) })
	gen, delivery := runNativeStreamDelivery(ctx, true, cancel, func(text, progress func(string) error) error {
		if err := text("x"); err != nil {
			return err
		}
		<-entered
		once.Do(func() { close(release) })
		return generationCause
	}, func(string) error { close(entered); <-release; return deliveryCause }, func(string) error { return nil })
	if !errors.Is(gen, generationCause) || !errors.Is(delivery, deliveryCause) {
		t.Fatalf("lost independent errors: %v / %v", gen, delivery)
	}
}

func TestNativeStreamDeliveryConsumerPanic(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var cancels atomic.Int32
	gen, delivery := runNativeStreamDelivery(ctx, true, func() { cancels.Add(1); cancel() }, func(text, progress func(string) error) error {
		if err := text("x"); err != nil {
			return err
		}
		<-ctx.Done()
		return ctx.Err()
	}, func(string) error { panic("synthetic writer panic") }, func(string) error { return nil })
	if !errors.Is(gen, context.Canceled) || !errors.Is(delivery, errStreamDelivery) || cancels.Load() != 1 {
		t.Fatalf("consumer panic closure: %v / %v cancel=%d", gen, delivery, cancels.Load())
	}
}
