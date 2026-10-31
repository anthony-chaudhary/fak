package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/sessionledger"
)

// A deadline-capable writer with a controlled slow first content write. The
// actual native handler, coalescer and write-deadline controller remain in use.
type nativeCollectorWriter struct {
	header                                http.Header
	body                                  bytes.Buffer
	mu                                    sync.Mutex
	blocked, release, interrupted         chan struct{}
	blockOnce, interruptOnce, releaseOnce sync.Once
	fail                                  bool
}

func (w *nativeCollectorWriter) Header() http.Header { return w.header }
func (w *nativeCollectorWriter) WriteHeader(int)     {}
func (w *nativeCollectorWriter) Flush()              {}
func (w *nativeCollectorWriter) FlushError() error   { return nil }
func (w *nativeCollectorWriter) SetWriteDeadline(d time.Time) error {
	if !d.IsZero() && !d.After(time.Now()) {
		w.interruptOnce.Do(func() { close(w.interrupted) })
	}
	return nil
}
func (w *nativeCollectorWriter) unblock() { w.releaseOnce.Do(func() { close(w.release) }) }
func (w *nativeCollectorWriter) Write(p []byte) (int, error) {
	block := false
	if bytes.Contains(p, []byte(`"type":"text_delta"`)) {
		w.blockOnce.Do(func() { close(w.blocked); block = true })
	}
	if block {
		select {
		case <-w.release:
		case <-w.interrupted:
			return 0, io.ErrClosedPipe
		}
		if w.fail {
			return 0, io.ErrClosedPipe
		}
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.body.Write(p)
}

// Method presence is not sufficient: failed deadline qualification must leave
// this writer on the original synchronous path.
type nativeCollectorUnsupportedWriter struct{ *nativeCollectorWriter }

func (w *nativeCollectorUnsupportedWriter) SetWriteDeadline(time.Time) error {
	return http.ErrNotSupported
}

// The channel handshake, not elapsed time, witnesses whether producing the
// remaining native deltas is independent of the blocked client write.
type nativeCollectorPlanner struct {
	blocked   <-chan struct{}
	generated chan struct{}
}

func (p *nativeCollectorPlanner) Model() string            { return "native-collector" }
func (p *nativeCollectorPlanner) StreamingSupported() bool { return true }
func (p *nativeCollectorPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return nil, errors.New("unexpected buffered planner")
}
func (p *nativeCollectorPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	if err := sink("hello "); err != nil {
		return nil, err
	}
	select {
	case <-p.blocked:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	for _, chunk := range []string{"world ST", "OPnever visible"} {
		if err := sink(chunk); err != nil {
			return nil, err
		}
	}
	close(p.generated)
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "hello world STOPnever visible"},
		FinishReason: "stop", Usage: agent.Usage{PromptTokens: 3, CompletionTokens: 7},
	}, nil
}

func TestNativeStreamCoalescesBehindSlowWriter(t *testing.T) {
	for _, mode := range []string{"complete", "cancel", "write_failure", "sync_unsupported"} {
		t.Run(mode, func(t *testing.T) {
			ledger, err := sessionledger.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			oldOpen := openNativeModelRequestLedger
			openNativeModelRequestLedger = func() (*sessionledger.Ledger, error) { return ledger, nil }
			t.Cleanup(func() { openNativeModelRequestLedger = oldOpen })
			srv := newTestServer(t)
			srv.native, srv.nativeMaxTurns = true, 1
			w := &nativeCollectorWriter{header: make(http.Header), blocked: make(chan struct{}), release: make(chan struct{}), interrupted: make(chan struct{}), fail: mode == "write_failure"}
			planner := &nativeCollectorPlanner{blocked: w.blocked, generated: make(chan struct{})}
			srv.planner = planner
			req, err := agent.DecodeAnthropicMessagesRequest([]byte(`{"model":"native-collector","max_tokens":32,"stream":true,"stop_sequences":["STOP"],"messages":[{"role":"user","content":"answer"}]}`))
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			r := httptest.NewRequest(http.MethodPost, "/v1/messages", nil).WithContext(ctx)
			done := make(chan struct{})
			var response http.ResponseWriter = w
			if mode == "sync_unsupported" {
				response = &nativeCollectorUnsupportedWriter{w}
			}
			go func() { defer close(done); srv.serveNativeMessagesStream(response, r, req, "collector-"+mode) }()
			t.Cleanup(func() { cancel(); w.unblock(); awaitNativeCollector(t, done) })
			awaitNativeCollector(t, w.blocked)
			select {
			case <-done:
				t.Fatal("handler finished before its blocked writer joined")
			default:
			}
			if mode == "sync_unsupported" {
				select {
				case <-planner.generated:
					t.Fatal("deadline-unsupported writer became asynchronous")
				default:
				}
				w.unblock()
				awaitNativeCollector(t, planner.generated)
			} else {
				awaitNativeCollector(t, planner.generated)
				if mode == "cancel" {
					cancel()
				} else {
					w.unblock()
				}
			}
			awaitNativeCollector(t, done)
			w.mu.Lock()
			body := w.body.String()
			w.mu.Unlock()
			if mode == "cancel" || mode == "write_failure" {
				if strings.Contains(body, "event: message_stop") || strings.Contains(body, "event: message_delta") {
					t.Fatalf("failed delivery emitted terminal success: %s", body)
				}
				return
			}
			var text strings.Builder
			for _, line := range strings.Split(body, "\n") {
				if !strings.HasPrefix(line, "data: ") {
					continue
				}
				var data struct{ Delta struct{ Type, Text string } }
				if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &data); err != nil {
					t.Fatal(err)
				}
				if data.Delta.Type == "text_delta" {
					text.WriteString(data.Delta.Text)
				}
			}
			if text.String() != "hello world " {
				t.Fatalf("stop holdback/content changed: %q", text.String())
			}
			if strings.Count(body, "event: message_stop\n") != 1 || strings.Count(body, "event: message_delta\n") != 1 {
				t.Fatalf("terminal frames not exactly once: %s", body)
			}
			textAt, progressAt, terminalAt := strings.LastIndex(body, "event: content_block_delta"), strings.Index(body, "event: turn_done"), strings.Index(body, "event: message_delta")
			if textAt < 0 || progressAt < textAt || terminalAt < progressAt {
				t.Fatalf("text/progress/terminal order changed: %s", body)
			}
		})
	}
}
