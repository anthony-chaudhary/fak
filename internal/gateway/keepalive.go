package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Dialect represents the wire protocol dialect for streaming responses.
type Dialect string

const (
	DialectAnthropic Dialect = "anthropic"
	DialectOpenAI    Dialect = "openai"
)

// KeepaliveConfig holds configuration for the keepalive filter and repetition tripwire.
type KeepaliveConfig struct {
	SilenceInterval time.Duration
	Dialect         Dialect
	AbortCallback   func()
	MaxRepetition   int
}

// ErrRepetitionTripwire is returned when degenerate repetition is detected.
var ErrRepetitionTripwire = errors.New("repetition tripwire: degenerate repetition detected")

// RepetitionError provides details on the degenerate repetition that tripped the wire.
type RepetitionError struct {
	Token string
	Count int
}

func (e *RepetitionError) Error() string {
	return fmt.Sprintf("repetition tripwire: detected %d consecutive repetitive tokens (%q)", e.Count, e.Token)
}

func (e *RepetitionError) Is(target error) bool {
	return target == ErrRepetitionTripwire
}

// KeepaliveFilter implements dialect-conforming SSE keepalive injection, active
// process cancellation on client disconnect, and degenerate repetition tripwire detection.
type KeepaliveFilter struct {
	ctx          context.Context
	cancel       context.CancelFunc
	w            io.Writer
	cfg          KeepaliveConfig
	mu           sync.Mutex
	lastActivity time.Time
	silenceTimer *time.Timer
	startOnce    sync.Once
	stopOnce     sync.Once
	stopCh       chan struct{}
	doneCh       chan struct{}
	stopped      bool
	aborted      bool
	tripped      bool
	tripErr      error
	lastToken    string
	repeatCount  int
}

// NewKeepaliveFilter creates and starts a new keepalive filter.
func NewKeepaliveFilter(ctx context.Context, w io.Writer, cfg KeepaliveConfig) *KeepaliveFilter {
	if ctx == nil {
		ctx = context.Background()
	}
	cCtx, cancel := context.WithCancel(ctx)
	f := &KeepaliveFilter{
		ctx:          cCtx,
		cancel:       cancel,
		w:            w,
		cfg:          cfg,
		stopCh:       make(chan struct{}),
		doneCh:       make(chan struct{}),
		lastActivity: time.Now(),
	}
	f.Start()
	return f
}

// Start starts the keepalive silence timer and client context cancellation listener.
// It is idempotent and safe to call multiple times.
func (f *KeepaliveFilter) Start() {
	f.startOnce.Do(func() {
		f.mu.Lock()
		if f.stopCh == nil {
			f.stopCh = make(chan struct{})
		}
		if f.doneCh == nil {
			f.doneCh = make(chan struct{})
		}
		if f.ctx == nil {
			f.ctx, f.cancel = context.WithCancel(context.Background())
		}
		if f.cfg.SilenceInterval <= 0 {
			f.cfg.SilenceInterval = 10 * time.Second
		}
		if f.cfg.MaxRepetition <= 0 {
			f.cfg.MaxRepetition = 16
		}
		if f.cfg.Dialect == "" {
			f.cfg.Dialect = DialectOpenAI
		}
		f.lastActivity = time.Now()
		f.silenceTimer = time.NewTimer(f.cfg.SilenceInterval)
		f.mu.Unlock()

		go f.loop()
	})
}

// StartTicker is an alias for Start.
func (f *KeepaliveFilter) StartTicker() {
	f.Start()
}

// StartKeepalive is an alias for Start.
func (f *KeepaliveFilter) StartKeepalive() {
	f.Start()
}

func (f *KeepaliveFilter) loop() {
	defer close(f.doneCh)
	for {
		var timerCh <-chan time.Time
		f.mu.Lock()
		if f.silenceTimer != nil {
			timerCh = f.silenceTimer.C
		}
		f.mu.Unlock()

		select {
		case <-f.stopCh:
			return
		case <-f.ctx.Done():
			f.triggerAbort()
			return
		case now := <-timerCh:
			f.mu.Lock()
			if f.stopped || f.tripped {
				f.mu.Unlock()
				return
			}
			elapsed := now.Sub(f.lastActivity)
			if elapsed >= f.cfg.SilenceInterval {
				_ = f.emitKeepaliveLocked()
				f.lastActivity = now
				if f.silenceTimer != nil && !f.stopped && !f.tripped {
					f.silenceTimer.Reset(f.cfg.SilenceInterval)
				}
			} else {
				remaining := f.cfg.SilenceInterval - elapsed
				if remaining <= 0 {
					remaining = f.cfg.SilenceInterval
				}
				if f.silenceTimer != nil && !f.stopped && !f.tripped {
					f.silenceTimer.Reset(remaining)
				}
			}
			f.mu.Unlock()
		}
	}
}

func (f *KeepaliveFilter) triggerAbort() {
	f.mu.Lock()
	if f.aborted {
		f.mu.Unlock()
		return
	}
	f.aborted = true
	cb := f.cfg.AbortCallback
	cancel := f.cancel
	if f.silenceTimer != nil {
		f.silenceTimer.Stop()
	}
	f.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	if cb != nil {
		cb()
	}
}

// Stop halts the keepalive timer and listener goroutines.
func (f *KeepaliveFilter) Stop() {
	f.stopOnce.Do(func() {
		f.mu.Lock()
		f.stopped = true
		if f.silenceTimer != nil {
			f.silenceTimer.Stop()
		}
		f.mu.Unlock()
		if f.stopCh != nil {
			close(f.stopCh)
			if f.doneCh != nil {
				<-f.doneCh
			}
		}
	})
}

// Close implements io.Closer by calling Stop.
func (f *KeepaliveFilter) Close() error {
	f.Stop()
	return nil
}

func (f *KeepaliveFilter) resetSilenceLocked() {
	f.lastActivity = time.Now()
	if f.silenceTimer != nil && !f.stopped && !f.tripped {
		f.silenceTimer.Reset(f.cfg.SilenceInterval)
	}
}

// RecordActivity resets the silence timer due to upstream or wire activity.
func (f *KeepaliveFilter) RecordActivity() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.resetSilenceLocked()
}

// ResetSilence is an alias for RecordActivity.
func (f *KeepaliveFilter) ResetSilence() {
	f.RecordActivity()
}

func (f *KeepaliveFilter) emitKeepaliveLocked() error {
	if f.stopped || f.tripped || f.w == nil {
		return nil
	}
	var frame []byte
	switch f.cfg.Dialect {
	case DialectAnthropic:
		frame = []byte("event: ping\ndata: {}\n\n")
	case DialectOpenAI:
		fallthrough
	default:
		frame = []byte("data: {\"choices\":[{\"index\":0,\"delta\":{}}]}\n\n")
	}
	if _, err := f.w.Write(frame); err != nil {
		return err
	}
	if flusher, ok := f.w.(http.Flusher); ok {
		flusher.Flush()
	}
	return nil
}

// EmitKeepalive manually emits a keepalive frame immediately.
func (f *KeepaliveFilter) EmitKeepalive() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.emitKeepaliveLocked()
}

func extractToken(chunk string) string {
	if chunk == "" || chunk == "[DONE]" {
		return ""
	}
	trimmed := strings.TrimSpace(chunk)
	if trimmed == "[DONE]" || strings.HasPrefix(trimmed, "data: [DONE]") {
		return ""
	}

	if strings.HasPrefix(trimmed, "event:") || strings.HasPrefix(trimmed, "data:") {
		payload := trimmed
		for strings.HasPrefix(payload, "event:") {
			idx := strings.Index(payload, "\n")
			if idx == -1 {
				return ""
			}
			payload = strings.TrimSpace(payload[idx+1:])
		}
		if strings.HasPrefix(payload, "data:") {
			payload = strings.TrimSpace(strings.TrimPrefix(payload, "data:"))
		}
		if payload == "" || payload == "[DONE]" {
			return ""
		}
		if strings.HasPrefix(payload, "{") && strings.HasSuffix(payload, "}") {
			var parsed struct {
				Choices []struct {
					Delta struct {
						Content string `json:"content"`
					} `json:"delta"`
					Text string `json:"text"`
				} `json:"choices"`
				Delta struct {
					Text string `json:"text"`
				} `json:"delta"`
				Text string `json:"text"`
			}
			if err := json.Unmarshal([]byte(payload), &parsed); err == nil {
				if len(parsed.Choices) > 0 {
					if parsed.Choices[0].Delta.Content != "" {
						return parsed.Choices[0].Delta.Content
					}
					if parsed.Choices[0].Text != "" {
						return parsed.Choices[0].Text
					}
				}
				if parsed.Delta.Text != "" {
					return parsed.Delta.Text
				}
				if parsed.Text != "" {
					return parsed.Text
				}
				return ""
			}
		}
		return payload
	}

	return chunk
}

func (f *KeepaliveFilter) tripLocked(token string, count int) (error, func()) {
	f.tripped = true
	f.tripErr = &RepetitionError{Token: token, Count: count}
	if f.silenceTimer != nil {
		f.silenceTimer.Stop()
	}

	if f.w != nil {
		msg := fmt.Sprintf("repetition tripwire: detected %d consecutive repetitive tokens (%q)", count, token)
		switch f.cfg.Dialect {
		case DialectAnthropic:
			errObj := map[string]any{
				"type": "error",
				"error": map[string]any{
					"type":    "api_error",
					"message": msg,
				},
			}
			b, _ := json.Marshal(errObj)
			_, _ = fmt.Fprintf(f.w, "event: error\ndata: %s\n\n", b)
		case DialectOpenAI:
			fallthrough
		default:
			errObj := map[string]any{
				"error": map[string]any{
					"message": msg,
					"type":    "server_error",
					"code":    "repetition_tripwire",
				},
			}
			b, _ := json.Marshal(errObj)
			_, _ = fmt.Fprintf(f.w, "data: %s\n\ndata: [DONE]\n\n", b)
		}
		if flusher, ok := f.w.(http.Flusher); ok {
			flusher.Flush()
		}
	}

	var cb func()
	if !f.aborted {
		f.aborted = true
		cb = f.cfg.AbortCallback
		if f.cancel != nil {
			f.cancel()
		}
	}

	return f.tripErr, cb
}

func (f *KeepaliveFilter) inspectChunkLocked(chunk string) (error, func()) {
	if f.tripped {
		return f.tripErr, nil
	}

	token := extractToken(chunk)
	if token == "" {
		return nil, nil
	}

	maxRep := f.cfg.MaxRepetition
	if maxRep <= 0 {
		maxRep = 16
	}

	runes := []rune(token)
	if len(runes) > 1 {
		allSame := true
		for _, r := range runes[1:] {
			if r != runes[0] {
				allSame = false
				break
			}
		}
		if allSame {
			single := string(runes[0])
			if single == f.lastToken {
				f.repeatCount += len(runes)
			} else {
				f.lastToken = single
				f.repeatCount = len(runes)
			}
			if f.repeatCount >= maxRep {
				err, cb := f.tripLocked(single, f.repeatCount)
				return err, cb
			}
			return nil, nil
		}
	}

	if token == f.lastToken {
		f.repeatCount++
	} else {
		f.lastToken = token
		f.repeatCount = 1
	}

	if f.repeatCount >= maxRep {
		err, cb := f.tripLocked(token, f.repeatCount)
		return err, cb
	}

	return nil, nil
}

// InspectChunk inspects an incoming chunk for repetition tripwire and resets the silence timer.
// If the repetition tripwire is triggered, it terminates the request with a structured error
// and invokes AbortCallback.
func (f *KeepaliveFilter) InspectChunk(chunk string) error {
	var cb func()
	f.mu.Lock()
	if f.tripped {
		err := f.tripErr
		f.mu.Unlock()
		return err
	}
	err, abortCb := f.inspectChunkLocked(chunk)
	if abortCb != nil {
		cb = abortCb
	}
	f.resetSilenceLocked()
	f.mu.Unlock()

	if cb != nil {
		cb()
	}
	return err
}

// InspectChunkBytes inspects byte chunk for repetition tripwire.
func (f *KeepaliveFilter) InspectChunkBytes(chunk []byte) error {
	return f.InspectChunk(string(chunk))
}

// WriteChunk inspects the chunk for repetition, resets silence, and writes the chunk to the underlying writer.
func (f *KeepaliveFilter) WriteChunk(chunk string) error {
	var cb func()
	f.mu.Lock()
	if f.tripped {
		err := f.tripErr
		f.mu.Unlock()
		return err
	}
	err, abortCb := f.inspectChunkLocked(chunk)
	if abortCb != nil {
		cb = abortCb
	}
	if err != nil {
		f.mu.Unlock()
		if cb != nil {
			cb()
		}
		return err
	}
	f.resetSilenceLocked()
	if f.w != nil {
		if _, wErr := io.WriteString(f.w, chunk); wErr != nil {
			f.mu.Unlock()
			return wErr
		}
		if flusher, ok := f.w.(http.Flusher); ok {
			flusher.Flush()
		}
	}
	f.mu.Unlock()
	return nil
}

// Write writes chunk bytes to the filter.
func (f *KeepaliveFilter) Write(p []byte) (int, error) {
	if err := f.WriteChunk(string(p)); err != nil {
		return 0, err
	}
	return len(p), nil
}

// IsTripped returns true if the repetition tripwire was tripped.
func (f *KeepaliveFilter) IsTripped() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.tripped
}

// IsAborted returns true if active cancellation or tripwire aborted the filter.
func (f *KeepaliveFilter) IsAborted() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.aborted
}

// TripError returns the repetition trip error if tripped, or nil.
func (f *KeepaliveFilter) TripError() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.tripErr
}

// UpstreamContext returns a context derived from the client context that is cancelled
// if the client disconnects or if the repetition tripwire is triggered.
func (f *KeepaliveFilter) UpstreamContext() context.Context {
	return f.ctx
}

// Context returns UpstreamContext.
func (f *KeepaliveFilter) Context() context.Context {
	return f.ctx
}
