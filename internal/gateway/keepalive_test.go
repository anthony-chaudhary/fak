package gateway

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"
	"time"
)

type safeBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (s *safeBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.buf.Write(p)
}

func (s *safeBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.buf.String()
}

func (s *safeBuffer) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.buf.Reset()
}

func TestKeepaliveSilenceAnthropic(t *testing.T) {
	buf := &safeBuffer{}
	cfg := KeepaliveConfig{
		SilenceInterval: 30 * time.Millisecond,
		Dialect:         DialectAnthropic,
	}
	filter := NewKeepaliveFilter(context.Background(), buf, cfg)
	defer filter.Stop()

	// Wait for silence interval to trigger keepalive ping
	time.Sleep(80 * time.Millisecond)

	out := buf.String()
	if !strings.Contains(out, "event: ping\ndata: {}\n\n") {
		t.Fatalf("expected Anthropic ping frame 'event: ping\\ndata: {}\\n\\n', got: %q", out)
	}
}

func TestKeepaliveSilenceOpenAI(t *testing.T) {
	buf := &safeBuffer{}
	cfg := KeepaliveConfig{
		SilenceInterval: 30 * time.Millisecond,
		Dialect:         DialectOpenAI,
	}
	filter := NewKeepaliveFilter(context.Background(), buf, cfg)
	defer filter.Stop()

	// Wait for silence interval to trigger keepalive ping
	time.Sleep(80 * time.Millisecond)

	out := buf.String()
	expected := "data: {\"choices\":[{\"index\":0,\"delta\":{}}]}\n\n"
	if !strings.Contains(out, expected) {
		t.Fatalf("expected OpenAI ping frame %q, got: %q", expected, out)
	}
}

func TestKeepaliveActivityResetsSilenceTimer(t *testing.T) {
	buf := &safeBuffer{}
	cfg := KeepaliveConfig{
		SilenceInterval: 50 * time.Millisecond,
		Dialect:         DialectAnthropic,
	}
	filter := NewKeepaliveFilter(context.Background(), buf, cfg)
	defer filter.Stop()

	// Emit activity every 15ms for 90ms (total time > SilenceInterval, but gap < SilenceInterval)
	for i := 0; i < 6; i++ {
		time.Sleep(15 * time.Millisecond)
		if err := filter.InspectChunk(fmt.Sprintf("unique_chunk_%d", i)); err != nil {
			t.Fatalf("unexpected InspectChunk error: %v", err)
		}
	}

	// Because chunks kept flowing, silence timer should have been reset and no ping emitted
	out := buf.String()
	if strings.Contains(out, "event: ping") {
		t.Fatalf("ping frame was emitted despite continuous activity: %q", out)
	}

	// Now stop activity and wait for silence threshold to elapse
	time.Sleep(80 * time.Millisecond)
	out = buf.String()
	if !strings.Contains(out, "event: ping") {
		t.Fatalf("expected ping frame after silence, got: %q", out)
	}
}

func TestKeepaliveClientContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	abortCalled := make(chan struct{})
	cfg := KeepaliveConfig{
		SilenceInterval: 10 * time.Second,
		Dialect:         DialectOpenAI,
		AbortCallback: func() {
			close(abortCalled)
		},
	}
	filter := NewKeepaliveFilter(ctx, io.Discard, cfg)
	defer filter.Stop()

	// Cancel the client context
	cancel()

	select {
	case <-abortCalled:
		// Succeeded
	case <-time.After(1 * time.Second):
		t.Fatal("AbortCallback was not invoked upon client context cancellation")
	}

	if !filter.IsAborted() {
		t.Fatal("expected filter.IsAborted() to be true")
	}
	if filter.Context().Err() == nil {
		t.Fatal("expected filter context to be cancelled")
	}
}

func TestKeepaliveRepetitionTripwire(t *testing.T) {
	t.Run("RepetitiveToken16Exclamation", func(t *testing.T) {
		buf := &safeBuffer{}
		aborted := false
		cfg := KeepaliveConfig{
			SilenceInterval: 10 * time.Second,
			Dialect:         DialectOpenAI,
			MaxRepetition:   16,
			AbortCallback: func() {
				aborted = true
			},
		}
		filter := NewKeepaliveFilter(context.Background(), buf, cfg)
		defer filter.Stop()

		// First 15 identical tokens should succeed without error
		for i := 0; i < 15; i++ {
			if err := filter.InspectChunk("!"); err != nil {
				t.Fatalf("chunk %d errored prematurely: %v", i, err)
			}
		}
		if aborted {
			t.Fatal("AbortCallback invoked prematurely before reaching threshold")
		}

		// 16th identical token must trip the wire
		err := filter.InspectChunk("!")
		if err == nil {
			t.Fatal("expected repetition tripwire to halt on 16th repetitive token, but err was nil")
		}
		if !errors.Is(err, ErrRepetitionTripwire) {
			t.Fatalf("expected ErrRepetitionTripwire, got: %v", err)
		}
		if !aborted {
			t.Fatal("expected AbortCallback to be invoked when tripwire tripped")
		}
		if !filter.IsTripped() {
			t.Fatal("expected filter.IsTripped() to be true")
		}

		out := buf.String()
		if !strings.Contains(out, "repetition_tripwire") {
			t.Fatalf("expected wire to contain structured error, got: %q", out)
		}
		if !strings.Contains(out, "data: [DONE]\n\n") {
			t.Fatalf("expected OpenAI wire to terminate with [DONE], got: %q", out)
		}

		// Subsequent chunks should also fail immediately
		if err := filter.InspectChunk("another_token"); err == nil {
			t.Fatal("expected subsequent chunk to fail after tripwire tripped")
		}
	})

	t.Run("RepetitiveWhitespace16", func(t *testing.T) {
		buf := &safeBuffer{}
		cfg := KeepaliveConfig{
			SilenceInterval: 10 * time.Second,
			Dialect:         DialectAnthropic,
			MaxRepetition:   16,
		}
		filter := NewKeepaliveFilter(context.Background(), buf, cfg)
		defer filter.Stop()

		for i := 0; i < 15; i++ {
			if err := filter.InspectChunk(" "); err != nil {
				t.Fatalf("whitespace chunk %d errored prematurely: %v", i, err)
			}
		}

		err := filter.InspectChunk(" ")
		if err == nil {
			t.Fatal("expected repetition tripwire to trip on 16th whitespace token")
		}

		out := buf.String()
		if !strings.Contains(out, "event: error\n") {
			t.Fatalf("expected Anthropic error event, got: %q", out)
		}
	})

	t.Run("RepetitiveSingleChunk16", func(t *testing.T) {
		buf := &safeBuffer{}
		cfg := KeepaliveConfig{
			SilenceInterval: 10 * time.Second,
			Dialect:         DialectOpenAI,
			MaxRepetition:   16,
		}
		filter := NewKeepaliveFilter(context.Background(), buf, cfg)
		defer filter.Stop()

		err := filter.InspectChunk("!!!!!!!!!!!!!!!!")
		if err == nil {
			t.Fatal("expected single chunk with 16 identical tokens to trip wire")
		}
	})

	t.Run("NonRepetitiveTokensDoNotTrip", func(t *testing.T) {
		buf := &safeBuffer{}
		cfg := KeepaliveConfig{
			SilenceInterval: 10 * time.Second,
			Dialect:         DialectOpenAI,
			MaxRepetition:   16,
		}
		filter := NewKeepaliveFilter(context.Background(), buf, cfg)
		defer filter.Stop()

		for i := 0; i < 30; i++ {
			if err := filter.InspectChunk(fmt.Sprintf("token_%d", i)); err != nil {
				t.Fatalf("unexpected tripwire error on distinct tokens at %d: %v", i, err)
			}
		}
		if filter.IsTripped() {
			t.Fatal("filter tripped unexpectedly on distinct tokens")
		}
	})

	t.Run("RepetitiveOpenAISSEChunks", func(t *testing.T) {
		buf := &safeBuffer{}
		cfg := KeepaliveConfig{
			SilenceInterval: 10 * time.Second,
			Dialect:         DialectOpenAI,
			MaxRepetition:   16,
		}
		filter := NewKeepaliveFilter(context.Background(), buf, cfg)
		defer filter.Stop()

		chunk := "data: {\"choices\":[{\"delta\":{\"content\":\"!\"}}]}\n\n"
		for i := 0; i < 15; i++ {
			if err := filter.InspectChunk(chunk); err != nil {
				t.Fatalf("chunk %d errored: %v", i, err)
			}
		}
		if err := filter.InspectChunk(chunk); err == nil {
			t.Fatal("expected repetition tripwire on 16th OpenAI SSE chunk")
		}
	})

	t.Run("RepetitiveAnthropicSSEChunks", func(t *testing.T) {
		buf := &safeBuffer{}
		cfg := KeepaliveConfig{
			SilenceInterval: 10 * time.Second,
			Dialect:         DialectAnthropic,
			MaxRepetition:   16,
		}
		filter := NewKeepaliveFilter(context.Background(), buf, cfg)
		defer filter.Stop()

		chunk := "event: content_block_delta\ndata: {\"delta\":{\"text\":\"!\"}}\n\n"
		for i := 0; i < 15; i++ {
			if err := filter.InspectChunk(chunk); err != nil {
				t.Fatalf("chunk %d errored: %v", i, err)
			}
		}
		if err := filter.InspectChunk(chunk); err == nil {
			t.Fatal("expected repetition tripwire on 16th Anthropic SSE chunk")
		}
	})
}

func TestKeepaliveWriteChunk(t *testing.T) {
	buf := &safeBuffer{}
	cfg := KeepaliveConfig{
		SilenceInterval: 10 * time.Second,
		Dialect:         DialectOpenAI,
		MaxRepetition:   16,
	}
	filter := NewKeepaliveFilter(context.Background(), buf, cfg)
	defer filter.Stop()

	for i := 0; i < 5; i++ {
		chunk := fmt.Sprintf("chunk_%d\n", i)
		if err := filter.WriteChunk(chunk); err != nil {
			t.Fatalf("WriteChunk error: %v", err)
		}
	}

	out := buf.String()
	for i := 0; i < 5; i++ {
		expected := fmt.Sprintf("chunk_%d\n", i)
		if !strings.Contains(out, expected) {
			t.Fatalf("expected buffer to contain %q, got: %q", expected, out)
		}
	}
}

func TestKeepaliveManualStartAndStop(t *testing.T) {
	f := &KeepaliveFilter{
		cfg: KeepaliveConfig{
			SilenceInterval: 100 * time.Millisecond,
			Dialect:         DialectOpenAI,
		},
	}
	f.Start()
	f.StartTicker()
	f.StartKeepalive()

	f.Stop()
	f.Stop()
	if err := f.Close(); err != nil {
		t.Fatalf("Close error: %v", err)
	}
}
