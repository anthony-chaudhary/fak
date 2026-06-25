package engine

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func lcCall(tool, args string) *abi.ToolCall {
	return &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)}}
}

// scriptedStream is a finite deterministic upstream token source: it returns its
// scripted ids then signals done — the offline stand-in for a vLLM/SGLang async
// token stream (the seam a real adapter reads).
type scriptedStream struct {
	toks []int
	i    int
}

func (s *scriptedStream) Next(ctx context.Context) (int, string, bool, error) {
	if err := ctx.Err(); err != nil {
		return 0, "", false, err
	}
	if s.i >= len(s.toks) {
		return 0, "", true, nil // upstream finished the turn
	}
	t := s.toks[s.i]
	s.i++
	return t, "", false, nil
}

// TestAdapterStreamsUpstream proves the external-adapter shape compiles against and
// satisfies the unchanged abi.LifecycleEngine, relaying the upstream's tokens one at
// a time and assembling the finished turn.
func TestAdapterStreamsUpstream(t *testing.T) {
	want := []int{5, 6, 7, 8}
	a := NewAdapterEngine("vllm-test", func(ctx context.Context, c *abi.ToolCall) (UpstreamStream, error) {
		return &scriptedStream{toks: append([]int(nil), want...)}, nil
	})
	if !abi.EngineSupportsLifecycle(a) {
		t.Fatal("adapter must report native lifecycle support")
	}
	if !abi.CapsHaveLifecycle(a.Caps()) {
		t.Fatal("adapter caps must advertise the lifecycle seam")
	}

	req, err := a.Admit(context.Background(), lcCall("ask", `{"q":"hi"}`))
	if err != nil {
		t.Fatalf("Admit: %v", err)
	}
	var streamed []int
	for tok := range req.Tokens() {
		streamed = append(streamed, tok.ID)
	}
	if len(streamed) != len(want) {
		t.Fatalf("streamed %v, want %v", streamed, want)
	}
	for i := range want {
		if streamed[i] != want[i] {
			t.Fatalf("token %d = %d, want %d", i, streamed[i], want[i])
		}
	}
	res, err := req.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if res == nil || res.Status != abi.StatusOK {
		t.Fatalf("result = %+v, want StatusOK", res)
	}
	if res.Meta["output_tokens"] != "4" {
		t.Fatalf("output_tokens = %q, want 4", res.Meta["output_tokens"])
	}
}

// countingStream is an unbounded upstream — only a ctx abort ends it. It stands in
// for an engine that would keep decoding until the client disconnects.
type countingStream struct{ n int }

func (s *countingStream) Next(ctx context.Context) (int, string, bool, error) {
	if err := ctx.Err(); err != nil {
		return 0, "", false, err
	}
	s.n++
	return s.n, "", false, nil
}

// TestAdapterAbortStopsMidStream proves ctx cancellation aborts the adapter
// mid-stream (it does not run to some fixed budget) and propagates to the upstream.
func TestAdapterAbortStopsMidStream(t *testing.T) {
	a := NewAdapterEngine("sglang-test", func(ctx context.Context, c *abi.ToolCall) (UpstreamStream, error) {
		return &countingStream{}, nil
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := a.Admit(ctx, lcCall("ask", `{}`))
	if err != nil {
		t.Fatalf("Admit: %v", err)
	}
	got := 0
	for range req.Tokens() {
		got++
		if got == 3 {
			cancel()
			break
		}
	}
	for range req.Tokens() {
		got++
	}
	if got >= 1000 {
		t.Fatalf("abort failed; streamed %d tokens from an unbounded upstream", got)
	}
	if _, err := req.Result(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Result err = %v, want context.Canceled", err)
	}
}
