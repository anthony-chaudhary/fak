package abi

import (
	"context"
	"testing"
)

// oneShotEngine implements ONLY EngineDriver (no Admit) — the wave-0 shape every
// existing engine has. It must still drive through the lifecycle seam via the shim.
type oneShotEngine struct{ body string }

func (oneShotEngine) Caps() []Capability { return nil }
func (e oneShotEngine) Complete(ctx context.Context, c *ToolCall) (*Result, error) {
	return &Result{Status: StatusOK, Meta: map[string]string{"body": e.body}}, nil
}

// TestAdmitOrShimDrivesOneShotEngine proves a one-shot-only engine "still registers
// and runs" through the lifecycle bridge: it is NOT reported as lifecycle-capable,
// its shim streams ZERO tokens (one-shot has no per-token stream) and closes, and
// Result() returns exactly what Complete produced.
func TestAdmitOrShimDrivesOneShotEngine(t *testing.T) {
	eng := oneShotEngine{body: "hi"}
	if EngineSupportsLifecycle(eng) {
		t.Fatal("a one-shot engine must not report native lifecycle support")
	}
	if CapsHaveLifecycle(eng.Caps()) {
		t.Fatal("a one-shot engine must not advertise the lifecycle cap")
	}
	req, err := AdmitOrShim(context.Background(), eng, &ToolCall{Tool: "t"})
	if err != nil {
		t.Fatalf("AdmitOrShim: %v", err)
	}
	n := 0
	for range req.Tokens() {
		n++
	}
	if n != 0 {
		t.Fatalf("one-shot shim streamed %d tokens, want 0", n)
	}
	res, err := req.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if res == nil || res.Meta["body"] != "hi" {
		t.Fatalf("shim dropped the one-shot result: %+v", res)
	}
}

// fakeLifecycle implements the full lifecycle: a native Admit that streams three
// tokens. It embeds oneShotEngine only to inherit a trivial Complete.
type fakeLifecycle struct{ oneShotEngine }

func (fakeLifecycle) Caps() []Capability { return []Capability{EngineLifecycleCap} }
func (fakeLifecycle) Admit(ctx context.Context, c *ToolCall) (EngineRequest, error) {
	r := &fakeReq{tokens: make(chan EngineToken, 3), done: make(chan struct{})}
	go func() {
		for i := 0; i < 3; i++ {
			r.tokens <- EngineToken{ID: i}
		}
		r.res = &Result{Status: StatusOK, Meta: map[string]string{"streamed": "yes"}}
		close(r.tokens)
		close(r.done)
	}()
	return r, nil
}

type fakeReq struct {
	tokens chan EngineToken
	done   chan struct{}
	res    *Result
	err    error
}

func (r *fakeReq) Tokens() <-chan EngineToken { return r.tokens }
func (r *fakeReq) Result() (*Result, error)   { <-r.done; return r.res, r.err }
func (r *fakeReq) Cancel()                    {}

// TestAdmitOrShimUsesNativeAdmit proves the bridge prefers a real lifecycle engine's
// native Admit (a streamed three-token decode), not the buffered shim.
func TestAdmitOrShimUsesNativeAdmit(t *testing.T) {
	eng := fakeLifecycle{}
	if !EngineSupportsLifecycle(eng) {
		t.Fatal("a lifecycle engine must report native support")
	}
	if !CapsHaveLifecycle(eng.Caps()) {
		t.Fatal("a lifecycle engine must advertise the lifecycle cap")
	}
	req, err := AdmitOrShim(context.Background(), eng, &ToolCall{Tool: "t"})
	if err != nil {
		t.Fatalf("AdmitOrShim: %v", err)
	}
	n := 0
	for range req.Tokens() {
		n++
	}
	if n != 3 {
		t.Fatalf("native admit streamed %d tokens, want 3", n)
	}
	res, err := req.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if res == nil || res.Meta["streamed"] != "yes" {
		t.Fatalf("native result lost: %+v", res)
	}
}
