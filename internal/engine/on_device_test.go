package engine

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// odCall builds a tool call routed to a given engine with a given share scope — the
// inputs the on-device adapter and the residency floor both read.
func odCall(tool, args, engine string, scope abi.ShareScope) *abi.ToolCall {
	return &abi.ToolCall{
		Tool:   tool,
		Engine: engine,
		Args:   abi.Ref{Kind: abi.RefInline, Inline: []byte(args), Scope: scope},
	}
}

// TestOnDeviceEngineCompletes proves the reference adapter satisfies EngineDriver and
// relays the on-device runtime's output as an OK result with the engine meta set.
func TestOnDeviceEngineCompletes(t *testing.T) {
	var gotPrompt string
	e := NewOnDeviceEngine("", OnDeviceRuntimeFunc(func(ctx context.Context, prompt string) (string, error) {
		gotPrompt = prompt
		return `{"battery":"87%"}`, nil
	}))
	if e.ID != OnDeviceEngineID {
		t.Fatalf("empty id should default to %q, got %q", OnDeviceEngineID, e.ID)
	}

	res, err := e.Complete(context.Background(), odCall("read_battery", `{"unit":"pct"}`, OnDeviceEngineID, abi.ScopeAgent))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if res.Status != abi.StatusOK {
		t.Fatalf("status = %v, want StatusOK", res.Status)
	}
	if got := string(refBytes(context.Background(), res.Payload)); got != `{"battery":"87%"}` {
		t.Fatalf("payload = %q, want the runtime output", got)
	}
	if res.Meta["engine"] != OnDeviceEngineID {
		t.Fatalf("engine meta = %q, want %q", res.Meta["engine"], OnDeviceEngineID)
	}
	if res.Meta["output_tokens"] == "" {
		t.Fatal("output_tokens meta missing")
	}
	// The runtime sees the rendered tool call, not raw bytes — the prompt seam works.
	if !strings.Contains(gotPrompt, "read_battery") || !strings.Contains(gotPrompt, "pct") {
		t.Fatalf("prompt = %q, want it to carry the tool + args", gotPrompt)
	}
}

// TestOnDeviceEngineFailsClosed proves a wedged / unwired runtime yields a StatusError
// result (not a panic, not a silent OK) so the dispatch chain sees the miss.
func TestOnDeviceEngineFailsClosed(t *testing.T) {
	// A runtime that errors.
	e := NewOnDeviceEngine(OnDeviceEngineID, OnDeviceRuntimeFunc(func(ctx context.Context, prompt string) (string, error) {
		return "", errors.New("npu busy")
	}))
	res, err := e.Complete(context.Background(), odCall("read_battery", `{}`, OnDeviceEngineID, abi.ScopeAgent))
	if err != nil {
		t.Fatalf("Complete should not return a transport error for a runtime miss: %v", err)
	}
	if res.Status != abi.StatusError {
		t.Fatalf("status = %v, want StatusError on a runtime error", res.Status)
	}
	if !strings.Contains(res.Meta["error"], "npu busy") {
		t.Fatalf("error meta = %q, want the runtime error", res.Meta["error"])
	}

	// A nil runtime fails closed the same way.
	res2, _ := (&OnDeviceEngine{ID: OnDeviceEngineID}).Complete(context.Background(), odCall("x", `{}`, OnDeviceEngineID, abi.ScopeAgent))
	if res2.Status != abi.StatusError {
		t.Fatalf("nil runtime status = %v, want StatusError", res2.Status)
	}
}

// TestOnDeviceResidencyKeepsBytesOnBox is the issue's hinge: a tenant-scoped payload
// routed to the on-device engine is NOT denied by the engine-residency floor (the
// route is on-box), whereas the SAME payload routed to a remote engine IS denied. This
// is what "the bytes never leave the device" means at the gate.
func TestOnDeviceResidencyKeepsBytesOnBox(t *testing.T) {
	g := residencyGate{}
	ctx := context.Background()

	onBox := g.Adjudicate(ctx, odCall("read_contacts", `{}`, OnDeviceEngineID, abi.ScopeTenant))
	if onBox.Kind != abi.VerdictDefer {
		t.Fatalf("tenant payload on-device = %v, want VerdictDefer (no leak — route is on-box)", onBox.Kind)
	}

	remote := g.Adjudicate(ctx, odCall("read_contacts", `{}`, "litellm/gpt-4o", abi.ScopeTenant))
	if remote.Kind != abi.VerdictDeny {
		t.Fatalf("tenant payload to a remote engine = %v, want VerdictDeny (residency leak)", remote.Kind)
	}

	// A "on-device:1" instance id keeps the on-box recognition (family-prefix form).
	inst := g.Adjudicate(ctx, odCall("read_contacts", `{}`, OnDeviceEngineID+":1", abi.ScopeTenant))
	if inst.Kind != abi.VerdictDefer {
		t.Fatalf("tenant payload on-device:1 = %v, want VerdictDefer", inst.Kind)
	}
}

// Example_onDeviceEngine is the runnable, in-process reference for the on-device
// topology: wire a phone-class runtime behind the EngineDriver seam, dispatch a tool
// call to it, and confirm the engine-residency floor keeps a tenant-scoped payload
// on-box (and denies the same payload bound for a remote engine). Run it with:
//
//	go test ./internal/engine -run Example_onDeviceEngine
func Example_onDeviceEngine() {
	ctx := context.Background()

	// A deterministic stand-in for a phone-class runtime (llama.cpp / Ollama on-device).
	// A real adapter swaps this for the device's local API; the seam is unchanged.
	eng := NewOnDeviceEngine(OnDeviceEngineID, OnDeviceRuntimeFunc(
		func(ctx context.Context, prompt string) (string, error) {
			return `{"contacts": 3}`, nil
		}))

	// The on-device model proposes a tool call over a tenant-scoped payload. fak
	// adjudicates it IN-PROCESS first: on-box → allowed to proceed; remote → denied.
	call := odCall("read_contacts", `{"limit":3}`, OnDeviceEngineID, abi.ScopeTenant)
	gate := residencyGate{}
	onBox := gate.Adjudicate(ctx, call) // the on-device route
	leaky := gate.Adjudicate(ctx, odCall("read_contacts", `{"limit":3}`, "litellm/gpt-4o", abi.ScopeTenant))

	res, _ := eng.Complete(ctx, call)

	fmt.Printf("engine: %s\n", res.Meta["engine"])
	fmt.Printf("on-device tenant call denied: %v\n", onBox.Kind == abi.VerdictDeny)
	fmt.Printf("remote tenant call denied:    %v\n", leaky.Kind == abi.VerdictDeny)
	fmt.Printf("result: %s\n", refBytes(ctx, res.Payload))
	// Output:
	// engine: on-device
	// on-device tenant call denied: false
	// remote tenant call denied:    true
	// result: {"contacts": 3}
}
