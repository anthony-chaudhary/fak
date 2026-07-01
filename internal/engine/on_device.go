package engine

// on_device.go — the reference phone-class on-device EngineDriver (issue #1041,
// epic #633 "edge/mobile/IoT", Tier 2 item 1).
//
// This is the ADAPTER, not new kernel surface: the EngineDriver seam already ships
// (abi.EngineDriver.Complete + the engine-residency floor in engine.go). What it adds
// is a runnable, tested reference for the "fak is the gate, your NPU runtime is the
// engine" topology — an on-device runtime (llama.cpp / Ollama on the phone) answers a
// tool call locally, and fak adjudicates that call IN-PROCESS before it can touch a
// camera / contacts / payment intent. The adapter proves the gate sits in front of a
// PHONE-NATIVE engine, not just a cloud one.
//
// The on-device runtime is a PLUGGABLE seam (OnDeviceRuntime) so this compiles and is
// fully testable with NO model, GPU, or NPU — a test drives it with a deterministic
// scripted runtime, exactly as the mock/cassette/adapter engines run the dispatch
// chain offline. A real adapter implements OnDeviceRuntime over a phone-class engine:
// llama.cpp embedded via CGo or its llama-server child process, or Ollama's local
// daemon. Per the issue's non-goals it does NOT re-grow an OpenAI HTTP client (the one
// live client lives in internal/agent, pinned by architest's TestSingleOpenAIChatClient)
// and it does NOT try to beat the vendor runtime on throughput — fak fronts it, never
// competes with it.
//
// RESIDENCY: the default id is "on-device", which the engine-residency floor
// (engine.go:localRoute) recognizes as an ON-BOX engine family. So a tenant-scoped /
// sensitivity-tagged payload routed to this engine is NOT denied by the remote-leak
// gate — the whole point of a phone-native engine is that the bytes never leave the
// device. A custom id MUST keep a local family prefix (on-device / local / kernel …)
// or the floor will treat the route as remote and deny sensitive payloads.

import (
	"context"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// OnDeviceEngineID is the default engine id the residency floor recognizes as on-box.
// Register the reference engine under it (or a "on-device:<instance>" variant) so a
// tenant-scoped payload routed to a phone-native runtime is not denied as a remote leak.
const OnDeviceEngineID = "on-device"

// OnDeviceRuntime is the minimal completion surface a phone-class on-device engine
// exposes: given the prompt for one tool call, return the generated text. It is the
// single-shot counterpart of the server-class UpstreamStream (lifecycle_adapter.go) —
// a phone runtime answers a whole turn locally rather than streaming an async token
// API. A real adapter implements it over llama.cpp (CGo or a llama-server child) or
// Ollama's local daemon; ctx bounds a stall so a wedged device surfaces as an error,
// not a hang.
type OnDeviceRuntime interface {
	Generate(ctx context.Context, prompt string) (string, error)
}

// OnDeviceRuntimeFunc adapts a plain function to an OnDeviceRuntime, so a reference or
// a test can wire a runtime without declaring a named type.
type OnDeviceRuntimeFunc func(ctx context.Context, prompt string) (string, error)

// Generate calls the wrapped function.
func (f OnDeviceRuntimeFunc) Generate(ctx context.Context, prompt string) (string, error) {
	return f(ctx, prompt)
}

// OnDeviceEngine is the reference EngineDriver wrapping a phone-class on-device
// runtime. It satisfies the bare abi.EngineDriver (Complete + Caps); the kernel
// dispatches an admitted tool call to it exactly as it would the in-kernel or mock
// engine. ID names the runtime for the result meta + Caps and keys residency; Runtime
// is the pluggable phone-class backend.
type OnDeviceEngine struct {
	ID      string
	Runtime OnDeviceRuntime
}

// NewOnDeviceEngine builds the reference on-device engine over a runtime. An empty id
// defaults to OnDeviceEngineID so the residency floor recognizes the route as on-box;
// pass a "on-device:<instance>" form to register several devices.
func NewOnDeviceEngine(id string, rt OnDeviceRuntime) *OnDeviceEngine {
	if id == "" {
		id = OnDeviceEngineID
	}
	return &OnDeviceEngine{ID: id, Runtime: rt}
}

// Caps advertises the on-device engine family plus this instance's id token. It does
// NOT advertise engine.openai: this adapter speaks to a local runtime, not the OpenAI
// HTTP wire (which has exactly one client, in internal/agent).
func (e *OnDeviceEngine) Caps() []abi.Capability {
	return []abi.Capability{"engine.ondevice", abi.Capability("engine.ondevice." + e.ID)}
}

// WeightBearing declares that the on-device runtime runs a local model-forward.
func (e *OnDeviceEngine) WeightBearing() bool { return true }

// Complete runs the tool call against the on-device runtime and returns the assembled
// result. It FAILS CLOSED: a nil runtime or a runtime error yields a StatusError result
// (never a panic, never a silent StatusOK), so the dispatch chain sees the miss rather
// than treating a wedged device as a successful empty answer.
func (e *OnDeviceEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	if e.Runtime == nil {
		return &abi.Result{Call: c, Status: abi.StatusError,
			Meta: map[string]string{"engine": e.ID, "error": "on-device runtime not wired"}}, nil
	}
	prompt := onDevicePrompt(c, refBytes(ctx, c.Args))
	out, err := e.Runtime.Generate(ctx, prompt)
	if err != nil {
		return &abi.Result{Call: c, Status: abi.StatusError,
			Meta: map[string]string{"engine": e.ID, "error": err.Error()}}, nil
	}
	ref := putBytes(ctx, []byte(out))
	// Token accounting is the same coarse offline estimate the mock engine emits (the
	// runtime returns text, not its own token ids); a real adapter should overwrite
	// these with the usage its phone-class engine reports. Kept for parity so the
	// dispatch chain's Meta consumers see the same keys for every offline engine.
	u := Usage{InputTokens: 50 + len(prompt)/4, OutputTokens: len(out) / 4}
	u.TotalTokens = u.InputTokens + u.OutputTokens
	return &abi.Result{
		Call:    c,
		Payload: ref,
		Status:  abi.StatusOK,
		Meta: map[string]string{
			"engine":        e.ID,
			"input_tokens":  itoa(u.InputTokens),
			"output_tokens": itoa(u.OutputTokens),
		},
	}, nil
}

// onDevicePrompt renders the tool call into the prompt the on-device runtime decodes.
// A real adapter formats this for its model's chat/tool template; the reference keeps
// it a deterministic "tool + args" string so the seam is testable without a tokenizer.
func onDevicePrompt(c *abi.ToolCall, args []byte) string {
	return fmt.Sprintf("tool=%s args=%s", c.Tool, string(args))
}

// OnDeviceEngine satisfies the bare EngineDriver — the same interface the in-kernel,
// mock, cassette, and external-adapter engines implement.
var _ abi.EngineDriver = (*OnDeviceEngine)(nil)
