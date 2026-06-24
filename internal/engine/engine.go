// Package engine is the inference-engine seam (the EngineDriver). It ships the
// deterministic, offline building blocks the dispatch chain runs on without a
// live model or a GPU: a mock engine (the offline fallback) and a cassette
// record/replay transport, plus the engine-residency adjudicator that denies a
// tenant-scoped payload routed to a remote engine. The default engine is no longer
// this mock but the fused in-kernel model (internal/modelengine, id "inkernel").
//
// The live OpenAI-compatible HTTP client lives in internal/agent (HTTPPlanner) —
// the single outbound /v1/chat/completions seam (one base_url drives local vLLM
// vs a remote provider). An earlier degenerate HTTPEngine here was a second,
// never-wired copy of that client that spoke a bespoke `tool=X args=Y` prompt
// instead of real tool-calling (TICKETS T4); it was deleted so there is exactly
// one OpenAI client, an invariant pinned by architest's TestSingleOpenAIChatClient.
package engine

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Usage is the token accounting extracted from a completion (unit 42).
type Usage struct {
	InputTokens  int `json:"prompt_tokens"`
	OutputTokens int `json:"completion_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// ---------------------------------------------------------------------------
// Mock engine — deterministic, offline. Registered as "mock" (the offline fallback;
// the default engine is the fused in-kernel model, id "inkernel").
// ---------------------------------------------------------------------------

// Mock answers any call with a deterministic synthetic result derived from the
// tool + args, and a simulated token usage proportional to payload size. It lets
// the whole dispatch chain run with zero network.
type Mock struct{ calls int64 }

// Caps reports the engine's capabilities; the mock declares none.
func (m *Mock) Caps() []abi.Capability { return nil }

func (m *Mock) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	m.calls++
	in := refBytes(ctx, c.Args)
	body := fmt.Sprintf(`{"tool":%q,"echo":%q,"ok":true}`, c.Tool, truncate(in, 256))
	ref := putBytes(ctx, []byte(body))
	u := Usage{InputTokens: 50 + len(in)/4, OutputTokens: len(body) / 4}
	u.TotalTokens = u.InputTokens + u.OutputTokens
	return &abi.Result{
		Call:    c,
		Payload: ref,
		Status:  abi.StatusOK,
		Meta: map[string]string{
			"engine":        "mock",
			"input_tokens":  itoa(u.InputTokens),
			"output_tokens": itoa(u.OutputTokens),
		},
	}, nil
}

// ---------------------------------------------------------------------------
// Cassette transport — record/replay. Deterministic, offline (unit 41).
// ---------------------------------------------------------------------------

// cassetteEntry is one recorded (request -> response) interaction keyed by a
// content hash of (tool, args).
type cassetteEntry struct {
	Key      string            `json:"key"`
	Tool     string            `json:"tool"`
	Response json.RawMessage   `json:"response"` // raw result payload bytes
	Usage    Usage             `json:"usage"`
	Meta     map[string]string `json:"meta,omitempty"`
}

// Cassette is a loaded set of recorded interactions. A miss in replay mode is an
// error (deterministic: a replay must cover its trace).
type Cassette struct {
	mu      sync.Mutex
	entries map[string]cassetteEntry
}

func callKey(tool string, args []byte) string {
	h := sha256.New()
	h.Write([]byte(tool))
	h.Write([]byte{0})
	h.Write(args)
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// CallKey is the exported content-address of a (tool, args) pair — used to author
// cassette entries whose keys the CassetteEngine will match on replay.
func CallKey(tool string, args []byte) string { return callKey(tool, args) }

// LoadCassette reads a cassette JSON file ({"entries":[...]}).
func LoadCassette(path string) (*Cassette, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var doc struct {
		Entries []cassetteEntry `json:"entries"`
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		return nil, err
	}
	c := &Cassette{entries: map[string]cassetteEntry{}}
	for _, e := range doc.Entries {
		c.entries[e.Key] = e
	}
	return c, nil
}

// CassetteEngine replays a cassette; on miss it errors (StatusError result).
type CassetteEngine struct{ c *Cassette }

// NewCassetteEngine wraps a loaded Cassette in an engine that replays its recorded
// (tool, args) -> response interactions.
func NewCassetteEngine(c *Cassette) *CassetteEngine { return &CassetteEngine{c} }

// Caps reports the engine's capabilities; the cassette replayer declares none.
func (e *CassetteEngine) Caps() []abi.Capability { return nil }

func (e *CassetteEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	args := refBytes(ctx, c.Args)
	key := callKey(c.Tool, args)
	e.c.mu.Lock()
	ent, ok := e.c.entries[key]
	e.c.mu.Unlock()
	if !ok {
		return &abi.Result{Call: c, Status: abi.StatusError,
			Meta: map[string]string{"engine": "cassette", "error": "cassette miss: " + key}}, nil
	}
	ref := putBytes(ctx, ent.Response)
	return &abi.Result{Call: c, Payload: ref, Status: abi.StatusOK,
		Meta: map[string]string{
			"engine":        "cassette",
			"input_tokens":  itoa(ent.Usage.InputTokens),
			"output_tokens": itoa(ent.Usage.OutputTokens),
		}}, nil
}

// ---------------------------------------------------------------------------
// helpers + registration
// ---------------------------------------------------------------------------

func refBytes(ctx context.Context, r abi.Ref) []byte {
	if r.Kind == abi.RefInline {
		return r.Inline
	}
	if res := abi.ActiveResolver(); res != nil {
		if b, err := res.Resolve(ctx, r); err == nil {
			return b
		}
	}
	return nil
}

func putBytes(ctx context.Context, b []byte) abi.Ref {
	if res := abi.ActiveResolver(); res != nil {
		if ref, err := res.Put(ctx, b); err == nil {
			return ref
		}
	}
	return abi.Ref{Kind: abi.RefInline, Inline: b, Len: int64(len(b))}
}

func truncate(b []byte, n int) string {
	if len(b) > n {
		return string(b[:n])
	}
	return string(b)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// MockEngine is the registered default (offline-safe). Tests + the bench can also
// register a CassetteEngine under another id and select it. The live
// OpenAI-compatible client is internal/agent's HTTPPlanner, not this package.
var MockEngine = &Mock{}

func init() {
	RegisterResidencyGate()
	abi.RegisterEngine("mock", MockEngine)
	abi.RegisterCapability("engine.route")
	abi.RegisterCapability("engine.residency")
	abi.RegisterCapability("engine.openai")
}

// RegisterResidencyGate installs the engine-residency adjudicator (rank 12) — the
// same registration init() performs at boot. Exported so a test that resets the ABI
// registry (abi.ResetForTest) can reinstall the REAL gate rather than a hand-rolled
// double: e.g. internal/gateway proving a per-call model route written pre-submit is
// still DENIED for a tenant payload bound for a remote engine. Idempotent (the
// adjudicator registry is keyed by rank).
func RegisterResidencyGate() { abi.RegisterAdjudicator(12, residencyGate{}) }

type residencyGate struct{}

// Caps reports the residency gate's capabilities: engine.route and engine.residency.
func (residencyGate) Caps() []abi.Capability {
	return []abi.Capability{"engine.route", "engine.residency"}
}

// Adjudicate denies a tenant-scoped (or otherwise sensitivity-tagged) call routed to a
// remote engine, deferring everything else — the engine-residency data-leak gate.
func (residencyGate) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	if c == nil || c.Engine == "" || !sensitiveRoute(c) || !remoteRoute(c.Engine) {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "engine-residency"}
	}
	return abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonTrustViolation,
		By:     "engine-residency",
		Payload: abi.WitnessPayload{
			Claim: "tenant-scoped payload routed to remote engine",
		},
		Meta: map[string]string{
			"engine_route": c.Engine,
			"scope":        "tenant",
		},
	}
}

func sensitiveRoute(c *abi.ToolCall) bool {
	if c.Args.Scope == abi.ScopeTenant {
		return true
	}
	tag := ""
	if c.Meta != nil {
		tag = c.Meta["sensitivity"]
		if tag == "" {
			tag = c.Meta["data_sensitivity"]
		}
	}
	switch strings.ToLower(strings.TrimSpace(tag)) {
	case "sensitive", "tenant", "confidential", "secret", "pii":
		return true
	default:
		return false
	}
}

// remoteRoute reports whether an engine route leaves this box — the residency
// floor's "is this engine remote?" predicate. It is FAIL-CLOSED: only a route the
// kernel can prove names a KNOWN on-box engine family (the fused in-kernel model,
// the offline mock, a cassette replay, or a route an operator explicitly marked
// local / on-device / kernel) is treated as local; EVERY other route is remote.
//
// This is the posture the model-routing wiring contract requires. A routing Plan
// member may bind to ANY backend in the ecosystem — a LiteLLM proxy fronting 100+
// providers, an OpenRouter model id, a direct provider wire (openai / anthropic /
// gemini / xai), or a user's own gateway ("their thing") — and the kernel cannot
// enumerate them. So an engine id the kernel cannot PROVE is local is assumed to
// leave the box, and a tenant-scoped / sensitivity-tagged payload routed to it is
// DENIED rather than leaked. The previous allow-list-of-remote-names form failed
// OPEN for exactly the aggregator/custom routes this integration makes first-class
// (e.g. "litellm/gpt-4o", "openrouter/…", "my-proxy" matched none of the known
// remote substrings, so a sensitive payload sailed past the floor). An empty route
// ("" => the in-kernel kernel default) is local.
func remoteRoute(route string) bool {
	route = strings.ToLower(strings.TrimSpace(route))
	if route == "" {
		return false // empty => the in-kernel kernel default (on-box)
	}
	return !localRoute(route)
}

// localRoute reports whether an engine id names a KNOWN on-box engine family: the
// fused in-kernel model, the offline mock, a cassette replay, or a route an
// operator explicitly marked local / on-device / kernel. It matches a bare family
// name ("inkernel"), a "family:instance" ("on-device:0"), a "family-suffix"
// ("local-gpu"), or a "family/path" ("local/llama") form; anything else is remote
// (fail-closed — see remoteRoute). A new on-box engine an operator wires SHOULD
// carry one of these prefixes so the residency floor recognizes it as local.
//
// MIRROR: internal/modelroute.IsRemoteRoute keeps a tier-1 copy of this same
// on-box family list (it cannot import this layer); the two MUST stay in sync so a
// `fak route` residency label never disagrees with the floor that enforces it.
func localRoute(route string) bool {
	for _, local := range []string{"inkernel", "mock", "cassette", "local", "on-device", "ondevice", "kernel"} {
		if route == local ||
			strings.HasPrefix(route, local+":") ||
			strings.HasPrefix(route, local+"-") ||
			strings.HasPrefix(route, local+"/") {
			return true
		}
	}
	return false
}
