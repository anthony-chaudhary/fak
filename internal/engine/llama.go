package engine

// llama.go — the llama.cpp / llama-server session-cache OBSERVATION adapter
// (cache-frontier "next 50" item 35, docs/cache-frontier/DEFAULT-ENABLEMENT-NEXT-50.md,
// issue #1553, under epic #1490 "turn the vCache gates ON by default + honest
// per-mechanism attribution").
//
// It is the local-engine sibling of the vLLM (#1551) and SGLang (#1552) adapters:
// each maps its engine's cache signal onto the SAME wire-neutral
// engine.CacheCapability contract (item 32 / #1550, cachecapability.go), so a
// llama-backed local session reports on identical axes to the server-class engines.
//
// What it reads: llama-server's PUBLIC read-only /slots telemetry. When --slots is
// enabled, /slots returns a JSON array of per-slot objects that carry prompt-cache
// accounting (n_past / cache_tokens / n_prompt_tokens_processed) — the SYMPTOM of a
// session/prompt cache. When --slots is disabled (the default on many builds), /slots
// answers 501 and no such surface exists.
//
// Honesty boundary — this is WHY the adapter exists and what it must never do:
//
//   - llama-server's session cache is largely PASSIVE. It exposes symptoms (the
//     /slots accounting counters) but no control surface to warm a named prefix or
//     evict an exact span. So the honest verdict when a signal IS present is
//     CachePassiveObserve, NEVER an active-warm/exact-evict class (item 36's
//     conformance test polices that "fronted != cache-integrated" boundary).
//   - When NO signal is present (endpoint disabled, empty, or slots without the
//     accounting field) the verdict is CacheUnknown with an explicit passive /
//     no-evidence Evidence string — never a fabricated cache-state number.
//   - The evidence provenance is ProvenanceProvider: an OBSERVED provider counter,
//     kept separate from a fak in-kernel witness (#1490's honest attribution).
//   - Observation changes NO serving path. This adapter does not generate, warm, or
//     evict, so ColdPathCorrect stays true: a session-cache miss still sends the full
//     required context (item 38's cold-path rule). It is not registered as an
//     abi.EngineDriver — it only produces a CacheCapability behind the item-32 seam.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
)

// LlamaEngineID is the opaque engine label a llama.cpp/llama-server capability is
// reported under. It matches the "llama.cpp" token the CacheCapability contract names
// as an example, so the report and the inventory (item 31) agree on engine identity.
const LlamaEngineID = "llama.cpp"

// LlamaConfig wires one llama.cpp/llama-server instance through its PUBLIC read-only
// surface only — the /slots telemetry endpoint. It deliberately adds no generation
// path: this adapter observes a cache signal, it does not serve.
type LlamaConfig struct {
	// BaseURL is the llama-server HTTP root (host:port llama-server binds). /slots is
	// derived from it when SlotsURL is empty.
	BaseURL string
	// SlotsURL overrides the derived <BaseURL>/slots endpoint when a proxy/bridge
	// exposes the slots telemetry at a non-default path.
	SlotsURL string
	// APIKey is an optional bearer token for a fronted llama-server.
	APIKey string
	// Client is an optional HTTP client; nil falls back to the shared default.
	Client *http.Client
}

// EnvLlamaConfig returns the default llama observation configuration from the
// FAK_LLAMA_* environment. FAK_LLAMA_BASE_URL should point at the llama-server HTTP
// root; FAK_LLAMA_SLOTS_URL overrides the derived /slots path.
func EnvLlamaConfig() LlamaConfig {
	return LlamaConfig{
		BaseURL:  os.Getenv("FAK_LLAMA_BASE_URL"),
		SlotsURL: os.Getenv("FAK_LLAMA_SLOTS_URL"),
		APIKey:   os.Getenv("FAK_LLAMA_API_KEY"),
	}
}

// LlamaSessionCache is the decoded session-cache signal read from a llama-server
// /slots response. It records only WHETHER a session-cache symptom surface is exposed
// and which accounting fields carried it — never a claimed reuse figure, so the
// passive/no-evidence honesty boundary cannot be crossed by accident.
type LlamaSessionCache struct {
	// Present is true when /slots exposed an observable session-cache symptom: at
	// least one slot carrying a prompt-cache accounting field. False when the endpoint
	// is disabled, empty, or exposes no such field.
	Present bool
	// Slots is the number of slots the response covered.
	Slots int
	// Fields names the prompt-cache accounting fields observed on the slots (e.g.
	// "cache_tokens", "n_past"), sorted and de-duplicated. It is the human anchor for
	// the passive-observe verdict, not a reuse count.
	Fields []string
	// Note explains a signal-ABSENT observation (endpoint disabled, empty body, no
	// accounting field). Empty when Present.
	Note string
}

// Capability maps the decoded signal onto the wire-neutral engine.CacheCapability
// model. A present symptom surface is CachePassiveObserve; an absent one is the
// safe CacheUnknown with an explicit passive/no-evidence Evidence string. Both
// carry ProvenanceProvider (an observed provider counter, not a kernel witness) and
// ColdPathCorrect true (observation never changes the request's cold path).
func (s LlamaSessionCache) Capability() CacheCapability {
	out := CacheCapability{
		Engine:          LlamaEngineID,
		Provenance:      ProvenanceProvider,
		ColdPathCorrect: true,
	}
	if s.Present {
		out.Verdict = CachePassiveObserve
		out.Evidence = fmt.Sprintf(
			"llama-server /slots exposes per-slot prompt-cache accounting across %d slot(s) (fields: %s); passive-observe, no warm/evict control surface",
			s.Slots, strings.Join(s.Fields, ","))
		return out
	}
	out.Verdict = CacheUnknown
	note := s.Note
	if note == "" {
		note = "no /slots session-cache signal"
	}
	out.Evidence = "llama-server exposed no session-cache signal (" + note + "); passive / no-evidence, no cache-state number fabricated"
	return out
}

// llamaSlotWire decodes the session-cache-relevant fields of one llama-server /slots
// entry. The accounting counters are pointers so a MISSING field (nil) is
// distinguishable from a present-but-zero field — the difference between "no symptom
// surface" and "a surface reporting zero reuse this turn".
type llamaSlotWire struct {
	ID               int    `json:"id"`
	State            *int   `json:"state"`
	CacheTokens      *int64 `json:"cache_tokens"`
	NPast            *int64 `json:"n_past"`
	NPromptProcessed *int64 `json:"n_prompt_tokens_processed"`
}

// decodeLlamaSlots turns a raw /slots body into a LlamaSessionCache signal. A non-array
// body is a decode error (a malformed 200 is a real fault). An empty array, or slots
// with no accounting field, is a signal-ABSENT observation, not an error.
func decodeLlamaSlots(raw []byte) (LlamaSessionCache, error) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" {
		return LlamaSessionCache{Note: "/slots returned an empty body"}, nil
	}
	var slots []llamaSlotWire
	if err := json.Unmarshal([]byte(trimmed), &slots); err != nil {
		return LlamaSessionCache{}, fmt.Errorf("llama: decode /slots JSON: %w", err)
	}
	sig := LlamaSessionCache{Slots: len(slots)}
	present := map[string]bool{}
	for _, sl := range slots {
		if sl.CacheTokens != nil {
			present["cache_tokens"] = true
		}
		if sl.NPast != nil {
			present["n_past"] = true
		}
		if sl.NPromptProcessed != nil {
			present["n_prompt_tokens_processed"] = true
		}
	}
	switch {
	case len(slots) == 0:
		sig.Note = "/slots returned no slots"
	case len(present) == 0:
		sig.Note = "/slots slots expose no prompt-cache accounting field"
	default:
		sig.Present = true
		sig.Fields = sortedKeys(present)
	}
	return sig, nil
}

// sortedKeys returns the keys of set in a stable order, so the Evidence string and any
// test assertion over Fields is deterministic.
func sortedKeys(set map[string]bool) []string {
	keys := make([]string, 0, len(set))
	for k := range set {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// ObserveLlamaSessionCache fetches one /slots snapshot and decodes its session-cache
// signal. A transport error is returned to the caller; a NON-200 status (llama-server
// answers 501 when --slots is disabled) is NOT an error — it resolves to the
// passive/no-evidence signal, because a disabled endpoint is a legitimate "engine
// exposes no cache surface" observation rather than a failed read.
func ObserveLlamaSessionCache(ctx context.Context, client *http.Client, slotsURL, apiKey string) (LlamaSessionCache, error) {
	if strings.TrimSpace(slotsURL) == "" {
		return LlamaSessionCache{}, errors.New("llama: slots URL is required (FAK_LLAMA_BASE_URL/FAK_LLAMA_SLOTS_URL or LlamaConfig)")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, slotsURL, nil)
	if err != nil {
		return LlamaSessionCache{}, err
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := client.Do(req)
	if err != nil {
		return LlamaSessionCache{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return LlamaSessionCache{Note: fmt.Sprintf("/slots returned %d (endpoint disabled)", resp.StatusCode)}, nil
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return LlamaSessionCache{}, err
	}
	return decodeLlamaSlots(raw)
}

// LlamaCacheObserver reads llama-server's session-cache signal and reports it as a
// wire-neutral engine.CacheCapability. It satisfies CacheCapabilityProducer behind the
// item-32 seam, so the gateway can report a llama-backed session's cache capability
// without importing this package.
type LlamaCacheObserver struct {
	cfg    LlamaConfig
	client *http.Client
	signal LlamaSessionCache
	read   bool
}

// NewLlamaCacheObserver builds an observer over a llama-server's public /slots surface.
func NewLlamaCacheObserver(cfg LlamaConfig) *LlamaCacheObserver {
	return &LlamaCacheObserver{cfg: cfg, client: defaultHTTPClient(cfg.Client)}
}

// slotsURL resolves the /slots endpoint: an explicit SlotsURL wins, else it is derived
// from BaseURL.
func (o *LlamaCacheObserver) slotsURL() (string, error) {
	if strings.TrimSpace(o.cfg.SlotsURL) != "" {
		return o.cfg.SlotsURL, nil
	}
	if strings.TrimSpace(o.cfg.BaseURL) == "" {
		return "", errors.New("llama: FAK_LLAMA_BASE_URL or LlamaConfig.BaseURL/SlotsURL is required")
	}
	return joinEndpoint(o.cfg.BaseURL, "/slots")
}

// Observe reads the /slots session-cache signal once and caches it for CacheCapability.
// A transport error is returned; a disabled/empty endpoint is not an error — it
// resolves to the passive/no-evidence signal. Safe to call again to refresh.
func (o *LlamaCacheObserver) Observe(ctx context.Context) (LlamaSessionCache, error) {
	url, err := o.slotsURL()
	if err != nil {
		return LlamaSessionCache{}, err
	}
	sig, err := ObserveLlamaSessionCache(ctx, o.client, url, o.cfg.APIKey)
	if err != nil {
		return LlamaSessionCache{}, err
	}
	o.signal, o.read = sig, true
	return sig, nil
}

// CacheCapability satisfies engine.CacheCapabilityProducer. Before Observe has resolved
// a signal it FAILS CLOSED to the unknown/no-evidence label rather than inferring a
// positive — the same fail-closed default the contract's zero verdict has.
func (o *LlamaCacheObserver) CacheCapability() CacheCapability {
	if !o.read {
		return LlamaSessionCache{Note: "not yet observed"}.Capability()
	}
	return o.signal.Capability()
}

// LlamaCacheObserver produces a wire-neutral CacheCapability behind the item-32 seam.
var _ CacheCapabilityProducer = (*LlamaCacheObserver)(nil)
