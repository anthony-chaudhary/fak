package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
)

// guard_local.go implements `fak guard --local`: auto-detect a local OpenAI-compatible
// model server the user is ALREADY running (Ollama / LM Studio / llama.cpp server) and
// wire guard's upstream to it with zero flags. It is the works-end-to-end-TODAY sibling
// of the in-kernel `--gguf` path (#1057, epic #1056): a server already running emits real
// tool calls, so an OpenAI-wire harness (codex/aider/opencode — and Claude Code over the
// OpenAI wire) becomes a governed local coding loop with no base-URL hunting.
//
// The detection DECISION (an ordered list of probe outcomes -> the chosen base + model)
// is split from the I/O (the actual HTTP probes) so the precedence rules are unit-tested
// without standing a server up, mirroring how guardLocalModelDecision keeps the --gguf
// precedence pure.

// localBackend is one conventional local model server fak knows how to probe. The order
// of guardLocalBackends defines the detection PRECEDENCE: the first live backend wins.
type localBackend struct {
	// name is the human label printed when this backend is detected ("Ollama").
	name string
	// base is the bare server base URL (no /v1), e.g. "http://127.0.0.1:11434". The chosen
	// upstream base is guardOpenAIV1Base(base) so it lands on the OpenAI /v1 surface.
	base string
	// modelsPath is the GET route that lists served models, relative to base. Ollama uses
	// its native /api/tags; the OpenAI-compatible servers use /v1/models.
	modelsPath string
	// parseModels extracts the served model ids from the modelsPath response body. Each
	// backend speaks a slightly different JSON shape, so the decode lives with the backend.
	parseModels func([]byte) []string
}

// guardLocalBackends is the ordered probe list. Precedence: Ollama (the most common local
// runner) first, then LM Studio, then a bare llama.cpp server. honored env overrides are
// applied in guardDetectLocalBackend (OLLAMA_HOST for Ollama) rather than baked here, so
// this stays a pure data table.
func guardLocalBackends() []localBackend {
	return []localBackend{
		{
			name:        "Ollama",
			base:        "http://127.0.0.1:11434",
			modelsPath:  "/api/tags",
			parseModels: parseOllamaTags,
		},
		{
			name:        "LM Studio",
			base:        "http://127.0.0.1:1234",
			modelsPath:  "/v1/models",
			parseModels: parseOpenAIModels,
		},
		{
			name:        "llama.cpp",
			base:        "http://127.0.0.1:8080",
			modelsPath:  "/v1/models",
			parseModels: parseOpenAIModels,
		},
	}
}

// localProbeResult is the outcome of probing one backend: whether it answered, and the
// model ids it reported. It is the pure-decision input — guardChooseLocalBackend takes a
// slice of these (one per backend, same order as guardLocalBackends) and picks the winner.
type localProbeResult struct {
	backend localBackend
	live    bool     // the modelsPath answered (any HTTP response, not a connection error)
	models  []string // served model ids, when the body parsed
}

// guardChooseLocalBackend is the PURE detection decision: given the ordered probe results,
// return the first live backend's OpenAI /v1 base URL and a chosen model id, plus a label
// for the banner. found=false when nothing answered. A live backend that reports zero
// models is still chosen (model=="") — the harness may still serve a default — but the
// caller logs that case. No I/O, no exit: the precedence is unit-tested directly.
func guardChooseLocalBackend(results []localProbeResult) (base, model, label string, found bool) {
	for _, r := range results {
		if !r.live {
			continue
		}
		return guardOpenAIV1Base(r.backend.base), guardPickLocalModel(r.models), r.backend.name, true
	}
	return "", "", "", false
}

// guardPickLocalModel selects ONE model id from a backend's served list. It prefers a
// coding-tuned name (one containing "coder" or "code") so a `--local` coding loop lands on
// the right model when several are loaded, else the first id, else "" (let the harness pick
// its own default). Deterministic: the candidate list is sorted before the preference scan
// so the choice does not depend on the server's map iteration order.
func guardPickLocalModel(models []string) string {
	if len(models) == 0 {
		return ""
	}
	sorted := append([]string(nil), models...)
	sort.Strings(sorted)
	for _, m := range sorted {
		l := strings.ToLower(m)
		if strings.Contains(l, "coder") || strings.Contains(l, "code") {
			return m
		}
	}
	return sorted[0]
}

// parseOllamaTags extracts model names from an Ollama GET /api/tags body:
// {"models":[{"name":"qwen2.5-coder:7b", ...}, ...]}.
func parseOllamaTags(body []byte) []string {
	var doc struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.Unmarshal(body, &doc); err != nil {
		return nil
	}
	out := make([]string, 0, len(doc.Models))
	for _, m := range doc.Models {
		if n := strings.TrimSpace(m.Name); n != "" {
			out = append(out, n)
		}
	}
	return out
}

// parseOpenAIModels extracts ids from an OpenAI GET /v1/models body:
// {"data":[{"id":"...", "object":"model"}, ...]}.
func parseOpenAIModels(body []byte) []string {
	var doc struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &doc); err != nil {
		return nil
	}
	out := make([]string, 0, len(doc.Data))
	for _, m := range doc.Data {
		if id := strings.TrimSpace(m.ID); id != "" {
			out = append(out, id)
		}
	}
	return out
}

// guardLocalProbeTimeout is the per-endpoint probe deadline. Deliberately short (the
// servers are local, so a live one answers in single-digit ms): a dead endpoint must
// fail-soft fast so probing all three backends stays well under a second total.
const guardLocalProbeTimeout = 300 * time.Millisecond

// guardDetectLocalBackend probes the conventional local endpoints in precedence order and
// returns the chosen OpenAI /v1 base + model + label, or found=false if none answered. It
// honors OLLAMA_HOST for the Ollama base. Each probe is fail-soft (a connection error just
// means "not running"); only a route that returns an HTTP response counts as live. This is
// the I/O half; the decision half is guardChooseLocalBackend.
func guardDetectLocalBackend() (base, model, label string, found bool) {
	backends := guardLocalBackends()
	// Apply env overrides on the backend bases (Ollama honors OLLAMA_HOST).
	for i := range backends {
		if backends[i].name == "Ollama" {
			if h := guardOllamaHostBase(os.Getenv("OLLAMA_HOST")); h != "" {
				backends[i].base = h
			}
		}
	}
	client := &http.Client{Timeout: guardLocalProbeTimeout}
	results := make([]localProbeResult, 0, len(backends))
	for _, b := range backends {
		results = append(results, guardProbeLocalBackend(client, b))
	}
	return guardChooseLocalBackend(results)
}

// guardProbeLocalBackend performs the single GET <base><modelsPath> probe for one backend
// and returns its result. A transport error -> not live; any HTTP status -> live (the
// server is up; even a 404 on the models route still proves a listener). The model list is
// parsed best-effort from a 200 body.
func guardProbeLocalBackend(client *http.Client, b localBackend) localProbeResult {
	resp, err := client.Get(strings.TrimRight(b.base, "/") + b.modelsPath)
	if err != nil {
		return localProbeResult{backend: b, live: false}
	}
	defer func() { _ = resp.Body.Close() }()
	res := localProbeResult{backend: b, live: true}
	if resp.StatusCode == http.StatusOK {
		// Cap the read so a misbehaving endpoint can't stream forever within the timeout.
		buf := make([]byte, 0, 64*1024)
		tmp := make([]byte, 8192)
		for len(buf) < 1<<20 {
			n, rerr := resp.Body.Read(tmp)
			buf = append(buf, tmp[:n]...)
			if rerr != nil {
				break
			}
		}
		res.models = b.parseModels(buf)
	}
	return res
}

// guardOllamaHostBase normalizes an OLLAMA_HOST value into a bare base URL fak can probe.
// OLLAMA_HOST is conventionally a "host:port" or a full URL; an empty value means "use the
// default". It returns "" when the value is empty (the caller keeps the 127.0.0.1:11434
// default) so a missing env var is not an error.
func guardOllamaHostBase(host string) string {
	h := strings.TrimSpace(host)
	if h == "" {
		return ""
	}
	if strings.HasPrefix(h, "http://") || strings.HasPrefix(h, "https://") {
		return strings.TrimRight(h, "/")
	}
	return "http://" + strings.TrimRight(h, "/")
}

// guardLocalNothingDetectedMessage is the fail-loud one-liner printed when --local found no
// running server and no --gguf was passed. It names the two ways forward: install/run a
// local server, or use the no-server in-kernel path.
func guardLocalNothingDetectedMessage() string {
	return fmt.Sprintf(
		"fak guard --local: no local model server detected on the conventional ports "+
			"(Ollama %s, LM Studio %s, llama.cpp %s).\n"+
			"  Start one (e.g. `ollama run %s`), or use the no-server in-kernel path: `fak guard --gguf %s -- <agent>`.",
		"127.0.0.1:11434", "127.0.0.1:1234", "127.0.0.1:8080",
		"qwen2.5-coder:7b", "qwen2.5-coder:3b",
	)
}
