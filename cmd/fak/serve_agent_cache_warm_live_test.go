//go:build fak_hitl

package main

// serve_agent_cache_warm_live_test.go — CW-26 (#13326) witness. The CW-08
// (#13329) producer witness pins the serve seam's INPUTS with a fake planner and
// the CW-14 (#13327) e2e witness pins the serve/gateway cold-vs-warm ADMISSION
// shape against an in-process fake warmer. Neither runs the REAL native backend:
// no receipt proves that a first natural request on the selected deployed native
// model is served from startup-warmed agent state.
//
// This file closes that gap. It is build-tagged (`fak_hitl`) so the ordinary
// `go test ./cmd/fak` suite never compiles or runs it — the normal suite must not
// require hardware. When explicitly requested it drives the ACTUAL serve startup
// path (load the configured native model, installServeAgentWarm, run the
// production background warm half runServeAgentWarmup) against a real
// gateway.Server, sends the first natural request through the real
// /v1/chat/completions handler, and compares it to a COLD reference run served
// without a warm.
//
// BIPARTITE DISCIPLINE: the physical cache-residency clause is [HW-WITNESSED]
// and is only ever satisfied on the real selected backend. This file asserts the
// [SW-VERIFIED] wiring/shape contract (the tagged witness fails loud on missing
// inputs or a skipped run — a skip is a failure), records bounded
// timeout/failure reasons, and emits raw receipts. It never fabricates a
// residency or throughput number: a counter the backend does not report stays
// explicitly unavailable.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// The explicit inputs a live qualification must be handed. No default model is
// assumed: a run without them FAILS (never skips), because "No-tests-to-run or a
// skipped named witness fails acceptance".
const (
	envLiveGGUFPath  = "FAK_HITL_GGUF"
	envLiveBackend   = "FAK_HITL_BACKEND"
	envLiveReceipt   = "FAK_HITL_RECEIPT"
	envLiveWorkspace = "FAK_HITL_WORKSPACE"
	// envLiveTimeout bounds the whole qualification so a wedged native load or
	// decode cannot hang the run. It is a bounded failure reason, not a skip.
	envLiveTimeout = "FAK_HITL_TIMEOUT"

	defaultLiveTimeout = 20 * time.Minute
	// liveTurnBudget bounds the single first-real request's generated tokens so the
	// witness stays bounded and deterministic.
	liveTurnBudget = 8
)

// liveWitnessReceipt is the durable evidence this witness retains. Every field is
// a scalar/copied value or a nested receipt; it deliberately records no prompt
// text and no model weights. Counters a backend did not report are left at their
// unavailable sentinel (-1 / empty), never guessed.
type liveWitnessReceipt struct {
	Schema    string `json:"schema"`
	Backend   string `json:"backend"`
	ModelRef  string `json:"model_ref"`
	Timestamp string `json:"timestamp_utc"`
	TimeoutMs int64  `json:"timeout_ms"`

	// Cold is the reference: a server that never installed a warm profile.
	Cold liveArmReceipt `json:"cold"`
	// Warm is the serve startup path: profile installed pre-readiness, warm
	// materialized on the background startup half, then the first real request.
	Warm liveArmReceipt `json:"warm"`

	ColdOutputDigest string `json:"cold_output_digest"`
	WarmOutputDigest string `json:"warm_output_digest"`
	// OutputParity reports the cold and warm first-request bodies were byte-equal.
	OutputParity bool `json:"output_parity"`
	// FirstRequestAdmitted reports the warm first request was served (200) with no
	// demand-time re-prime.
	FirstRequestAdmitted bool `json:"first_request_admitted"`

	// Notes records bounded, closed failure/refusal reasons. Non-empty notes do not
	// by themselves fail the witness when the shape held.
	Notes []string `json:"notes,omitempty"`
	// Reasons records the closed reasons any admission held/refused, in order.
	Reasons []string `json:"admission_reasons,omitempty"`
}

// liveArmReceipt is one arm's (cold or warm) observed shape. Missing counters are
// explicit sentinels.
type liveArmReceipt struct {
	// Armed reports the profile was installed through the production seam.
	Armed bool `json:"armed"`
	// PreWarmHeld reports readiness was held (503) before materialization.
	PreWarmHeld bool `json:"pre_warm_held"`
	// WarmStatus is the observed /healthz agent_warm status.
	WarmStatus string `json:"warm_status"`
	// WarmReason is the closed admission reason, when non-empty.
	WarmReason string `json:"warm_reason"`
	// WarmReady is the planner's receipt Ready bit.
	WarmReady bool `json:"warm_ready"`
	// WarmReceiptStatus is the planner receipt's closed status token.
	WarmReceiptStatus string `json:"warm_receipt_status"`
	// WarmIdentity is the receipt's descriptor identity.
	WarmIdentity string `json:"warm_identity"`
	// RequestedTokens / RestoredTokens are the receipt's stable boundary and the
	// readback that decides Ready.
	RequestedTokens int `json:"requested_tokens"`
	RestoredTokens  int `json:"restored_tokens"`
	// WarmBytes is the receipt's claimed resident payload, or -1 when absent.
	WarmBytes int64 `json:"warm_bytes"`
	// FirstHTTP is the first real request's status code.
	FirstHTTP int `json:"first_request_http"`
	// PromptTokens / CompletionTokens are the served turn's reported usage, or -1
	// when the backend did not report it (never fabricated).
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

func liveEnvOrFail(t *testing.T, key string) string {
	t.Helper()
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		t.Fatalf("live witness requires %s (explicit model artifact/backend input); "+
			"a missing input is a FAILURE, not a skip", key)
	}
	return v
}

func liveTimeout() time.Duration {
	raw := strings.TrimSpace(os.Getenv(envLiveTimeout))
	if raw == "" {
		return defaultLiveTimeout
	}
	if d, err := time.ParseDuration(raw); err == nil && d > 0 {
		return d
	}
	return defaultLiveTimeout
}

// TestServeAgentCacheWarmLive is the CW-26 (#13326) live qualification witness.
//
// It is a FAIL-LOUD witness: without explicit native inputs it fails rather than
// skipping, and its build tag keeps it out of the normal suite entirely. The
// [HW-WITNESSED] residency clause is only ever established on the real backend;
// on any host the [SW-VERIFIED] wiring/shape contract still holds.
func TestServeAgentCacheWarmLive(t *testing.T) {
	gguf := liveEnvOrFail(t, envLiveGGUFPath)
	backendName := liveEnvOrFail(t, envLiveBackend)
	if _, err := os.Stat(gguf); err != nil {
		t.Fatalf("live model artifact %q is not readable: %v", gguf, err)
	}

	timeout := liveTimeout()
	deadline := time.Now().Add(timeout)
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()

	workspace := strings.TrimSpace(os.Getenv(envLiveWorkspace))
	if workspace == "" {
		workspace = t.TempDir()
	}
	instruction := []byte("# CW-26 live warm witness\n\nStable effective agent instructions.\n")
	if err := os.WriteFile(filepath.Join(workspace, "AGENTS.md"), instruction, 0o644); err != nil {
		t.Fatalf("write AGENTS.md: %v", err)
	}

	receipt := liveWitnessReceipt{
		Schema:    "fak.agent.cache_warm.live.v1",
		Backend:   backendName,
		ModelRef:  gguf,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		TimeoutMs: timeout.Milliseconds(),
	}

	// --- Load the real native model ONCE and share the resident engine across
	// both arms, so the cold/warm comparison differs only in the warm profile,
	// never in a second load's residency.
	planner := liveNativePlanner(t, ctx, gguf, backendName)

	// --- COLD reference: a server that never installs a warm profile. The first
	// real request is served with no startup-warmed prefix.
	coldSrv, err := gateway.New(gateway.Config{Model: planner.Model()})
	if err != nil {
		t.Fatalf("cold gateway.New: %v", err)
	}
	coldSrv.SetPlanner(planner)
	coldSrv.MarkWarmupComplete(0)
	coldTS := httptest.NewServer(coldSrv.Handler())
	defer coldTS.Close()

	coldCode, coldBody := livePostChat(t, ctx, coldTS, planner.Model(), "hello live witness", deadline)
	receipt.Cold.FirstHTTP = coldCode
	receipt.ColdOutputDigest = liveDigest(coldBody)
	if coldCode != http.StatusOK {
		receipt.Notes = append(receipt.Notes, fmt.Sprintf("cold_first_request_http=%d", coldCode))
	}

	// --- WARM arm: the actual serve startup path on a fresh server over the SAME
	// resident planner. installServeAgentWarm arms the profile before readiness;
	// runServeAgentWarmup is the production background half that materializes it.
	warmSrv, err := gateway.New(gateway.Config{Model: planner.Model()})
	if err != nil {
		t.Fatalf("warm gateway.New: %v", err)
	}
	warmSrv.SetPlanner(planner)
	warmSrv.MarkWarmupComplete(0)

	armed := installServeAgentWarm(warmSrv, workspace, io.Discard)
	receipt.Warm.Armed = armed
	if !armed {
		t.Fatalf("installServeAgentWarm did not arm a profile for a readable AGENTS.md under %q", workspace)
	}

	warmTS := httptest.NewServer(warmSrv.Handler())
	defer warmTS.Close()

	// Pre-materialization: readiness must be HELD (the probe is refused), never
	// silently served cold.
	if code, aw := liveHealthz(t, ctx, warmTS); aw != nil {
		receipt.Reasons = append(receipt.Reasons, fmt.Sprintf("pre_warm_status=%v", aw["status"]))
		if code == http.StatusOK {
			t.Fatalf("pre-warm /healthz returned 200 before the warm materialized; readiness must be held")
		}
		receipt.Warm.PreWarmHeld = true
	}

	// The production background warm half. Its returned receipt is the durable
	// evidence; an error is recorded, not hidden.
	warmReceipt, warmErr := warmSrv.RunAgentWarmup(ctx)
	if warmErr != nil {
		receipt.Notes = append(receipt.Notes, fmt.Sprintf("run_agent_warmup_error=%v", warmErr))
	}
	receipt.Warm.WarmReady = warmReceipt.Ready
	receipt.Warm.WarmReceiptStatus = warmReceipt.Status
	receipt.Warm.WarmIdentity = warmReceipt.Identity
	receipt.Warm.RequestedTokens = warmReceipt.RequestedTokens
	receipt.Warm.RestoredTokens = warmReceipt.RestoredTokens
	if warmReceipt.Claim != nil {
		receipt.Warm.WarmBytes = warmReceipt.Claim.Bytes
	} else {
		receipt.Warm.WarmBytes = -1
	}

	// The first REAL request after readiness, through the real handler.
	warmCode, warmBody := livePostChat(t, ctx, warmTS, planner.Model(), "hello live witness", deadline)
	receipt.Warm.FirstHTTP = warmCode
	receipt.WarmOutputDigest = liveDigest(warmBody)
	receipt.OutputParity = receipt.ColdOutputDigest != "" && receipt.ColdOutputDigest == receipt.WarmOutputDigest

	if status, reason := liveHealthzStatus(t, ctx, warmTS); status != "" {
		receipt.Warm.WarmStatus = status
		receipt.Warm.WarmReason = reason
		if reason != "" {
			receipt.Reasons = append(receipt.Reasons, reason)
		}
	}
	if pt, ct := liveUsage(warmBody); pt >= 0 || ct >= 0 {
		receipt.Warm.PromptTokens, receipt.Warm.CompletionTokens = pt, ct
	} else {
		receipt.Warm.PromptTokens, receipt.Warm.CompletionTokens = -1, -1
	}

	receipt.FirstRequestAdmitted = warmCode == http.StatusOK

	if warmCode != http.StatusOK {
		liveRetainReceipt(t, receipt, workspace)
		t.Fatalf("warm first real request code = %d (cold was %d); readiness held the first request", warmCode, coldCode)
	}
	if !warmReceipt.Ready {
		liveRetainReceipt(t, receipt, workspace)
		t.Fatalf("warm receipt was not Ready (status=%q reason=%q); the first request cannot be a startup-warmed serve",
			warmReceipt.Status, warmReceipt.Reason)
	}
	if warmReceipt.RestoredTokens < warmReceipt.RequestedTokens || warmReceipt.RestoredTokens <= 0 {
		liveRetainReceipt(t, receipt, workspace)
		t.Fatalf("warm restored %d/%d tokens; the first request was not served from a full startup warm",
			warmReceipt.RestoredTokens, warmReceipt.RequestedTokens)
	}

	path := liveRetainReceipt(t, receipt, workspace)
	t.Logf("[SW-VERIFIED] CW-26 live shape: cold_http=%d warm_http=%d restored=%d/%d identity=%s parity=%v receipt=%s",
		coldCode, warmCode, warmReceipt.RestoredTokens, warmReceipt.RequestedTokens,
		warmReceipt.Identity, receipt.OutputParity, path)
}

// liveNativePlanner loads the configured native artifact through the SAME helper
// the `fak run`/`fak serve` path uses and builds the real in-kernel planner.
func liveNativePlanner(t *testing.T, ctx context.Context, gguf, backendName string) *agent.InKernelPlanner {
	t.Helper()
	backend, err := resolveServeChatBackend(backendName)
	if err != nil {
		t.Fatalf("resolve backend %q: %v", backendName, err)
	}
	m, q4k, _, _ := loadServeInKernelModel(gguf, backend, false, 0, nil, 1, nil)
	if m == nil {
		t.Fatalf("failed to load %q into the in-kernel engine", gguf)
	}
	tok, ok := resolveServeTokenizer("", gguf)
	if !ok || tok == nil {
		t.Fatalf("%q has no usable tokenizer; pass a GGUF with an embedded tokenizer", gguf)
	}
	if err := ctx.Err(); err != nil {
		t.Fatalf("live qualification timed out before the planner was built: %v", err)
	}
	return agent.NewInKernelPlannerWithConfig(m, tok, gguf, q4k, backend, false, agent.InKernelPlannerConfig{})
}

// livePostChat sends one natural chat completion through the real server handler
// with a bounded client deadline. A non-2xx is returned, not fatal, so the caller
// can assert the held/admitted shapes.
func livePostChat(t *testing.T, ctx context.Context, ts *httptest.Server, model, content string, deadline time.Time) (int, map[string]any) {
	t.Helper()
	payload := []byte(`{"model":` + liveJSONString(model) + `,"messages":[{"role":"user","content":` +
		liveJSONString(content) + `}],"max_tokens":` + fmt.Sprint(liveTurnBudget) + `}`)
	reqCtx, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, ts.URL+"/v1/chat/completions", bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("build POST: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	defer res.Body.Close()
	raw, _ := io.ReadAll(res.Body)
	var body map[string]any
	_ = json.Unmarshal(raw, &body)
	return res.StatusCode, body
}

// liveHealthz serves one real /healthz and returns the agent_warm block (nil when
// absent).
func liveHealthz(t *testing.T, ctx context.Context, ts *httptest.Server) (int, map[string]any) {
	t.Helper()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/healthz", nil)
	if err != nil {
		t.Fatalf("build GET /healthz: %v", err)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("GET /healthz: %v", err)
	}
	defer res.Body.Close()
	var body map[string]any
	_ = json.NewDecoder(res.Body).Decode(&body)
	aw, _ := body["agent_warm"].(map[string]any)
	return res.StatusCode, aw
}

// liveHealthzStatus returns the observed agent_warm status and reason (empty when
// absent), without asserting a specific code.
func liveHealthzStatus(t *testing.T, ctx context.Context, ts *httptest.Server) (string, string) {
	t.Helper()
	_, aw := liveHealthz(t, ctx, ts)
	if aw == nil {
		return "", ""
	}
	status, _ := aw["status"].(string)
	reason, _ := aw["reason"].(string)
	return status, reason
}

// liveUsage extracts the served turn's token usage from the response body, or -1
// each when the backend did not report it (never fabricated).
func liveUsage(body map[string]any) (promptTokens, completionTokens int) {
	promptTokens, completionTokens = -1, -1
	usage, _ := body["usage"].(map[string]any)
	if usage == nil {
		return promptTokens, completionTokens
	}
	if v, ok := usage["prompt_tokens"].(float64); ok {
		promptTokens = int(v)
	}
	if v, ok := usage["completion_tokens"].(float64); ok {
		completionTokens = int(v)
	}
	return promptTokens, completionTokens
}

// liveRetainReceipt writes the raw receipt durably so the evidence outlives the
// process, and returns the path written.
func liveRetainReceipt(t *testing.T, receipt liveWitnessReceipt, workspace string) string {
	t.Helper()
	path := strings.TrimSpace(os.Getenv(envLiveReceipt))
	if path == "" {
		path = filepath.Join(workspace, "cw26-live-receipt.json")
	}
	raw, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatalf("marshal live receipt: %v", err)
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatalf("write live receipt %q: %v", path, err)
	}
	return path
}

func liveDigest(body map[string]any) string {
	if body == nil {
		return ""
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return ""
	}
	return "sha256:" + sha256Hex(raw)
}

func liveJSONString(s string) string {
	raw, err := json.Marshal(s)
	if err != nil {
		return `""`
	}
	return string(raw)
}
