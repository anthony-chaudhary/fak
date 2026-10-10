package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

type piHaloCompactionResult struct {
	models  map[string]piFakModel
	reserve int
	keep    int
}

func runPiHaloCompactionCase(t *testing.T, rows []piRouterFakeRow, modelsDoc func(url string) string) piHaloCompactionResult {
	t.Helper()
	pinPiRouterTestEnv(t, piRouterTestKey)
	srv := newPiRouterFake(t, rows)
	dir := t.TempDir()
	modelsPath := filepath.Join(dir, "models.json")
	settingsPath := filepath.Join(dir, "settings.json")
	settings := `{"defaultProvider":"fak","defaultModel":"` + rows[0]["id"].(string) + `","compaction":{"enabled":true,"reserveTokens":27648,"keepRecentTokens":20000}}`
	if err := os.WriteFile(modelsPath, []byte(modelsDoc(srv.URL+"/v1")), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(settingsPath, []byte(settings), 0o644); err != nil {
		t.Fatal(err)
	}
	if code, _, stderr := runPiConfigRouter(t, "--from-router", srv.URL+"/v1", "--write", "--path", modelsPath, "--settings-path", settingsPath); code != 0 {
		t.Fatalf("write exit=%d stderr=%s", code, stderr)
	}
	out := piHaloCompactionResult{models: map[string]piFakModel{}}
	for _, m := range readPiFakModels(t, modelsPath) {
		out.models[m.ID] = m
	}
	raw, err := os.ReadFile(settingsPath)
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Compaction struct {
			Reserve int `json:"reserveTokens"`
			Keep    int `json:"keepRecentTokens"`
		} `json:"compaction"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	out.reserve, out.keep = doc.Compaction.Reserve, doc.Compaction.Keep
	return out
}

func derivedOutputTokens(t *testing.T, served int) int {
	t.Helper()
	env, err := harnesskit.DeriveContextEnvelope(harnesskit.ContextEnvelopeInput{ServedWindow: served, Source: harnesskit.WindowServed})
	if err != nil {
		t.Fatal(err)
	}
	return env.OutputTokens
}

// assertSharedCompactionSafe checks every model against Pi's own compaction
// arithmetic under the one global settings block: the answer and the summary
// request fit the window, and a compaction reclaims at least a quarter.
func assertSharedCompactionSafe(t *testing.T, r piHaloCompactionResult) {
	t.Helper()
	for id, m := range r.models {
		trigger := m.ContextWindow - r.reserve
		summary := min(r.reserve*4/5, m.MaxTokens)
		reclaim := trigger - harnesskit.DefaultFixedPromptTokens - summary - r.keep
		switch {
		case trigger+m.MaxTokens > m.ContextWindow:
			t.Fatalf("model %s: answer overflows: trigger %d + maxTokens %d > window %d", id, trigger, m.MaxTokens, m.ContextWindow)
		case trigger+harnesskit.SummaryPromptOverheadTokens+summary > m.ContextWindow:
			t.Fatalf("model %s: summary request %d exceeds window %d", id, trigger+harnesskit.SummaryPromptOverheadTokens+summary, m.ContextWindow)
		case reclaim < trigger/4:
			t.Fatalf("model %s: compaction reclaims %d of trigger %d", id, reclaim, trigger)
		}
	}
}

// The live Halo shape: a hand-set maxTokens 4096 on the Halo Qwen slot and a
// reserve inflated by 163840-window cloud models. The Halo slot's maxTokens must
// follow the derivation, and its trigger must stay inside what the Halo's
// deadline admission prices for a cold prompt: the gateway gives no prompt-cache
// credit at admission, pi declares the OpenAI SDK's 600s, admission refuses
// above 0.9 of it, and the measured cold prefill is 162 tok/s with decode
// 21 tok/s (a 1024-token completion) after one 4096-token turn of growth.
// fak-test:runtime fast est=1s
func TestPiConfigFromRouterHaloCompactionFitsColdAdmission(t *testing.T) {
	rows := []piRouterFakeRow{
		{"id": "halo-qwen", "owned_by": "halo-node-a", "context_length": 131072},
		{"id": "cloud-big", "owned_by": "opencode-go", "context_length": 1000000},
	}
	r := runPiHaloCompactionCase(t, rows, func(url string) string {
		return `{"providers":{"fak":{"baseUrl":"` + url + `","apiKey":"fak","api":"openai-completions","models":[` +
			`{"id":"halo-qwen","contextWindow":131072,"maxTokens":4096},` +
			`{"id":"cloud-big","contextWindow":163840,"maxTokens":20480}]}}}`
	})
	halo := r.models["halo-qwen"]
	if want := derivedOutputTokens(t, 131072); halo.ContextWindow != 131072 || halo.MaxTokens != want {
		t.Fatalf("halo-qwen = %+v, want contextWindow 131072 maxTokens %d (the derivation, not the stale 4096)", halo, want)
	}
	trigger := halo.ContextWindow - r.reserve
	coldSeconds := float64(trigger+4096)/162 + 1024.0/21
	if coldSeconds > 0.9*600 {
		t.Fatalf("halo trigger %d (reserve %d) prices %.0fs cold, above the 540s admission bound", trigger, r.reserve, coldSeconds)
	}
	if r.reserve != 56320 || r.keep != 4608 {
		t.Fatalf("shared compaction reserve=%d keep=%d, want 56320/4608 (Halo trigger 74752)", r.reserve, r.keep)
	}
	assertSharedCompactionSafe(t, r)
}

// Without a deadline-capped model, the global reserve is the binding (lowest
// trigger) model's own reserve, raised only as far as every other model's
// answer and summary need. A large-window cloud model's own envelope reserve
// (27648 for 163840) must not become the shared reserve.
// fak-test:runtime fast est=1s
func TestPiConfigFromRouterReserveFollowsBindingModel(t *testing.T) {
	rows := []piRouterFakeRow{
		{"id": "slot-128k", "owned_by": "appliance-a", "context_length": 131072},
		{"id": "cloud-big", "owned_by": "opencode-go", "context_length": 1000000},
	}
	r := runPiHaloCompactionCase(t, rows, func(url string) string {
		return `{"providers":{"fak":{"baseUrl":"` + url + `","apiKey":"fak","api":"openai-completions","models":[]}}}`
	})
	if r.reserve >= 27648 {
		t.Fatalf("shared reserve=%d, want below the cloud model's own 27648 envelope reserve", r.reserve)
	}
	if r.reserve != 23552 || r.keep != 20000 {
		t.Fatalf("shared reserve=%d keep=%d, want 23552 (cloud answer 20480 + summary prompt 2048 + margin 1024) keep 20000", r.reserve, r.keep)
	}
	assertSharedCompactionSafe(t, r)
}
