package agent

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"
)

// LlamaSlotTask is what llama-server's GET /slots reports about its only slot:
// the last task it ran, whether it is running one now, and that task's prompt
// length. A task id that has not moved proves no other request has used the slot.
type LlamaSlotTask struct {
	IDTask       int
	Processing   bool
	PromptTokens int
}

// LlamaSingleSlot reads the upstream's /slots within timeout. ok is false unless
// the upstream reports exactly one slot carrying an id_task: with several slots
// the slot a request lands on is not known before it runs.
func (p *HTTPPlanner) LlamaSingleSlot(ctx context.Context, timeout time.Duration) (LlamaSlotTask, bool) {
	if p == nil || p.Provider != ProviderOpenAI || strings.TrimSpace(p.BaseURL) == "" {
		return LlamaSlotTask{}, false
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	base := strings.TrimSuffix(strings.TrimRight(strings.TrimSpace(p.BaseURL), "/"), "/v1")
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/slots", nil)
	if err != nil {
		return LlamaSlotTask{}, false
	}
	if key := p.effectiveAPIKey(); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := llamaSlotHTTPClient.Do(req)
	if err != nil {
		return LlamaSlotTask{}, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return LlamaSlotTask{}, false
	}
	var slots []struct {
		IDTask       *int `json:"id_task"`
		IsProcessing bool `json:"is_processing"`
		NPrompt      int  `json:"n_prompt_tokens"`
	}
	if json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&slots) != nil || len(slots) != 1 || slots[0].IDTask == nil {
		return LlamaSlotTask{}, false
	}
	return LlamaSlotTask{IDTask: *slots[0].IDTask, Processing: slots[0].IsProcessing, PromptTokens: slots[0].NPrompt}, true
}
