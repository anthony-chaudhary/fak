package gateway

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestChatCompletionsForwardsMaxCompletionTokens proves end to end that the OpenAI
// chat wire honours `max_completion_tokens` (the current OpenAI field; max_tokens is
// its deprecated alias): a request carrying only max_completion_tokens (as Pi sends)
// reaches the planner as the output cap instead of falling to the planner default,
// it wins over max_tokens when both are sent, and a negative value is a 400.
func TestChatCompletionsForwardsMaxCompletionTokens(t *testing.T) {
	t.Parallel()
	const msgs = `,"messages":[{"role":"user","content":"hi"}]`
	cases := []struct {
		name       string
		body       string
		wantStatus int
		wantMax    int
	}{
		{"only_max_completion_tokens", `{"model":"m","max_completion_tokens":600` + msgs + `}`, http.StatusOK, 600},
		{"only_max_tokens", `{"model":"m","max_tokens":300` + msgs + `}`, http.StatusOK, 300},
		{"both_prefers_max_completion_tokens", `{"model":"m","max_tokens":300,"max_completion_tokens":700` + msgs + `}`, http.StatusOK, 700},
		{"negative_max_completion_tokens", `{"model":"m","max_completion_tokens":-5` + msgs + `}`, http.StatusBadRequest, 0},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			srv := newTestServer(t)
			rp := &recordingPlanner{comp: &agent.Completion{
				Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
				FinishReason: "stop",
			}}
			srv.planner = rp
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()

			r, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader([]byte(tc.body)))
			if err != nil {
				t.Fatal(err)
			}
			defer r.Body.Close()
			if r.StatusCode != tc.wantStatus {
				t.Fatalf("status = %d, want %d", r.StatusCode, tc.wantStatus)
			}
			if tc.wantStatus != http.StatusOK {
				return
			}
			if rp.got.MaxTokens == nil || *rp.got.MaxTokens != tc.wantMax {
				t.Fatalf("planner got max_tokens = %v, want %d", rp.got.MaxTokens, tc.wantMax)
			}
		})
	}
}
