package logprobparity_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/logprobparity"
)

// A fak endpoint that honors the native_decode_token_ids opt-in is scored on
// its own token ids, not on text segmented against the reference pieces.
func TestScoreOpenAIUsesNativeDecodeTokenIDs(t *testing.T) {
	refs := logprobparity.OraclePrompts(readV41(t))[:1]
	want := make([]int, len(refs[0].Positions))
	for i, p := range refs[0].Positions {
		want[i] = p.ID
	}
	want[3] = -42
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Trace bool `json:"fak_decode_trace"`
			Fak   struct {
				TokenIDs bool `json:"native_decode_token_ids"`
			} `json:"fak"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		if !body.Trace || !body.Fak.TokenIDs {
			t.Errorf("client did not request native decode token ids")
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"text": "irrelevant"}},
			"usage":   map[string]any{"prompt_tokens": 7},
			"fak":     map[string]any{"native_decode_token_ids": map[string]any{"token_ids": want}},
		})
	}))
	defer srv.Close()

	prs, err := logprobparity.Score(context.Background(), logprobparity.Client{BaseURL: srv.URL + "/v1", API: logprobparity.APIOpenAI}, refs, 5)
	if err != nil {
		t.Fatal(err)
	}
	if prs[0].Method != logprobparity.TokenIDs || prs[0].GreedyMatchTokens != 3 || prs[0].DivergedAt == nil || *prs[0].DivergedAt != 3 {
		t.Fatalf("token-id scoring: %+v", prs[0])
	}
}

// An endpoint that refuses the extension with a 400 is retried on the plain
// OpenAI wire and scored by text pieces.
func TestScoreOpenAIFallsBackWhenExtensionRefused(t *testing.T) {
	refs := logprobparity.OraclePrompts(readV41(t))[:1]
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if _, ext := body["fak_decode_trace"]; ext {
			http.Error(w, `{"error":"fak_decode_trace requires a fak-native model route"}`, http.StatusBadRequest)
			return
		}
		if _, ext := body["fak"]; ext {
			http.Error(w, "fak key leaked into plain retry", http.StatusTeapot)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"text": " \"Au\" from"}}})
	}))
	defer srv.Close()

	prs, err := logprobparity.Score(context.Background(), logprobparity.Client{BaseURL: srv.URL + "/v1", API: logprobparity.APIOpenAI}, refs, 5)
	if err != nil || calls != 2 || prs[0].Method != logprobparity.TextPieces || prs[0].GreedyMatchTokens == 0 {
		t.Fatalf("fallback: calls=%d %+v %v", calls, prs, err)
	}
}
