package logprobparity_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/logprobparity"
)

func singleRef(id int, piece string, top ...logprobparity.Candidate) []logprobparity.RefPrompt {
	return []logprobparity.RefPrompt{{Text: "prompt", Positions: []logprobparity.Position{{ID: id, Piece: piece, TopK: top}}}}
}

func TestScoreOpenAIUsesReportedLogprobRows(t *testing.T) {
	refs := singleRef(7, "A",
		logprobparity.Candidate{ID: 7, Piece: "A", Logprob: -0.1},
		logprobparity.Candidate{ID: 8, Piece: "B", Logprob: -1},
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if got := body["logprobs"]; got != float64(5) {
			t.Errorf("logprobs request = %#v, want 5", got)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"text": "B",
				"logprobs": map[string]any{
					"tokens":         []string{"B"},
					"token_logprobs": []float64{-0.2},
					"top_logprobs":   []map[string]float64{{"B": -0.2, "A": -0.8}},
				},
			}},
		})
	}))
	defer srv.Close()

	prs, err := logprobparity.Score(context.Background(), logprobparity.Client{BaseURL: srv.URL, API: logprobparity.APIOpenAI}, refs, 5)
	if err != nil {
		t.Fatal(err)
	}
	got := prs[0]
	if got.Method != logprobparity.TextPieces || got.Verdict != logprobparity.Close || got.RefInActualTopK == nil || !*got.RefInActualTopK {
		t.Fatalf("reported logprob rows not used: %+v", got)
	}
}

func TestScoreOpenAIRefusesMisalignedLogprobRows(t *testing.T) {
	refs := []logprobparity.RefPrompt{{Text: "prompt", Positions: []logprobparity.Position{{ID: 7}, {ID: 8}}}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"text": "AB",
				"logprobs": map[string]any{
					"tokens":         []string{"A", "B"},
					"token_logprobs": []float64{-0.1},
					"top_logprobs":   []map[string]float64{{"A": -0.1}, {"B": -0.2}},
				},
			}},
		})
	}))
	defer srv.Close()

	prs, err := logprobparity.Score(context.Background(), logprobparity.Client{BaseURL: srv.URL, API: logprobparity.APIOpenAI}, refs, 5)
	if !errors.Is(err, logprobparity.ErrEndpoint) || len(prs) != 1 || prs[0].Verdict != logprobparity.Empty {
		t.Fatalf("misaligned rows: results=%+v err=%v", prs, err)
	}
}

func TestScoreOpenAINullLogprobsFallsBackToText(t *testing.T) {
	refs := singleRef(7, "A")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"text": "A", "logprobs": nil}}})
	}))
	defer srv.Close()

	prs, err := logprobparity.Score(context.Background(), logprobparity.Client{BaseURL: srv.URL, API: logprobparity.APIOpenAI}, refs, 5)
	if err != nil || prs[0].Method != logprobparity.TextPieces || prs[0].Verdict != logprobparity.Match {
		t.Fatalf("null logprobs fallback: %+v err=%v", prs, err)
	}
}

func TestScoreOpenAIReportDepthDoesNotProveLargerMembershipK(t *testing.T) {
	refs := singleRef(7, "A",
		logprobparity.Candidate{ID: 7, Piece: "A", Logprob: -0.1},
		logprobparity.Candidate{ID: 8, Piece: "B", Logprob: -0.2},
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if got := body["logprobs"]; got != float64(5) {
			t.Errorf("wire logprobs = %#v, want cap 5", got)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"text": "B",
				"logprobs": map[string]any{
					"tokens":         []string{"B"},
					"token_logprobs": []float64{-0.2},
					"top_logprobs":   []map[string]float64{{"B": -0.2, "C": -0.3}},
				},
			}},
		})
	}))
	defer srv.Close()

	prs, err := logprobparity.Score(context.Background(), logprobparity.Client{BaseURL: srv.URL, API: logprobparity.APIOpenAI}, refs, 10)
	if err != nil {
		t.Fatal(err)
	}
	got := prs[0]
	if got.Verdict != logprobparity.Unchecked || got.ActualInRefTopK == nil || !*got.ActualInRefTopK || got.RefInActualTopK != nil {
		t.Fatalf("partial report depth claimed reciprocal evidence: %+v", got)
	}
}

func TestScoreOpenAINativeIDsTakePrecedence(t *testing.T) {
	refs := singleRef(7, "A")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"text": "wrong",
				"logprobs": map[string]any{
					"tokens":         []string{"wrong", "extra"},
					"token_logprobs": []float64{-0.1},
					"top_logprobs":   []map[string]float64{{"wrong": -0.1}},
				},
			}},
			"fak": map[string]any{"native_decode_token_ids": map[string]any{"token_ids": []int{7}}},
		})
	}))
	defer srv.Close()

	prs, err := logprobparity.Score(context.Background(), logprobparity.Client{BaseURL: srv.URL, API: logprobparity.APIOpenAI}, refs, 10)
	if err != nil || prs[0].Method != logprobparity.TokenIDs || prs[0].Verdict != logprobparity.Match {
		t.Fatalf("native ids did not take precedence: %+v err=%v", prs, err)
	}
}

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
