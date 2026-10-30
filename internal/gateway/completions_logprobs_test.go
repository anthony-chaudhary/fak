package gateway

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func postNativeCompletion(t *testing.T, srv *Server, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rr, req)
	return rr
}

type completionLogprobsWire struct {
	Choices []struct {
		Text     string `json:"text"`
		Logprobs *struct {
			Tokens        []string             `json:"tokens"`
			TokenLogprobs []float64            `json:"token_logprobs"`
			TopLogprobs   []map[string]float64 `json:"top_logprobs"`
		} `json:"logprobs"`
	} `json:"choices"`
	Usage struct {
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
	Fak *struct {
		TokenIDs *struct {
			TokenIDs []int `json:"token_ids"`
		} `json:"native_decode_token_ids"`
		Top5 *struct {
			Rows [][]struct {
				TokenID *int     `json:"token_id"`
				Logprob *float64 `json:"logprob"`
			} `json:"rows"`
		} `json:"native_decode_top5"`
	} `json:"fak"`
}

func TestCompletionsNativeDecodeTop5AndLogprobsWire(t *testing.T) {
	srv := nativeReceiptServer(t)
	const base = `"model":"synthetic-live","prompt":"What is the capital of France?","max_tokens":8,"temperature":0,"fak_decode_trace":true`
	plain := postNativeCompletion(t, srv, `{`+base+`,"fak":{"native_decode_token_ids":true}}`)
	rich := postNativeCompletion(t, srv, `{`+base+`,"logprobs":3,"fak":{"native_decode_token_ids":true,"native_decode_top5":true}}`)
	if plain.Code != http.StatusOK || rich.Code != http.StatusOK {
		t.Fatalf("status plain=%d rich=%d body=%s", plain.Code, rich.Code, rich.Body.String())
	}
	if strings.Contains(plain.Body.String(), `"logprobs"`) || strings.Contains(plain.Body.String(), `"native_decode_top5"`) {
		t.Fatalf("unrequested logprobs leaked into response: %s", plain.Body.String())
	}
	var off, on completionLogprobsWire
	if err := json.Unmarshal(plain.Body.Bytes(), &off); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(rich.Body.Bytes(), &on); err != nil {
		t.Fatal(err)
	}
	ids := on.Fak.TokenIDs.TokenIDs
	if on.Choices[0].Text != off.Choices[0].Text || !reflect.DeepEqual(ids, off.Fak.TokenIDs.TokenIDs) || len(ids) == 0 {
		t.Fatalf("greedy output changed with logprobs: %q/%v vs %q/%v", on.Choices[0].Text, ids, off.Choices[0].Text, off.Fak.TokenIDs.TokenIDs)
	}
	rows := on.Fak.Top5.Rows
	if len(rows) != len(ids) || on.Usage.CompletionTokens != len(ids) {
		t.Fatalf("top5 rows=%d ids=%d usage=%d, want equal", len(rows), len(ids), on.Usage.CompletionTokens)
	}
	for i, row := range rows {
		if len(row) != 5 || row[0].TokenID == nil || *row[0].TokenID != ids[i] {
			t.Fatalf("row %d=%v, want 5 candidates led by greedy id %d", i, row, ids[i])
		}
		for j, c := range row {
			if c.TokenID == nil || c.Logprob == nil || *c.Logprob > 0 || math.IsNaN(*c.Logprob) || (j > 0 && *c.Logprob > *row[j-1].Logprob) {
				t.Fatalf("row %d=%v: want token_id/logprob, non-positive, non-increasing", i, row)
			}
		}
	}
	lp := on.Choices[0].Logprobs
	if lp == nil || len(lp.Tokens) != len(ids) || len(lp.TokenLogprobs) != len(ids) || len(lp.TopLogprobs) != len(ids) {
		t.Fatalf("openai logprobs=%+v, want %d aligned rows", lp, len(ids))
	}
	for i := range ids {
		if lp.TokenLogprobs[i] != *rows[i][0].Logprob || len(lp.TopLogprobs[i]) == 0 || len(lp.TopLogprobs[i]) > 3 {
			t.Fatalf("openai row %d token_logprob=%v top=%v, want greedy logprob %v and <=3 candidates", i, lp.TokenLogprobs[i], lp.TopLogprobs[i], *rows[i][0].Logprob)
		}
	}
}

func TestCompletionsLogprobsValidation(t *testing.T) {
	srv := nativeReceiptServer(t)
	for _, body := range []string{
		`{"model":"synthetic-live","prompt":"x","logprobs":21}`,
		`{"model":"synthetic-live","prompt":"x","logprobs":-1}`,
		`{"model":"synthetic-live","prompt":"x","stream":true,"fak":{"native_decode_top5":true}}`,
	} {
		if rr := postNativeCompletion(t, srv, body); rr.Code != http.StatusBadRequest {
			t.Fatalf("body %s status=%d, want 400", body, rr.Code)
		}
	}
	chat := `{"model":"synthetic-live","messages":[{"role":"user","content":"x"}],"fak":{"native_decode_top5":true}}`
	if rr := postNativeReceipt(t, srv, chat); rr.Code != http.StatusBadRequest {
		t.Fatalf("chat native_decode_top5 status=%d, want 400 (completions-only extension)", rr.Code)
	}
}
