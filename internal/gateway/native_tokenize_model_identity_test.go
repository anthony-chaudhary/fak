package gateway

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type nativeTokenizeIdentityPlanner struct {
	nativeTokenizeProbePlanner
	encodingModel string
}

func (p *nativeTokenizeIdentityPlanner) EncodePrompt(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (agent.PromptEncoding, error) {
	encoding, err := p.nativeTokenizeProbePlanner.EncodePrompt(ctx, messages, tools, opts...)
	encoding.ModelID = p.encodingModel
	return encoding, err
}

// fak-test:runtime fast est=1s
func TestGatewayNativeTokenizeModelIdentity(t *testing.T) {
	const served = "qwen38:27b"
	cases := []struct {
		name          string
		requested     string
		encodingModel string
		wantStatus    int
		wantEncode    int
	}{
		{"exact", served, served, http.StatusOK, 1},
		{"casefold", "Qwen38:27B", served, http.StatusOK, 1},
		{"same artifact alias", "qwen38:27b-q4", served, http.StatusOK, 1},
		{"different quant", "qwen38:27b-q2k", served, http.StatusBadRequest, 0},
		{"different size", "qwen38:70b", served, http.StatusBadRequest, 0},
		{"foreign model", "deepseek-ai/DeepSeek-V4.1-Flash", served, http.StatusBadRequest, 0},
		{"encoder identity stays exact", served, "qwen38:27b-q4", http.StatusInternalServerError, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			planner := &nativeTokenizeIdentityPlanner{encodingModel: tc.encodingModel}
			srv := newTestServer(t)
			srv.model = served
			srv.planner = planner
			ts := httptest.NewServer(srv.Handler())
			t.Cleanup(ts.Close)
			body := `{"model":"` + tc.requested + `","messages":[{"role":"user","content":"hi"}]}`
			resp, err := http.Post(ts.URL+"/v1/fak/tokenize", "application/json", strings.NewReader(body))
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			raw, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatal(err)
			}
			if resp.StatusCode != tc.wantStatus || planner.encodeCalls != tc.wantEncode || planner.completeCalls != 0 {
				t.Fatalf("status/encode/decode = %d/%d/%d, want %d/%d/0; body=%s", resp.StatusCode, planner.encodeCalls, planner.completeCalls, tc.wantStatus, tc.wantEncode, raw)
			}
			if tc.wantStatus == http.StatusBadRequest && chatErrorCode(t, raw) != "model_mismatch" {
				t.Fatalf("error code = %q, want model_mismatch", chatErrorCode(t, raw))
			}
			if tc.wantStatus == http.StatusOK {
				var encoding agent.PromptEncoding
				if err := json.Unmarshal(raw, &encoding); err != nil {
					t.Fatal(err)
				}
				if encoding.ModelID != served || encoding.PromptTokens != 3 {
					t.Fatalf("encoding identity/count = %q/%d, want %q/3", encoding.ModelID, encoding.PromptTokens, served)
				}
			}
		})
	}
}
