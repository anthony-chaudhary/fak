package gateway

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type responsesSchemaFanoutPlanner struct{ calls atomic.Int32 }

func (p *responsesSchemaFanoutPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls.Add(1)
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, FinishReason: "stop"}, nil
}

func (*responsesSchemaFanoutPlanner) Model() string { return "test-model" }

// fak-test:runtime fast est=200ms lane=default
func TestResponsesSchemaRejectedBeforeEPFanout(t *testing.T) {
	const schemaBase = `{"description":"","properties":{}}`
	schema := func(size int) string {
		return `{"description":"` + strings.Repeat("x", size-len(schemaBase)) + `","properties":{}}`
	}
	for _, tc := range []struct {
		name, text, code, previous, message string
		status                              int
	}{
		{name: "missing_schema", text: `{"format":{"type":"json_schema"}}`, status: http.StatusUnprocessableEntity, code: "json_schema_missing"},
		{name: "oversized_schema", text: `{"format":{"type":"json_schema","schema":` + schema(65) + `}}`, status: http.StatusUnprocessableEntity, code: "json_schema_too_large"},
		{name: "malformed_text_format", text: `{"format":[]}`, status: http.StatusBadRequest, code: "invalid_text_format"},
		{name: "valid_at_cap", text: `{"format":{"type":"json_schema","schema":` + schema(64) + `}}`, status: http.StatusOK},
		{name: "continuation_refusal_precedes_schema", text: `{"format":{"type":"json_schema"}}`, previous: "previous", status: http.StatusBadRequest, message: "previous_response_id is unavailable with expert-parallel request fanout"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("FAK_GATEWAY_JSON_SCHEMA_MAX_BYTES", "64")
			t.Setenv("FAK_GATEWAY_JSON_SCHEMA_MAX_DEPTH", "32")
			var followerHits atomic.Int32
			type mirroredRequest struct{ path, follower, body string }
			mirrored := make(chan mirroredRequest, 1)
			follower := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				followerHits.Add(1)
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Errorf("read follower body: %v", err)
				}
				select {
				case mirrored <- mirroredRequest{r.URL.Path, r.Header.Get(epFollowerHeader), string(body)}:
				default:
				}
				_, _ = io.WriteString(w, `{}`)
			}))
			t.Cleanup(follower.Close)
			t.Setenv("FAK_EP_FANOUT_ADDRS", follower.URL)
			srv := newTestServer(t)
			if srv.roster != nil {
				t.Fatal("witness requires the eager no-roster fanout path")
			}
			planner := &responsesSchemaFanoutPlanner{}
			srv.planner = planner
			front := httptest.NewServer(srv.Handler())
			t.Cleanup(front.Close)
			body := `{"model":"test-model","input":"hello","max_output_tokens":16,"text":` + tc.text
			if tc.previous != "" {
				body += `,"previous_response_id":"` + tc.previous + `"`
			}
			body += `}`
			client := &http.Client{Timeout: 5 * time.Second}
			resp, err := client.Post(front.URL+"/v1/responses", "application/json", strings.NewReader(body))
			if err != nil {
				t.Fatal(err)
			}
			raw, err := io.ReadAll(resp.Body)
			resp.Body.Close()
			if err != nil {
				t.Fatal(err)
			}
			var envelope struct {
				Error struct{ Code string } `json:"error"`
			}
			if tc.code != "" {
				if err := json.Unmarshal(raw, &envelope); err != nil {
					t.Fatalf("decode refusal: %v: %s", err, raw)
				}
			}
			if resp.StatusCode != tc.status || envelope.Error.Code != tc.code || (tc.message != "" && !strings.Contains(string(raw), tc.message)) {
				t.Errorf("status=%d body=%s, want status=%d code=%s message=%s", resp.StatusCode, raw, tc.status, tc.code, tc.message)
			}
			if tc.status != http.StatusOK {
				if followerHits.Load() != 0 || planner.calls.Load() != 0 {
					t.Fatalf("refused format dispatched: follower calls=%d planner calls=%d", followerHits.Load(), planner.calls.Load())
				}
				return
			}
			if followerHits.Load() != 1 || planner.calls.Load() != 1 {
				t.Fatalf("valid format did not dispatch exactly once: follower calls=%d planner calls=%d", followerHits.Load(), planner.calls.Load())
			}
			select {
			case got := <-mirrored:
				if got.path != "/v1/responses" || got.follower != "1" || got.body != body {
					t.Fatalf("valid request mirror changed: got=%+v want body=%s", got, body)
				}
			default:
				t.Fatal("valid request has no recorded follower body")
			}
		})
	}
}
