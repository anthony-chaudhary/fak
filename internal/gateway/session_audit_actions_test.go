package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

func TestHandleFakSessionAuditActionsReportsPressure(t *testing.T) {
	root := t.TempDir()
	const ns = "C--work-fak"
	writeGatewaySessionAuditJSONL(t, filepath.Join(root, ns, "heavy.jsonl"), []map[string]any{
		gatewaySessionAuditAssistantDetailed("opus", 200, 0, 900_000, 50_000, "claude-opus-4-8"),
	})
	writeGatewaySessionAuditJSONL(t, filepath.Join(root, ns, "fable.jsonl"), []map[string]any{
		gatewaySessionAuditAssistantDetailed("fable", 300, 0, 20_000, 1_000, "claude-fable-5"),
	})
	q := url.Values{}
	q.Set("root", root)
	q.Set("ns_prefix", ns)
	q.Set("since_days", "-1")
	q.Set("max", "2")
	q.Set("gate", "high")
	rec := httptest.NewRecorder()
	(&Server{}).handleFakSessionAuditActions(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/session-audit/actions?"+q.Encode(), nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 body=%s", rec.Code, rec.Body.String())
	}
	var plan sessionaudit.CompactActionPlan
	if err := json.Unmarshal(rec.Body.Bytes(), &plan); err != nil {
		t.Fatalf("decode: %v\n%s", err, rec.Body.String())
	}
	if plan.Schema != sessionaudit.CompactActionPlanSchema ||
		plan.Scope.NamespaceFilter != ns ||
		plan.Counts.Total != 2 ||
		plan.Counts.High != 2 ||
		plan.Gate.Threshold != "high" ||
		plan.Gate.Verdict != "refuse" ||
		plan.Gate.Refused != 2 {
		t.Fatalf("plan = %+v", plan)
	}
	if len(plan.Actions) != 2 || plan.Actions[1].Session != "heavy" || !strings.Contains(plan.Actions[1].Target, ns+"/heavy") {
		t.Fatalf("actions = %+v", plan.Actions)
	}
}

func TestHandleFakSessionAuditActionsRejectsInvalidGate(t *testing.T) {
	rec := httptest.NewRecorder()
	(&Server{}).handleFakSessionAuditActions(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/session-audit/actions?gate=urgent", nil))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d want 400 body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "invalid gate") {
		t.Fatalf("error body missing invalid gate:\n%s", rec.Body.String())
	}
}

func TestHandleFakSessionAuditActionsRejectsNonGet(t *testing.T) {
	rec := httptest.NewRecorder()
	(&Server{}).handleFakSessionAuditActions(rec, httptest.NewRequest(http.MethodPost, "/v1/fak/session-audit/actions", nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST: status=%d want 405", rec.Code)
	}
}

func gatewaySessionAuditAssistantDetailed(id string, out, input, cacheRead, cacheCreate int64, model string) map[string]any {
	return map[string]any{
		"type":      "assistant",
		"timestamp": "2026-06-20T00:00:00.000Z",
		"message": map[string]any{
			"id":    id,
			"model": model,
			"usage": map[string]any{
				"input_tokens":                input,
				"output_tokens":               out,
				"cache_read_input_tokens":     cacheRead,
				"cache_creation_input_tokens": cacheCreate,
			},
			"content": []any{},
		},
	}
}

func writeGatewaySessionAuditJSONL(t *testing.T, path string, records []map[string]any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o777); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	enc := json.NewEncoder(f)
	for _, r := range records {
		if err := enc.Encode(r); err != nil {
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}
