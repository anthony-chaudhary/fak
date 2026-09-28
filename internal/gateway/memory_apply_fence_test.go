package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/recall"
)

// TestMCPMemoryRunApplyFailsClosedBeforePersist proves the public MCP route cannot
// apply a storage mutation while it has no act-bound lease epoch.
func TestMCPMemoryRunApplyFailsClosedBeforePersist(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	recorder := recall.NewRecorder("memory-apply-fence")
	recorder.Record(ctx, "clock", []byte("it is 3:47pm right now"))
	recorder.Record(ctx, "preference", []byte("the user prefers afternoon meetings in general"))
	if err := recorder.Persist(dir); err != nil {
		t.Fatal(err)
	}
	assertNoTombstones := func() {
		t.Helper()
		session, err := recall.Load(dir)
		if err != nil {
			t.Fatal(err)
		}
		for _, page := range session.Pages() {
			if session.Tombstoned(page.Step) {
				t.Fatalf("page %d was persisted as tombstoned", page.Step)
			}
		}
	}
	assertNoTombstones()

	frame, err := json.Marshal(map[string]any{
		"jsonrpc": "2.0", "id": 1, "method": "tools/call",
		"params": map[string]any{
			"name":      "fak_memory_run",
			"arguments": map[string]any{"driver": "clean", "image_dir": dir, "apply": true},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/mcp", bytes.NewReader(frame))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	newTestServer(t).Handler().ServeHTTP(rec, req)
	var response struct {
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode MCP response: %v; body=%s", err, rec.Body.String())
	}
	if response.Error == nil || !strings.Contains(response.Error.Message, "act-bound lease") {
		t.Fatalf("mutating fak_memory_run response = %s, want act-bound lease refusal", rec.Body.String())
	}
	assertNoTombstones()
}
