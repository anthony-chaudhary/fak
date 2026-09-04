package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
)

func TestHealthAdvertisesNativeCodeWorkspaceWithoutLeakingRoot(t *testing.T) {
	root := filepath.Join(t.TempDir(), "private-workspace-name")
	srv, err := New(Config{EngineID: "test", Model: "test", Native: true, NativeCodeWorkspace: root})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	workspace, ok := body["native_code_workspace"].(map[string]any)
	if !ok || workspace["armed"] != true {
		t.Fatalf("body=%v", body)
	}
	tools, ok := workspace["tools"].([]any)
	if !ok || len(tools) < 6 {
		t.Fatalf("workspace=%v", workspace)
	}
	if _, leaked := workspace["root"]; leaked {
		t.Fatalf("health leaked workspace root: %v", workspace)
	}
}
