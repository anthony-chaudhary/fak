package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestWorkspaceIdentityStableAndDoesNotRevealRoot(t *testing.T) {
	root := t.TempDir()
	got := workspaceIdentity(root)
	if got == "" || got != workspaceIdentity(filepath.Clean(root)) {
		t.Fatalf("identity=%q", got)
	}
	if filepath.IsAbs(got) || got == root {
		t.Fatalf("identity leaked root: %q", got)
	}
}

func TestStatusCarriesBoundWorkspaceIdentityAcrossStores(t *testing.T) {
	root := t.TempDir()
	identity := workspaceIdentity(root)
	gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"native_code_workspace": map[string]any{
			"armed": true, "tools": []string{"Read", "Write", "Edit", "Bash", "Grep", "Glob"},
		}})
	}))
	defer gateway.Close()
	for _, state := range []string{"first.json", "second.json"} {
		adapter := &liveAdapter{baseURL: gateway.URL, client: gateway.Client(), identity: identity}
		if err := adapter.probeWorkspace(t.Context()); err != nil {
			t.Fatal(err)
		}
		s := newStore()
		s.persist = filepath.Join(root, state)
		req := httptest.NewRequest(http.MethodGet, "/api/status", nil)
		rec := httptest.NewRecorder()
		handlerWithLive(s, adapter).ServeHTTP(rec, req)
		var body struct {
			Workspace workspaceStatus `json:"workspace"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatal(err)
		}
		if !body.Workspace.Armed || body.Workspace.Identity != identity {
			t.Fatalf("status=%+v", body.Workspace)
		}
		if string(rec.Body.Bytes()) == root {
			t.Fatal("status leaked workspace root")
		}
	}
	_ = os.Remove(filepath.Join(root, "first.json"))
}
