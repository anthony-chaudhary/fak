package harnessweb

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestLiveAdapterDiscoversArmedWorkspaceAndStatusProjectsIt(t *testing.T) {
	fak := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "native_code_workspace": map[string]any{"armed": true, "tools": []string{"Read", "Write", "Edit", "Bash", "Grep", "Glob"}}})
		case "/debug/vars":
			_ = json.NewEncoder(w).Encode(map[string]any{})
		default:
			http.NotFound(w, r)
		}
	}))
	defer fak.Close()
	live := &liveAdapter{baseURL: fak.URL, client: fak.Client()}
	if err := live.probeWorkspace(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !live.workspace.Armed || !reflect.DeepEqual(live.workspace.Tools, []string{"Read", "Write", "Edit", "Bash", "Grep", "Glob"}) {
		t.Fatalf("workspace=%+v", live.workspace)
	}
	ui := httptest.NewServer(handlerWithLive(newStore(), live))
	defer ui.Close()
	resp, err := ui.Client().Get(ui.URL + "/api/status")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var status struct {
		Mode      string          `json:"mode"`
		Workspace workspaceStatus `json:"workspace"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		t.Fatal(err)
	}
	if status.Mode != "live" || !status.Workspace.Armed || len(status.Workspace.Tools) != 6 {
		t.Fatalf("status=%+v", status)
	}
}
