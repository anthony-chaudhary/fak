package gateway

import (
	"encoding/json"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func TestModelsAdvertisesRosterCatalog(t *testing.T) {
	roster := &modelroute.Roster{Version: modelroute.RosterVersion, Accounts: []modelroute.Account{{ID: "local", Kind: modelroute.KindLocal}}, Bindings: []modelroute.Binding{{Model: "zeta", Account: "local"}, {Model: "boot-model", Account: "local"}, {Model: "alpha", Account: "local"}}}
	got, raw := modelsCatalog(t, &Server{model: "boot-model", roster: roster})
	want := []string{"alpha", "boot-model", "zeta"}
	if !reflect.DeepEqual(got.data, want) || !reflect.DeepEqual(got.codex, want) {
		t.Fatalf("catalog data=%v codex=%v, want %v", got.data, got.codex, want)
	}
	for _, field := range []string{"account", "base_url", "cred_env", "upstream_model"} {
		if strings.Contains(raw, `"`+field+`"`) {
			t.Fatalf("catalog leaks roster field %q: %s", field, raw)
		}
	}
	plain, _ := modelsCatalog(t, &Server{model: "boot-model"})
	if !reflect.DeepEqual(plain.data, []string{"boot-model"}) || !reflect.DeepEqual(plain.codex, []string{"boot-model"}) {
		t.Fatalf("no-roster catalog changed: data=%v codex=%v", plain.data, plain.codex)
	}
}

type modelCatalogIDs struct{ data, codex []string }

func modelsCatalog(t *testing.T, s *Server) (modelCatalogIDs, string) {
	t.Helper()
	rec := httptest.NewRecorder()
	s.handleModels(rec, httptest.NewRequest("GET", "/v1/models", nil))
	if rec.Code != 200 {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var body struct {
		Data   []map[string]any `json:"data"`
		Models []map[string]any `json:"models"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	var ids modelCatalogIDs
	for _, row := range body.Data {
		ids.data = append(ids.data, row["id"].(string))
	}
	for _, row := range body.Models {
		ids.codex = append(ids.codex, row["slug"].(string))
	}
	return ids, rec.Body.String()
}
