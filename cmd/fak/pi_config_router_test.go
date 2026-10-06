package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

const piRouterTestKey = "test-gateway-key"

type piRouterFakeRow map[string]any

// newPiRouterFake serves GET /v1/models with rows and requires the test bearer key.
func newPiRouterFake(t *testing.T, rows []piRouterFakeRow) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer "+piRouterTestKey {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		if rows == nil {
			rows = []piRouterFakeRow{}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": rows})
	}))
	t.Cleanup(srv.Close)
	return srv
}

func pinPiRouterTestEnv(t *testing.T, key string) {
	t.Helper()
	isolatePiHome(t)
	prevKey, prevNow := piConfigRouterKey, piConfigNow
	piConfigRouterKey = func(string) string { return key }
	piConfigNow = func() time.Time { return time.Date(2026, 10, 6, 12, 0, 0, 0, time.UTC) }
	t.Cleanup(func() { piConfigRouterKey, piConfigNow = prevKey, prevNow })
}

// multiModelRouterRows: two models on one appliance backend, one cloud model with
// no advertised window, a same-leaf alias, and an explicitly marked alias.
func multiModelRouterRows() []piRouterFakeRow {
	return []piRouterFakeRow{
		{"id": "org/model-a", "owned_by": "appliance-a", "context_window": 131072, "context_length": 262144},
		{"id": "model-b", "owned_by": "appliance-a", "context_length": 65536},
		{"id": "cloud-c", "owned_by": "cloud"},
		{"id": "model-a", "owned_by": "appliance-a", "context_length": 131072},
		{"id": "fast", "owned_by": "appliance-a", "alias_of": "org/model-a", "context_length": 131072},
	}
}

const piRouterOtherProviderBlock = `    "other-router": {
      "baseUrl": "http://127.0.0.1:9/v1",
      "models": [ { "id": "z", "contextWindow": 500000 } ],
      "apiKey": "x"
    },`

const piRouterKeptEntry = `"id": "org/model-a",
          "name": "keep my name",
          "x-note": "operator field",
          "contextWindow": %CW%,
          "maxTokens": %MT%`

func stalePiModelsJSON(routerURL string) string {
	keep := strings.NewReplacer("%CW%", "500000", "%MT%", "32768").Replace(piRouterKeptEntry)
	return `{
  "zz-unknown": {"keep": [1, 2, 3]},
  "providers": {
` + piRouterOtherProviderBlock + `
    "fak": {
      "baseUrl": "` + routerURL + `",
      "apiKey": "fak",
      "api": "openai-completions",
      "models": [
        {
          "id": "qwen38:27b-q4",
          "contextWindow": 65536
        },
        {
          ` + keep + `
        },
        {
          "id": "/var/lib/models/stale.gguf"
        },
        {
          "id": "custom-model"
        }
      ]
    },
    "tail-provider": { "baseUrl": "http://example.invalid/v1" }
  }
}
`
}

type piFakModel struct {
	ID            string `json:"id"`
	ContextWindow int    `json:"contextWindow"`
	MaxTokens     int    `json:"maxTokens"`
}

func readPiFakModels(t *testing.T, path string) []piFakModel {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Providers map[string]struct {
			Models []piFakModel `json:"models"`
		} `json:"providers"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatalf("written config is not valid JSON: %v\n%s", err, data)
	}
	return doc.Providers["fak"].Models
}

func piBackups(t *testing.T, path string) []string {
	t.Helper()
	matches, err := filepath.Glob(path + ".bak-*")
	if err != nil {
		t.Fatal(err)
	}
	return matches
}

func runPiConfigRouter(t *testing.T, args ...string) (int, string, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := runPi(&stdout, &stderr, append([]string{"config"}, args...))
	return code, stdout.String(), stderr.String()
}

// fak-test:runtime fast est=1s
func TestPiConfigFromRouterPlanThenWriteIsIdempotent(t *testing.T) {
	pinPiRouterTestEnv(t, piRouterTestKey)
	srv := newPiRouterFake(t, multiModelRouterRows())
	routerURL := srv.URL + "/v1"
	dir := t.TempDir()
	modelsPath := filepath.Join(dir, "models.json")
	settingsPath := filepath.Join(dir, "settings.json")
	original := []byte(stalePiModelsJSON(routerURL))
	if err := os.WriteFile(modelsPath, original, 0o644); err != nil {
		t.Fatal(err)
	}
	settingsOrig := []byte(`{"defaultProvider": "fak", "defaultModel": "custom-model", "theme": "dark"}`)
	if err := os.WriteFile(settingsPath, settingsOrig, 0o644); err != nil {
		t.Fatal(err)
	}

	// Plan (no --write): the diff is reported and nothing on disk changes.
	rows, _, err := fetchPiRouterCatalog(routerURL, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := buildPiRouterPlan(rows, routerURL, modelsPath, settingsPath)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := plan.Removed, []string{"qwen38:27b-q4", "/var/lib/models/stale.gguf", "custom-model"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Removed = %v, want %v", got, want)
	}
	if got, want := plan.Added, []string{"model-b", "cloud-c"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Added = %v, want %v", got, want)
	}
	if len(plan.Changed) != 1 || plan.Changed[0].ID != "org/model-a" || plan.Changed[0].FromWindow != 500000 || plan.Changed[0].ToWindow != 65536 {
		t.Fatalf("Changed = %+v, want org/model-a 500000 -> 65536", plan.Changed)
	}
	if plan.Default.Served || plan.Default.Pick != "org/model-a" {
		t.Fatalf("Default = %+v, want stale custom-model replaced by org/model-a", plan.Default)
	}
	if code, _, stderr := runPiConfigRouter(t, "--from-router", routerURL, "--path", modelsPath, "--settings-path", settingsPath); code != 0 {
		t.Fatalf("plan exit = %d, stderr=%s", code, stderr)
	}
	if got, _ := os.ReadFile(modelsPath); !bytes.Equal(got, original) {
		t.Fatal("plan mode modified models.json")
	}
	if got, _ := os.ReadFile(settingsPath); !bytes.Equal(got, settingsOrig) {
		t.Fatal("plan mode modified settings.json")
	}
	if len(piBackups(t, modelsPath)) != 0 {
		t.Fatal("plan mode wrote a backup")
	}

	// --write applies the plan.
	if code, _, stderr := runPiConfigRouter(t, "--from-router", routerURL, "--write", "--path", modelsPath, "--settings-path", settingsPath); code != 0 {
		t.Fatalf("write exit = %d, stderr=%s", code, stderr)
	}
	want := []piFakModel{
		{ID: "org/model-a", ContextWindow: 65536, MaxTokens: 16384},
		{ID: "model-b", ContextWindow: 32768, MaxTokens: 8192},
		// No advertised window: the conservative default is capped by the smallest
		// window the router does advertise (65536), never the larger prior.
		{ID: "cloud-c", ContextWindow: 32768, MaxTokens: 8192},
	}
	if got := readPiFakModels(t, modelsPath); !reflect.DeepEqual(got, want) {
		t.Fatalf("fak models = %+v, want %+v", got, want)
	}
	written, _ := os.ReadFile(modelsPath)
	keptEntry := strings.NewReplacer("%CW%", "65536", "%MT%", "16384").Replace(piRouterKeptEntry)
	for _, verbatim := range []string{
		`"zz-unknown": {"keep": [1, 2, 3]},`,
		piRouterOtherProviderBlock,
		`"tail-provider": { "baseUrl": "http://example.invalid/v1" }`,
		keptEntry,
	} {
		if !bytes.Contains(written, []byte(verbatim)) {
			t.Fatalf("written config lost byte-stable content %q:\n%s", verbatim, written)
		}
	}
	backups := piBackups(t, modelsPath)
	if len(backups) != 1 {
		t.Fatalf("models backups = %v, want exactly one", backups)
	}
	if got, _ := os.ReadFile(backups[0]); !bytes.Equal(got, original) {
		t.Fatal("models backup does not hold the original bytes")
	}
	var settings map[string]any
	settingsNow, _ := os.ReadFile(settingsPath)
	if err := json.Unmarshal(settingsNow, &settings); err != nil {
		t.Fatal(err)
	}
	if settings["defaultModel"] != "org/model-a" || settings["defaultProvider"] != "fak" || settings["theme"] != "dark" {
		t.Fatalf("settings = %v, want defaultModel org/model-a with theme preserved", settings)
	}
	sBackups := piBackups(t, settingsPath)
	if len(sBackups) != 1 {
		t.Fatalf("settings backups = %v, want exactly one", sBackups)
	}
	if got, _ := os.ReadFile(sBackups[0]); !bytes.Equal(got, settingsOrig) {
		t.Fatal("settings backup does not hold the original bytes")
	}

	// A second --write is a no-op: same bytes, no new backup.
	if code, _, stderr := runPiConfigRouter(t, "--from-router", routerURL, "--write", "--path", modelsPath, "--settings-path", settingsPath); code != 0 {
		t.Fatalf("second write exit = %d, stderr=%s", code, stderr)
	}
	if again, _ := os.ReadFile(modelsPath); !bytes.Equal(again, written) {
		t.Fatalf("second run changed models.json:\n%s", again)
	}
	if again, _ := os.ReadFile(settingsPath); !bytes.Equal(again, settingsNow) {
		t.Fatal("second run changed settings.json")
	}
	if n, m := len(piBackups(t, modelsPath)), len(piBackups(t, settingsPath)); n != 1 || m != 1 {
		t.Fatalf("second run wrote backups: models=%d settings=%d", n, m)
	}
	plan2, err := buildPiRouterPlan(rows, routerURL, modelsPath, settingsPath)
	if err != nil {
		t.Fatal(err)
	}
	if plan2.configChanged() || len(plan2.Added)+len(plan2.Removed)+len(plan2.Changed) != 0 || !plan2.Default.Served {
		t.Fatalf("second plan not empty: %+v", plan2)
	}
}

// fak-test:runtime fast est=1s
func TestPiConfigFromRouterFreshFileAndDefaultFromConfiguredURL(t *testing.T) {
	pinPiRouterTestEnv(t, piRouterTestKey)
	srv := newPiRouterFake(t, multiModelRouterRows())
	dir := t.TempDir()
	modelsPath := filepath.Join(dir, "models.json")
	settingsPath := filepath.Join(dir, "settings.json")
	if code, _, stderr := runPiConfigRouter(t, "--from-router", srv.URL, "--write", "--path", modelsPath, "--settings-path", settingsPath); code != 0 {
		t.Fatalf("exit = %d, stderr=%s", code, stderr)
	}
	got := readPiFakModels(t, modelsPath)
	if len(got) != 3 || got[0].ID != "org/model-a" || got[1].ID != "model-b" || got[2].ID != "cloud-c" {
		t.Fatalf("fresh models = %+v", got)
	}
	if len(piBackups(t, modelsPath)) != 0 {
		t.Fatal("fresh write produced a backup of a file that did not exist")
	}
	// Bare --from-router reads the provider baseUrl that the first run wrote.
	if code, stdout, stderr := runPiConfigRouter(t, "--from-router", "--path", modelsPath, "--settings-path", settingsPath); code != 0 || !strings.Contains(stdout, srv.URL+"/v1") {
		t.Fatalf("bare --from-router exit = %d, stdout=%s stderr=%s", code, stdout, stderr)
	}
}

// fak-test:runtime fast est=1s
func TestPiConfigFromRouterRefusals(t *testing.T) {
	dir := t.TempDir()
	modelsPath := filepath.Join(dir, "models.json")
	original := []byte(`{"providers": {"fak": {"models": [{"id": "x"}]}}}`)
	if err := os.WriteFile(modelsPath, original, 0o644); err != nil {
		t.Fatal(err)
	}

	t.Run("empty catalog", func(t *testing.T) {
		pinPiRouterTestEnv(t, piRouterTestKey)
		srv := newPiRouterFake(t, nil)
		if _, _, err := fetchPiRouterCatalog(srv.URL+"/v1", time.Second); !errors.Is(err, errPiRouterEmptyCatalog) {
			t.Fatalf("err = %v, want errPiRouterEmptyCatalog", err)
		}
		if code, _, _ := runPiConfigRouter(t, "--from-router", srv.URL, "--write", "--path", modelsPath); code != 1 {
			t.Fatalf("exit = %d, want 1", code)
		}
	})
	t.Run("unauthorized", func(t *testing.T) {
		pinPiRouterTestEnv(t, "")
		srv := newPiRouterFake(t, multiModelRouterRows())
		if _, _, err := fetchPiRouterCatalog(srv.URL+"/v1", time.Second); !errors.Is(err, errPiRouterUnauthorized) {
			t.Fatalf("err = %v, want errPiRouterUnauthorized", err)
		}
	})
	t.Run("malformed config", func(t *testing.T) {
		pinPiRouterTestEnv(t, piRouterTestKey)
		srv := newPiRouterFake(t, multiModelRouterRows())
		bad := filepath.Join(t.TempDir(), "models.json")
		if err := os.WriteFile(bad, []byte(`[1, 2]`), 0o644); err != nil {
			t.Fatal(err)
		}
		rows, url, err := fetchPiRouterCatalog(srv.URL+"/v1", time.Second)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := buildPiRouterPlan(rows, url, bad, filepath.Join(t.TempDir(), "settings.json")); !errors.Is(err, errPiConfigMalformed) {
			t.Fatalf("err = %v, want errPiConfigMalformed", err)
		}
	})
	t.Run("model flag conflicts", func(t *testing.T) {
		pinPiRouterTestEnv(t, piRouterTestKey)
		if code, _, _ := runPiConfigRouter(t, "--from-router", "--model", "x", "--path", modelsPath); code != 2 {
			t.Fatalf("exit = %d, want 2", code)
		}
	})
	if got, _ := os.ReadFile(modelsPath); !bytes.Equal(got, original) {
		t.Fatal("a refused run modified models.json")
	}
}

func TestPiRouterCatalogCollapseRule(t *testing.T) {
	rows := []piRouterRow{
		{ID: "a/x", OwnedBy: "b1", Window: 131072},
		{ID: "x", OwnedBy: "b2", Window: 65536},
		{ID: "X", OwnedBy: "b1", Window: 32768},
		{ID: "y", OwnedBy: "b1"},
		{ID: "p", OwnedBy: "b1", AliasOf: "q"},
		{ID: "q", OwnedBy: "b1", AliasOf: "p"},
		{ID: "r", OwnedBy: "b1", AliasOf: "a/x"},
		{ID: "s", OwnedBy: "b1", AliasOf: "missing"},
		{ID: "a/x", OwnedBy: "b9"},
	}
	type view struct {
		ID      string
		Window  int
		Aliases []string
	}
	var got []view
	for _, m := range collapsePiRouterCatalog(rows) {
		got = append(got, view{m.ID, m.Window, m.Aliases})
	}
	want := []view{
		{"a/x", 32768, []string{"X", "r"}},
		{"x", 65536, nil},
		{"y", 0, nil},
		{"p", 0, []string{"q"}},
		{"s", 0, nil},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("collapse = %+v\nwant     %+v", got, want)
	}
}

func TestPiRouterDefaultPick(t *testing.T) {
	models := []piRouterModel{{ID: "first"}, {ID: "marked", Default: true}}
	if got := piRouterDefaultModel(models); got != "marked" {
		t.Fatalf("marked default = %q", got)
	}
	if got := piRouterDefaultModel(models[:1]); got != "first" {
		t.Fatalf("unmarked default = %q", got)
	}

	dir := t.TempDir()
	plan := &piRouterPlan{Models: []piRouterModel{{ID: "org/model-a", Aliases: []string{"fast"}}, {ID: "model-b"}}}
	cases := []struct {
		settings string
		served   bool
		pick     string
		skipped  bool
	}{
		{`{"defaultProvider": "fak", "defaultModel": "model-b"}`, true, "", false},
		{`{"defaultProvider": "fak", "defaultModel": "fast"}`, false, "org/model-a", false},
		{`{"defaultProvider": "other", "defaultModel": "gone"}`, false, "", true},
		{`{}`, false, "org/model-a", false},
	}
	for i, tc := range cases {
		path := filepath.Join(dir, "settings"+string(rune('a'+i))+".json")
		if err := os.WriteFile(path, []byte(tc.settings), 0o644); err != nil {
			t.Fatal(err)
		}
		def, _, err := planPiRouterDefault(path, plan)
		if err != nil {
			t.Fatal(err)
		}
		if def.Served != tc.served || def.Pick != tc.pick || (def.Skipped != "") != tc.skipped {
			t.Fatalf("case %d: %+v", i, def)
		}
	}
}

func TestFoldPiFromRouterArg(t *testing.T) {
	got := foldPiFromRouterArg([]string{"--from-router", "http://r:1/v1", "--write", "--from-router", "--path", "p"})
	want := []string{"--from-router=http://r:1/v1", "--write", "--from-router", "--path", "p"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("fold = %v, want %v", got, want)
	}
}
