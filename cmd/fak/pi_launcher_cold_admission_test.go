package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/projectassets"
)

// fak-test:runtime fast est=3s
func TestRunPiSelectedHaloColdAdmission(t *testing.T) {
	for _, tc := range []struct {
		name, selected, catalog, window string
		configured                      bool
		served                          int
		halo                            bool
	}{
		{name: "selected_halo", selected: "custom", catalog: `[{"id":"custom","context_length":131072,"owned_by":"halo-a"}]`, served: 131072, halo: true},
		{name: "selected_cloud", selected: "custom", catalog: `[{"id":"custom","context_length":131072,"owned_by":"cloud"}]`, served: 131072},
		{name: "selected_unknown", selected: "custom", catalog: `[{"id":"custom","context_length":131072}]`, served: 131072},
		{name: "unrelated_halo", selected: "custom", catalog: `[{"id":"local","context_length":262144,"owned_by":"halo-a"},{"id":"custom","context_length":131072,"owned_by":"cloud"}]`, served: 131072},
		{name: "exact_cloud_over_alias", selected: "custom", catalog: `[{"id":"vendor/custom","context_length":131072,"owned_by":"halo-a"},{"id":"custom","context_length":131072,"owned_by":"cloud"}]`, served: 131072},
		{name: "exact_unknown_over_alias", selected: "custom", catalog: `[{"id":"vendor/custom","context_length":131072,"owned_by":"halo-a"},{"id":"custom","context_length":131072}]`, served: 131072},
		{name: "malformed_owner", selected: "custom", catalog: `[{"id":"custom","context_length":65536,"owned_by":123}]`, served: 65536},
		{name: "qualified_alias", selected: "VENDOR/custom", catalog: `[{"id":"custom","context_length":131072,"owned_by":" HALO-a "}]`, served: 131072, halo: true},
		{name: "ambiguous_alias", selected: "request/custom", catalog: `[{"id":"cloud/custom","context_length":262144,"owned_by":"cloud"},{"id":"device/custom","context_length":131072,"owned_by":"halo-a"}]`, served: 131072, halo: true},
		{name: "duplicate_halo_first", selected: "custom", catalog: `[{"id":"custom","context_length":262144,"owned_by":"halo-a"},{"id":"custom","context_length":131072,"owned_by":"cloud"}]`, served: 131072, halo: true},
		{name: "duplicate_halo_last", selected: "custom", catalog: `[{"id":"custom","context_length":131072,"owned_by":"cloud"},{"id":"custom","context_length":262144,"owned_by":"halo-a"}]`, served: 131072, halo: true},
		{name: "explicit_smaller_window", selected: "custom", catalog: `[{"id":"custom","context_length":131072,"owned_by":"halo-a"}]`, window: "65536", served: 65536, halo: true},
		{name: "registry_window", selected: "Qwen3.8-27B-UD-Q2_K_XL", catalog: `[{"id":"Qwen3.8-27B-UD-Q2_K_XL","context_length":262144,"owned_by":"halo-a"}]`, served: 131072, halo: true},
		{name: "configured_route", selected: "custom", catalog: `[{"id":"local","context_length":65536,"owned_by":"cloud"},{"id":"custom","context_length":131072,"owned_by":"halo-a"}]`, configured: true, served: 131072, halo: true},
		{name: "unadvertised_selection", selected: "custom", catalog: `[{"id":"local","context_length":262144,"owned_by":"halo-a"}]`, served: 131072},
		{name: "nonpositive_window", selected: "custom", catalog: `[{"id":"custom","context_length":0,"owned_by":"halo-a"}]`, served: 131072, halo: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			isolatePiHome(t)
			dir := t.TempDir()
			t.Chdir(dir)
			for _, key := range []string{"HOME", "USERPROFILE", "APPDATA", "TMPDIR", "TMP", "TEMP"} {
				t.Setenv(key, dir)
			}
			t.Setenv("FAK_GATEWAY_KEY", "")
			var health, models atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				switch r.URL.Path {
				case "/healthz":
					health.Add(1)
					fmt.Fprint(w, `{"ok":true,"model":"local"}`)
				case "/v1/models":
					models.Add(1)
					fmt.Fprintf(w, `{"data":%s}`, tc.catalog)
				default:
					http.NotFound(w, r)
				}
			}))
			defer upstream.Close()
			modelsPath := filepath.Join(dir, "models.json")
			settingsPath := filepath.Join(dir, "settings.json")
			seed := map[string]any{"fixtureSetting": "preserved"}
			if tc.configured {
				seed["defaultProvider"], seed["defaultModel"] = "fak", tc.selected
			}
			data, err := json.Marshal(seed)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(settingsPath, data, 0o600); err != nil {
				t.Fatal(err)
			}
			args := []string{"--dry-run", "--base-url", upstream.URL + "/v1", "--write-config", "--safe-settings", "--config-path", modelsPath, "--settings-path", settingsPath}
			if !tc.configured {
				args = append(args, "--model", tc.selected)
			}
			if tc.window != "" {
				args = append(args, "--window", tc.window)
			}
			var stdout, stderr bytes.Buffer
			if code := runPi(&stdout, &stderr, args); code != 0 {
				t.Fatalf("runPi=%d: %s", code, stderr.String())
			}
			ordinary := projectassets.PiModelContextBudget(tc.selected, tc.served)
			want := ordinary
			if tc.halo {
				// Use the current router admission authority, never the issue's
				// historical numeric cap or a second copy of its formula.
				want = projectassets.PiContextBudgetFor(ordinary.ServedWindow, projectassets.HaloPiColdAdmission.MaxTrigger())
			}
			if tc.name == "selected_halo" && want.Envelope.CompactTrigger >= ordinary.Envelope.CompactTrigger {
				t.Fatal("fixture no longer distinguishes cold-admission cap from ordinary budget")
			}
			wantDiagnostic := fmt.Sprintf("compaction  = reserve %d, keep %d, fires at %d", want.ReserveTokens, want.KeepRecentTokens, want.Envelope.CompactTrigger)
			if !strings.Contains(stderr.String(), wantDiagnostic) {
				t.Errorf("selected route compaction omitted %q: %s", wantDiagnostic, stderr.String())
			}
			var settings struct {
				FixtureSetting string `json:"fixtureSetting"`
				DefaultModel   string `json:"defaultModel"`
				Compaction     struct {
					ReserveTokens    int `json:"reserveTokens"`
					KeepRecentTokens int `json:"keepRecentTokens"`
				} `json:"compaction"`
			}
			data, err = os.ReadFile(settingsPath)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(data, &settings); err != nil {
				t.Fatal(err)
			}
			if settings.Compaction.ReserveTokens != want.ReserveTokens || settings.Compaction.KeepRecentTokens != want.KeepRecentTokens {
				t.Fatalf("selected route settings=%+v want reserve=%d keep=%d", settings.Compaction, want.ReserveTokens, want.KeepRecentTokens)
			}
			if settings.FixtureSetting != "preserved" || settings.DefaultModel != tc.selected {
				t.Fatalf("unrelated settings or selected model changed: %+v", settings)
			}
			var config struct {
				Providers map[string]struct {
					Models []struct {
						ID            string `json:"id"`
						ContextWindow int    `json:"contextWindow"`
						MaxTokens     int    `json:"maxTokens"`
					} `json:"models"`
				} `json:"providers"`
			}
			data, err = os.ReadFile(modelsPath)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(data, &config); err != nil {
				t.Fatal(err)
			}
			entries := config.Providers["fak"].Models
			if len(entries) != 1 || entries[0].ID != tc.selected || entries[0].ContextWindow != ordinary.ResidentTarget || entries[0].MaxTokens != ordinary.MaxOutputTokens {
				t.Fatalf("provider changed with compaction cap: %+v ordinary=%+v", entries, ordinary)
			}
			if health.Load() != 1 || models.Load() != 1 {
				t.Fatalf("probe requests health=%d models=%d; want existing single probe", health.Load(), models.Load())
			}
		})
	}
}
