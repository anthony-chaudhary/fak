package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/projectassets"
)

// fak-test:runtime fast est=250ms
func TestRunPiUsesSelectedModelWindowEverywhere(t *testing.T) {
	for _, tc := range []struct {
		name, health, catalog, configuredModel, explicitModel, explicitWindow string
		wantModel                                                             string
		wantWindow                                                            int
	}{
		{
			name: "detected custom model", health: "custom",
			catalog:   `[{"id":"custom","context_length":65536}]`,
			wantModel: "custom", wantWindow: 65536,
		},
		{
			name: "explicit window overrides catalog", health: "custom", explicitWindow: "98304",
			catalog:   `[{"id":"custom","context_length":65536}]`,
			wantModel: "custom", wantWindow: 98304,
		},
		{
			name: "configured routed model", health: "local", configuredModel: "routed",
			catalog:   `[{"id":"local","context_length":65536},{"id":"routed","context_length":98304}]`,
			wantModel: "routed", wantWindow: 98304,
		},
		{
			name: "explicit routed model", health: "local", explicitModel: "routed",
			catalog:   `[{"id":"local","context_length":131072},{"id":"routed","context_length":65536}]`,
			wantModel: "routed", wantWindow: 65536,
		},
		{
			name: "configured alias uses smallest matching bound", health: "local", configuredModel: "VENDOR/custom",
			catalog:   `[{"id":"local","context_length":131072},{"id":"other/custom","context_length":98304},{"id":"custom","context_length":65536}]`,
			wantModel: "VENDOR/custom", wantWindow: 65536,
		},
		{
			name: "known route missing from catalog uses registry", health: "local", configuredModel: "deepseek-v41-flash",
			catalog:   `[{"id":"local","context_length":65536}]`,
			wantModel: "deepseek-v41-flash", wantWindow: 1000000,
		},
		{
			name: "health only uses prior", health: "custom",
			wantModel: "custom", wantWindow: 131072,
		},
		{
			name:      "catalog only detects model and window",
			catalog:   `[{"id":"custom","context_length":65536}]`,
			wantModel: "custom", wantWindow: 65536,
		},
		{
			name: "nonpositive selected bounds do not inherit local window", health: "local", explicitModel: "custom",
			catalog:   `[{"id":"local","context_length":65536},{"id":"custom","context_length":0},{"id":"custom","context_length":-1}]`,
			wantModel: "custom", wantWindow: 131072,
		},
		{
			name: "duplicate exact ids use smallest positive bound", health: "custom",
			catalog:   `[{"id":"custom","context_length":98304},{"id":"custom","context_length":65536}]`,
			wantModel: "custom", wantWindow: 65536,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			isolatePiHome(t)
			dir := t.TempDir()
			t.Chdir(dir)
			for _, key := range []string{"HOME", "USERPROFILE", "APPDATA", "TMPDIR", "TMP", "TEMP"} {
				t.Setenv(key, dir)
			}
			t.Setenv("FAK_GATEWAY_KEY", "")
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				switch {
				case r.URL.Path == "/healthz" && tc.health != "":
					fmt.Fprintf(w, `{"ok":true,"model":%q}`, tc.health)
				case r.URL.Path == "/v1/models" && tc.catalog != "":
					fmt.Fprintf(w, `{"data":%s}`, tc.catalog)
				default:
					http.NotFound(w, r)
				}
			}))
			defer server.Close()

			modelsPath := filepath.Join(dir, "models.json")
			settingsPath := filepath.Join(dir, "settings.json")
			if tc.configuredModel != "" {
				seed := fmt.Sprintf(`{"defaultProvider":"fak","defaultModel":%q}`, tc.configuredModel)
				if err := os.WriteFile(settingsPath, []byte(seed), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			args := []string{
				"--base-url", server.URL + "/v1", "--write-config", "--safe-settings",
				"--config-path", modelsPath, "--settings-path", settingsPath,
			}
			if tc.explicitModel != "" {
				args = append(args, "--model", tc.explicitModel)
			}
			if tc.explicitWindow != "" {
				args = append(args, "--window", tc.explicitWindow)
			}
			want := projectassets.PiModelContextBudget(tc.wantModel, tc.wantWindow)
			var stdout, stderr bytes.Buffer
			if code := runPi(&stdout, &stderr, append([]string{"--dry-run"}, args...)); code != 0 {
				t.Fatalf("dry-run returned %d: %s", code, stderr.String())
			}
			for _, text := range []string{
				fmt.Sprintf("contextWindow %d tokens of %d served window, maxTokens %d", want.ResidentTarget, want.ServedWindow, want.MaxOutputTokens),
				fmt.Sprintf("compaction  = reserve %d, keep %d, fires at %d", want.ReserveTokens, want.KeepRecentTokens, want.Envelope.CompactTrigger),
			} {
				if !strings.Contains(stderr.String(), text) {
					t.Errorf("diagnostic omitted %q:\n%s", text, stderr.String())
				}
			}
			var models struct {
				Providers map[string]struct {
					Models []struct {
						ID            string `json:"id"`
						ContextWindow int    `json:"contextWindow"`
						MaxTokens     int    `json:"maxTokens"`
					} `json:"models"`
				} `json:"providers"`
			}
			var settings struct {
				Compaction struct {
					ReserveTokens    int `json:"reserveTokens"`
					KeepRecentTokens int `json:"keepRecentTokens"`
				} `json:"compaction"`
			}
			for path, target := range map[string]any{modelsPath: &models, settingsPath: &settings} {
				data, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(data, target); err != nil {
					t.Fatal(err)
				}
			}
			entries := models.Providers["fak"].Models
			if len(entries) != 1 || entries[0].ID != tc.wantModel || entries[0].ContextWindow != want.ResidentTarget || entries[0].MaxTokens != want.MaxOutputTokens {
				t.Errorf("provider models = %+v, want %s window=%d output=%d", entries, tc.wantModel, want.ResidentTarget, want.MaxOutputTokens)
			}
			if settings.Compaction.ReserveTokens != want.ReserveTokens || settings.Compaction.KeepRecentTokens != want.KeepRecentTokens {
				t.Errorf("compaction = %+v, want reserve=%d keep=%d", settings.Compaction, want.ReserveTokens, want.KeepRecentTokens)
			}
			if tc.wantWindow == 65536 && (settings.Compaction.ReserveTokens != 12288 || settings.Compaction.KeepRecentTokens != 4096) {
				t.Errorf("64KiB regression: compaction = %+v, want reserve=12288 keep=4096", settings.Compaction)
			}

			originalRun := piLaunchRun
			t.Cleanup(func() { piLaunchRun = originalRun })
			calls := 0
			piLaunchRun = func(_, _ io.Writer, argv, _ []string) int {
				calls++
				for i := 1; i+1 < len(argv); i++ {
					if argv[i] != "-e" {
						continue
					}
					data, err := os.ReadFile(argv[i+1])
					if err != nil {
						t.Fatal(err)
					}
					for _, text := range []string{
						fmt.Sprintf("contextWindow: %d", want.ResidentTarget),
						fmt.Sprintf("maxTokens: %d", want.MaxOutputTokens),
					} {
						if !strings.Contains(string(data), text) {
							t.Errorf("session provider omitted %q:\n%s", text, data)
						}
					}
					return 0
				}
				t.Fatal("session provider extension missing from launch argv")
				return 1
			}
			stdout.Reset()
			stderr.Reset()
			if code := runPi(&stdout, &stderr, args); code != 0 || calls != 1 {
				t.Fatalf("launch returned %d, child calls=%d: %s", code, calls, stderr.String())
			}
		})
	}
}
