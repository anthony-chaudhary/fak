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
	"testing"

	"github.com/anthony-chaudhary/fak/internal/projectassets"
)

// newPiConfigFakeGateway serves /healthz and /v1/models for the non-router
// `fak pi config` catalog probe.
func newPiConfigFakeGateway(t *testing.T, catalog string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/healthz":
			fmt.Fprint(w, `{"ok":true,"model":"local"}`)
		case "/v1/models":
			fmt.Fprintf(w, `{"data":%s}`, catalog)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// runPiConfigNonRouter drives the plain (non --from-router) `fak pi config`
// path against a fake gateway at addr.
func runPiConfigNonRouter(t *testing.T, addr string, args ...string) (int, string, string) {
	t.Helper()
	full := append([]string{"config", "--addr", addr}, args...)
	var stdout, stderr bytes.Buffer
	code := runPi(&stdout, &stderr, full)
	return code, stdout.String(), stderr.String()
}

// #13779 witness: the non-router `fak pi config` route must apply the Halo
// cold-admission cap exactly like the launcher and --from-router routes: a
// fake /v1/models row with owned_by halo-* caps the reported trigger and the
// written compaction block, while cloud/unowned models keep the ordinary
// budget. The provider contextWindow stays uncapped on every route.
// fak-test:runtime fast est=3s
func TestRunPiConfigNonRouterHaloColdAdmission(t *testing.T) {
	for _, tc := range []struct {
		name    string
		catalog string
		halo    bool
	}{
		{name: "halo_owner", catalog: `[{"id":"custom","context_length":131072,"owned_by":"halo-node-a"}]`, halo: true},
		{name: "cloud_owner", catalog: `[{"id":"custom","context_length":131072,"owned_by":"cloud"}]`},
		{name: "no_owner", catalog: `[{"id":"custom","context_length":131072}]`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			isolatePiHome(t)
			dir := t.TempDir()
			srv := newPiConfigFakeGateway(t, tc.catalog)
			modelsPath := filepath.Join(dir, "models.json")
			settingsPath := filepath.Join(dir, "settings.json")
			ordinary := projectassets.PiModelContextBudget("custom", 131072)
			want := ordinary
			if tc.halo {
				// Use the current router admission authority, never a second copy of its formula.
				want = projectassets.PiContextBudgetFor(ordinary.ServedWindow, projectassets.HaloPiColdAdmission.MaxTrigger())
				if want.Envelope.CompactTrigger >= ordinary.Envelope.CompactTrigger {
					t.Fatal("fixture no longer distinguishes cold-admission cap from ordinary budget")
				}
			}
			// Plan path: the reported trigger must carry the cap.
			code, _, stderr := runPiConfigNonRouter(t, srv.URL, "--model", "custom", "--window", "131072")
			if code != 0 {
				t.Fatalf("plan runPi=%d: %s", code, stderr)
			}
			wantLine := fmt.Sprintf("compaction reserve %d, keep %d, fires at %d", want.ReserveTokens, want.KeepRecentTokens, want.Envelope.CompactTrigger)
			if !strings.Contains(stderr, wantLine) {
				t.Errorf("plan banner omitted %q: %s", wantLine, stderr)
			}
			// Write path: the persisted compaction block must carry the cap.
			code, _, stderr = runPiConfigNonRouter(t, srv.URL, "--model", "custom", "--window", "131072", "--write", "--path", modelsPath, "--settings-path", settingsPath)
			if code != 0 {
				t.Fatalf("write runPi=%d: %s", code, stderr)
			}
			var settings struct {
				Compaction struct {
					ReserveTokens    int `json:"reserveTokens"`
					KeepRecentTokens int `json:"keepRecentTokens"`
				} `json:"compaction"`
			}
			data, err := os.ReadFile(settingsPath)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(data, &settings); err != nil {
				t.Fatal(err)
			}
			if settings.Compaction.ReserveTokens != want.ReserveTokens || settings.Compaction.KeepRecentTokens != want.KeepRecentTokens {
				t.Fatalf("written compaction=%+v want reserve=%d keep=%d", settings.Compaction, want.ReserveTokens, want.KeepRecentTokens)
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
			if len(entries) != 1 || entries[0].ID != "custom" || entries[0].ContextWindow != ordinary.ResidentTarget || entries[0].MaxTokens != ordinary.MaxOutputTokens {
				t.Fatalf("provider entry changed with compaction cap: %+v ordinary=%+v", entries, ordinary)
			}
		})
	}
}
