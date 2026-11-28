package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func TestRouteInitProfilesValid(t *testing.T) {
	profiles := []string{"hybrid", "local-only", "multi-cloud"}
	for _, prof := range profiles {
		t.Run(prof, func(t *testing.T) {
			outDir := filepath.Join(t.TempDir(), "router-"+prof)
			code, out, errb := runRT("init", "--profile", prof, "--out-dir", outDir)
			if code != 0 {
				t.Fatalf("fak route init --profile %s failed: exit=%d stderr=%s", prof, code, errb)
			}
			if !strings.Contains(out, "fak serve --route-accounts") || !strings.Contains(out, "/v1/router/status") {
				t.Fatalf("missing next-step command or status check in output:\n%s", out)
			}

			accountsPath := filepath.Join(outDir, "accounts.json")
			manifestPath := filepath.Join(outDir, "manifest.json")

			roster, err := modelroute.LoadRoster(accountsPath)
			if err != nil {
				t.Fatalf("LoadRoster(%s): %v", accountsPath, err)
			}
			manifest, err := modelroute.LoadManifest(manifestPath)
			if err != nil {
				t.Fatalf("LoadManifest(%s): %v", manifestPath, err)
			}

			for _, m := range manifest.Default.Members {
				if _, err := roster.ResolveChain(m.Model); err != nil {
					t.Fatalf("ResolveChain(default %q): %v", m.Model, err)
				}
			}
			for _, rule := range manifest.Rules {
				for _, m := range rule.Plan.Members {
					if _, err := roster.ResolveChain(m.Model); err != nil {
						t.Fatalf("ResolveChain(rule %s %q): %v", rule.Name, m.Model, err)
					}
				}
			}
		})
	}
}

func TestRouteInitOverwriteGuardAndForce(t *testing.T) {
	outDir := filepath.Join(t.TempDir(), "router")
	code, _, errb := runRT("init", "--out-dir", outDir)
	if code != 0 {
		t.Fatalf("initial init failed: exit=%d stderr=%s", code, errb)
	}

	accountsPath := filepath.Join(outDir, "accounts.json")
	sentinel := []byte("{\"sentinel\":true}\n")
	if err := os.WriteFile(accountsPath, sentinel, 0o644); err != nil {
		t.Fatalf("write sentinel: %v", err)
	}

	code, _, errb = runRT("init", "--out-dir", outDir)
	if code == 0 {
		t.Fatalf("expected overwrite refusal without --force, got exit=0")
	}
	if !strings.Contains(errb, "--force") {
		t.Fatalf("expected --force hint in stderr, got: %s", errb)
	}
	got, err := os.ReadFile(accountsPath)
	if err != nil {
		t.Fatalf("read sentinel: %v", err)
	}
	if !bytes.Equal(got, sentinel) {
		t.Fatalf("existing file was modified without --force")
	}

	code, out, errb := runRT("init", "--out-dir", outDir, "--force", "--json")
	if code != 0 {
		t.Fatalf("init --force failed: exit=%d stderr=%s", code, errb)
	}
	var receipt routeInitReceipt
	if err := json.Unmarshal([]byte(out), &receipt); err != nil {
		t.Fatalf("parse json receipt: %v\n%s", err, out)
	}
	if receipt.Profile != "hybrid" || !strings.Contains(receipt.NextCommand, "--route-accounts") || !strings.Contains(receipt.StatusCheck, "/v1/router/status") {
		t.Fatalf("unexpected receipt: %+v", receipt)
	}
	if _, err := modelroute.LoadRoster(accountsPath); err != nil {
		t.Fatalf("LoadRoster after --force: %v", err)
	}
}

func TestRouteInitUnknownProfile(t *testing.T) {
	outDir := filepath.Join(t.TempDir(), "router")
	code, _, errb := runRT("init", "--profile", "does-not-exist", "--out-dir", outDir)
	if code != 2 {
		t.Fatalf("expected exit 2 for unknown profile, got %d (stderr=%s)", code, errb)
	}
}

func TestRouteExamplesLocalRouterZeroDrift(t *testing.T) {
	exDir := filepath.Join("..", "..", "examples", "local-router")
	accountsPath := filepath.Join(exDir, "accounts.json")
	manifestPath := filepath.Join(exDir, "manifest.json")

	diskRoster, err := modelroute.LoadRoster(accountsPath)
	if err != nil {
		t.Fatalf("LoadRoster(%s): %v", accountsPath, err)
	}
	diskManifest, err := modelroute.LoadManifest(manifestPath)
	if err != nil {
		t.Fatalf("LoadManifest(%s): %v", manifestPath, err)
	}

	wantRoster, wantManifest, err := routeInitProfile("hybrid")
	if err != nil {
		t.Fatalf("routeInitProfile(hybrid): %v", err)
	}
	if !bytes.Equal(diskRoster.JSON(), wantRoster.JSON()) {
		t.Fatalf("examples/local-router/accounts.json drifted from hybrid profile:\ngot:\n%s\nwant:\n%s", diskRoster.JSON(), wantRoster.JSON())
	}
	if !bytes.Equal(diskManifest.JSON(), wantManifest.JSON()) {
		t.Fatalf("examples/local-router/manifest.json drifted from hybrid profile:\ngot:\n%s\nwant:\n%s", diskManifest.JSON(), wantManifest.JSON())
	}
}
