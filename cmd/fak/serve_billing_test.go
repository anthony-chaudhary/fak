// serve_billing_test.go — proves `fak serve --billing` loads and reports the
// billing plane (PLAN-C Phase 3 item 10, ticket bm-p3-10-manifest): the valid
// manifest case loads all four pieces and reports their state; each fault class
// refuses with the closed-vocabulary root sentinel naming the offending path;
// and the no-flag default path stays byte-for-byte unchanged.

package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	billingmanifest "github.com/anthony-chaudhary/fak/pkg/billing/manifest"
)

// serveBillingPriceBookFixture is one minimal valid v1 price book: a single
// card with two priced lines and two unpriced (null) ones.
func serveBillingPriceBookFixture() map[string]any {
	return map[string]any{
		"schema_version": 1,
		"models": []map[string]any{{
			"id":               "m1",
			"version":          1,
			"source_revision":  "rev-1",
			"source_time":      "2026-01-01T00:00:00Z",
			"currency":         "USD",
			"units":            "per_million_tokens",
			"fresh_input_rate": 1,
			"cache_read_rate":  nil,
			"cache_write_rate": nil,
			"output_rate":      2,
		}},
	}
}

// serveBillingManifestFixture is one minimal valid v1 manifest referencing the
// price book at pricesPath with state under stateDir.
func serveBillingManifestFixture(stateDir, pricesPath string) map[string]any {
	return map[string]any{
		"version":     billingmanifest.ManifestVersionV1,
		"state_dir":   stateDir,
		"prices_path": pricesPath,
		"models": []map[string]any{
			{"name": "served-a", "upstream": "alpha", "upstream_model": "up-a"},
			{"name": "served-b", "upstream": "beta", "upstream_model": "up-b"},
		},
		"upstreams": []map[string]any{
			{"name": "alpha", "base_url": "http://alpha.local:8000/v1"},
			{"name": "beta", "base_url": "https://beta.example.com/v1"},
		},
		"limits": map[string]any{
			"requests_per_minute": 0,
			"tokens_per_minute":   120000,
			"daily_token_cap":     0,
			"session_token_cap":   0,
		},
	}
}

func writeServeBillingFixtures(t *testing.T, dir string) string {
	t.Helper()
	pricesPath := filepath.Join(dir, "prices.json")
	writeServeBillingJSON(t, pricesPath, serveBillingPriceBookFixture())
	manifestPath := filepath.Join(dir, "billing.json")
	writeServeBillingJSON(t, manifestPath, serveBillingManifestFixture(filepath.Join(dir, "state"), pricesPath))
	return manifestPath
}

func writeServeBillingJSON(t *testing.T, path string, v any) {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal %s: %v", path, err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// TestServeBillingLoadsAllFourPiecesAndReportsState is the positive half: a
// valid manifest loads the manifest, its price book, and the durable ledger and
// key store under the manifest's state dir, and the startup block reports the
// loaded state neutrally.
func TestServeBillingLoadsAllFourPiecesAndReportsState(t *testing.T) {
	dir := t.TempDir()
	manifestPath := writeServeBillingFixtures(t, dir)

	rt, err := loadServeBilling(manifestPath)
	if err != nil {
		t.Fatalf("loadServeBilling(valid manifest) = %v, want success", err)
	}
	if rt.Manifest == nil || rt.PriceBook == nil || rt.Ledger == nil || rt.Keys == nil {
		t.Fatalf("loaded runtime must retain all four pieces, got %+v", rt)
	}
	if rt.Manifest.Version != billingmanifest.ManifestVersionV1 {
		t.Fatalf("manifest version = %q, want %q", rt.Manifest.Version, billingmanifest.ManifestVersionV1)
	}
	if want := filepath.Join(dir, "state"); rt.StateDir != want {
		t.Fatalf("state dir = %q, want %q", rt.StateDir, want)
	}
	if want := filepath.Join(dir, "state", "ledger.jsonl"); rt.LedgerPath != want {
		t.Fatalf("ledger path = %q, want %q", rt.LedgerPath, want)
	}
	if want := filepath.Join(dir, "state", "keys.json"); rt.KeysPath != want {
		t.Fatalf("key store path = %q, want %q", rt.KeysPath, want)
	}

	block := serveBillingStartupBlock(rt)
	for _, want := range []string{
		"fak serve: billing armed:",
		billingmanifest.ManifestVersionV1,
		"models     2",
		"upstreams  2: alpha, beta",
		"requests_per_minute=0",
		"tokens_per_minute=120000",
		"daily_token_cap=0",
		"session_token_cap=0",
		"(0 = uncapped)",
	} {
		if !strings.Contains(block, want) {
			t.Errorf("startup block missing %q:\n%s", want, block)
		}
	}
	wantStamp := rt.PriceBook.VersionStamp()
	if len(wantStamp) != 64 || !strings.Contains(block, "sha256:"+wantStamp) {
		t.Errorf("startup block must carry the price book's 64-hex version stamp:\n%s", block)
	}
}

// TestServeBillingMissingFileRefusesTyped pins the fail-fast half for an absent
// manifest: the refusal carries the closed-vocabulary root sentinel and names
// the offending path.
func TestServeBillingMissingFileRefusesTyped(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "absent.json")
	_, err := loadServeBilling(missing)
	if err == nil {
		t.Fatal("a missing manifest must refuse, not load")
	}
	if !errors.Is(err, ErrServeBillingUnusable) {
		t.Fatalf("refusal must wrap the closed-vocabulary root sentinel; got %v", err)
	}
	if !strings.Contains(err.Error(), missing) {
		t.Fatalf("refusal must name the offending path %q; got %v", missing, err)
	}
}

// TestServeBillingInvalidVersionRefusesTyped pins the typed refusal for a
// manifest whose version this package does not validate: the specific package
// sentinel must survive the wrap so operators can match it.
func TestServeBillingInvalidVersionRefusesTyped(t *testing.T) {
	dir := t.TempDir()
	manifestPath := writeServeBillingFixtures(t, dir)
	bad := map[string]any{
		"version":     "fak-billing-manifest/v9",
		"state_dir":   filepath.Join(dir, "state"),
		"prices_path": filepath.Join(dir, "prices.json"),
	}
	writeServeBillingJSON(t, manifestPath, bad)
	_, err := loadServeBilling(manifestPath)
	if err == nil {
		t.Fatal("a wrong-version manifest must refuse")
	}
	if !errors.Is(err, ErrServeBillingUnusable) {
		t.Fatalf("refusal must wrap the closed-vocabulary root sentinel; got %v", err)
	}
	if !errors.Is(err, billingmanifest.ErrUnknownVersion) {
		t.Fatalf("refusal must carry the specific package sentinel; got %v", err)
	}
}

// TestServeBillingMissingPriceBookRefusesTyped pins the typed refusal for a
// manifest whose prices_path points at a nonexistent file: the offending path
// named is the price book's, not the manifest's.
func TestServeBillingMissingPriceBookRefusesTyped(t *testing.T) {
	dir := t.TempDir()
	manifestPath := writeServeBillingFixtures(t, dir)
	ghost := filepath.Join(dir, "ghost-prices.json")
	manifest := serveBillingManifestFixture(filepath.Join(dir, "state"), ghost)
	writeServeBillingJSON(t, manifestPath, manifest)

	_, err := loadServeBilling(manifestPath)
	if err == nil {
		t.Fatal("a missing price book must refuse")
	}
	if !errors.Is(err, ErrServeBillingUnusable) {
		t.Fatalf("refusal must wrap the closed-vocabulary root sentinel; got %v", err)
	}
	if !strings.Contains(err.Error(), ghost) {
		t.Fatalf("refusal must name the offending price-book path %q; got %v", ghost, err)
	}
}

// TestServeBillingNoFlagLeavesDefaultPathUnchanged pins the opt-in guarantee:
// without --billing the flag stays empty, no billing runtime is ever resolved,
// and the flag set parses a typical argv exactly as before.
func TestServeBillingNoFlagLeavesDefaultPathUnchanged(t *testing.T) {
	fs, sf := newServeFlagSet()
	f := fs.Lookup("billing")
	if f == nil {
		t.Fatal("--billing must be registered on the serve flag set")
	}
	if f.DefValue != "" {
		t.Fatalf("--billing default = %q, want empty (no billing = today's behavior)", f.DefValue)
	}
	if !parseFlags(fs, []string{"--model", "mock", "--mock"}) {
		t.Fatal("parsing a --billing-free argv must succeed")
	}
	if *sf.billing != "" {
		t.Fatalf("no --billing must leave the flag empty, got %q", *sf.billing)
	}
	if sf.billingRuntime != nil {
		t.Fatalf("no --billing must never resolve a billing runtime, got %+v", sf.billingRuntime)
	}
}

// TestServeSourceWiresBillingLoadIntoRunPath is the drift guard on the link
// itself: the behavioral tests above call loadServeBilling directly, so they
// would still pass if cmdServe quietly stopped calling it — the exact
// regression that made --key-principal parse-only (#5332). This one reads the
// real serve.go and requires the chain: flag registered, the load called in
// the run path, and the loaded runtime retained on the serve config.
func TestServeSourceWiresBillingLoadIntoRunPath(t *testing.T) {
	root := repoRootFromTest(t)
	body, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "serve.go"))
	if err != nil {
		t.Fatalf("read serve.go: %v", err)
	}
	src := string(body)
	for _, want := range []string{
		`fs.String("billing", "", `,
		"loadServeBilling(*sf.billing)",
		"sf.billingRuntime = billingRT",
		"printServeBillingStartup(",
	} {
		if !strings.Contains(src, want) {
			t.Fatalf("serve.go must contain %q — --billing is parse-only without it", want)
		}
	}
}
