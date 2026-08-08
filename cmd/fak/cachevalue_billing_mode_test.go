package main

import (
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// setBillingModeForTest installs a seat for one test and restores the previous one, so a
// process-global stamp cannot leak between tests in this package.
func setBillingModeForTest(t *testing.T, mode string) {
	t.Helper()
	prev := resolvedBillingMode()
	t.Cleanup(func() { recordBillingMode(prev) })
	recordBillingMode(mode)
}

// TestBillingModeFromCredentialClass pins the mapping #3664 depends on: only a resolved api
// key with NO subscription token pinned counts as per-token billing. Everything fak cannot
// prove — subscription OAuth, plain passthrough, the local in-kernel branch — must classify
// away from real-dollar, because the fleet report counts that column as real money.
func TestBillingModeFromCredentialClass(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   guardManagedCacheInputs
		want string
	}{
		{"api key billing", guardManagedCacheInputs{provider: "anthropic", apiKey: "sk-live"}, cachevaluereport.BillingModeAPIKey},
		{"keychain api key is still api billing", guardManagedCacheInputs{provider: "anthropic", apiKey: "sk-live", keychainAPIKey: true}, cachevaluereport.BillingModeAPIKey},
		{"subscription oauth is flat rate", guardManagedCacheInputs{provider: "anthropic", apiKey: "oauth-token", oauthSource: "claude-cli"}, cachevaluereport.BillingModeOAuth},
		{"passthrough credential is unprovable", guardManagedCacheInputs{provider: "anthropic"}, cachevaluereport.BillingModeUnknown},
		{"blank credential is unprovable", guardManagedCacheInputs{provider: "anthropic", apiKey: "   "}, cachevaluereport.BillingModeUnknown},
		{"local in-kernel model bills nothing", guardManagedCacheInputs{localModel: true}, cachevaluereport.BillingModeUnknown},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := billingModeFrom(tc.in); got != tc.want {
				t.Fatalf("billingModeFrom(%+v) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// TestBillingModeIndependentOfForcedManagedCache is the overstatement fence: `--managed-cache
// on` forces the posture ACTIVE even on a credential whose billing fak cannot see, and the
// seat stamp must NOT follow that flag. Deriving the ledger's real-$ column from the active
// bit would let an operator flag turn unprovable spend into "real dollars".
func TestBillingModeIndependentOfForcedManagedCache(t *testing.T) {
	in := guardManagedCacheInputs{provider: "anthropic"} // passthrough: billing unknown
	posture, err := resolveGuardManagedCache(guardManagedCacheOn, in)
	if err != nil {
		t.Fatalf("resolve managed cache: %v", err)
	}
	if !posture.active {
		t.Fatalf("precondition: --managed-cache on should force ACTIVE here, got %+v", posture)
	}
	if got := billingModeFrom(in); got != cachevaluereport.BillingModeUnknown {
		t.Fatalf("forced-active passthrough classified %q, want %q — an operator flag must not mint real dollars",
			got, cachevaluereport.BillingModeUnknown)
	}
}

// TestAppendObservedCacheSavingsStampsResolvedSeat is the end-to-end write witness: the seat
// this process resolved lands on every durable Track-2 row, so the fleet fold can split the
// reduction it prints.
func TestAppendObservedCacheSavingsStampsResolvedSeat(t *testing.T) {
	clearCachevaluePriceEnv(t)
	setBillingModeForTest(t, billingModeFrom(guardManagedCacheInputs{provider: "anthropic", apiKey: "sk-live"}))

	path := filepath.Join(t.TempDir(), "cache-savings.jsonl")
	sum := gateway.AdjudicationSummary{
		InputTokens: 20, CachedPromptTokens: 60, CacheCreationTokens: 20,
		OutputTokens: 3, CompactionShedTokens: 9,
	}
	if res := appendObservedCacheSavingsTo(path, "guard", "anthropic", "claude", sum, time.Now()); res.Err != nil {
		t.Fatalf("append savings: %v", res.Err)
	}
	rows := cachevaluereport.ReadSavingsLedgerFile(path)
	if len(rows) == 0 {
		t.Fatal("no rows written")
	}
	for _, row := range rows {
		if row.BillingMode != cachevaluereport.BillingModeAPIKey {
			t.Fatalf("row %q billing mode = %q, want %q", row.Mechanism, row.BillingMode, cachevaluereport.BillingModeAPIKey)
		}
		if !cachevaluereport.RealDollarBillingMode(row.BillingMode) {
			t.Fatalf("row %q should fold into the real-$ column: %+v", row.Mechanism, row)
		}
	}
}

// TestAppendObservedCacheSavingsUnresolvedSeatStaysBlank keeps the back-compat direction: a
// front door that never classified its credential writes rows with no seat at all, which the
// fleet fold reads as notional — the same as every row written before #3664.
func TestAppendObservedCacheSavingsUnresolvedSeatStaysBlank(t *testing.T) {
	clearCachevaluePriceEnv(t)
	setBillingModeForTest(t, "")

	path := filepath.Join(t.TempDir(), "cache-savings.jsonl")
	sum := gateway.AdjudicationSummary{InputTokens: 20, CachedPromptTokens: 60, CacheCreationTokens: 20, OutputTokens: 3}
	if res := appendObservedCacheSavingsTo(path, "serve", "anthropic", "claude", sum, time.Now()); res.Err != nil {
		t.Fatalf("append savings: %v", res.Err)
	}
	rows := cachevaluereport.ReadSavingsLedgerFile(path)
	if len(rows) == 0 {
		t.Fatal("no rows written")
	}
	for _, row := range rows {
		if row.BillingMode != "" {
			t.Fatalf("unresolved seat stamped %q, want blank", row.BillingMode)
		}
		if cachevaluereport.RealDollarBillingMode(row.BillingMode) {
			t.Fatalf("an unstamped row must never fold real-$: %+v", row)
		}
	}
}

// TestGuardRecordsBillingModeFromResolvedCredential pins the WIRE, not just the helper: the
// stamp is only honest if cmdGuard actually publishes it off the same inputs struct it hands
// resolveGuardManagedCache. Detection without enforcement is the failure mode this catches —
// billingModeFrom can be perfect while no live session ever calls it.
func TestGuardRecordsBillingModeFromResolvedCredential(t *testing.T) {
	src := readEntrypoint(t, "guard.go")
	if !strings.Contains(src, "recordBillingMode(billingModeFrom(mcIn))") {
		t.Fatal("guard.go must publish the resolved seat via recordBillingMode(billingModeFrom(mcIn))")
	}
	if !strings.Contains(src, "resolveGuardManagedCache(*managedCacheMode, mcIn)") {
		t.Fatal("the seat must be derived from the SAME inputs struct resolveGuardManagedCache resolved")
	}
	if !strings.Contains(readEntrypoint(t, "cachevalue_savings.go"), "BillingMode: resolvedBillingMode()") {
		t.Fatal("cachevalue_savings.go must stamp the resolved seat onto the Track-2 observation")
	}
}
