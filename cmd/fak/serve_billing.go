// serve_billing.go arms the `fak serve --billing` plane (PLAN-C Phase 3 item 10,
// ticket bm-p3-10-manifest): it loads the operator's billing manifest, the price
// book the manifest references, and opens the durable ledger and key store under
// the manifest's state dir — all before the listener binds. This file wires
// PRESENCE only: the loaded runtime is retained on the serve config for the
// gateway chain (item 12) to consume; no admission, proxy, or settle logic lives
// here.

package main

import (
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	billingkeystore "github.com/anthony-chaudhary/fak/pkg/billing/keystore"
	billingledger "github.com/anthony-chaudhary/fak/pkg/billing/ledger"
	billingmanifest "github.com/anthony-chaudhary/fak/pkg/billing/manifest"
	billingpricebook "github.com/anthony-chaudhary/fak/pkg/billing/pricebook"
)

// ErrServeBillingUnusable is the closed startup-refusal root for a --billing
// manifest that cannot be loaded or opened: every fault below wraps it, so
// errors.Is(err, ErrServeBillingUnusable) identifies the refusal class while
// the wrapped package sentinel (manifest.ErrInvalidManifest,
// pricebook.ErrInvalidPriceBook, ledger.ErrCorruptLog, keystore.ErrCorruptStore,
// or a plain filesystem fault) names the specific fault.
var ErrServeBillingUnusable = errors.New("billing: manifest unusable")

// billingRuntime holds the four pieces one --billing manifest resolves to at
// boot: the validated manifest, the validated price book it references, and
// the durable ledger and key store opened under the manifest's state dir.
// Item 12's gateway chain consumes these; until then they ride the serve
// config untouched.
type billingRuntime struct {
	Manifest     *billingmanifest.Manifest
	PriceBook    *billingpricebook.PriceBook
	Ledger       *billingledger.Ledger
	Keys         *billingkeystore.Store
	ManifestPath string
	StateDir     string
	LedgerPath   string
	KeysPath     string
}

// loadServeBilling resolves one --billing manifest end to end:
// manifest.LoadFile -> Validate -> pricebook.LoadFile(manifest.PricesPath) ->
// Validate -> ledger.Open(<state_dir>/ledger.jsonl) ->
// keystore.Open(<state_dir>/keys.json). It performs no admission, settlement,
// or routing: both Open calls create nothing for a missing file (an empty
// ledger / an empty store), so a fresh state dir boots clean and the first
// mutation writes the file.
func loadServeBilling(path string) (*billingRuntime, error) {
	m, err := billingmanifest.LoadFile(path)
	if err != nil {
		return nil, fmt.Errorf("%w: manifest %q: %w", ErrServeBillingUnusable, path, err)
	}
	if err := m.Validate(); err != nil {
		return nil, fmt.Errorf("%w: manifest %q: %w", ErrServeBillingUnusable, path, err)
	}
	pb, err := billingpricebook.LoadFile(m.PricesPath)
	if err != nil {
		return nil, fmt.Errorf("%w: price book %q: %w", ErrServeBillingUnusable, m.PricesPath, err)
	}
	if err := pb.Validate(); err != nil {
		return nil, fmt.Errorf("%w: price book %q: %w", ErrServeBillingUnusable, m.PricesPath, err)
	}
	ledgerPath := filepath.Join(m.StateDir, "ledger.jsonl")
	lg, err := billingledger.Open(ledgerPath)
	if err != nil {
		return nil, fmt.Errorf("%w: ledger %q: %w", ErrServeBillingUnusable, ledgerPath, err)
	}
	keysPath := filepath.Join(m.StateDir, "keys.json")
	ks, err := billingkeystore.Open(keysPath)
	if err != nil {
		return nil, fmt.Errorf("%w: key store %q: %w", ErrServeBillingUnusable, keysPath, err)
	}
	return &billingRuntime{
		Manifest:     m,
		PriceBook:    pb,
		Ledger:       lg,
		Keys:         ks,
		ManifestPath: path,
		StateDir:     m.StateDir,
		LedgerPath:   ledgerPath,
		KeysPath:     keysPath,
	}, nil
}

// serveBillingStartupBlock renders the neutral startup report for a loaded
// billing runtime: what was read and where, how many models and upstreams the
// manifest declares, the limits as configured (0 = uncapped), and the price
// book's version stamp. Mechanism reporting only — no rates, no vendor names,
// no policy claims.
func serveBillingStartupBlock(rt *billingRuntime) string {
	if rt == nil || rt.Manifest == nil || rt.PriceBook == nil {
		return ""
	}
	m := rt.Manifest
	upstreams := make([]string, len(m.Upstreams))
	for i, u := range m.Upstreams {
		upstreams[i] = u.Name
	}
	var b strings.Builder
	fmt.Fprintf(&b, "fak serve: billing armed:\n")
	fmt.Fprintf(&b, "  manifest   %s (%s)\n", m.Version, rt.ManifestPath)
	fmt.Fprintf(&b, "  state_dir  %s\n", m.StateDir)
	fmt.Fprintf(&b, "  models     %d\n", len(m.Models))
	fmt.Fprintf(&b, "  upstreams  %d: %s\n", len(m.Upstreams), strings.Join(upstreams, ", "))
	fmt.Fprintf(&b, "  limits     requests_per_minute=%d tokens_per_minute=%d daily_token_cap=%d session_token_cap=%d (0 = uncapped)\n",
		m.Limits.RequestsPerMinute, m.Limits.TokensPerMinute, m.Limits.DailyTokenCap, m.Limits.SessionTokenCap)
	fmt.Fprintf(&b, "  price_book sha256:%s\n", rt.PriceBook.VersionStamp())
	return b.String()
}

// printServeBillingStartup writes the neutral billing startup block to w.
func printServeBillingStartup(rt *billingRuntime, w io.Writer) {
	if block := serveBillingStartupBlock(rt); block != "" {
		fmt.Fprintln(w, block)
	}
}
