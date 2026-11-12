package main

// `fak keys` is the operator surface for the billing key registry and the
// neutral unit ledger (PLAN-C Phase 3 item 11, ticket bm-p3-11-keyscli). It
// rides pkg/billing/keystore (hash-at-rest tenant keys) and
// pkg/billing/ledger (append-only integer-unit balances):
//
//	fak keys mint     --manifest M --tenant T [--store P]
//	fak keys topup    --manifest M --tenant T --units N [--ledger P]
//	fak keys balance  --manifest M --tenant T [--ledger P]
//	fak keys revoke   --manifest M --key KEY-OR-PREFIX [--store P]
//	fak keys list     --manifest M [--tenant T] [--store P]
//
// --manifest is the primary location source: its state_dir locates the key
// store (<state_dir>/keys.json) and the ledger (<state_dir>/ledger.jsonl);
// --store / --ledger override the derived paths for direct use. Units are
// integer ledger units: no rates, currency, or vendor names on this surface.
//
// Exit codes: 0 success; 1 data/runtime refusal (bad or missing manifest,
// unknown or ambiguous key, already-revoked key, ledger fault); 2 usage
// fault (missing subcommand or required flag, non-positive --units).

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/billing/keystore"
	"github.com/anthony-chaudhary/fak/pkg/billing/ledger"
	"github.com/anthony-chaudhary/fak/pkg/billing/manifest"
)

const (
	keysStoreFileName  = "keys.json"
	keysLedgerFileName = "ledger.jsonl"
)

func cmdKeys(argv []string) {
	os.Exit(runKeys(os.Stdout, os.Stderr, argv))
}

func runKeys(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		keysUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "mint":
		return runKeysMint(stdout, stderr, argv[1:])
	case "topup":
		return runKeysTopup(stdout, stderr, argv[1:])
	case "balance":
		return runKeysBalance(stdout, stderr, argv[1:])
	case "revoke":
		return runKeysRevoke(stdout, stderr, argv[1:])
	case "list":
		return runKeysList(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		keysUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak keys: unknown subcommand %q\n", argv[0])
		keysUsage(stderr)
		return 2
	}
}

func keysUsage(w io.Writer) {
	fmt.Fprint(w, `usage: fak keys <subcommand> [flags]

The billing key registry and unit-ledger operator surface. --manifest is the
primary location source: its state_dir locates the key store
(<state_dir>/keys.json) and the ledger (<state_dir>/ledger.jsonl); --store /
--ledger override the derived paths for direct use.

  fak keys mint --manifest M --tenant T
      mint a key for the tenant; prints the full key exactly once (it cannot
      be recovered afterwards) plus its display prefix.

  fak keys topup --manifest M --tenant T --units N
      deposit N positive ledger units to the tenant and print the resulting
      balance in ledger units.

  fak keys balance --manifest M --tenant T
      print the tenant's available balance in ledger units (0 when the
      tenant has no recorded entries).

  fak keys revoke --manifest M --key KEY-OR-PREFIX
      durably revoke the stored key; refuses with ErrUnknownKey,
      ErrAmbiguousKey, or ErrKeyRevoked (exit 1).

  fak keys list --manifest M [--tenant T]
      list stored keys: prefix, tenant, created/revoked timestamps, state.
      Never prints hashes or plaintext.

Units are integer ledger units.
`)
}

// keysRegisterLocationFlags registers the shared --manifest/--store/--ledger
// flags. The manifest is the primary location source; the explicit paths
// override the derived ones for direct use.
func keysRegisterLocationFlags(fs *flag.FlagSet) (manifestPath, storePath, ledgerPath *string) {
	manifestPath = fs.String("manifest", "", "billing manifest path (its state_dir locates the key store and ledger)")
	storePath = fs.String("store", "", "key store path (overrides the manifest-derived <state_dir>/keys.json)")
	ledgerPath = fs.String("ledger", "", "ledger path (overrides the manifest-derived <state_dir>/ledger.jsonl)")
	return manifestPath, storePath, ledgerPath
}

// keysLoadManifest loads path and wraps file-level faults under the manifest
// sentinel, so every bad-or-missing manifest reads as ErrInvalidManifest.
func keysLoadManifest(path string) (*manifest.Manifest, error) {
	m, err := manifest.LoadFile(path)
	if err != nil {
		if errors.Is(err, manifest.ErrInvalidManifest) {
			return nil, err
		}
		return nil, fmt.Errorf("%w: %s: %v", manifest.ErrInvalidManifest, path, err)
	}
	return m, nil
}

// keysManifestStateDir loads the manifest at manifestPath and ensures its
// state directory exists, returning it. A bad or missing manifest is a typed
// exit-1 refusal naming the manifest sentinel.
func keysManifestStateDir(manifestPath string, stderr io.Writer, verb string) (string, int) {
	m, err := keysLoadManifest(manifestPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak keys %s: %v\n", verb, err)
		return "", 1
	}
	if err := os.MkdirAll(m.StateDir, 0o755); err != nil {
		fmt.Fprintf(stderr, "fak keys %s: create state dir %s: %v\n", verb, m.StateDir, err)
		return "", 1
	}
	return m.StateDir, 0
}

// keysStorePath resolves the key-store location: an explicit --store wins,
// else the manifest's state_dir, else a usage refusal (exit 2).
func keysStorePath(manifestPath, storeOverride string, stderr io.Writer, verb string) (string, int) {
	if storeOverride != "" {
		return storeOverride, 0
	}
	if manifestPath != "" {
		dir, rc := keysManifestStateDir(manifestPath, stderr, verb)
		if rc != 0 {
			return "", rc
		}
		return filepath.Join(dir, keysStoreFileName), 0
	}
	fmt.Fprintf(stderr, "fak keys %s: --manifest or --store is required\n", verb)
	keysUsage(stderr)
	return "", 2
}

// keysLedgerPath resolves the ledger location: an explicit --ledger wins,
// else the manifest's state_dir, else a usage refusal (exit 2).
func keysLedgerPath(manifestPath, ledgerOverride string, stderr io.Writer, verb string) (string, int) {
	if ledgerOverride != "" {
		return ledgerOverride, 0
	}
	if manifestPath != "" {
		dir, rc := keysManifestStateDir(manifestPath, stderr, verb)
		if rc != 0 {
			return "", rc
		}
		return filepath.Join(dir, keysLedgerFileName), 0
	}
	fmt.Fprintf(stderr, "fak keys %s: --manifest or --ledger is required\n", verb)
	keysUsage(stderr)
	return "", 2
}

// keysRandomEventID mints a fresh dedup key for a ledger deposit: 16 bytes of
// crypto/rand as lowercase hex, tagged for readability.
func keysRandomEventID() (string, error) {
	raw := make([]byte, 16)
	if _, err := rand.Read(raw); err != nil {
		return "", fmt.Errorf("fak keys: generate event id: %w", err)
	}
	return "evt-" + hex.EncodeToString(raw), nil
}

// keysAvailableBalance returns the tenant's available balance, reporting 0
// for a tenant with no recorded entries (a fresh account) rather than an
// error.
func keysAvailableBalance(lg *ledger.Ledger, tenantID string) (int64, error) {
	acct, err := lg.GetAccount(tenantID)
	if err != nil {
		if errors.Is(err, ledger.ErrAccountNotFound) {
			return 0, nil
		}
		return 0, err
	}
	return acct.Available, nil
}

// --- mint -------------------------------------------------------------------

func runKeysMint(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak keys mint", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifestPath, storePath, _ := keysRegisterLocationFlags(fs)
	tenant := fs.String("tenant", "", "tenant id to mint the key for")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	tenantID := strings.TrimSpace(*tenant)
	if tenantID == "" {
		fmt.Fprintln(stderr, "fak keys mint: --tenant is required")
		return 2
	}
	path, rc := keysStorePath(*manifestPath, *storePath, stderr, "mint")
	if rc != 0 {
		return rc
	}
	store, err := keystore.Open(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak keys mint: open store: %v\n", err)
		return 1
	}
	key, rec, err := store.Mint(tenantID)
	if err != nil {
		fmt.Fprintf(stderr, "fak keys mint: %v\n", err)
		return 1
	}
	// The full key is delivered exactly once, here, on stdout only.
	fmt.Fprintln(stdout, "WARNING: the full key is shown exactly once and cannot be recovered afterwards.")
	fmt.Fprintf(stdout, "key: %s\n", key)
	fmt.Fprintf(stdout, "prefix: %s\n", rec.Prefix)
	fmt.Fprintf(stdout, "tenant: %s\n", tenantID)
	return 0
}

// --- topup ------------------------------------------------------------------

func runKeysTopup(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak keys topup", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifestPath, _, ledgerPath := keysRegisterLocationFlags(fs)
	tenant := fs.String("tenant", "", "tenant id to credit")
	units := fs.Int64("units", 0, "positive ledger units to deposit")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	tenantID := strings.TrimSpace(*tenant)
	if tenantID == "" {
		fmt.Fprintln(stderr, "fak keys topup: --tenant is required")
		return 2
	}
	if *units <= 0 {
		fmt.Fprintf(stderr, "fak keys topup: --units must be a positive integer (got %d)\n", *units)
		return 2
	}
	path, rc := keysLedgerPath(*manifestPath, *ledgerPath, stderr, "topup")
	if rc != 0 {
		return rc
	}
	lg, err := ledger.Open(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak keys topup: open ledger: %v\n", err)
		return 1
	}
	eventID, err := keysRandomEventID()
	if err != nil {
		fmt.Fprintf(stderr, "fak keys topup: %v\n", err)
		return 1
	}
	if err := lg.Deposit(eventID, tenantID, *units); err != nil {
		fmt.Fprintf(stderr, "fak keys topup: %v\n", err)
		return 1
	}
	balance, err := keysAvailableBalance(lg, tenantID)
	if err != nil {
		fmt.Fprintf(stderr, "fak keys topup: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "fak keys topup: tenant=%s units=%d event=%s\n", tenantID, *units, eventID)
	fmt.Fprintf(stdout, "balance: %d ledger units\n", balance)
	return 0
}

// --- balance ----------------------------------------------------------------

func runKeysBalance(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak keys balance", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifestPath, _, ledgerPath := keysRegisterLocationFlags(fs)
	tenant := fs.String("tenant", "", "tenant id to read")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	tenantID := strings.TrimSpace(*tenant)
	if tenantID == "" {
		fmt.Fprintln(stderr, "fak keys balance: --tenant is required")
		return 2
	}
	path, rc := keysLedgerPath(*manifestPath, *ledgerPath, stderr, "balance")
	if rc != 0 {
		return rc
	}
	lg, err := ledger.Open(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak keys balance: open ledger: %v\n", err)
		return 1
	}
	balance, err := keysAvailableBalance(lg, tenantID)
	if err != nil {
		fmt.Fprintf(stderr, "fak keys balance: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "fak keys balance: tenant=%s\n", tenantID)
	fmt.Fprintf(stdout, "balance: %d ledger units\n", balance)
	return 0
}

// --- revoke -----------------------------------------------------------------

func runKeysRevoke(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak keys revoke", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifestPath, storePath, _ := keysRegisterLocationFlags(fs)
	key := fs.String("key", "", "full key or display prefix to revoke")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if strings.TrimSpace(*key) == "" {
		fmt.Fprintln(stderr, "fak keys revoke: --key is required")
		return 2
	}
	path, rc := keysStorePath(*manifestPath, *storePath, stderr, "revoke")
	if rc != 0 {
		return rc
	}
	store, err := keystore.Open(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak keys revoke: open store: %v\n", err)
		return 1
	}
	if err := store.Revoke(*key); err != nil {
		// Closed vocabulary: name the sentinel, no prose sprawl.
		switch {
		case errors.Is(err, keystore.ErrUnknownKey):
			fmt.Fprintln(stderr, "fak keys revoke: ErrUnknownKey")
		case errors.Is(err, keystore.ErrAmbiguousKey):
			fmt.Fprintln(stderr, "fak keys revoke: ErrAmbiguousKey")
		case errors.Is(err, keystore.ErrKeyRevoked):
			fmt.Fprintln(stderr, "fak keys revoke: ErrKeyRevoked")
		default:
			fmt.Fprintf(stderr, "fak keys revoke: %v\n", err)
		}
		return 1
	}
	fmt.Fprintln(stdout, "fak keys revoke: revoked")
	return 0
}

// --- list -------------------------------------------------------------------

func runKeysList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak keys list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	manifestPath, storePath, _ := keysRegisterLocationFlags(fs)
	tenant := fs.String("tenant", "", "filter to one tenant id (default: all tenants)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	path, rc := keysStorePath(*manifestPath, *storePath, stderr, "list")
	if rc != 0 {
		return rc
	}
	store, err := keystore.Open(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak keys list: open store: %v\n", err)
		return 1
	}
	var records []keystore.KeyRecord
	if strings.TrimSpace(*tenant) != "" {
		records = store.TenantKeys(strings.TrimSpace(*tenant))
	} else {
		records = store.List()
	}
	fmt.Fprintf(stdout, "fak keys list: %d key(s)\n", len(records))
	if len(records) == 0 {
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "prefix\ttenant\tcreated\trevoked\tstate")
	for i := range records {
		r := &records[i]
		state := "active"
		revoked := "-"
		if r.RevokedAt != nil {
			state = "revoked"
			revoked = r.RevokedAt.UTC().Format(time.RFC3339)
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\n",
			r.Prefix, r.TenantID, r.CreatedAt.UTC().Format(time.RFC3339), revoked, state)
	}
	tw.Flush()
	return 0
}
