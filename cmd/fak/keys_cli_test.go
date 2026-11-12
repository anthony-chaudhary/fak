package main

// Verb-level tests for `fak keys` (PLAN-C Phase 3 item 11, ticket
// bm-p3-11-keyscli): mint/topup/balance/revoke/list over real temp-dir files,
// through the same runKeys entry point the neighboring cmd/fak verb tests use.

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

var keysFullKeyRE = regexp.MustCompile(`sk-fak-[0-9a-f]{64}`)

// keysWriteManifest writes a minimal valid v1 manifest whose state_dir sits
// under dir, returning its path and the state dir.
func keysWriteManifest(t *testing.T, dir string) (string, string) {
	t.Helper()
	stateDir := filepath.Join(dir, "state")
	m := map[string]any{
		"version":     "fak-billing-manifest/v1",
		"state_dir":   stateDir,
		"prices_path": filepath.Join(dir, "prices.json"),
	}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, "manifest.json")
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p, stateDir
}

// keysMint runs `fak keys mint` and returns the full key from stdout.
func keysMint(t *testing.T, manifestPath, tenant string) string {
	t.Helper()
	var out, errb bytes.Buffer
	rc := runKeys(&out, &errb, []string{"mint", "--manifest", manifestPath, "--tenant", tenant})
	if rc != 0 {
		t.Fatalf("mint rc=%d stderr=%s", rc, errb.String())
	}
	key := keysFullKeyRE.FindString(out.String())
	if key == "" {
		t.Fatalf("no full key in mint output:\n%s", out.String())
	}
	return key
}

// TestRunKeysMintThenBalanceRoundTrip pins the mint-then-balance round trip
// through REAL temp-dir files: the full key is printed exactly once with the
// unrecoverable warning and display prefix, the store file holds no
// plaintext, and a fresh tenant reads a numeric zero balance.
func TestRunKeysMintThenBalanceRoundTrip(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	manifestPath, stateDir := keysWriteManifest(t, dir)

	var out, errb bytes.Buffer
	rc := runKeys(&out, &errb, []string{"mint", "--manifest", manifestPath, "--tenant", "t1"})
	if rc != 0 {
		t.Fatalf("mint rc=%d stderr=%s", rc, errb.String())
	}
	stdout := out.String()
	key := keysFullKeyRE.FindString(stdout)
	if key == "" {
		t.Fatalf("no full key in mint output:\n%s", stdout)
	}
	if strings.Count(stdout, key) != 1 {
		t.Fatalf("full key must appear exactly once on stdout, got %d:\n%s", strings.Count(stdout, key), stdout)
	}
	if !strings.Contains(stdout, "cannot be recovered") {
		t.Fatalf("mint output missing the unrecoverable-key warning:\n%s", stdout)
	}
	if !strings.Contains(stdout, "prefix: "+key[:12]) {
		t.Fatalf("mint output missing the display prefix:\n%s", stdout)
	}
	// Hash-at-rest: the durable store file carries the hash + prefix, never the plaintext.
	raw, err := os.ReadFile(filepath.Join(stateDir, "keys.json"))
	if err != nil {
		t.Fatalf("store file not created at <state_dir>/keys.json: %v", err)
	}
	if bytes.Contains(raw, []byte(key)) {
		t.Fatalf("store file leaked the full key plaintext:\n%s", raw)
	}

	out.Reset()
	errb.Reset()
	rc = runKeys(&out, &errb, []string{"balance", "--manifest", manifestPath, "--tenant", "t1"})
	if rc != 0 {
		t.Fatalf("balance rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "balance: 0 ledger units") {
		t.Fatalf("fresh tenant should read a numeric zero balance:\n%s", out.String())
	}
}

// TestRunKeysTopupIncrementsBalance pins topup deposits accumulating across
// invocations (each replaying the durable log) and the balance verb reading
// the persisted result.
func TestRunKeysTopupIncrementsBalance(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	manifestPath, stateDir := keysWriteManifest(t, dir)

	var out, errb bytes.Buffer
	rc := runKeys(&out, &errb, []string{"topup", "--manifest", manifestPath, "--tenant", "t1", "--units", "500"})
	if rc != 0 {
		t.Fatalf("topup rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "balance: 500 ledger units") {
		t.Fatalf("first topup should print balance 500:\n%s", out.String())
	}

	out.Reset()
	errb.Reset()
	rc = runKeys(&out, &errb, []string{"topup", "--manifest", manifestPath, "--tenant", "t1", "--units", "250"})
	if rc != 0 {
		t.Fatalf("second topup rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "balance: 750 ledger units") {
		t.Fatalf("second topup should print balance 750:\n%s", out.String())
	}

	// Durable: a fresh process reopens the log and reads the accumulated balance.
	out.Reset()
	errb.Reset()
	rc = runKeys(&out, &errb, []string{"balance", "--manifest", manifestPath, "--tenant", "t1"})
	if rc != 0 {
		t.Fatalf("balance rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "balance: 750 ledger units") {
		t.Fatalf("replayed balance should be 750:\n%s", out.String())
	}

	logRaw, err := os.ReadFile(filepath.Join(stateDir, "ledger.jsonl"))
	if err != nil {
		t.Fatalf("ledger file not created at <state_dir>/ledger.jsonl: %v", err)
	}
	lines := 0
	for _, l := range strings.Split(string(logRaw), "\n") {
		if strings.TrimSpace(l) != "" {
			lines++
		}
	}
	if lines != 2 {
		t.Fatalf("ledger should hold exactly 2 deposit entries, got %d:\n%s", lines, logRaw)
	}
}

// TestRunKeysRevokeThenListShowsRevoked pins revoke durability and the list
// surface: the revoked row shows state=revoked with its timestamp, and list
// never prints the full key or any stored hash.
func TestRunKeysRevokeThenListShowsRevoked(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	manifestPath, _ := keysWriteManifest(t, dir)

	key1 := keysMint(t, manifestPath, "t1")
	key2 := keysMint(t, manifestPath, "t2")

	var out, errb bytes.Buffer
	rc := runKeys(&out, &errb, []string{"revoke", "--manifest", manifestPath, "--key", key1})
	if rc != 0 {
		t.Fatalf("revoke rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "revoked") {
		t.Fatalf("revoke success output missing 'revoked':\n%s", out.String())
	}

	out.Reset()
	errb.Reset()
	rc = runKeys(&out, &errb, []string{"list", "--manifest", manifestPath})
	if rc != 0 {
		t.Fatalf("list rc=%d stderr=%s", rc, errb.String())
	}
	listOut := out.String()
	if !strings.Contains(listOut, "2 key(s)") {
		t.Fatalf("list should show both keys:\n%s", listOut)
	}
	if !strings.Contains(listOut, key1[:12]) || !strings.Contains(listOut, key2[:12]) {
		t.Fatalf("list should show both prefixes:\n%s", listOut)
	}
	if !strings.Contains(listOut, "revoked") {
		t.Fatalf("list should mark the revoked key:\n%s", listOut)
	}
	if strings.Contains(listOut, key1) || strings.Contains(listOut, key2) {
		t.Fatalf("list must never print a full key:\n%s", listOut)
	}
	for _, k := range []string{key1, key2} {
		sum := sha256.Sum256([]byte(k))
		if strings.Contains(listOut, hex.EncodeToString(sum[:])) {
			t.Fatalf("list must never print a stored hash:\n%s", listOut)
		}
	}

	// Re-revoking the same key is a typed ErrKeyRevoked refusal.
	out.Reset()
	errb.Reset()
	rc = runKeys(&out, &errb, []string{"revoke", "--manifest", manifestPath, "--key", key1})
	if rc != 1 || !strings.Contains(errb.String(), "ErrKeyRevoked") {
		t.Fatalf("re-revoke: rc=%d stderr=%s (want rc=1 + ErrKeyRevoked)", rc, errb.String())
	}
}

// TestRunKeysRevokeUnknownKeyRefusesTyped pins the unknown-key refusal: exit
// 1 with the sentinel name in the message, closed vocabulary.
func TestRunKeysRevokeUnknownKeyRefusesTyped(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	manifestPath, _ := keysWriteManifest(t, dir)

	var out, errb bytes.Buffer
	rc := runKeys(&out, &errb, []string{"revoke", "--manifest", manifestPath, "--key", "sk-fak-" + strings.Repeat("0", 64)})
	if rc != 1 || !strings.Contains(errb.String(), "ErrUnknownKey") {
		t.Fatalf("unknown key: rc=%d stderr=%s (want rc=1 + ErrUnknownKey)", rc, errb.String())
	}
}

// TestRunKeysRevokeAmbiguousPrefixRefusesTyped pins the ambiguous-prefix
// refusal: two stored records sharing one display prefix make a prefix revoke
// refuse with ErrAmbiguousKey (exit 1).
func TestRunKeysRevokeAmbiguousPrefixRefusesTyped(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	manifestPath, stateDir := keysWriteManifest(t, dir)
	if err := os.MkdirAll(stateDir, 0o755); err != nil {
		t.Fatal(err)
	}
	const prefix = "sk-fak-abcde" // 12 chars: "sk-fak-" marker + 5 lowercase hex
	store := `{
  "version": "fak-billing-keystore/v1",
  "keys": [
    {"prefix":"sk-fak-abcde","hash":"` + strings.Repeat("a", 64) + `","tenant_id":"t1","created_at":"2026-11-12T00:00:00Z"},
    {"prefix":"sk-fak-abcde","hash":"` + strings.Repeat("b", 64) + `","tenant_id":"t2","created_at":"2026-11-12T00:00:00Z"}
  ]
}`
	if err := os.WriteFile(filepath.Join(stateDir, "keys.json"), []byte(store), 0o600); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	rc := runKeys(&out, &errb, []string{"revoke", "--manifest", manifestPath, "--key", prefix})
	if rc != 1 || !strings.Contains(errb.String(), "ErrAmbiguousKey") {
		t.Fatalf("ambiguous prefix: rc=%d stderr=%s (want rc=1 + ErrAmbiguousKey)", rc, errb.String())
	}
}

// TestRunKeysMissingManifestRefusesTyped pins the bad-or-missing manifest
// refusal: exit 1 naming the manifest sentinel, for both an absent file and a
// malformed payload.
func TestRunKeysMissingManifestRefusesTyped(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	missing := filepath.Join(dir, "nope.json")

	var out, errb bytes.Buffer
	rc := runKeys(&out, &errb, []string{"list", "--manifest", missing})
	if rc != 1 || !strings.Contains(errb.String(), "manifest: invalid billing manifest") {
		t.Fatalf("missing manifest: rc=%d stderr=%s (want rc=1 + manifest sentinel)", rc, errb.String())
	}

	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errb.Reset()
	rc = runKeys(&out, &errb, []string{"list", "--manifest", bad})
	if rc != 1 || !strings.Contains(errb.String(), "manifest: invalid billing manifest") {
		t.Fatalf("malformed manifest: rc=%d stderr=%s (want rc=1 + manifest sentinel)", rc, errb.String())
	}
}

// TestRunKeysTopupUnitsRefusals pins the non-positive --units refusal: exit 2
// with a typed message, before any ledger write happens.
func TestRunKeysTopupUnitsRefusals(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	manifestPath, stateDir := keysWriteManifest(t, dir)

	for _, u := range []string{"0", "-5"} {
		var out, errb bytes.Buffer
		rc := runKeys(&out, &errb, []string{"topup", "--manifest", manifestPath, "--tenant", "t1", "--units", u})
		if rc != 2 || !strings.Contains(errb.String(), "--units must be a positive integer") {
			t.Fatalf("--units %s: rc=%d stderr=%s (want rc=2 + typed refusal)", u, rc, errb.String())
		}
	}
	if _, err := os.Stat(filepath.Join(stateDir, "ledger.jsonl")); !os.IsNotExist(err) {
		t.Fatalf("refused topups must not create the ledger file")
	}
}

// TestRunKeysUsageRefusals pins the usage-class refusals: missing subcommand,
// unknown subcommand, missing required flags, and stray positionals all exit
// 2 with a usage line.
func TestRunKeysUsageRefusals(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	manifestPath, _ := keysWriteManifest(t, dir)

	cases := []struct {
		name    string
		argv    []string
		wantErr string
	}{
		{"no subcommand", nil, "usage: fak keys"},
		{"unknown subcommand", []string{"frobnicate"}, "unknown subcommand"},
		{"mint without tenant", []string{"mint", "--manifest", manifestPath}, "--tenant is required"},
		{"mint without location", []string{"mint", "--tenant", "t1"}, "--manifest or --store is required"},
		{"topup without location", []string{"topup", "--tenant", "t1", "--units", "5"}, "--manifest or --ledger is required"},
		{"balance without tenant", []string{"balance", "--manifest", manifestPath}, "--tenant is required"},
		{"revoke without key", []string{"revoke", "--manifest", manifestPath}, "--key is required"},
		{"stray positional", []string{"list", "--manifest", manifestPath, "extra"}, "unexpected argument"},
	}
	for _, tc := range cases {
		var out, errb bytes.Buffer
		rc := runKeys(&out, &errb, tc.argv)
		if rc != 2 {
			t.Fatalf("%s: rc=%d want 2; stderr=%s", tc.name, rc, errb.String())
		}
		if !strings.Contains(errb.String(), tc.wantErr) {
			t.Fatalf("%s: stderr missing %q:\n%s", tc.name, tc.wantErr, errb.String())
		}
	}
}

// TestRunKeysExplicitStoreAndLedgerOverrides pins direct use without a
// manifest: --store / --ledger locate the state files verbatim.
func TestRunKeysExplicitStoreAndLedgerOverrides(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	storePath := filepath.Join(dir, "custom-keys.json")
	ledgerPath := filepath.Join(dir, "custom-ledger.jsonl")

	key := keysMintWithStore(t, storePath, "t1")

	var out, errb bytes.Buffer
	rc := runKeys(&out, &errb, []string{"topup", "--ledger", ledgerPath, "--tenant", "t1", "--units", "7"})
	if rc != 0 {
		t.Fatalf("topup rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "balance: 7 ledger units") {
		t.Fatalf("direct-use topup should print balance 7:\n%s", out.String())
	}

	out.Reset()
	errb.Reset()
	rc = runKeys(&out, &errb, []string{"balance", "--ledger", ledgerPath, "--tenant", "t1"})
	if rc != 0 {
		t.Fatalf("balance rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "balance: 7 ledger units") {
		t.Fatalf("direct-use balance should print 7:\n%s", out.String())
	}

	out.Reset()
	errb.Reset()
	rc = runKeys(&out, &errb, []string{"list", "--store", storePath, "--tenant", "t1"})
	if rc != 0 {
		t.Fatalf("list rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), key[:12]) || !strings.Contains(out.String(), "active") {
		t.Fatalf("direct-use list should show the active key:\n%s", out.String())
	}
}

// keysMintWithStore mints against an explicit --store path (no manifest).
func keysMintWithStore(t *testing.T, storePath, tenant string) string {
	t.Helper()
	var out, errb bytes.Buffer
	rc := runKeys(&out, &errb, []string{"mint", "--store", storePath, "--tenant", tenant})
	if rc != 0 {
		t.Fatalf("mint --store rc=%d stderr=%s", rc, errb.String())
	}
	key := keysFullKeyRE.FindString(out.String())
	if key == "" {
		t.Fatalf("no full key in mint output:\n%s", out.String())
	}
	return key
}
