package fleetaccounts

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// picker_parity_test.go — #3000's cross-surface drift gate.
//
// The account picker is transcribed TWICE on purpose: this package is the Go port of the
// read-only roster/resolve fold in tools/fleet_accounts.py (see the package doc), and the
// Python surface is still the live one for the launcher/watchdog tools that import it. So
// "one implementation owns the lifecycle registry index, tombstone exclusion, and
// same-identity status reconciliation" cannot be enforced by deleting a copy here; it is
// enforced by a gate that drives BOTH surfaces from ONE fixture and fails the lane the
// moment their verdicts diverge. That is #3000's stated witness ("the same tombstone +
// stale-auth + usage-cap fixtures passing identically on both surfaces").
//
// testdata/parity_check.sh proved the idea by hand and was never wired to `go test` for two
// reasons this test fixes: its fixture exercised none of the three cases #3000 regressed on,
// and a whole-envelope diff is permanently red on three INTENTIONAL deltas (see
// parityValueExempt / parityKnownAsymmetric). Absorbing exactly those, and nothing else, is
// what makes the gate honest instead of either red-by-design or vacuous.
//
// The fixture is the union of the three cases in one account tree:
//
//	tombstone  — .claude-gem8NEW-netra is tombstoned in ~/.claude-accounts/registry.json and
//	             must never be offered, by name/dir/identity.
//	stale-auth — .claude-july4-netra (canonical) carries an auth-ONLY block while its
//	             same-UUID peer .claude (duplicate) has a newer successful session, so the
//	             canonical row must be repaired to available with status_source=identity-peer.
//	usage-cap  — .claude-capped-acct carries a usage block, which must stay fail-closed.

// parityKnownAsymmetric is the CLOSED set of row keys one surface emits and the other does
// not, each for a documented reason. Any other key present on one side only is new drift and
// fails the gate — that asymmetry check is what keeps this test from silently absorbing a
// field the two pickers stop agreeing about.
var parityKnownAsymmetric = map[string]string{
	"login_status":     "Go-only by design: the port adds the credential-safe login readiness verdict (package doc)",
	"can_serve":        "Go-only by design: the served half of the login_status verdict",
	"discovery_source": "Go-only: native dispatch publishes the sanctioned root that contributed the census row (#8482)",
	"root_state":       "Go-only: native dispatch publishes the account root's structural verdict (#8482)",
	"throttled_since":  "Python-only: extra provenance on an already-agreed throttled row",
}

// parityValueExempt keys are emitted by BOTH surfaces but legitimately differ in value, so
// only their presence is compared. Every other shared key must match exactly.
var parityValueExempt = map[string]string{
	"dir":              "path separators only — Python joins the raw env home, Go filepath-cleans it",
	"registry_age_min": "wall-clock derived from the sessions.json generated_utc stamp",
}

// legacyPickerExe resolves the interpreter the repo's hooks use: python3 → python → py -3.
func legacyPickerExe() (string, []string) {
	for _, c := range []struct {
		bin  string
		args []string
	}{
		{"python3", nil}, {"python", nil}, {"py", []string{"-3"}},
	} {
		if _, err := exec.LookPath(c.bin); err == nil {
			return c.bin, c.args
		}
	}
	return "", nil
}

// writePickerParityFixture builds the tombstone + stale-auth + usage-cap account tree and
// points BOTH surfaces at it through the env knobs they share, so neither can fall back to
// the operator's real home or registry. Returns the roster inputs the Go fold needs.
func writePickerParityFixture(t *testing.T) (home, cfg, sessions string) {
	t.Helper()
	root := t.TempDir()
	home = filepath.Join(root, "home")
	cfg = filepath.Join(root, "cfg")
	regDir := filepath.Join(root, "reg")
	for _, d := range []string{
		filepath.Join(home, ".claude", "projects"),
		filepath.Join(home, ".claude-july4-netra", "projects"),
		filepath.Join(home, ".claude-gem8NEW-netra", "projects"),
		filepath.Join(home, ".claude-capped-acct", "projects"),
		filepath.Join(home, ".claude-accounts"),
		cfg, regDir,
	} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	write := func(p, body string) {
		t.Helper()
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	id := func(uuid, email, org string) string {
		return `{"oauthAccount":{"accountUuid":"` + uuid + `","emailAddress":"` + email +
			`","organizationUuid":"` + org + `","organizationType":"claude_max"}}`
	}
	// .claude and .claude-july4-netra are the SAME Anthropic account (one uuid, two dirs) —
	// the duplicate/canonical pair the stale-auth repair is about.
	for _, dir := range []string{".claude", ".claude-july4-netra"} {
		write(filepath.Join(home, dir, ".claude.json"), id("uuid-july4", "july4@example.com", "org-july4"))
		write(filepath.Join(home, dir, ".credentials.json"), `{}`)
	}
	write(filepath.Join(home, ".claude-gem8NEW-netra", ".claude.json"),
		id("uuid-gem8", "gem8@example.com", "org-gem8"))
	write(filepath.Join(home, ".claude-gem8NEW-netra", ".credentials.json"), `{}`)
	write(filepath.Join(home, ".claude-capped-acct", ".claude.json"),
		id("uuid-capped", "capped@example.com", "org-capped"))
	write(filepath.Join(home, ".claude-capped-acct", ".credentials.json"), `{}`)

	acctReg := filepath.Join(home, ".claude-accounts", "registry.json")
	write(acctReg, `{"version":"fak-config-homes/v1","homes":[`+
		`{"name":"default","dir":"`+jsonPath(filepath.Join(home, ".claude"))+`"},`+
		`{"name":"july4-netra","dir":"`+jsonPath(filepath.Join(home, ".claude-july4-netra"))+`"},`+
		`{"name":"capped-acct","dir":"`+jsonPath(filepath.Join(home, ".claude-capped-acct"))+`"},`+
		`{"name":"gem8NEW-netra","dir":"`+jsonPath(filepath.Join(home, ".claude-gem8NEW-netra"))+`",`+
		`"status":"tombstoned","rehome_to":"july4-netra",`+
		`"tombstone_reason":"retired in fak accounts registry"}`+
		`],"roles":{"active":"july4-netra","anchor":"default"}}`)

	// An EMPTY probe ledger beside sessions.json, so the fixture's registry is a complete
	// one: a prober is wired here, it has simply recorded nothing about these accounts.
	// Both surfaces now grade the dir the same way — Go through
	// accountprobe.ResolveRegDir().BlocksDerivable(), and the legacy picker through
	// _registry_blocks_derivable(), which the Python half of #5439 substituted for the old
	// "FLEET_REG_DIR is merely SET" test. Drop this line and they still agree, but they agree
	// on blocks-unknown: the dir grades accountprobe.RegHealthBlocksUnknown, and both folds
	// publish an unblocked seat as status_source=registry-unknown (see markUnknownHealth and
	// its Python mirror _mark_unknown_health). That is a different state than this fixture is
	// built to compare. The divergence this comment used to record — Go publishing
	// registry-unknown while the picker published registry — is closed; it was never absorbed
	// into parityValueExempt, because doing so would have stopped this gate comparing
	// status_source at all. Naming the prober's dir and giving it no ledger was always an
	// inconsistent fixture; this line makes the on-disk state match the claim the
	// FLEET_REG_DIR above was already making.
	write(filepath.Join(regDir, "probe_ledger.jsonl"), "")

	sessions = filepath.Join(regDir, "sessions.json")
	write(sessions, `{"generated_utc":"2026-07-06T02:24:02Z",`+
		`"auth":{".claude-july4-netra":{"block_kind":"auth","block_reason":"auth/login required",`+
		`"seen_utc":"2026-07-06T02:02:02Z"}},`+
		`"throttle":{".claude-capped-acct":{"reset":"Dec 31, 1pm","block_kind":"usage",`+
		`"block_reason":"usage limit reached"}},`+
		`"sessions":[`+
		`{"account":".claude","project":"work","disp":"DONE","age_min":1},`+
		`{"account":".claude-july4-netra","project":"work","disp":"INFRA_AUTH",`+
		`"action":"BLOCKED_AUTH","age_min":22,"last":"Not logged in"}]}`)

	// Both surfaces read these; setting them on the test process means the Python child
	// inherits the identical view. FLEET_POLICY_PATH names a file that does not exist so the
	// legacy picker falls back to its built-in policy, matching DefaultPolicy() here.
	t.Setenv("FLEET_USER_HOME", home)
	t.Setenv("FLEET_CONFIG_HOME", cfg)
	t.Setenv("FLEET_REG_DIR", regDir)
	t.Setenv("FLEET_POLICY_PATH", filepath.Join(root, "no-policy.json"))
	t.Setenv("FAK_ACCOUNTS_REGISTRY", acctReg)
	return home, cfg, sessions
}

// runLegacyPicker invokes tools/fleet_accounts.py and decodes its JSON envelope. The script
// is run with cwd=tools because that is how every caller in the repo invokes it.
func runLegacyPicker(t *testing.T, mode string, args ...string) map[string]any {
	t.Helper()
	py, pyArgs := legacyPickerExe()
	if py == "" {
		t.Skip("no python interpreter on PATH — cannot witness legacy-picker parity")
	}
	toolsDir := filepath.Join("..", "..", "tools")
	script := filepath.Join(toolsDir, "fleet_accounts.py")
	if _, err := os.Stat(script); err != nil {
		t.Skipf("legacy picker not present at %s: %v", script, err)
	}
	cmd := exec.Command(py, append(append(pyArgs, "fleet_accounts.py", mode), args...)...)
	cmd.Dir = toolsDir
	out, err := cmd.Output()
	if err != nil {
		var stderr string
		if ee, ok := err.(*exec.ExitError); ok {
			stderr = string(ee.Stderr)
		}
		// `resolve` exits non-zero when it cannot place the request; that is a verdict, not a
		// harness failure, so decode anyway and let the assertions speak.
		if len(out) == 0 {
			t.Fatalf("legacy picker %s failed: %v\n%s", mode, err, stderr)
		}
	}
	var doc map[string]any
	if err := json.Unmarshal(out, &doc); err != nil {
		t.Fatalf("legacy picker %s emitted undecodable JSON: %v\n%s", mode, err, out)
	}
	return doc
}

// goRosterRows re-encodes the Go roster through Account.MarshalJSON, so the comparison runs
// over the bytes the Go surface actually publishes rather than over its in-memory structs.
func goRosterRows(t *testing.T, rows []Account) map[string]map[string]any {
	t.Helper()
	out := map[string]map[string]any{}
	for _, r := range rows {
		b, err := json.Marshal(r)
		if err != nil {
			t.Fatalf("marshal %s: %v", r.Account, err)
		}
		var m map[string]any
		if err := json.Unmarshal(b, &m); err != nil {
			t.Fatalf("unmarshal %s: %v", r.Account, err)
		}
		out[r.Account] = m
	}
	return out
}

// legacyRosterRows keys the legacy envelope's rows by account for row-wise comparison.
func legacyRosterRows(t *testing.T, doc map[string]any) map[string]map[string]any {
	t.Helper()
	raw, ok := doc["accounts"].([]any)
	if !ok {
		t.Fatalf("legacy envelope has no accounts list: %v", doc)
	}
	out := map[string]map[string]any{}
	for _, r := range raw {
		m, ok := r.(map[string]any)
		if !ok {
			t.Fatalf("legacy row is not an object: %v", r)
		}
		out[asString(m["account"])] = m
	}
	return out
}

// symmetricKeys returns a row's keys minus the documented one-sided ones, so the two
// surfaces' field sets can be compared for NEW asymmetry.
func symmetricKeys(row map[string]any) []string {
	var out []string
	for k := range row {
		if _, known := parityKnownAsymmetric[k]; known {
			continue
		}
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// accountNames lists the account ids in a roster map, sorted.
func accountNames(rows map[string]map[string]any) []string {
	var out []string
	for k := range rows {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// assertPickerInvariants checks #3000's acceptance on ONE surface's rows: the tombstoned row
// is excluded and unoffered, the canonical stale-auth row is repaired from its identity peer,
// and the usage-capped row stays fail-closed. Run against both surfaces so the gate proves
// the behavior, not merely that two implementations agree on a shared mistake.
func assertPickerInvariants(t *testing.T, surface string, rows map[string]map[string]any, offered []string) {
	t.Helper()

	tomb := rows[".claude-gem8NEW-netra"]
	if tomb == nil {
		t.Fatalf("%s: tombstoned row not discovered", surface)
	}
	if got := asString(tomb["kind"]); got != string(KindExcluded) {
		t.Errorf("%s: tombstoned row kind = %q, want %q", surface, got, KindExcluded)
	}
	if got := asString(tomb["reason"]); !strings.Contains(got, "retired in fak accounts registry") {
		t.Errorf("%s: tombstoned row reason = %q, want the registry tombstone reason", surface, got)
	}
	for _, name := range offered {
		if name == ".claude-gem8NEW-netra" {
			t.Errorf("%s: tombstoned row was OFFERED (%v)", surface, offered)
		}
	}

	canonical := rows[".claude-july4-netra"]
	if canonical == nil {
		t.Fatalf("%s: canonical row not discovered", surface)
	}
	if canonical["available"] != true {
		t.Errorf("%s: canonical stale-auth row available = %v, want true via its identity peer",
			surface, canonical["available"])
	}
	if got := asString(canonical["status_source"]); got != "identity-peer" {
		t.Errorf("%s: canonical status_source = %q, want identity-peer", surface, got)
	}
	if canonical["block_kind"] != nil {
		t.Errorf("%s: canonical block_kind = %v, want cleared by the identity-peer repair",
			surface, canonical["block_kind"])
	}

	capped := rows[".claude-capped-acct"]
	if capped == nil {
		t.Fatalf("%s: usage-capped row not discovered", surface)
	}
	if capped["available"] != false {
		t.Errorf("%s: usage-capped row available = %v, want false (usage stays fail-closed)",
			surface, capped["available"])
	}
	if got := asString(capped["block_kind"]); got != "usage" {
		t.Errorf("%s: usage-capped row block_kind = %q, want usage", surface, got)
	}
	if want := []string{".claude-july4-netra"}; !reflect.DeepEqual(offered, want) {
		t.Errorf("%s: offered accounts = %v, want %v", surface, offered, want)
	}
}

// TestTombstoneStaleAuthAndUsageCapFixtureMatchesLegacyPicker is the #3000 gate: one fixture,
// both pickers, identical verdicts. A divergence in registry indexing, tombstone exclusion, or
// same-identity status reconciliation fails the fleetaccounts lane instead of shipping as the
// next silent picker drift.
func TestTombstoneStaleAuthAndUsageCapFixtureMatchesLegacyPicker(t *testing.T) {
	home, cfg, sessions := writePickerParityFixture(t)

	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), LoadRegistry(sessions))
	goRows := goRosterRows(t, rows)
	var goOffered []string
	for _, r := range Available(rows) {
		goOffered = append(goOffered, r.Account)
	}
	assertPickerInvariants(t, "go", goRows, goOffered)

	legacyDoc := runLegacyPicker(t, "json")
	legacyRows := legacyRosterRows(t, legacyDoc)
	var legacyOffered []string
	if raw, ok := legacyDoc["available_accounts"].([]any); ok {
		for _, r := range raw {
			if m, ok := r.(map[string]any); ok {
				legacyOffered = append(legacyOffered, asString(m["account"]))
			}
		}
	}
	assertPickerInvariants(t, "legacy", legacyRows, legacyOffered)

	// Same roster, row for row.
	if g, l := accountNames(goRows), accountNames(legacyRows); !reflect.DeepEqual(g, l) {
		t.Fatalf("roster membership differs:\n go     %v\n legacy %v", g, l)
	}
	if !reflect.DeepEqual(goOffered, legacyOffered) {
		t.Errorf("offered set differs:\n go     %v\n legacy %v", goOffered, legacyOffered)
	}
	for _, account := range accountNames(goRows) {
		g, l := goRows[account], legacyRows[account]
		if gk, lk := symmetricKeys(g), symmetricKeys(l); !reflect.DeepEqual(gk, lk) {
			t.Errorf("%s: field sets diverged (a one-sided key that is not in parityKnownAsymmetric"+
				" is new drift)\n go     %v\n legacy %v", account, gk, lk)
			continue
		}
		for _, key := range symmetricKeys(g) {
			if _, exempt := parityValueExempt[key]; exempt {
				continue
			}
			if !reflect.DeepEqual(g[key], l[key]) {
				t.Errorf("%s: %s diverged — go %#v, legacy %#v", account, key, g[key], l[key])
			}
		}
	}
}

// TestResolveDefaultLaunchPicksCanonicalRowOnBothSurfaces witnesses #3000's named command —
// `fleet-accounts resolve --product claude --task "default launch"` — on both surfaces. The
// duplicate .claude dir's newer successful session proves the account is currently usable
// without consuming its live-session slot, so the CANONICAL row is the one that must be
// handed to the launch, not the duplicate and not a tombstoned peer.
func TestResolveDefaultLaunchPicksCanonicalRowOnBothSurfaces(t *testing.T) {
	home, cfg, sessions := writePickerParityFixture(t)
	const want = ".claude-july4-netra"

	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), LoadRegistry(sessions))
	got := Resolve(rows, home, ResolveRequest{Product: "claude", TaskText: "default launch"}, DefaultPolicy())
	if !got.OK || got.Account != want {
		t.Fatalf("go resolve = {ok:%v account:%q reason:%q}, want %s",
			got.OK, got.Account, got.Reason, want)
	}

	doc := runLegacyPicker(t, "resolve", "--product", "claude", "--task", "default launch")
	if doc["ok"] != true || asString(doc["account"]) != want {
		t.Fatalf("legacy resolve = {ok:%v account:%q reason:%q}, want %s",
			doc["ok"], asString(doc["account"]), asString(doc["reason"]), want)
	}
}
