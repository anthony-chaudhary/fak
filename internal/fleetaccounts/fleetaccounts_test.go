package fleetaccounts

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fixture builds a hermetic two-product account tree + a sessions.json registry under
// t.TempDir, mirroring the Python fixtures used to capture the golden contract:
//
//	<home>/.claude                 worker  (logged in as uuid-default; AVAILABLE)
//	<home>/.claude-gem8-acct       worker  (logged in as uuid-gem8; THROTTLED)
//	<home>/.claude-dup-acct        worker  (ALSO logged in as uuid-default -> duplicate)
//	<home>/.claude-backup-acct     excluded by default policy (backup)
//	<home>/.claude-noproj          non-account (no projects/ subdir)
//	<home>/.claude.json            non-account (a plain file)
//	<cfg>/opencode-glm             worker  (opencode tier-2 GLM; 1 live session)
//	<cfg>/opencode-noconfig        non-account (no opencode.json)
func fixture(t *testing.T) (home, cfg, regPath string) {
	t.Helper()
	root := t.TempDir()
	home = filepath.Join(root, "home")
	cfg = filepath.Join(root, "cfg")
	reg := filepath.Join(root, "reg")
	for _, d := range []string{
		filepath.Join(home, ".claude", "projects"),
		filepath.Join(home, ".claude-gem8-acct", "projects"),
		filepath.Join(home, ".claude-dup-acct", "projects"),
		filepath.Join(home, ".claude-backup-acct", "projects"),
		filepath.Join(home, ".claude-noproj"),
		filepath.Join(cfg, "opencode-glm"),
		filepath.Join(cfg, "opencode-noconfig"),
		reg,
	} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	write := func(p, body string) {
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.Join(home, ".claude", ".claude.json"),
		`{"oauthAccount":{"accountUuid":"uuid-default","emailAddress":"default@example.com","organizationUuid":"org-1","organizationType":"team","seatTier":"pro"}}`)
	write(filepath.Join(home, ".claude", ".credentials.json"), `{}`)
	write(filepath.Join(home, ".claude-gem8-acct", ".claude.json"),
		`{"oauthAccount":{"accountUuid":"uuid-gem8","emailAddress":"gem8user@example.com","organizationUuid":"org-2","organizationType":"team"}}`)
	write(filepath.Join(home, ".claude-gem8-acct", ".credentials.json"), `{}`)
	// dup shares uuid-default with the .claude dir (its tag does NOT match the login)
	write(filepath.Join(home, ".claude-dup-acct", ".claude.json"),
		`{"oauthAccount":{"accountUuid":"uuid-default","emailAddress":"default@example.com","organizationUuid":"org-1","organizationType":"team"}}`)
	write(filepath.Join(home, ".claude-dup-acct", ".credentials.json"), `{}`)
	// a plain file named .claude.json under home -> non-account
	write(filepath.Join(home, ".claude.json"), `{}`)
	write(filepath.Join(cfg, "opencode-glm", "opencode.json"),
		`{"model":"zai-coding-plan/glm-5.2","small_model":"zai-coding-plan/glm-5.2-air"}`)
	// gem8 throttled with a clearly-future dated reset; opencode-glm has one live session.
	regBody := `{"generated_utc":"2026-06-29T12:00:00Z",` +
		`"throttle":{".claude-gem8-acct":{"reset":"Dec 31, 1pm"}},` +
		`"auth":{},` +
		`"sessions":[{"account":"opencode-glm","project":"work","disp":"LIVE","age_min":3}]}`
	write(filepath.Join(reg, "sessions.json"), regBody)
	return home, cfg, filepath.Join(reg, "sessions.json")
}

func find(rows []Account, account string) *Account {
	for i := range rows {
		if rows[i].Account == account {
			return &rows[i]
		}
	}
	return nil
}

func TestDiscoverClassifiesBothProducts(t *testing.T) {
	home, cfg, _ := fixture(t)
	rows := Discover(home, cfg, DefaultPolicy())

	cases := map[string]struct {
		kind    Kind
		tag     string
		product string
	}{
		".claude":             {KindWorker, "default", "claude"},
		".claude-gem8-acct":   {KindWorker, "gem8", "claude"},
		".claude-dup-acct":    {KindWorker, "dup", "claude"},
		".claude-backup-acct": {KindExcluded, "backup", "claude"},
		".claude-noproj":      {KindNonAccount, "noproj", "claude"},
		".claude.json":        {KindNonAccount, ".json", "claude"},
		"opencode-glm":        {KindWorker, "glm", "opencode"},
		"opencode-noconfig":   {KindNonAccount, "noconfig", "opencode"},
	}
	for acct, want := range cases {
		r := find(rows, acct)
		if r == nil {
			t.Fatalf("account %q not discovered", acct)
		}
		if r.Kind != want.kind {
			t.Errorf("%s: kind=%q want %q (reason=%q)", acct, r.Kind, want.kind, r.Reason)
		}
		if r.Tag != want.tag {
			t.Errorf("%s: tag=%q want %q", acct, r.Tag, want.tag)
		}
		if r.Product != want.product {
			t.Errorf("%s: product=%q want %q", acct, r.Product, want.product)
		}
	}
}

func TestProfileTierInference(t *testing.T) {
	home, cfg, _ := fixture(t)
	rows := Discover(home, cfg, DefaultPolicy())

	def := find(rows, ".claude")
	if def.ModelTier == nil || *def.ModelTier != 1 || derefStr(def.Model) != "opus" {
		t.Errorf("claude default: tier/model = %v/%q want 1/opus", def.ModelTier, derefStr(def.Model))
	}
	if derefStr(def.ProfileSource) != "default:claude-opus" {
		t.Errorf("claude default profile_source = %q", derefStr(def.ProfileSource))
	}
	glm := find(rows, "opencode-glm")
	if glm.ModelTier == nil || *glm.ModelTier != 2 {
		t.Errorf("opencode glm: tier = %v want 2", glm.ModelTier)
	}
	if derefStr(glm.Model) != "zai-coding-plan/glm-5.2" {
		t.Errorf("opencode glm model = %q", derefStr(glm.Model))
	}
}

func TestIdentityReconciliation(t *testing.T) {
	home, cfg, _ := fixture(t)
	rows := Discover(home, cfg, DefaultPolicy())

	def := find(rows, ".claude")
	dup := find(rows, ".claude-dup-acct")
	gem8 := find(rows, ".claude-gem8-acct")

	// .claude and .claude-dup-acct share uuid-default. One is canonical, the other dup.
	roles := map[string]string{
		derefStr(def.IdentityRole): "",
		derefStr(dup.IdentityRole): "",
	}
	if _, ok := roles["canonical"]; !ok {
		t.Errorf("expected one of .claude/.claude-dup to be canonical; got def=%q dup=%q",
			derefStr(def.IdentityRole), derefStr(dup.IdentityRole))
	}
	if _, ok := roles["duplicate"]; !ok {
		t.Errorf("expected one of .claude/.claude-dup to be duplicate; got def=%q dup=%q",
			derefStr(def.IdentityRole), derefStr(dup.IdentityRole))
	}
	// gem8 has a unique account.
	if derefStr(gem8.IdentityRole) != "unique" {
		t.Errorf("gem8 identity_role = %q want unique", derefStr(gem8.IdentityRole))
	}
	// the duplicate must not be routable.
	var theDup *Account
	if derefStr(def.IdentityRole) == "duplicate" {
		theDup = def
	} else {
		theDup = dup
	}
	if RoutableWorker(*theDup) {
		t.Errorf("duplicate-identity dir %s must not be routable", theDup.Account)
	}
}

func TestRuntimeStatusFold(t *testing.T) {
	home, cfg, regPath := fixture(t)
	reg := LoadRegistry(regPath)
	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), reg)

	def := find(rows, ".claude")
	if !derefBool(def.Available) || derefBool(def.Blocked) {
		t.Errorf(".claude should be available; available=%v blocked=%v", def.Available, def.Blocked)
	}
	if derefStr(def.StatusSource) != "registry" {
		t.Errorf(".claude status_source = %q want registry", derefStr(def.StatusSource))
	}

	gem8 := find(rows, ".claude-gem8-acct")
	if derefBool(gem8.Available) || !derefBool(gem8.Blocked) {
		t.Errorf("gem8 should be throttled/blocked; available=%v blocked=%v", gem8.Available, gem8.Blocked)
	}
	if derefStr(gem8.BlockKind) != "usage" || !derefBool(gem8.Throttled) {
		t.Errorf("gem8 block_kind=%q throttled=%v want usage/true", derefStr(gem8.BlockKind), gem8.Throttled)
	}
	if derefStr(gem8.BlockReason) != "usage limit; resets Dec 31, 1pm" {
		t.Errorf("gem8 block_reason = %q", derefStr(gem8.BlockReason))
	}

	glm := find(rows, "opencode-glm")
	if derefInt(glm.LiveSessions) != 1 || derefInt(glm.ActiveSessions) != 1 {
		t.Errorf("opencode-glm live/active = %d/%d want 1/1",
			derefInt(glm.LiveSessions), derefInt(glm.ActiveSessions))
	}
}

func TestAvailableExcludesBlockedAndDuplicate(t *testing.T) {
	home, cfg, regPath := fixture(t)
	reg := LoadRegistry(regPath)
	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), reg)
	avail := Available(rows)

	got := map[string]bool{}
	for _, r := range avail {
		got[r.Account] = true
	}
	// .claude (canonical) and opencode-glm are available; gem8 throttled, dup not routable,
	// backup excluded, non-accounts not workers.
	if !got[".claude"] || !got["opencode-glm"] {
		t.Errorf("available should include .claude + opencode-glm; got %v", got)
	}
	if got[".claude-gem8-acct"] {
		t.Errorf("throttled gem8 must not be available")
	}
	if got[".claude-dup-acct"] {
		t.Errorf("duplicate-identity dir must not be available")
	}
	if got[".claude-backup-acct"] {
		t.Errorf("excluded backup must not be available")
	}
}

func TestClaudeWorkerWithoutCredentialsIsBlockedByLoginStatus(t *testing.T) {
	home, cfg, _ := fixture(t)
	acctDir := filepath.Join(home, ".claude-needslogin-acct")
	if err := os.MkdirAll(filepath.Join(acctDir, "projects"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(acctDir, ".claude.json"),
		[]byte(`{"oauthAccount":{"accountUuid":"uuid-needs","emailAddress":"needs@example.com"}}`), 0o644); err != nil {
		t.Fatal(err)
	}

	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), Registry{})
	needs := find(rows, ".claude-needslogin-acct")
	if needs == nil {
		t.Fatal("needs-login account not discovered")
	}
	if derefStr(needs.LoginStatus) != "needs_login" || derefBool(needs.CanServe) {
		t.Fatalf("login_status/can_serve = %q/%v, want needs_login/false",
			derefStr(needs.LoginStatus), derefBool(needs.CanServe))
	}
	if derefBool(needs.Available) || !derefBool(needs.Blocked) {
		t.Fatalf("available/blocked = %v/%v, want false/true", derefBool(needs.Available), derefBool(needs.Blocked))
	}
	if derefStr(needs.BlockKind) != "auth" || !strings.Contains(derefStr(needs.BlockReason), "no live credentials") {
		t.Fatalf("block kind/reason = %q/%q, want auth/no-live-credentials",
			derefStr(needs.BlockKind), derefStr(needs.BlockReason))
	}
	for _, r := range Available(rows) {
		if r.Account == ".claude-needslogin-acct" {
			t.Fatalf("needs-login account was offered as available")
		}
	}
}

func TestExcludeReasonUsesNote(t *testing.T) {
	home, cfg, _ := fixture(t)
	rows := Discover(home, cfg, DefaultPolicy())
	backup := find(rows, ".claude-backup-acct")
	if backup.Kind != KindExcluded {
		t.Fatalf("backup kind = %q", backup.Kind)
	}
	if backup.Reason != "break-glass backup account; never auto-resume" {
		t.Errorf("backup reason = %q (should be the policy note)", backup.Reason)
	}
}

func TestPolicyExcludeMatchesClaudeLoginEmail(t *testing.T) {
	home, cfg, _ := fixture(t)
	pol := DefaultPolicy()
	pol.Exclude = append(pol.Exclude, "default@example.com")
	pol.Notes["default@example.com"] = "retired login identity"

	rows := Discover(home, cfg, pol)
	def := find(rows, ".claude")
	if def == nil {
		t.Fatal("default account not discovered")
	}
	if def.Kind != KindExcluded {
		t.Fatalf("default kind = %q want %q", def.Kind, KindExcluded)
	}
	if def.Reason != "retired login identity" {
		t.Errorf("default reason = %q", def.Reason)
	}
}

// TestJSONShapeMatchesPythonContract proves the marshaled worker/non-worker row keeps
// the stable key order, including the Go-only login readiness fields on Claude workers.
func TestJSONShapeMatchesPythonContract(t *testing.T) {
	home, cfg, regPath := fixture(t)
	reg := LoadRegistry(regPath)
	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), reg)

	workerKeys := []string{
		"dir", "product", "account", "tag", "kind", "reason", "notes",
		"model_tier", "model", "small_model", "model_effort", "agent", "profile_source", "route_weight",
		"account_uuid", "login_email", "org_uuid", "org_type", "plan",
		"tag_login_match", "identity_peers", "identity_role", "login_status", "can_serve",
		"available", "blocked", "block_kind", "block_reason", "reset", "weekly", "throttled",
		"active_sessions", "live_sessions", "auth_blocked_sessions", "status_source", "registry_age_min",
	}
	nonAccountKeys := []string{
		"dir", "product", "account", "tag", "kind", "reason", "notes",
		"available", "blocked", "block_kind", "block_reason", "reset", "weekly", "throttled",
		"active_sessions", "live_sessions", "auth_blocked_sessions", "status_source", "registry_age_min",
	}
	opencodeWorkerKeys := []string{
		"dir", "product", "account", "tag", "kind", "reason", "notes",
		"model_tier", "model", "small_model", "model_effort", "agent", "profile_source", "route_weight",
		"available", "blocked", "block_kind", "block_reason", "reset", "weekly", "throttled",
		"active_sessions", "live_sessions", "auth_blocked_sessions", "status_source", "registry_age_min",
	}

	assertKeyOrder(t, *find(rows, ".claude"), workerKeys, "claude worker")
	assertKeyOrder(t, *find(rows, ".claude-noproj"), nonAccountKeys, "non-account")
	assertKeyOrder(t, *find(rows, "opencode-glm"), opencodeWorkerKeys, "opencode worker (no claude identity)")
}

// assertKeyOrder marshals one Account and checks its top-level keys equal want, in order.
func assertKeyOrder(t *testing.T, a Account, want []string, label string) {
	t.Helper()
	data, err := json.Marshal(a)
	if err != nil {
		t.Fatalf("%s: marshal: %v", label, err)
	}
	got := topLevelKeysInOrder(t, data)
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("%s key order mismatch:\n got: %v\nwant: %v", label, got, want)
	}
}

// topLevelKeysInOrder extracts top-level object keys in document order using a streaming
// decoder (encoding/json preserves source order at the token level).
func topLevelKeysInOrder(t *testing.T, data []byte) []string {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader(data))
	tok, err := dec.Token()
	if err != nil || tok != json.Delim('{') {
		t.Fatalf("expected object, got %v (err=%v)", tok, err)
	}
	var keys []string
	depth := 0
	for dec.More() || depth > 0 {
		tk, err := dec.Token()
		if err != nil {
			t.Fatalf("token: %v", err)
		}
		if depth == 0 {
			if s, ok := tk.(string); ok {
				keys = append(keys, s)
				// consume the value (which may be a nested object/array)
				v, err := dec.Token()
				if err != nil {
					t.Fatalf("value token: %v", err)
				}
				if d, ok := v.(json.Delim); ok && (d == '{' || d == '[') {
					skipNested(t, dec, d)
				}
			}
			continue
		}
	}
	return keys
}

func skipNested(t *testing.T, dec *json.Decoder, open json.Delim) {
	t.Helper()
	close := json.Delim('}')
	if open == '[' {
		close = json.Delim(']')
	}
	for {
		tk, err := dec.Token()
		if err != nil {
			t.Fatalf("skip token: %v", err)
		}
		if d, ok := tk.(json.Delim); ok {
			switch d {
			case '{', '[':
				skipNested(t, dec, d)
			case close:
				return
			}
		}
	}
}

func TestResolvePinAndRoute(t *testing.T) {
	home, cfg, regPath := fixture(t)
	reg := LoadRegistry(regPath)
	pol := DefaultPolicy()
	rows := AnnotatedRoster(home, cfg, pol, reg)

	// pin an available account
	r := Resolve(rows, home, ResolveRequest{Pin: "default"}, pol)
	if !r.OK || r.Account != ".claude" || r.Reason != "pinned account" {
		t.Errorf("pin default: ok=%v account=%q reason=%q", r.OK, r.Account, r.Reason)
	}
	if r.SelectedTier == nil || *r.SelectedTier != 1 {
		t.Errorf("pin default selected_tier = %v want 1", r.SelectedTier)
	}

	// pin a throttled account -> blocked
	rb := Resolve(rows, home, ResolveRequest{Pin: "gem8"}, pol)
	if rb.OK || !strings.Contains(rb.Reason, "blocked") {
		t.Errorf("pin gem8: ok=%v reason=%q (want blocked)", rb.OK, rb.Reason)
	}

	// pin a non-existent account
	rn := Resolve(rows, home, ResolveRequest{Pin: "nope"}, pol)
	if rn.OK || !strings.Contains(rn.Reason, "not an offered worker") {
		t.Errorf("pin nope: ok=%v reason=%q", rn.OK, rn.Reason)
	}

	// route engineering work -> tier 1 (the available .claude)
	re := Resolve(rows, home, ResolveRequest{WorkKind: "engineering"}, pol)
	if !re.OK || re.Account != ".claude" {
		t.Errorf("route engineering: ok=%v account=%q", re.OK, re.Account)
	}
	if re.TargetTier == nil || *re.TargetTier != 1 {
		t.Errorf("route engineering target_tier = %v want 1", re.TargetTier)
	}

	// route gardening -> tier 2 target; opencode-glm (tier 2) is available.
	rg := Resolve(rows, home, ResolveRequest{WorkKind: "gardening"}, pol)
	if !rg.OK || rg.Account != "opencode-glm" {
		t.Errorf("route gardening: ok=%v account=%q want opencode-glm", rg.OK, rg.Account)
	}
}

func TestSeatPoolBindingAndHeadroom(t *testing.T) {
	home, cfg, regPath := fixture(t)
	reg := LoadRegistry(regPath)
	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), reg)

	// One live worker leased onto opencode-glm by tag.
	leases := []Lease{{Worker: "resolve-1", Tag: "glm", Dir: filepath.Join(cfg, "opencode-glm")}}
	pool := BuildSeatPool(rows, leases, "")

	// routable seats: .claude (canonical), gem8, opencode-glm. dup is NOT a seat (one pool).
	if pool.TotalSeats != 3 {
		t.Errorf("total_seats = %d want 3 (dup collapsed)", pool.TotalSeats)
	}
	// opencode-glm leased, .claude free, gem8 blocked.
	if pool.LeasedSeats != 1 || pool.FreeSeats != 1 || pool.BlockedSeats != 1 {
		t.Errorf("seat states leased/free/blocked = %d/%d/%d want 1/1/1",
			pool.LeasedSeats, pool.FreeSeats, pool.BlockedSeats)
	}
	if pool.Depleted {
		t.Errorf("pool should not be depleted (one free seat)")
	}
	if pool.Schema != SeatPoolSchema {
		t.Errorf("schema = %q", pool.Schema)
	}
	// the leased seat is opencode-glm.
	var glmSeat *Seat
	for i := range pool.Seats {
		if pool.Seats[i].Account == "opencode-glm" {
			glmSeat = &pool.Seats[i]
		}
	}
	if glmSeat == nil || glmSeat.State != "leased" || len(glmSeat.Workers) != 1 {
		t.Errorf("opencode-glm seat not leased correctly: %+v", glmSeat)
	}
}

func TestResetParsingExpiredVsFuture(t *testing.T) {
	// a clearly expired dated reset (last year, >180d back rolls forward — so use a
	// recent past bare time within window vs a far-future one).
	now := mustParse(t, "2026-06-29T18:00:00Z")
	// bare time already passed beyond the window -> expired (false)
	if got := resetIsFuture("3am", now); got == nil || *got {
		t.Errorf("'3am' at 6pm should be expired; got %v", got)
	}
	// bare time still ahead today -> future (true)
	if got := resetIsFuture("11pm", now); got == nil || !*got {
		t.Errorf("'11pm' at 6pm should be future; got %v", got)
	}
	// dated future -> future (true)
	if got := resetIsFuture("Dec 31, 1pm", now); got == nil || !*got {
		t.Errorf("'Dec 31, 1pm' should be future; got %v", got)
	}
	// unknown format -> nil
	if got := resetIsFuture("whenever", now); got != nil {
		t.Errorf("unknown reset should be nil; got %v", *got)
	}
}

func mustParse(t *testing.T, s string) time.Time {
	t.Helper()
	p := parseUTC(s)
	if p == nil {
		t.Fatalf("parse %q", s)
	}
	return *p
}
