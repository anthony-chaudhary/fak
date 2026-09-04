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

	// Pin the registry this fold RESOLVES to, and give it an empty probe ledger, so the
	// fixture is hermetic. Without both lines shouldConsultProbeLedger falls through to
	// accountprobe.ResolveRegDir()'s ambient discovery, which finds a real ledger-bearing
	// Fleet dir on an operator's box and nothing at all in CI — so markUnknownHealth
	// publishes an unblocked seat as status_source "registry" here and "registry-unknown"
	// there, and the fold's verdict becomes a fact about whose machine ran it. The empty
	// ledger is the same shape picker_parity_test.go pins for the same reason (#5439): a
	// prober IS wired to this dir, it has simply recorded nothing about these accounts.
	t.Setenv("FLEET_REG_DIR", reg)
	write(filepath.Join(reg, "probe_ledger.jsonl"), "")

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

func jsonPath(s string) string {
	b, _ := json.Marshal(s)
	return strings.Trim(string(b), `"`)
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
	if glm.LoginStatus == nil || *glm.LoginStatus != "ready" || glm.CanServe == nil || !*glm.CanServe {
		t.Fatalf("opencode readiness = %v/%v, want ready/true", glm.LoginStatus, glm.CanServe)
	}
	if glm.Reason != "configured opencode account; serving requires active inference probe" {
		t.Fatalf("opencode readiness reason = %q", glm.Reason)
	}
	if glm.ModelTier == nil || *glm.ModelTier != 2 {
		t.Errorf("opencode glm: tier = %v want 2", glm.ModelTier)
	}
	if derefStr(glm.Model) != "zai-coding-plan/glm-5.2" {
		t.Errorf("opencode glm model = %q", derefStr(glm.Model))
	}
}

// TestModelTierFromNameGeminiFlashIsTier2 pins the v1 model taxonomy, including the
// Gemini 3.5 Flash → tier 2 classification (the GCP Vertex lightweight seat). It also
// guards the existing tiers against regression, and asserts the Vertex OpenAI-compat
// upstream id (`google/gemini-3.5-flash`) — the one the dispatch seat actually sends —
// resolves to tier 2, not tier 3.
func TestModelTierFromNameGeminiFlashIsTier2(t *testing.T) {
	cases := []struct {
		model string
		tier  int
	}{
		// Gemini 3.5 Flash: the new tier-2 lightweight seat, in every id shape it appears.
		{"gemini-3.5-flash", 2},
		{"google/gemini-3.5-flash", 2},
		{"Gemini 3.5 Flash", 2},
		{"gemini_3.5_flash", 2},
		// Fable 5 is the restricted apex model: tier 0, in every id shape it appears.
		{"claude-fable-5", 0},
		{"fable-5", 0},
		{"fable5", 0},
		{"Fable 5", 0},
		{"claude-fable-5-preview", 0},
		// GLM-5.2 stays tier 2; the frontier set stays tier 1; unknowns stay tier 3.
		{"glm-5.2", 2},
		{"zai-coding-plan/glm-5.2", 2},
		{"gpt-5.5", 1},
		// GPT-6 generation: Astra (flagship, bare gpt-6 and astra aliases) ranks frontier.
		{"gpt-6", 1},
		{"gpt-6-astra", 1},
		{"gpt 6 astra", 1},
		{"gpt6astra", 1},
		{"astra", 1},
		{"openai/gpt-6-astra", 1},
		// GPT-5.6 generation: Sol (bare alias) and Terra rank frontier; Luna is the
		// fast/cheap lightweight seat.
		{"gpt-5.6", 1},
		{"gpt-5.6-sol", 1},
		{"gpt-5.6-terra", 1},
		{"openai/gpt-5.6-sol", 1},
		{"gpt-5.6-luna", 2},
		// The Opus FAMILY ranks frontier in every generation and id shape. Enumerating only
		// opus-4.6 here used to leave the SHIPPED fleet default (claude-opus-4-8) at tier 3,
		// so the router treated the strongest seat as "everything else"; these rows pin the
		// family match that fixed it, and keep the next model bump from re-opening the hole.
		{"opus-4.6", 1},
		{"claude-opus-5", 1},
		{"claude-opus-4-8", 1},
		{"claude-opus-4.8", 1},
		{"anthropic/claude-opus-5", 1},
		{"Claude Opus 5", 1},
		{"claudeopus5", 1},
		{"opus", 1},
		{"claude-opus", 1},
		// …but only as a NAME, never as a substring: an unrelated id that merely CONTAINS
		// the letters must not be promoted into the frontier set.
		{"octopus-7b", 3},
		{"deepseek-v4-pro", 1},
		{"kimi-k2.6", 1},
		{"gemini-3.5-pro", 3}, // only Flash is tier 2; Pro is not classified here
		{"llama3.2", 3},
		{"", 3},
	}
	for _, tc := range cases {
		if got := modelTierFromName(tc.model); got != tc.tier {
			t.Errorf("modelTierFromName(%q) = %d, want %d", tc.model, got, tc.tier)
		}
	}
}

func TestLegacyNIMDemoSeatsAreRestrictedFromEngineeringRoutes(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "home")
	cfg := filepath.Join(root, "cfg")
	if err := os.MkdirAll(home, 0o755); err != nil {
		t.Fatal(err)
	}
	seats := []struct {
		account string
		tag     string
		model   string
		weight  int
	}{
		{"opencode-nim-deepseek-v4-pro", "nim-deepseek-v4-pro", NIMDeepSeekV4ProModel, 0},
		{"opencode-nim-kimi-k26", "nim-kimi-k26", NIMKimiK26Model, 0},
		{"opencode-nim-glm52", "nim-glm52", NIMGLM52Model, 10},
	}
	for _, seat := range seats {
		dir := filepath.Join(cfg, seat.account)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "opencode.json"), []byte(`{}`), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	pol := DefaultPolicy()
	rows := AnnotatedRoster(home, cfg, pol, Registry{})
	for _, seat := range seats {
		row := find(rows, seat.account)
		if row == nil {
			t.Fatalf("%s not discovered", seat.account)
		}
		if row.Kind != KindWorker || row.Product != "opencode" || row.Tag != seat.tag {
			t.Fatalf("%s classified as kind/product/tag=%q/%q/%q",
				seat.account, row.Kind, row.Product, row.Tag)
		}
		wantTier := TierOther
		if seat.tag == "nim-glm52" {
			wantTier = TierFrontier
		}
		if row.ModelTier == nil || *row.ModelTier != wantTier {
			t.Errorf("%s tier = %v want %d", seat.account, row.ModelTier, wantTier)
		}
		if derefStr(row.Model) != seat.model {
			t.Errorf("%s model = %q want %q", seat.account, derefStr(row.Model), seat.model)
		}
		if derefStr(row.ProfileSource) != "default:nvidia-nim-coding:"+seat.tag {
			t.Errorf("%s profile_source = %q", seat.account, derefStr(row.ProfileSource))
		}
		if row.RouteWeight == nil || *row.RouteWeight != seat.weight {
			t.Errorf("%s route_weight = %v want %d", seat.account, row.RouteWeight, seat.weight)
		}
	}

	route := RouteAccount(rows, "implement the feature", "engineering", false, false, "opencode", pol)
	if !route.OK || route.Account == nil || route.Account.Account != "opencode-nim-glm52" {
		t.Fatalf("engineering opencode route = %+v, want audited frontier seat; legacy API-demo homes must be excluded", route)
	}
	explicit := RouteAccount(rows, "literal one-file edit with supplied test", "tier3", false, true, "opencode", pol)
	if !explicit.OK || explicit.Account == nil || explicit.Account.Account != "opencode-nim-deepseek-v4-pro" {
		t.Fatalf("explicit tier-3 route = %+v, want restricted legacy API-demo seat", explicit)
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

func TestPickerUsesAccountsRegistryAndIdentityPeerAvailability(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "home")
	cfg := filepath.Join(root, "cfg")
	for _, d := range []string{
		filepath.Join(home, ".claude", "projects"),
		filepath.Join(home, ".claude-july4-netra", "projects"),
		filepath.Join(home, ".claude-gem8NEW-netra", "projects"),
		filepath.Join(cfg, "empty"),
		filepath.Join(home, ".claude-accounts"),
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
	july4ID := `{"oauthAccount":{"accountUuid":"uuid-july4","emailAddress":"july4@example.com","organizationUuid":"org-july4","organizationType":"claude_max"}}`
	gem8ID := `{"oauthAccount":{"accountUuid":"uuid-gem8","emailAddress":"gem8@example.com","organizationUuid":"org-gem8","organizationType":"claude_max"}}`
	for _, dir := range []string{".claude", ".claude-july4-netra"} {
		write(filepath.Join(home, dir, ".claude.json"), july4ID)
		write(filepath.Join(home, dir, ".credentials.json"), `{}`)
	}
	write(filepath.Join(home, ".claude-gem8NEW-netra", ".claude.json"), gem8ID)
	write(filepath.Join(home, ".claude-gem8NEW-netra", ".credentials.json"), `{}`)
	accountsReg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"default","dir":"` + jsonPath(filepath.Join(home, ".claude")) + `"},` +
		`{"name":"july4-netra","dir":"` + jsonPath(filepath.Join(home, ".claude-july4-netra")) + `"},` +
		`{"name":"gem8NEW-netra","dir":"` + jsonPath(filepath.Join(home, ".claude-gem8NEW-netra")) + `","status":"tombstoned","rehome_to":"july4-netra","tombstone_reason":"retired in fak accounts registry"}` +
		`],"roles":{"active":"july4-netra","anchor":"default"}}`
	write(filepath.Join(home, ".claude-accounts", "registry.json"), accountsReg)

	reg := Registry{
		GeneratedUTC: "2026-07-06T02:24:02Z",
		Auth: map[string]any{
			".claude-july4-netra": map[string]any{
				"block_kind": "auth", "block_reason": "auth/login required",
				"seen_utc": "2026-07-06T02:02:02Z",
			},
		},
		Sessions: []Session{
			{Account: ".claude", Project: "work", Disp: "DONE", AgeMin: 1, hasAge: true},
			{Account: ".claude-july4-netra", Project: "work", Disp: "INFRA_AUTH", Action: "BLOCKED_AUTH", AgeMin: 22, hasAge: true, Last: "Not logged in"},
		},
	}
	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), reg)
	gem8 := find(rows, ".claude-gem8NEW-netra")
	if gem8 == nil || gem8.Kind != KindExcluded || !strings.Contains(gem8.Reason, "retired in fak accounts registry") {
		t.Fatalf("gem8NEW row = %+v, want excluded by fak accounts registry", gem8)
	}
	july4 := find(rows, ".claude-july4-netra")
	if july4 == nil || !derefBool(july4.Available) || derefStr(july4.StatusSource) != "identity-peer" {
		t.Fatalf("july4 row = %+v, want available via identity-peer", july4)
	}
	if derefInt(july4.LiveSessions) != 0 || derefInt(july4.AuthBlockedSessions) != 1 {
		t.Fatalf("july4 sessions live/auth = %d/%d, want 0/1",
			derefInt(july4.LiveSessions), derefInt(july4.AuthBlockedSessions))
	}
	resolved := Resolve(rows, home, ResolveRequest{WorkKind: "engineering", Product: "claude"}, DefaultPolicy())
	if !resolved.OK || resolved.Account != ".claude-july4-netra" {
		t.Fatalf("resolved = %+v, want july4-netra", resolved)
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

func TestCanServeFalseBlocksSwitcherEvenWithoutLoginStatus(t *testing.T) {
	rows := []Account{{
		Dir:       "C:/Users/u/.claude-stale",
		Product:   "claude",
		Account:   ".claude-stale",
		Tag:       "stale",
		Kind:      KindWorker,
		Reason:    "real offered account",
		ModelTier: intp(1),
		Available: boolp(true),
		CanServe:  boolp(false),
	}}

	annotated := Annotate(rows, Registry{})
	got := annotated[0]
	if derefBool(got.Available) || !derefBool(got.Blocked) ||
		derefStr(got.BlockKind) != "auth" || derefStr(got.BlockReason) == "" {
		t.Fatalf("annotated row = %+v, want auth-blocked by can_serve=false", got)
	}
	if len(Available(rows)) != 0 || len(Available(annotated)) != 0 {
		t.Fatalf("can_serve=false row was offered: raw=%+v annotated=%+v", Available(rows), Available(annotated))
	}

	route := RouteAccount(rows, "ship feature", "engineering", false, false, "claude", DefaultPolicy())
	if route.OK || len(route.BlockedTargetAccounts) != 1 ||
		route.BlockedTargetAccounts[0].CanServe == nil || *route.BlockedTargetAccounts[0].CanServe {
		t.Fatalf("route = %+v, want blocked target with can_serve=false", route)
	}

	resolved := Resolve(rows, t.TempDir(), ResolveRequest{Pin: "stale"}, DefaultPolicy())
	if resolved.OK || !strings.Contains(resolved.Reason, "blocked") || resolved.BlockReason == "" {
		t.Fatalf("resolve = %+v, want pinned account blocked by can_serve=false", resolved)
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
	discovered := Discover(home, cfg, DefaultPolicy())
	before := find(discovered, "opencode-glm")
	rows := Annotate(discovered, reg)
	after := find(rows, "opencode-glm")
	if before.LoginStatus == nil || before.CanServe == nil || after.LoginStatus == nil || after.CanServe == nil {
		t.Fatalf("opencode readiness lost across annotation: before=%v/%v after=%v/%v", before.LoginStatus, before.CanServe, after.LoginStatus, after.CanServe)
	}

	workerKeys := []string{
		"dir", "product", "account", "tag", "kind", "reason", "notes",
		"discovery_source", "root_state",
		"model_tier", "model", "small_model", "model_effort", "agent", "profile_source", "route_weight",
		"account_uuid", "login_email", "org_uuid", "org_type", "plan",
		"tag_login_match", "identity_peers", "identity_role", "login_status", "can_serve",
		"available", "blocked", "block_kind", "block_reason", "reset", "weekly", "throttled",
		"active_sessions", "live_sessions", "auth_blocked_sessions", "status_source", "registry_age_min",
	}
	nonAccountKeys := []string{
		"dir", "product", "account", "tag", "kind", "reason", "notes",
		"discovery_source", "root_state",
		"available", "blocked", "block_kind", "block_reason", "reset", "weekly", "throttled",
		"active_sessions", "live_sessions", "auth_blocked_sessions", "status_source", "registry_age_min",
	}
	opencodeWorkerKeys := []string{
		"dir", "product", "account", "tag", "kind", "reason", "notes",
		"discovery_source", "root_state",
		"model_tier", "model", "small_model", "model_effort", "agent", "profile_source", "route_weight",
		"login_status", "can_serve",
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
	if r.LoginStatus == nil || *r.LoginStatus != "ready" || r.CanServe == nil || !*r.CanServe {
		t.Errorf("pin default login_status/can_serve = %v/%v want ready/true", r.LoginStatus, r.CanServe)
	}

	// pin a throttled account -> blocked
	rb := Resolve(rows, home, ResolveRequest{Pin: "gem8"}, pol)
	if rb.OK || !strings.Contains(rb.Reason, "blocked") {
		t.Errorf("pin gem8: ok=%v reason=%q (want blocked)", rb.OK, rb.Reason)
	}
	if rb.LoginStatus == nil || *rb.LoginStatus != "ready" || rb.CanServe == nil || !*rb.CanServe {
		t.Errorf("pin blocked gem8 login_status/can_serve = %v/%v want ready/true (usage-blocked, not logged out)",
			rb.LoginStatus, rb.CanServe)
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

	// route gardening -> tier 2 target; opencode-glm (tier 2) is available but its
	// one live session fills the opencode session cap, so the route falls back up
	// to tier 1 instead of overfilling the glm seat.
	rg := Resolve(rows, home, ResolveRequest{WorkKind: "gardening"}, pol)
	if !rg.OK || rg.Account != ".claude" {
		t.Errorf("route gardening: ok=%v account=%q want tier-1 fallback .claude (glm seat at session cap)", rg.OK, rg.Account)
	}
}

func TestResolveSurfacesLoginBlockedTarget(t *testing.T) {
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
	got := Resolve(rows, home, ResolveRequest{Pin: "needslogin"}, DefaultPolicy())
	if got.OK || got.LoginStatus == nil || *got.LoginStatus != "needs_login" ||
		got.CanServe == nil || *got.CanServe {
		t.Fatalf("resolve needslogin = %+v, want blocked with login_status=needs_login can_serve=false", got)
	}
	if !strings.Contains(got.BlockReason, "no live credentials") {
		t.Fatalf("block_reason = %q, want no live credentials", got.BlockReason)
	}
}

func TestAllocateWaveGrantsDistinctPoolsAndUnderfills(t *testing.T) {
	rows := []Account{
		{Dir: "C:/seats/a", Product: "claude", Account: ".claude-seat-a", Tag: "seat-a", Kind: KindWorker, Reason: "real offered account", AccountUUID: strp("acct-a"), ModelTier: intp(1), Available: boolp(true)},
		{Dir: "C:/seats/b", Product: "claude", Account: ".claude-seat-b", Tag: "seat-b", Kind: KindWorker, Reason: "real offered account", AccountUUID: strp("acct-b"), ModelTier: intp(1), Available: boolp(true)},
		{Dir: "C:/seats/c", Product: "claude", Account: ".claude-seat-c", Tag: "seat-c", Kind: KindWorker, Reason: "real offered account", AccountUUID: strp("acct-c"), ModelTier: intp(1), Available: boolp(true)},
		{Dir: "C:/seats/blocked", Product: "claude", Account: ".claude-blocked", Tag: "blocked", Kind: KindWorker, Reason: "real offered account", AccountUUID: strp("acct-blocked"), ModelTier: intp(1), Available: boolp(false), BlockReason: strp("usage limit")},
	}
	wave := AllocateWave(rows, WaveRequest{Count: 3, WorkKind: "engineering", Product: "claude"}, DefaultPolicy())
	if !wave.OK || wave.Requested != 3 || wave.Granted != 3 || wave.Shortfall != 0 ||
		wave.DistinctPools != 3 || wave.Size != 3 || wave.WaveID == "" {
		t.Fatalf("wave = %+v, want 3 slots on 3 independent identities", wave)
	}
	if len(wave.Lanes) != 3 || wave.Lanes[0].Rank != 0 || wave.Lanes[1].Rank != 1 ||
		wave.Lanes[2].Rank != 2 || wave.Lanes[0].WaveID != wave.WaveID || wave.Lanes[2].Size != 3 {
		t.Fatalf("lane membership = %+v, want rank-stamped shared wave", wave.Lanes)
	}
	for i, lane := range wave.Lanes {
		if lane.SessionSlot != 1 || lane.SessionCap != 1 {
			t.Fatalf("lane %d = %+v, want one slot on an independent Claude pool", i, lane)
		}
	}
	if len(wave.BlockedTargetAccounts) != 1 || wave.BlockedTargetAccounts[0].Tag != "blocked" {
		t.Fatalf("blocked target accounts = %+v, want blocked identity", wave.BlockedTargetAccounts)
	}
}

func TestAllocateWaveSubtractsLiveLeaseSlots(t *testing.T) {
	rows := []Account{
		{Dir: "C:/seats/a", Product: "claude", Account: ".claude-seat-a", Tag: "seat-a", Kind: KindWorker, Reason: "real offered account", AccountUUID: strp("acct-a"), ModelTier: intp(1), Available: boolp(true)},
		{Dir: "C:/seats/b", Product: "claude", Account: ".claude-seat-b", Tag: "seat-b", Kind: KindWorker, Reason: "real offered account", AccountUUID: strp("acct-b"), ModelTier: intp(1), Available: boolp(true)},
		{Dir: "C:/seats/c", Product: "claude", Account: ".claude-seat-c", Tag: "seat-c", Kind: KindWorker, Reason: "real offered account", AccountUUID: strp("acct-c"), ModelTier: intp(1), Available: boolp(true)},
		{Dir: "C:/seats/d", Product: "claude", Account: ".claude-seat-d", Tag: "seat-d", Kind: KindWorker, Reason: "real offered account", AccountUUID: strp("acct-d"), ModelTier: intp(1), Available: boolp(true)},
	}
	leases := []Lease{
		{Worker: "resolve-1", Tag: "seat-a", Dir: "C:/seats/a"},
		{Worker: "resolve-2", Tag: "seat-b", Dir: "C:/seats/b"},
		{Worker: "resolve-3", Tag: "seat-c", Dir: "C:/seats/c"},
	}
	wave := AllocateWave(rows, WaveRequest{Count: 4, WorkKind: "engineering", Product: "claude", Leases: leases}, DefaultPolicy())
	if !wave.OK || wave.Granted != 1 || wave.Shortfall != 3 || len(wave.Lanes) != 1 {
		t.Fatalf("wave = %+v, want only the fourth identity free", wave)
	}
	got := wave.Lanes[0]
	if got.Tag != "seat-d" || got.SessionSlot != 1 || got.SessionCap != 1 {
		t.Fatalf("lane = %+v, want seat-d slot/cap 1/1", got)
	}
}

func TestAllocateWaveSkipsCanServeFalse(t *testing.T) {
	rows := []Account{{
		Dir:       "C:/Users/u/.claude-stale",
		Product:   "claude",
		Account:   ".claude-stale",
		Tag:       "stale",
		Kind:      KindWorker,
		Reason:    "real offered account",
		ModelTier: intp(1),
		Available: boolp(true),
		CanServe:  boolp(false),
	}}

	wave := AllocateWave(rows, WaveRequest{Count: 1, WorkKind: "engineering", Product: "claude"}, DefaultPolicy())
	if wave.OK || wave.Granted != 0 || len(wave.Lanes) != 0 || len(wave.BlockedTargetAccounts) != 1 {
		t.Fatalf("wave = %+v, want no lane and one blocked target", wave)
	}
	if wave.BlockedTargetAccounts[0].CanServe == nil || *wave.BlockedTargetAccounts[0].CanServe {
		t.Fatalf("blocked target can_serve = %+v, want false", wave.BlockedTargetAccounts[0])
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
		t.Errorf("total_seats = %d want 3 (two Claude identities + one opencode slot; dup collapsed)", pool.TotalSeats)
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
	if glmSeat == nil || glmSeat.State != "leased" || len(glmSeat.Workers) != 1 ||
		glmSeat.SessionCap != 1 || glmSeat.FreeSlots != 0 || glmSeat.LeasedSlots != 1 {
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

func TestResetParsingUsesLosAngelesHint(t *testing.T) {
	// 19:00 UTC is noon in Los Angeles on July 3, 2026. A reset string that says
	// "2pm (America/Los_Angeles)" is still two hours away, even though bare 2pm UTC
	// would already be expired on a UTC-anchored process.
	now := mustParse(t, "2026-07-03T19:00:00Z")
	if got := resetIsFuture("2pm (America/Los_Angeles)", now); got == nil || !*got {
		t.Fatalf("2pm LA at noon LA should be future; got %v", got)
	}
	if got := resetIsFuture("10am (America/Los_Angeles)", now); got == nil || *got {
		t.Fatalf("10am LA at noon LA should be expired; got %v", got)
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
