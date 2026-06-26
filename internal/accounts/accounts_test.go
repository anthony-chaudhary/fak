package accounts

import (
	"os"
	"path/filepath"
	"testing"
)

// fixture is a small well-formed registry: gem8 is the live default, q is tombstoned
// and rehomes to gem8, and a two-hop chain (old -> mid -> gem8) exercises transitivity.
func fixture() Registry {
	return Registry{
		Version: RegistryVersion,
		Homes: []Home{
			{Name: "gem8-seat", Dir: "/h/.claude-gem8-seat", Default: true},
			{Name: "day24-seat", Dir: "/h/.claude-day24-seat", Status: StatusActive},
			{Name: "q", Status: StatusTombstoned, RehomeTo: "gem8-seat"},
			{Name: "old", Status: StatusTombstoned, RehomeTo: "mid"},
			{Name: "mid", Status: StatusTombstoned, RehomeTo: "gem8-seat"},
		},
	}
}

func TestValidateAccepts(t *testing.T) {
	if err := fixture().Validate(); err != nil {
		t.Fatalf("fixture should validate: %v", err)
	}
}

func TestResolveActivePassesThrough(t *testing.T) {
	h, chain, err := fixture().Resolve("gem8-seat")
	if err != nil {
		t.Fatalf("resolve active: %v", err)
	}
	if h.Name != "gem8-seat" || len(chain) != 0 {
		t.Fatalf("active resolve = %q chain=%v, want gem8-seat with empty chain", h.Name, chain)
	}
}

func TestResolveTombstoneRehomes(t *testing.T) {
	h, chain, err := fixture().Resolve("q")
	if err != nil {
		t.Fatalf("resolve tombstone: %v", err)
	}
	if h.Name != "gem8-seat" {
		t.Fatalf("q resolved to %q, want gem8-seat", h.Name)
	}
	if len(chain) != 1 || chain[0] != "q" {
		t.Fatalf("rehome chain = %v, want [q]", chain)
	}
}

func TestResolveTombstoneTransitive(t *testing.T) {
	h, chain, err := fixture().Resolve("old")
	if err != nil {
		t.Fatalf("resolve transitive: %v", err)
	}
	if h.Name != "gem8-seat" {
		t.Fatalf("old resolved to %q, want gem8-seat", h.Name)
	}
	if len(chain) != 2 || chain[0] != "old" || chain[1] != "mid" {
		t.Fatalf("chain = %v, want [old mid]", chain)
	}
}

func TestResolveUnknown(t *testing.T) {
	if _, _, err := fixture().Resolve("nope"); err == nil {
		t.Fatalf("resolving an unknown name should fail")
	}
}

func TestResolveCycleFailsLoud(t *testing.T) {
	r := Registry{Homes: []Home{
		{Name: "a", Status: StatusTombstoned, RehomeTo: "b"},
		{Name: "b", Status: StatusTombstoned, RehomeTo: "a"},
	}}
	if _, _, err := r.Resolve("a"); err == nil {
		t.Fatalf("a rehome cycle should fail, not loop forever")
	}
}

func TestValidateRejections(t *testing.T) {
	cases := map[string]Registry{
		"no homes":              {Homes: nil},
		"empty name":            {Homes: []Home{{Name: "", Dir: "/d"}}},
		"duplicate name":        {Homes: []Home{{Name: "a", Dir: "/d"}, {Name: "a", Dir: "/e"}}},
		"unknown status":        {Homes: []Home{{Name: "a", Dir: "/d", Status: "retired"}}},
		"active without dir":    {Homes: []Home{{Name: "a"}}},
		"tombstone no rehome":   {Homes: []Home{{Name: "a", Dir: "/d"}, {Name: "b", Status: StatusTombstoned}}},
		"tombstone self rehome": {Homes: []Home{{Name: "a", Status: StatusTombstoned, RehomeTo: "a"}}},
		"dangling rehome":       {Homes: []Home{{Name: "a", Dir: "/d"}, {Name: "b", Status: StatusTombstoned, RehomeTo: "ghost"}}},
		"two defaults":          {Homes: []Home{{Name: "a", Dir: "/d", Default: true}, {Name: "b", Dir: "/e", Default: true}}},
		"default tombstoned":    {Homes: []Home{{Name: "a", Dir: "/d"}, {Name: "b", Status: StatusTombstoned, Default: true, RehomeTo: "a"}}},
		"foreign version":        {Version: "some-other-roster/v1", Homes: []Home{{Name: "a", Dir: "/d"}}},
		"rehome cycle":          {Homes: []Home{{Name: "a", Status: StatusTombstoned, RehomeTo: "b"}, {Name: "b", Status: StatusTombstoned, RehomeTo: "a"}}},
	}
	for name, r := range cases {
		if err := r.Validate(); err == nil {
			t.Errorf("Validate(%s) should fail, got nil", name)
		}
	}
}

func TestDefault(t *testing.T) {
	h, ok := fixture().Default()
	if !ok || h.Name != "gem8-seat" {
		t.Fatalf("Default = %q,%v, want gem8-seat,true", h.Name, ok)
	}
	if _, ok := (Registry{Homes: []Home{{Name: "a", Dir: "/d"}}}).Default(); ok {
		t.Fatalf("no default marked should report ok=false")
	}
}

func TestNameLie(t *testing.T) {
	cases := []struct {
		name  string
		email string
		lie   bool
	}{
		{"q-seat", "gem8@example.test", true},                                 // named q, logged in as gem8
		{"gem8-seat", "gem8@example.test", false},                             // suffix ignored
		{"jack-barker-claude-seat", "jack.barker.claude@example.test", false}, // separators normalize
		{"alex-agent-seat", "alex.agent@example.test", false},                 // all name tokens present in email
		{"day24-seat", "gem5@example.test", true},                             // different person
		{"default", "gem8@example.test", false},                               // role name, never a lie
		{"whatever", "", false},                                               // no identity -> never a lie
	}
	for _, c := range cases {
		h := Home{Name: c.name, Identity: Identity{Email: c.email}}
		if got := h.NameLie(); got != c.lie {
			t.Errorf("NameLie(name=%q email=%q) = %v, want %v", c.name, c.email, got, c.lie)
		}
	}
}

func TestJSONRoundTrip(t *testing.T) {
	r := fixture()
	got, err := ParseRegistry(r.JSON())
	if err != nil {
		t.Fatalf("round-trip parse: %v", err)
	}
	if len(got.Homes) != len(r.Homes) {
		t.Fatalf("round-trip homes = %d, want %d", len(got.Homes), len(r.Homes))
	}
	if _, _, err := got.Resolve("old"); err != nil {
		t.Fatalf("round-tripped registry should still resolve: %v", err)
	}
}

// TestSameFamilyVersionAccepted proves the version check is family-based: a later minor/
// major in the same fak-config-homes/* family validates (so additive, omitempty schema
// growth never strands an existing file), while only a FOREIGN family is refused.
func TestSameFamilyVersionAccepted(t *testing.T) {
	r := fixture()
	r.Version = "fak-config-homes/v2"
	if err := r.Validate(); err != nil {
		t.Fatalf("same-family v2 should validate: %v", err)
	}
}

// TestEnabledOrDefault pins the default-true semantics of the optional Enabled pointer: a
// nil pointer (the field omitted, as in every v1 registry) reads as enabled; only an
// explicit false disables. This is what keeps an old registry's accounts fully enrolled.
func TestEnabledOrDefault(t *testing.T) {
	tru, fal := true, false
	if (Home{}).EnabledOrDefault() != true {
		t.Fatalf("nil Enabled should read as enabled (default true)")
	}
	if (Home{Enabled: &tru}).EnabledOrDefault() != true {
		t.Fatalf("explicit true should read as enabled")
	}
	if (Home{Enabled: &fal}).EnabledOrDefault() != false {
		t.Fatalf("explicit false should read as disabled")
	}
}

// TestPolicyFieldsRoundTrip proves the new policy attributes (Enabled/Reserved/
// ChromeProfile) survive a JSON round-trip, and that a registry WITHOUT them (the v1 shape)
// still parses under the new code — the additive-growth guarantee.
func TestPolicyFieldsRoundTrip(t *testing.T) {
	disabled := false
	r := Registry{
		Homes: []Home{
			{Name: "live", Dir: "/h/.claude-live", Default: true, Reserved: true, ChromeProfile: "Profile 9"},
			{Name: "off", Dir: "/h/.claude-off", Enabled: &disabled},
		},
	}
	got, err := ParseRegistry(r.JSON())
	if err != nil {
		t.Fatalf("policy round-trip parse: %v", err)
	}
	if !got.Homes[0].Reserved || got.Homes[0].ChromeProfile != "Profile 9" {
		t.Fatalf("reserved/chrome_profile lost in round-trip: %+v", got.Homes[0])
	}
	if got.Homes[0].EnabledOrDefault() != true {
		t.Fatalf("home with no enabled field should read enabled after round-trip")
	}
	if got.Homes[1].EnabledOrDefault() != false {
		t.Fatalf("home with enabled:false should read disabled after round-trip")
	}

	// A literal v1-shaped registry (no new keys at all) must still parse.
	v1 := []byte(`{"version":"fak-config-homes/v1","homes":[{"name":"a","dir":"/d"}]}`)
	if _, err := ParseRegistry(v1); err != nil {
		t.Fatalf("v1-shaped registry should parse under new code: %v", err)
	}
}

// TestSaveRegistryRoundTrips proves SaveRegistry writes a file that LoadRegistry reads back
// to an equivalent registry, and that it refuses to persist an invalid one.
func TestSaveRegistryRoundTrips(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "registry.json")
	r := fixture()
	if err := SaveRegistry(path, r); err != nil {
		t.Fatalf("SaveRegistry: %v", err)
	}
	got, err := LoadRegistry(path)
	if err != nil {
		t.Fatalf("LoadRegistry after save: %v", err)
	}
	if len(got.Homes) != len(r.Homes) {
		t.Fatalf("saved homes = %d, want %d", len(got.Homes), len(r.Homes))
	}
	// An invalid registry (no homes) must be refused, leaving no file behind at a fresh path.
	bad := filepath.Join(dir, "bad.json")
	if err := SaveRegistry(bad, Registry{}); err == nil {
		t.Fatalf("SaveRegistry should refuse an invalid registry")
	}
	if _, err := os.Stat(bad); !os.IsNotExist(err) {
		t.Fatalf("refused registry should not have been written")
	}
}

// serveFixture has disk-derived Identity populated so Serve's creds checks have meaning:
// gem8 is the serveable default, throttled is active-but-logged-out, q is tombstoned.
func serveFixture() Registry {
	live := Identity{Email: "x@y", Exists: true, HasCreds: true}
	noCreds := Identity{Email: "x@y", Exists: true, HasCreds: false}
	return Registry{
		Homes: []Home{
			{Name: "gem8-seat", Dir: "/h/.claude-gem8-seat", Default: true, Identity: live},
			{Name: "throttled", Dir: "/h/.claude-throttled", Identity: noCreds},                // active but can't serve
			{Name: "q", Status: StatusTombstoned, RehomeTo: "gem8-seat"},                       // tombstoned -> gem8
			{Name: "stale", Dir: "/h/.claude-stale", Identity: noCreds, RehomeTo: "gem8-seat"}, // no creds, explicit rehome
		},
	}
}

func TestServeReturnsServeableAsIs(t *testing.T) {
	h, chain, err := serveFixture().Serve("gem8-seat")
	if err != nil {
		t.Fatalf("serve: %v", err)
	}
	if h.Name != "gem8-seat" || len(chain) != 0 {
		t.Fatalf("serveable seat = %q chain=%v, want gem8-seat with no rehome", h.Name, chain)
	}
}

func TestServeRehomesTombstone(t *testing.T) {
	h, chain, err := serveFixture().Serve("q")
	if err != nil {
		t.Fatalf("serve q: %v", err)
	}
	if h.Name != "gem8-seat" || len(chain) != 1 || chain[0] != "q" {
		t.Fatalf("serve q = %q chain=%v, want gem8-seat via [q]", h.Name, chain)
	}
}

func TestServeRehomesUnserveableToDefault(t *testing.T) {
	// "throttled" is active but has no creds, and no explicit rehome_to -> falls forward
	// to the registry default rather than pinning to a seat that can't serve.
	h, chain, err := serveFixture().Serve("throttled")
	if err != nil {
		t.Fatalf("serve throttled: %v", err)
	}
	if h.Name != "gem8-seat" || len(chain) != 1 || chain[0] != "throttled" {
		t.Fatalf("serve throttled = %q chain=%v, want gem8-seat via [throttled]", h.Name, chain)
	}
}

func TestServeRehomesUnserveableViaExplicit(t *testing.T) {
	h, _, err := serveFixture().Serve("stale")
	if err != nil || h.Name != "gem8-seat" {
		t.Fatalf("serve stale = %q,%v, want gem8-seat", h.Name, err)
	}
}

func TestServeUnknownFailsLoud(t *testing.T) {
	if _, _, err := serveFixture().Serve("ghost"); err == nil {
		t.Fatalf("serving an unknown name should fail")
	}
}

func TestPlanPullsSharedHistory(t *testing.T) {
	r := fixture()
	r.SharedHistory = filepath.Join("/store")

	// Active seat: nothing to pull.
	p, err := r.Plan("gem8-seat")
	if err != nil {
		t.Fatalf("plan active: %v", err)
	}
	if p.Into.Name != "gem8-seat" || len(p.From) != 0 {
		t.Fatalf("active plan = into %q from %v, want gem8-seat with no pulls", p.Into.Name, p.From)
	}

	// One-hop tombstone: pull q's bundle into gem8.
	p, err = r.Plan("q")
	if err != nil {
		t.Fatalf("plan q: %v", err)
	}
	if p.Into.Name != "gem8-seat" {
		t.Fatalf("plan q into %q, want gem8-seat", p.Into.Name)
	}
	if len(p.From) != 1 || p.From[0] != filepath.Join("/store", "q") {
		t.Fatalf("plan q from = %v, want [%s]", p.From, filepath.Join("/store", "q"))
	}

	// Transitive: pull both tombstone bundles, nearest first.
	p, err = r.Plan("old")
	if err != nil {
		t.Fatalf("plan old: %v", err)
	}
	want := []string{filepath.Join("/store", "old"), filepath.Join("/store", "mid")}
	if len(p.From) != 2 || p.From[0] != want[0] || p.From[1] != want[1] {
		t.Fatalf("plan old from = %v, want %v", p.From, want)
	}
}

func TestPlanHistoryAtOverride(t *testing.T) {
	r := Registry{
		SharedHistory: "/store",
		Homes: []Home{
			{Name: "gem8-seat", Dir: "/h/.claude-gem8-seat", Default: true},
			{Name: "q", Status: StatusTombstoned, RehomeTo: "gem8-seat", HistoryAt: "q-archive-2026-06-25"},
		},
	}
	p, err := r.Plan("q")
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	if len(p.From) != 1 || p.From[0] != filepath.Join("/store", "q-archive-2026-06-25") {
		t.Fatalf("plan from = %v, want history_at bundle", p.From)
	}
}

func TestPlanNoStoreFailsLoud(t *testing.T) {
	r := fixture() // tombstones present, but no SharedHistory set
	if _, err := r.Plan("q"); err == nil {
		t.Fatalf("planning a tombstone pull with no shared_history store should fail")
	}
}

func TestParseRejectsUnknownField(t *testing.T) {
	if _, err := ParseRegistry([]byte(`{"homes":[{"name":"a","dir":"/d","bogus":1}]}`)); err == nil {
		t.Fatalf("unknown field should be rejected")
	}
}

func TestDiscover(t *testing.T) {
	home := t.TempDir()
	// A config home logged in as gem8 (has .claude.json + creds).
	mk := func(dir, email, uuid string, creds, projects bool) {
		full := filepath.Join(home, dir)
		if err := os.MkdirAll(full, 0o755); err != nil {
			t.Fatal(err)
		}
		if email != "" {
			body := `{"oauthAccount":{"emailAddress":"` + email + `","accountUuid":"` + uuid + `"},"numStartups":3}`
			if err := os.WriteFile(filepath.Join(full, ".claude.json"), []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		if creds {
			if err := os.WriteFile(filepath.Join(full, ".credentials.json"), []byte(`{}`), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		if projects {
			if err := os.MkdirAll(filepath.Join(full, "projects"), 0o755); err != nil {
				t.Fatal(err)
			}
		}
	}
	mk(".claude", "q@example.test", "uuid-q", true, true) // default home
	mk(".claude-gem8-seat", "gem8@example.test", "uuid-8", true, true)
	mk(".claude-q-seat", "gem8@example.test", "uuid-8", true, true) // the lie
	mk(".claude-account-backups", "", "", false, false)             // NOT a config home
	mk(".claude-monitor", "", "", false, false)                     // NOT a config home

	homes, err := Discover(home)
	if err != nil {
		t.Fatalf("discover: %v", err)
	}
	byName := map[string]Home{}
	for _, h := range homes {
		byName[h.Name] = h
	}
	if _, ok := byName["account-backups"]; ok {
		t.Errorf("account-backups should be skipped (not a config home)")
	}
	if _, ok := byName["monitor"]; ok {
		t.Errorf("monitor should be skipped (not a config home)")
	}
	if d, ok := byName["default"]; !ok || d.Identity.Email != "q@example.test" {
		t.Errorf("default home identity = %+v, want q@", d.Identity)
	}
	qn, ok := byName["q-seat"]
	if !ok {
		t.Fatalf("q-seat not discovered")
	}
	if qn.Identity.Email != "gem8@example.test" {
		t.Errorf("q-seat identity = %q, want gem8@ (disk truth)", qn.Identity.Email)
	}
	if !qn.NameLie() {
		t.Errorf("q-seat (logged in as gem8) should be flagged a name-lie")
	}
	if !qn.Identity.HasCreds || !qn.Identity.Exists {
		t.Errorf("q-seat should have creds + exist: %+v", qn.Identity)
	}
}

// TestMergeDiscovered proves the regenerator is non-destructive: authored policy fields on a
// known home survive a rescan, identity is refreshed from disk, a brand-new config dir is
// added as an active seat, and a registry entry whose dir vanished is kept (not silently
// dropped).
func TestMergeDiscovered(t *testing.T) {
	home := t.TempDir()
	mk := func(dir, email, uuid string) {
		full := filepath.Join(home, dir)
		if err := os.MkdirAll(full, 0o755); err != nil {
			t.Fatal(err)
		}
		body := `{"oauthAccount":{"emailAddress":"` + email + `","accountUuid":"` + uuid + `"}}`
		if err := os.WriteFile(filepath.Join(full, ".claude.json"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(full, ".credentials.json"), []byte(`{}`), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	mk(".claude-keep-seat", "keep@example.test", "uuid-keep") // known home, has authored policy
	mk(".claude-new-seat", "new@example.test", "uuid-new")    // brand new, registry doesn't know it

	reserved := true
	base := Registry{
		Homes: []Home{
			// Known home: authored Reserved + ChromeProfile must survive; identity gets refreshed.
			{Name: "keep-seat", Dir: filepath.Join(home, ".claude-keep-seat"), Reserved: reserved, ChromeProfile: "Profile 9"},
			// A tombstone whose dir never existed on disk — must be kept verbatim.
			{Name: "gone", Status: StatusTombstoned, RehomeTo: "keep-seat"},
		},
	}
	merged, err := base.MergeDiscovered(home)
	if err != nil {
		t.Fatalf("MergeDiscovered: %v", err)
	}
	byName := map[string]Home{}
	for _, h := range merged.Homes {
		byName[h.Name] = h
	}
	keep, ok := byName["keep-seat"]
	if !ok {
		t.Fatalf("keep-seat missing after merge")
	}
	if !keep.Reserved || keep.ChromeProfile != "Profile 9" {
		t.Errorf("authored policy lost on merge: %+v", keep)
	}
	if keep.Identity.Email != "keep@example.test" {
		t.Errorf("identity not refreshed from disk: %q", keep.Identity.Email)
	}
	nw, ok := byName["new-seat"]
	if !ok {
		t.Fatalf("new-seat (brand-new dir) should have been added")
	}
	if nw.Identity.Email != "new@example.test" || !nw.EnabledOrDefault() {
		t.Errorf("new seat should be active with disk identity: %+v", nw)
	}
	if _, ok := byName["gone"]; !ok {
		t.Errorf("vanished-dir tombstone should be kept, not dropped")
	}
	// The merged registry must still be valid (gone resolves to keep-seat).
	if err := merged.Validate(); err != nil {
		t.Errorf("merged registry should validate: %v", err)
	}
}
