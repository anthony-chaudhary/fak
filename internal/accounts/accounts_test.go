package accounts

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// fixture is a small well-formed registry: gem8 is the live default, q is tombstoned
// and rehomes to gem8, and a two-hop chain (old -> mid -> gem8) exercises transitivity.
func fixture() Registry {
	return Registry{
		Version: RegistryVersion,
		Roles:   map[string]string{RoleAnchor: "gem8-seat"},
		Homes: []Home{
			{Name: "gem8-seat", Dir: "/h/.claude-gem8-seat"},
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
		"role names ghost":      {Homes: []Home{{Name: "a", Dir: "/d"}}, Roles: map[string]string{RoleActive: "ghost"}},
		"role on tombstone":     {Homes: []Home{{Name: "a", Dir: "/d"}, {Name: "b", Status: StatusTombstoned, RehomeTo: "a"}}, Roles: map[string]string{RoleAnchor: "b"}},
		"foreign version":       {Version: "some-other-roster/v1", Homes: []Home{{Name: "a", Dir: "/d"}}},
		"rehome cycle":          {Homes: []Home{{Name: "a", Status: StatusTombstoned, RehomeTo: "b"}, {Name: "b", Status: StatusTombstoned, RehomeTo: "a"}}},
	}
	for name, r := range cases {
		if err := r.Validate(); err == nil {
			t.Errorf("Validate(%s) should fail, got nil", name)
		}
	}
}

func TestDefault(t *testing.T) {
	// Default() is the compatibility shim for the rehome ANCHOR role.
	h, ok := fixture().Default()
	if !ok || h.Name != "gem8-seat" {
		t.Fatalf("Default = %q,%v, want gem8-seat,true", h.Name, ok)
	}
	if _, ok := (Registry{Homes: []Home{{Name: "a", Dir: "/d"}}}).Default(); ok {
		t.Fatalf("no anchor role set should report ok=false")
	}
}

// TestActiveMemoryDir is the affordance toward #1313: ActiveMemoryDir yields the ACTIVE
// seat's resolved per-workspace agent-memory dir (so a recall can read the active store
// instead of the hardcoded ~/.claude default), and fails CLOSED — ("", false) — when no
// active seat is set, never a guessed path.
func TestActiveMemoryDir(t *testing.T) {
	r := Registry{
		Roles: map[string]string{RoleActive: "day27-seat", RoleAnchor: "gem8-seat"},
		Homes: []Home{
			{Name: "gem8-seat", Dir: "/h/.claude-gem8-seat"},
			{Name: "day27-seat", Dir: "/h/.claude-day27-seat"},
		},
	}
	// A forward-slash workspace path is stable across OSes (filepath.Clean leaves it as-is
	// on POSIX, and on Windows it normalizes to backslashes that the slug then collapses to
	// the same '-' — so the per-component slug agrees). Every non-alphanumeric collapses to
	// one '-': "/work/fak" -> "-work-fak".
	const ws = "/work/fak"
	got, ok := r.ActiveMemoryDir(ws)
	if !ok {
		t.Fatalf("ActiveMemoryDir(active seat set) ok=false, want true")
	}
	want := filepath.Join("/h/.claude-day27-seat", "projects", projectSlug(ws), "memory")
	if got != want {
		t.Fatalf("ActiveMemoryDir = %q, want %q", got, want)
	}
	// The active seat's dir, not the anchor's, is the base.
	if !strings.Contains(got, "day27-seat") || strings.Contains(got, "gem8-seat") {
		t.Fatalf("ActiveMemoryDir = %q, want it rooted at the ACTIVE (day27) seat, not the anchor", got)
	}

	// Fail-closed: no active role -> ("", false), never a guessed path.
	noActive := Registry{
		Roles: map[string]string{RoleAnchor: "gem8-seat"},
		Homes: []Home{{Name: "gem8-seat", Dir: "/h/.claude-gem8-seat"}},
	}
	if got, ok := noActive.ActiveMemoryDir(ws); ok || got != "" {
		t.Fatalf("ActiveMemoryDir(no active seat) = %q,%v, want \"\",false", got, ok)
	}
}

// TestProjectSlug pins the Claude Code session-store key derivation: every non-alphanumeric
// rune of the cleaned path collapses to a single '-', matching the fleet's Python tools so
// the Go and Python views agree on which projects/<key> dir a workspace owns.
func TestProjectSlug(t *testing.T) {
	cases := map[string]string{
		"/work/fak":  "-work-fak",
		"/home/u/p":  "-home-u-p",
		"a.b c@d":    "a-b-c-d",
		"/a//b/../c": "-a-c", // Clean folds "//" and ".." before slugging
	}
	for in, want := range cases {
		if got := projectSlug(in); got != want {
			t.Errorf("projectSlug(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestRolesAreIndependent is the regression this whole change exists for: the launch ACTIVE
// seat and the rehome ANCHOR are separate roles, so pointing one never disturbs the other.
// Under the old single `default: true` boolean, setting the active account silently moved the
// rehome anchor; that conflation is what is fixed here.
func TestRolesAreIndependent(t *testing.T) {
	r := Registry{
		Roles: map[string]string{RoleAnchor: "gem8-seat", RoleActive: "day24-seat"},
		Homes: []Home{
			{Name: "gem8-seat", Dir: "/h/.claude-gem8-seat"},
			{Name: "day24-seat", Dir: "/h/.claude-day24-seat"},
			{Name: "day27-seat", Dir: "/h/.claude-day27-seat"},
		},
	}
	if err := r.Validate(); err != nil {
		t.Fatalf("two-role registry should validate: %v", err)
	}
	// Rotate the ACTIVE seat to day27 — the way `set-role active` does.
	r.Roles[RoleActive] = "day27-seat"
	if err := r.Validate(); err != nil {
		t.Fatalf("after rotating active: %v", err)
	}
	act, _ := r.Role(RoleActive)
	if act.Name != "day27-seat" {
		t.Fatalf("active = %q, want day27-seat", act.Name)
	}
	// The anchor MUST be untouched — the entire point of separating the roles.
	anchor, ok := r.Role(RoleAnchor)
	if !ok || anchor.Name != "gem8-seat" {
		t.Fatalf("anchor moved when only active was set: anchor=%q,%v want gem8-seat", anchor.Name, ok)
	}
}

// TestMigrateLegacyDefaultIsIdempotent proves a pre-roles `default: true` folds into RoleAnchor
// exactly once and re-running migrate (or re-parsing) is a no-op.
func TestMigrateLegacyDefaultIsIdempotent(t *testing.T) {
	r := Registry{Homes: []Home{
		{Name: "anchor-seat", Dir: "/h/.claude-anchor", Default: true},
		{Name: "other", Dir: "/h/.claude-other"},
	}}
	r.migrate()
	if got := r.Roles[RoleAnchor]; got != "anchor-seat" {
		t.Fatalf("migrate: RoleAnchor = %q, want anchor-seat", got)
	}
	if r.Homes[0].Default {
		t.Fatalf("migrate should clear the legacy default bool")
	}
	// An explicit anchor role wins over a stray legacy bool (no clobber on re-migrate).
	r.Roles[RoleAnchor] = "other"
	r.Homes[0].Default = true // simulate a stray legacy flag reappearing
	r.migrate()
	if got := r.Roles[RoleAnchor]; got != "other" {
		t.Fatalf("migrate must not clobber an explicit anchor: got %q, want other", got)
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
		{"gem8NEW-netra", "gem8@example.test", false},                         // restore suffix is operational, not identity
		{"gem8NEW-netra", "day26@example.test", true},                         // still catches a restored dir logged into another account
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
	// The legacy `default: true` on "live" migrated to RoleAnchor and the bool cleared, so the
	// registry has ONE representation of the anchor after a round-trip.
	if anchor, ok := got.Role(RoleAnchor); !ok || anchor.Name != "live" {
		t.Fatalf("legacy default:true should migrate to RoleAnchor=live, got %q,%v", anchor.Name, ok)
	}
	if got.Homes[0].Default {
		t.Fatalf("legacy default:true should be cleared after migration, still set on %+v", got.Homes[0])
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
// gem8 is the serveable anchor (Serve's fall-forward target), throttled is
// active-but-logged-out, q is tombstoned.
func serveFixture() Registry {
	live := Identity{Email: "gem8@example.test", Exists: true, HasCreds: true}
	noCreds := Identity{Email: "throttled@example.test", Exists: true, HasCreds: false}
	return Registry{
		Roles: map[string]string{RoleAnchor: "gem8-seat"},
		Homes: []Home{
			{Name: "gem8-seat", Dir: "/h/.claude-gem8-seat", Identity: live},
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

// cooldownServeFixture is serveFixture's cooldown-aware sibling (#4673): every live seat
// has creds and a distinct account UUID, so the ONLY thing that can make one unserveable
// is the CooldownStore overlay. "gone" is the tombstoned seat whose rehome chain points
// at "sink" — the throttled-sink shape from the live audit — and "anchor-seat" is the
// fall-forward anchor.
func cooldownServeFixture() Registry {
	id := func(name, uuid string) Identity {
		return Identity{Email: name + "@example.test", AccountUUID: uuid, Exists: true, HasCreds: true}
	}
	return Registry{
		Roles: map[string]string{RoleAnchor: "anchor-seat"},
		Homes: []Home{
			{Name: "anchor-seat", Dir: "/h/.claude-anchor-seat", Identity: id("anchor", "u-anchor")},
			{Name: "sink", Dir: "/h/.claude-sink", Identity: id("sink", "u-sink")},
			{Name: "gone", Status: StatusTombstoned, RehomeTo: "sink"},
		},
	}
}

// TestServeAtWalksPastCooledDownSeat is the #4673 repro: a tombstone whose rehome chain
// points at a throttled seat. Cooldown-blind Serve stops ON the throttled sink (the bug
// shape — documented here so the baseline is explicit); ServeAt with the store walks
// PAST it to the serving anchor.
func TestServeAtWalksPastCooledDownSeat(t *testing.T) {
	r := cooldownServeFixture()
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	cd := &CooldownStore{entries: map[string]CooldownEntry{}}
	cd.Cool(UUIDBucketKey("u-sink"), CooldownUsageLimit, "weekly limit", now, now.Add(2*time.Hour))

	h, _, err := r.Serve("gone")
	if err != nil || h.Name != "sink" {
		t.Fatalf("cooldown-blind Serve = %q,%v, want the throttled sink (the documented blind baseline)", h.Name, err)
	}

	h, chain, entry, err := r.ServeAt("gone", cd, now)
	if err != nil {
		t.Fatalf("ServeAt: %v", err)
	}
	if h.Name != "anchor-seat" {
		t.Fatalf("ServeAt landed on %q, want anchor-seat (must not stop on the throttled sink)", h.Name)
	}
	if strings.Join(chain, ",") != "gone,sink" {
		t.Fatalf("chain = %v, want [gone sink] (the cooled hop is walked past, so it is a hop)", chain)
	}
	if entry != nil {
		t.Fatalf("a serving seat was reachable, so the all-cooled entry must be nil, got %+v", entry)
	}
}

// TestServeAtRehomesCooledRequestedSeat covers the directly-pinned case: the requested
// seat itself is otherwise Ready but throttled, so it falls forward like any other
// unserveable seat instead of serving into the wall.
func TestServeAtRehomesCooledRequestedSeat(t *testing.T) {
	r := cooldownServeFixture()
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	cd := &CooldownStore{entries: map[string]CooldownEntry{}}
	cd.Cool(UUIDBucketKey("u-sink"), CooldownUsageLimit, "weekly limit", now, now.Add(2*time.Hour))

	h, chain, entry, err := r.ServeAt("sink", cd, now)
	if err != nil {
		t.Fatalf("ServeAt: %v", err)
	}
	if h.Name != "anchor-seat" || strings.Join(chain, ",") != "sink" || entry != nil {
		t.Fatalf("ServeAt(sink) = %q chain=%v entry=%v, want anchor-seat via [sink] with nil entry", h.Name, chain, entry)
	}
}

// TestServeAtNilStoreMatchesServe proves the DoD delegation contract: Serve IS
// ServeAt(name, nil, time.Time{}) — no overlay, byte-for-byte the historical behavior —
// across every serve shape the existing fixtures exercise (serveable as-is, tombstone
// rehome, unserveable fall-forward, explicit rehome, unknown name).
func TestServeAtNilStoreMatchesServe(t *testing.T) {
	r := serveFixture()
	for _, name := range []string{"gem8-seat", "q", "throttled", "stale", "ghost"} {
		sh, sc, serr := r.Serve(name)
		ah, ac, entry, aerr := r.ServeAt(name, nil, time.Time{})
		if entry != nil {
			t.Fatalf("ServeAt(%q, nil, zero) returned all-cooled entry %+v; a nil store applies no overlay", name, entry)
		}
		if sh.Name != ah.Name || strings.Join(sc, ",") != strings.Join(ac, ",") || (serr == nil) != (aerr == nil) {
			t.Fatalf("ServeAt(%q, nil, zero) = %q %v %v diverges from Serve = %q %v %v",
				name, ah.Name, ac, aerr, sh.Name, sc, serr)
		}
	}
}

// TestServeableAtMatchesCanServeAndCooldown pins serveableAt's contract: identical to
// CanServe with no store (or before/after the window), false exactly while the seat's
// account holds an active cooldown at now.
func TestServeableAtMatchesCanServeAndCooldown(t *testing.T) {
	ready := Home{Name: "sink", Dir: "/h/.claude-sink",
		Identity: Identity{Email: "sink@example.test", AccountUUID: "u-sink", Exists: true, HasCreds: true}}
	dead := Home{Name: "gone", Status: StatusTombstoned, RehomeTo: "sink"}
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	cd := &CooldownStore{entries: map[string]CooldownEntry{}}
	cd.Cool(UUIDBucketKey("u-sink"), CooldownUsageLimit, "weekly limit", now, now.Add(time.Hour))

	if got := serveableAt(ready, nil, time.Time{}); got != ready.CanServe() {
		t.Fatalf("serveableAt(ready, nil, zero) = %v, want CanServe() = %v", got, ready.CanServe())
	}
	if got := serveableAt(dead, cd, now); got != dead.CanServe() {
		t.Fatalf("serveableAt(dead, cd, now) = %v, want CanServe() = %v (cooldown never rescues a static failure)", got, dead.CanServe())
	}
	if serveableAt(ready, cd, now) {
		t.Fatalf("an active cooldown at now must make an otherwise-ready seat non-serveable")
	}
	if !serveableAt(ready, cd, now.Add(2*time.Hour)) {
		t.Fatalf("an elapsed cooldown must restore serveability with no manual action")
	}
}

// TestServeAtAllCooledDownResolvesSoonestReset pins the terminal case: when every
// reachable, otherwise-serveable seat is throttled, ServeAt neither hard-errors nor
// silently lands — it returns the soonest-reset seat WITH its cooldown entry as the
// explicit all-cooled/degraded signal.
func TestServeAtAllCooledDownResolvesSoonestReset(t *testing.T) {
	r := cooldownServeFixture()
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	cd := &CooldownStore{entries: map[string]CooldownEntry{}}
	cd.Cool(UUIDBucketKey("u-sink"), CooldownUsageLimit, "weekly limit", now, now.Add(2*time.Hour))
	cd.Cool(UUIDBucketKey("u-anchor"), CooldownUsageLimit, "weekly limit", now, now.Add(20*time.Minute))

	h, chain, entry, err := r.ServeAt("gone", cd, now)
	if err != nil {
		t.Fatalf("all-cooled chain must degrade, not hard-error: %v", err)
	}
	if h.Name != "anchor-seat" {
		t.Fatalf("all-cooled resolve landed on %q, want anchor-seat (soonest reset: 20m vs 2h)", h.Name)
	}
	if entry == nil {
		t.Fatalf("all reachable seats are cooled: the returned entry must be the explicit signal, got nil")
	}
	if !entry.ResetAt.Equal(now.Add(20 * time.Minute)) {
		t.Fatalf("entry.ResetAt = %v, want the soonest reset %v", entry.ResetAt, now.Add(20*time.Minute))
	}
	if strings.Join(chain, ",") != "gone,sink" {
		t.Fatalf("chain = %v, want [gone sink] (the hops that reached the served seat)", chain)
	}
}

// TestServeAtStructuralFailureStaysLoud pins that the degraded fallback never masks a
// genuinely broken registry: with no cooled candidate in the walk, an unresolvable chain
// still fails exactly as Serve always has.
func TestServeAtStructuralFailureStaysLoud(t *testing.T) {
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	cd := &CooldownStore{entries: map[string]CooldownEntry{}}
	r := Registry{Homes: []Home{
		{Name: "a", Status: StatusTombstoned, RehomeTo: "b"},
		{Name: "b", Status: StatusTombstoned, RehomeTo: "a"},
	}}
	if _, _, entry, err := r.ServeAt("a", cd, now); err == nil || entry != nil {
		t.Fatalf("a rehome cycle with no serveable seat must stay fail-loud, got entry=%v err=%v", entry, err)
	}
	if _, _, entry, err := r.ServeAt("ghost", cd, now); err == nil || entry != nil {
		t.Fatalf("an unknown name must stay fail-loud, got entry=%v err=%v", entry, err)
	}
}

// deadRolePairFixture is the shape that broke the `f` launcher on the maintainers' box:
// BOTH role seats are live-but-unserveable — the active seat's config dir had been
// re-logged into another account and the anchor needed a login — while ready seats sat in
// the pool. The role-only fall-forward bounced active->anchor->active and the walk died on
// its OWN cycle guard ("accounts: rehome cycle through ..."), so a bare `fak accounts
// launch` refused to start anything at all.
func deadRolePairFixture() Registry {
	ready := func(name, uuid string) Home {
		return Home{Name: name, Dir: "/h/.claude-" + name,
			Identity: Identity{Email: name + "@example.test", AccountUUID: uuid, Exists: true, HasCreds: true}}
	}
	// Live (never tombstoned) but cannot serve: the dir is there, the credentials are not.
	dead := func(name string) Home {
		return Home{Name: name, Dir: "/h/.claude-" + name, Identity: Identity{Exists: true}}
	}
	return Registry{
		Roles: map[string]string{RoleActive: "active-dead", RoleAnchor: "anchor-dead"},
		// pool-ready sits BETWEEN the two dead role seats so the sweep is proven to be a
		// registry-order scan of the whole roster, not a lucky first/last hit.
		Homes: []Home{dead("anchor-dead"), ready("pool-ready", "u-pool"), dead("active-dead")},
	}
}

// TestServeAtDeadRolePairFallsForwardToPool pins the fix: once both roles are exhausted the
// walk sweeps the POOL instead of dead-ending on its cycle guard.
func TestServeAtDeadRolePairFallsForwardToPool(t *testing.T) {
	r := deadRolePairFixture()

	h, chain, entry, err := r.ServeAt("active-dead", nil, time.Time{})
	if err != nil {
		t.Fatalf("a dead active+anchor pair must fall forward to a ready pool seat, got: %v", err)
	}
	if h.Name != "pool-ready" {
		t.Fatalf("served %q, want pool-ready", h.Name)
	}
	if entry != nil {
		t.Fatalf("nothing is cooled here: entry must be nil, got %v", entry)
	}
	if got := strings.Join(chain, ","); got != "active-dead,anchor-dead" {
		t.Fatalf("chain = %v, want [active-dead anchor-dead] (both refused role hops)", chain)
	}
	// Serve (the cooldown-blind wrapper the older callers still use) gets the same rescue.
	if h, _, err := r.Serve("active-dead"); err != nil || h.Name != "pool-ready" {
		t.Fatalf("Serve = %q,%v, want pool-ready,<nil>", h.Name, err)
	}
}

// TestServeAtDeadRolePairDegradesOntoCooledPool pins the next rung down: when the only
// pool seat is held back solely by an active cooldown, the walk still REACHES it, so the
// all-cooled terminal degrades onto it (with its entry) rather than refusing to resolve.
func TestServeAtDeadRolePairDegradesOntoCooledPool(t *testing.T) {
	r := deadRolePairFixture()
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	cd := &CooldownStore{entries: map[string]CooldownEntry{}}
	cd.Cool(UUIDBucketKey("u-pool"), CooldownUsageLimit, "weekly limit", now, now.Add(20*time.Minute))

	h, chain, entry, err := r.ServeAt("active-dead", cd, now)
	if err != nil {
		t.Fatalf("an all-cooled pool must degrade, not hard-error: %v", err)
	}
	if h.Name != "pool-ready" {
		t.Fatalf("served %q, want pool-ready", h.Name)
	}
	if entry == nil || !entry.ResetAt.Equal(now.Add(20*time.Minute)) {
		t.Fatalf("the cooled landing must carry its entry as the degraded signal, got %v", entry)
	}
	if got := strings.Join(chain, ","); got != "active-dead,anchor-dead" {
		t.Fatalf("chain = %v, want [active-dead anchor-dead]", chain)
	}
	// Once the window elapses the same registry resolves clean, with no manual action.
	if _, _, entry, err := r.ServeAt("active-dead", cd, now.Add(time.Hour)); err != nil || entry != nil {
		t.Fatalf("an elapsed cooldown must resolve clean, got entry=%v err=%v", entry, err)
	}
}

// TestServeAtEmptyPoolStaysLoud pins that the pool sweep is a rescue, not a silent landing:
// strip the one ready seat and the same walk fails loud again.
func TestServeAtEmptyPoolStaysLoud(t *testing.T) {
	r := deadRolePairFixture()
	r.Homes = slices.DeleteFunc(slices.Clone(r.Homes), func(h Home) bool { return h.Name == "pool-ready" })

	if h, _, entry, err := r.ServeAt("active-dead", nil, time.Time{}); err == nil || entry != nil {
		t.Fatalf("no seat can serve: must stay fail-loud, got seat=%q entry=%v err=%v", h.Name, entry, err)
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
			{Name: "gem8-seat", Dir: "/h/.claude-gem8-seat"},
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
	// An archived (`--archive`) seat: the rename keeps .claude.json + projects/ intact, so
	// the config-home markers alone would re-admit it. Discover must honor the `.DELETED`
	// tombstone and drop it — the deleted-seat-resurfaces-in-the-switcher regression.
	mk(".claude-gem8-seat.DELETED-2026-06-26", "gem8@example.test", "uuid-8", true, true)

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
	if _, ok := byName["gem8-seat.DELETED-2026-06-26"]; ok {
		t.Errorf("tombstoned (.DELETED) dir must not be discovered as a live seat")
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

// TestDeriveIdentityDefaultSeatFreshestStateFile pins the two-writer reality of the DEFAULT
// home: a bare `claude` writes the profile-root ~/.claude.json while an explicit
// CLAUDE_CONFIG_DIR=~/.claude launch writes ~/.claude/.claude.json. After a bare /login the
// in-dir copy is STALE — identity must follow the freshest file that names an account, and a
// named seat must never read the profile-root file at all.
func TestDeriveIdentityDefaultSeatFreshestStateFile(t *testing.T) {
	home := t.TempDir()
	def := filepath.Join(home, ".claude")
	if err := os.MkdirAll(def, 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(path, email, uuid string, mtime time.Time) {
		body := `{"oauthAccount":{"emailAddress":"` + email + `","accountUuid":"` + uuid + `"}}`
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(path, mtime, mtime); err != nil {
			t.Fatal(err)
		}
	}
	old := time.Now().Add(-time.Hour)
	inDir := filepath.Join(def, ".claude.json")
	root := filepath.Join(home, ".claude.json")

	// Bare /login: the profile-root file is fresher → its identity wins over the stale seed.
	write(inDir, "stale@example.test", "uuid-stale", old)
	write(root, "fresh@example.test", "uuid-fresh", old.Add(30*time.Minute))
	if id := DeriveIdentity(def); id.Email != "fresh@example.test" || id.AccountUUID != "uuid-fresh" {
		t.Errorf("default seat should follow the fresher profile-root state file, got %+v", id)
	}

	// Explicit CLAUDE_CONFIG_DIR login: the in-dir file is fresher → it wins.
	write(inDir, "indir@example.test", "uuid-indir", old.Add(45*time.Minute))
	if id := DeriveIdentity(def); id.Email != "indir@example.test" {
		t.Errorf("fresher in-dir state file should win, got %+v", id)
	}

	// A fresher file with NO oauthAccount never outranks one that names an account.
	if err := os.WriteFile(inDir, []byte(`{"numStartups":1}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if id := DeriveIdentity(def); id.Email != "fresh@example.test" {
		t.Errorf("identity-less in-dir file must fall back to the root identity, got %+v", id)
	}

	// A named seat never reads the profile-root file, even when it is fresher.
	named := filepath.Join(home, ".claude-worker-seat")
	if err := os.MkdirAll(named, 0o755); err != nil {
		t.Fatal(err)
	}
	write(filepath.Join(named, ".claude.json"), "worker@example.test", "uuid-worker", old)
	if id := DeriveIdentity(named); id.Email != "worker@example.test" {
		t.Errorf("named seat must ignore the profile-root state file, got %+v", id)
	}
}

func TestDeriveIdentityRejectsPlaceholderClaudeOAuthCreds(t *testing.T) {
	home := t.TempDir()
	dir := filepath.Join(home, ".claude-july4-netra")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(dir, ".claude.json"),
		[]byte(`{"oauthAccount":{"emailAddress":"july4@example.test","accountUuid":"uuid-july4"}}`),
		0o644,
	); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(dir, ".credentials.json"),
		[]byte(`{"claudeAiOauth":{"expiresAt":0,"scopes":["user:profile"],"subscriptionType":"max"}}`),
		0o644,
	); err != nil {
		t.Fatal(err)
	}

	id := DeriveIdentity(dir)
	if !id.Exists || id.Email != "july4@example.test" || id.AccountUUID != "uuid-july4" {
		t.Fatalf("identity = %+v, want july4 identity", id)
	}
	if id.HasCreds {
		t.Fatalf("placeholder .credentials.json without access/refresh token must not read as live creds: %+v", id)
	}

	if err := os.WriteFile(
		filepath.Join(dir, ".credentials.json"),
		[]byte(`{"claudeAiOauth":{"accessToken":"access","refreshToken":"refresh"}}`),
		0o644,
	); err != nil {
		t.Fatal(err)
	}
	if id := DeriveIdentity(dir); !id.HasCreds {
		t.Fatalf("credentials with access/refresh token should read as live creds: %+v", id)
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

// TestMergeDiscoveredSkipsTombstonedDir proves the regenerator (`fak accounts discover
// --write`) does NOT resurrect an archived seat: a leftover `.DELETED` config dir keeps its
// .claude.json + projects/ intact, but MergeDiscovered must treat it as absent, never fold
// it back into the registry as a brand-new active seat — the durable form of the
// deleted-seat-resurfaces regression.
func TestMergeDiscoveredSkipsTombstonedDir(t *testing.T) {
	home := t.TempDir()
	full := filepath.Join(home, ".claude-gem8-seat.DELETED-2026-06-26")
	if err := os.MkdirAll(filepath.Join(full, "projects"), 0o755); err != nil {
		t.Fatal(err)
	}
	body := `{"oauthAccount":{"emailAddress":"gem8@example.test","accountUuid":"uuid-8"}}`
	if err := os.WriteFile(filepath.Join(full, ".claude.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(full, ".credentials.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}

	base := Registry{Homes: []Home{{Name: "live-seat", Dir: filepath.Join(home, ".claude-live-seat")}}}
	merged, err := base.MergeDiscovered(home)
	if err != nil {
		t.Fatalf("MergeDiscovered: %v", err)
	}
	for _, h := range merged.Homes {
		if strings.Contains(strings.ToLower(h.Name), ".deleted") {
			t.Fatalf("archived .DELETED dir was resurrected into the registry as seat %q", h.Name)
		}
	}
}

// TestMergeDiscoveredSkipsTombstonedIdentity proves a retired account cannot resurface by
// re-logging into a NEW, differently-named dir with no `.DELETED` marker — the case the
// basename filter cannot see. The registry tombstones seat "gem8" (uuid-8); a live dir
// ".claude-gem8NEW" logged into the SAME uuid-8 must NOT be admitted as a fresh active seat.
// This is the identity-keyed form of the deleted-seat-resurfaces regression: the tombstone
// binds to the ACCOUNT identity, not to the mutable dir/registry name.
func TestMergeDiscoveredSkipsTombstonedIdentity(t *testing.T) {
	home := t.TempDir()
	// The retired account, re-logged-into a brand-new dir with no `.DELETED` suffix.
	full := filepath.Join(home, ".claude-gem8NEW")
	if err := os.MkdirAll(filepath.Join(full, "projects"), 0o755); err != nil {
		t.Fatal(err)
	}
	body := `{"oauthAccount":{"emailAddress":"gem8@example.test","accountUuid":"uuid-8"}}`
	if err := os.WriteFile(filepath.Join(full, ".claude.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(full, ".credentials.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}

	// A live seat plus a tombstone carrying the retired account's identity (uuid-8). The
	// tombstone's own dir is gone (archived elsewhere), so only its cached Identity remains.
	base := Registry{Homes: []Home{
		{Name: "live-seat", Dir: filepath.Join(home, ".claude-live-seat")},
		{
			Name: "gem8", Status: StatusTombstoned, RehomeTo: "live-seat",
			Identity: Identity{Email: "gem8@example.test", AccountUUID: "uuid-8"},
		},
	}}
	merged, err := base.MergeDiscovered(home)
	if err != nil {
		t.Fatalf("MergeDiscovered: %v", err)
	}
	for _, h := range merged.Homes {
		if h.Name == "gem8NEW" {
			t.Fatalf("re-login into a new dir resurrected tombstoned account uuid-8 as active seat %q", h.Name)
		}
		if h.Active() && h.Identity.AccountUUID == "uuid-8" {
			t.Fatalf("tombstoned account uuid-8 came back as active seat %q (dir %q)", h.Name, h.Dir)
		}
	}
	// The tombstone itself must be untouched, and the registry must still validate.
	if err := merged.Validate(); err != nil {
		t.Errorf("merged registry should validate: %v", err)
	}
}
