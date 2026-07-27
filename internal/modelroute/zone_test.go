package modelroute

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestZonesPartitionTheKnownKinds is the totality witness for the placement
// vocabulary: EVERY kind in the closed ProviderKind set maps to exactly one VALID
// zone, and the three zones each actually claim a kind. A kind added without a
// ZoneOfKind arm would silently fall through to ZoneVendor; this test does not
// catch that by itself, so it is paired with the explicit expectation table below.
func TestZonesPartitionTheKnownKinds(t *testing.T) {
	want := map[ProviderKind]PlacementZone{
		KindLocal:           ZoneDevice,
		KindFleet:           ZoneFleet,
		KindOpenAI:          ZoneVendor,
		KindOpenAIResponses: ZoneVendor,
		KindAnthropic:       ZoneVendor,
		KindGemini:          ZoneVendor,
		KindXAI:             ZoneVendor,
		KindDeepSeek:        ZoneVendor,
	}
	for k, wantZone := range want {
		if !knownKind(k) {
			t.Fatalf("kind %q is in the zone table but not in the closed kind set", k)
		}
		got := ZoneOfKind(k)
		if got != wantZone {
			t.Fatalf("ZoneOfKind(%q) = %q, want %q", k, got, wantZone)
		}
		if !got.Valid() {
			t.Fatalf("ZoneOfKind(%q) = %q which is not a valid zone", k, got)
		}
	}
	// Every kind the closed set knows must appear above — a new kind must make an
	// explicit zone choice, never inherit ZoneVendor by omission.
	for _, k := range []ProviderKind{KindOpenAI, KindOpenAIResponses, KindAnthropic, KindGemini, KindXAI, KindDeepSeek, KindLocal, KindFleet} {
		if _, ok := want[k]; !ok {
			t.Fatalf("kind %q is in the closed set but has no declared zone in this test — declare it", k)
		}
	}
}

// TestZoneLadderIsOrderedCheapestFirst pins the escalation order the whole design
// rests on: device before fleet before vendor. An escalation that compared zones
// any other way would send routine work to a frontier API.
func TestZoneLadderIsOrderedCheapestFirst(t *testing.T) {
	ladder := Zones()
	if len(ladder) != 3 {
		t.Fatalf("Zones() returned %d zones, want 3", len(ladder))
	}
	for i := 1; i < len(ladder); i++ {
		if ladder[i-1].Rank() >= ladder[i].Rank() {
			t.Fatalf("ladder is not strictly ascending at %d: %q(rank %d) then %q(rank %d)",
				i, ladder[i-1], ladder[i-1].Rank(), ladder[i], ladder[i].Rank())
		}
	}
	if ladder[0] != ZoneDevice || ladder[2] != ZoneVendor {
		t.Fatalf("ladder must run device -> ... -> vendor, got %v", ladder)
	}
	// An UNKNOWN zone must rank above vendor: an unrecognized placement is never
	// mistaken for the cheap rung.
	if got := PlacementZone("mystery").Rank(); got <= ZoneVendor.Rank() {
		t.Fatalf("unknown zone rank = %d, must be strictly above vendor (%d)", got, ZoneVendor.Rank())
	}
}

// TestSelfHostedCountsDeviceAndFleetOnly pins the predicate the token-economics
// goal is measured against. Getting this wrong in either direction misreports the
// headline number: counting vendor tokens as self-hosted inflates the saving,
// dropping fleet tokens erases the entire reason to own the hardware.
func TestSelfHostedCountsDeviceAndFleetOnly(t *testing.T) {
	cases := map[PlacementZone]bool{
		ZoneDevice:               true,
		ZoneFleet:                true,
		ZoneVendor:               false,
		PlacementZone("garbage"): false, // fail-closed: unattributable is not a saving
	}
	for z, want := range cases {
		if got := z.SelfHosted(); got != want {
			t.Fatalf("zone %q SelfHosted() = %v, want %v", z, got, want)
		}
	}
}

// TestOnBoxIsNarrowerThanSelfHosted is the security-shaped half of the vocabulary:
// owning the hardware (SelfHosted) must NOT imply the bytes stayed on the machine
// (OnBox). If ZoneFleet ever answered true to OnBox, declaring a company rack
// would silently buy a tenant-scoped payload a trip off the box — the exact
// fail-OPEN this zone split exists to avoid.
func TestOnBoxIsNarrowerThanSelfHosted(t *testing.T) {
	if !ZoneFleet.SelfHosted() {
		t.Fatal("ZoneFleet must be self-hosted (the org owns the silicon)")
	}
	if ZoneFleet.OnBox() {
		t.Fatal("ZoneFleet must NOT be OnBox — org-owned hardware is still off the calling machine; " +
			"answering true here would exempt a fleet route from the residency floor")
	}
	if !ZoneDevice.OnBox() {
		t.Fatal("ZoneDevice must be OnBox")
	}
	if ZoneVendor.OnBox() || ZoneVendor.SelfHosted() {
		t.Fatal("ZoneVendor is neither on-box nor self-hosted")
	}
}

// TestZoneOfRouteReadsTheLeadingPrefix pins the parse-back every downstream
// consumer of abi.ToolCall.Engine depends on: the LEADING family decides, in all
// four accepted shapes, and anything unplaceable falls closed to vendor.
func TestZoneOfRouteReadsTheLeadingPrefix(t *testing.T) {
	cases := []struct {
		route string
		want  PlacementZone
	}{
		{"", ZoneDevice},                      // unset Engine => the in-kernel default
		{"inkernel", ZoneDevice},              // bare family
		{"local:box/llama3.2", ZoneDevice},    // family:instance
		{"local-gpu", ZoneDevice},             // family-suffix
		{"local/llama", ZoneDevice},           // family/path
		{"on-device:0", ZoneDevice},           //
		{"mock", ZoneDevice},                  //
		{"cassette:replay", ZoneDevice},       //
		{"fleet:gpu07/glm-5.2", ZoneFleet},    // the new middle rung
		{"FLEET:gpu07/glm-5.2", ZoneFleet},    // case-insensitive, like the floor
		{"  fleet:gpu07/x  ", ZoneFleet},      // trimmed, like the floor
		{"anthropic:work/claude", ZoneVendor}, //
		{"openai:acct/gpt-5.5", ZoneVendor},   //
		{"deepseek:acct/v4", ZoneVendor},      //
		{"something-unknown", ZoneVendor},     // fail-closed
		{"fleetwood:acct/mac", ZoneVendor},    // NOT a fleet prefix — no delimiter after the family
		{"notlocal:acct/m", ZoneVendor},       // suffix match must not count
	}
	for _, c := range cases {
		if got := ZoneOfRoute(c.route); got != c.want {
			t.Fatalf("ZoneOfRoute(%q) = %q, want %q", c.route, got, c.want)
		}
		if want := !c.want.OnBox(); IsRemoteRoute(c.route) != want {
			t.Fatalf("IsRemoteRoute(%q) = %v, want %v (it must be exactly !ZoneOfRoute().OnBox())",
				c.route, IsRemoteRoute(c.route), want)
		}
	}
}

// TestEngineRouteCarriesTheZoneBackOut is the round-trip that makes the zone
// survive the kernel boundary: for every kind, the route string a Target stamps
// into abi.ToolCall.Engine must parse back to that Target's own zone. Without
// this, a usage ledger reading only the Engine string could not attribute a token
// to the right rung.
func TestEngineRouteCarriesTheZoneBackOut(t *testing.T) {
	for _, k := range []ProviderKind{KindLocal, KindFleet, KindOpenAI, KindOpenAIResponses, KindAnthropic, KindGemini, KindXAI, KindDeepSeek} {
		tg := Target{Kind: k, Account: "acct", UpstreamModel: "the-model"}
		route := tg.EngineRoute()
		if got, want := ZoneOfRoute(route), tg.Zone(); got != want {
			t.Fatalf("kind %q: EngineRoute()=%q parses back to zone %q but Target.Zone()=%q", k, route, got, want)
		}
		// And the locality answer must still agree with the floor's contract.
		if got, want := IsRemoteRoute(route), tg.Remote(); got != want {
			t.Fatalf("kind %q: IsRemoteRoute(%q)=%v but Target.Remote()=%v", k, route, got, want)
		}
	}
}

// TestFleetTargetIsRemoteToTheFloor is the fail-closed boundary of this slice,
// stated as an executable claim. Adding the fleet zone must change NOTHING about
// what the residency floor denies: an org-operated host is off-box until an
// operator-declared trust boundary says otherwise, which is a separate change.
func TestFleetTargetIsRemoteToTheFloor(t *testing.T) {
	tg := Target{Kind: KindFleet, Account: "gpu07", UpstreamModel: "glm-5.2"}
	if tg.Local() {
		t.Fatal("a fleet target must not report Local() — it is off-box")
	}
	if !tg.Remote() {
		t.Fatal("a fleet target must report Remote() so the residency floor still denies a sensitive payload routed to it")
	}
	if !tg.Zone().SelfHosted() {
		t.Fatal("a fleet target must still be attributable as self-hosted")
	}
	if got := tg.EngineRoute(); !strings.HasPrefix(got, string(ZoneFleet)+":") {
		t.Fatalf("a fleet target must lead with %q:, got %q", ZoneFleet, got)
	}
}

// TestFleetAccountValidation pins the invariants that keep the zones DISJOINT: a
// fleet account needs an explicit, parseable, http(s), NON-loopback endpoint, and
// unlike a vendor account it does not need a credential.
func TestFleetAccountValidation(t *testing.T) {
	fleet := func(a Account) Roster {
		return Roster{Accounts: []Account{a}, Default: a.ID}
	}
	ok := Account{ID: "gpu07", Kind: KindFleet, BaseURL: "http://gpu-07.corp.internal:8000/v1"}
	if err := fleet(ok).Validate(); err != nil {
		t.Fatalf("a well-formed credential-free fleet account must validate: %v", err)
	}
	withCred := ok
	withCred.CredEnv = "CORP_INFER_TOKEN"
	if err := fleet(withCred).Validate(); err != nil {
		t.Fatalf("a fleet account MAY carry a credential: %v", err)
	}

	bad := []struct {
		name string
		acct Account
		want string
	}{
		{"no base_url", Account{ID: "gpu07", Kind: KindFleet}, "needs a base_url"},
		{"loopback base_url", Account{ID: "gpu07", Kind: KindFleet, BaseURL: "http://127.0.0.1:8000/v1"}, "loopback"},
		{"localhost base_url", Account{ID: "gpu07", Kind: KindFleet, BaseURL: "http://localhost:8000/v1"}, "loopback"},
		{"no host", Account{ID: "gpu07", Kind: KindFleet, BaseURL: "/v1/chat"}, "naming a host"},
		{"wrong scheme", Account{ID: "gpu07", Kind: KindFleet, BaseURL: "ftp://gpu-07.corp.internal/v1"}, "http or https"},
		{"pasted secret in cred_env", Account{ID: "gpu07", Kind: KindFleet, BaseURL: "http://gpu-07.corp.internal:8000/v1", CredEnv: "sk-ant-abc123"}, "not an env-var name"},
	}
	for _, c := range bad {
		err := fleet(c.acct).Validate()
		if err == nil {
			t.Fatalf("%s: expected a validation error, got nil", c.name)
		}
		if !strings.Contains(err.Error(), c.want) {
			t.Fatalf("%s: error %q does not mention %q", c.name, err, c.want)
		}
	}
}

// TestFleetAccountResolvesToItsOwnEndpoint pins that a fleet account round-trips
// through the roster: KindBaseURL has no public default for it, so Resolve must
// carry the account's explicit endpoint rather than an empty string.
func TestFleetAccountResolvesToItsOwnEndpoint(t *testing.T) {
	if got := KindBaseURL(KindFleet); got != "" {
		t.Fatalf("KindFleet must have no public default base URL, got %q", got)
	}
	r := Roster{
		Accounts: []Account{
			{ID: "laptop", Kind: KindLocal, BaseURL: "http://127.0.0.1:11434/v1"},
			{ID: "gpu07", Kind: KindFleet, BaseURL: "http://gpu-07.corp.internal:8000/v1"},
			{ID: "frontier", Kind: KindAnthropic, CredEnv: "ANTHROPIC_API_KEY"},
		},
		Bindings: []Binding{
			{Model: "small", Account: "laptop", UpstreamModel: "qwen3.6-4b"},
			{Model: "mid", Account: "gpu07", UpstreamModel: "glm-5.2"},
			{Model: "large", Account: "frontier", UpstreamModel: "claude-opus-5"},
		},
		Default: "laptop",
	}
	if err := r.Validate(); err != nil {
		t.Fatalf("the three-zone roster must validate: %v", err)
	}
	want := map[string]PlacementZone{"small": ZoneDevice, "mid": ZoneFleet, "large": ZoneVendor}
	for model, zone := range want {
		tg, err := r.Resolve(model)
		if err != nil {
			t.Fatalf("resolve %q: %v", model, err)
		}
		if got := tg.Zone(); got != zone {
			t.Fatalf("model %q resolved to zone %q, want %q", model, got, zone)
		}
		if got := ZoneOfRoute(tg.EngineRoute()); got != zone {
			t.Fatalf("model %q engine route %q parses to zone %q, want %q", model, tg.EngineRoute(), got, zone)
		}
	}
	mid, err := r.Resolve("mid")
	if err != nil {
		t.Fatalf("resolve mid: %v", err)
	}
	if mid.BaseURL != "http://gpu-07.corp.internal:8000/v1" {
		t.Fatalf("fleet target base URL = %q, want the account's explicit endpoint", mid.BaseURL)
	}
}

// TestShippedExampleRosterCarriesAllThreeZones is the end-to-end witness that the
// deployment shape this vocabulary exists for is actually EXPRESSIBLE in a file a
// user is handed: examples/model-accounts.example.json must parse, Validate, and
// resolve a model in each of the three zones. Before KindFleet existed the middle
// rung could not be written down at all — a company-operated server had to either
// lie as a loopback account (refused by Validate) or masquerade as a vendor
// account (admitted, but then indistinguishable from a third-party API in every
// downstream record). This test is what keeps that regression visible.
//
// It also pins the two claims the routing seam rests on, per zone: the route
// string round-trips back to the same zone, and Remote() — what the residency
// floor enforces — stays true for everything but the on-box rung.
func TestShippedExampleRosterCarriesAllThreeZones(t *testing.T) {
	path := filepath.Join("..", "..", "examples", "model-accounts.example.json")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	r, err := ParseRoster(b) // ParseRoster validates
	if err != nil {
		t.Fatalf("the shipped example roster must parse and validate: %v", err)
	}
	want := []struct {
		model      string
		zone       PlacementZone
		selfHosted bool
		remote     bool
	}{
		{"zone-device", ZoneDevice, true, false},
		{"zone-fleet", ZoneFleet, true, true},
		{"zone-fleet-agentic", ZoneFleet, true, true},
		{"medium", ZoneVendor, false, true},
		{"large", ZoneVendor, false, true},
	}
	seen := map[PlacementZone]bool{}
	for _, c := range want {
		tg, err := r.Resolve(c.model)
		if err != nil {
			t.Fatalf("resolve %q: %v", c.model, err)
		}
		if got := tg.Zone(); got != c.zone {
			t.Fatalf("model %q zone = %q, want %q", c.model, got, c.zone)
		}
		if got := tg.Zone().SelfHosted(); got != c.selfHosted {
			t.Fatalf("model %q SelfHosted = %v, want %v", c.model, got, c.selfHosted)
		}
		if got := tg.Remote(); got != c.remote {
			t.Fatalf("model %q Remote = %v, want %v (the residency floor reads this)", c.model, got, c.remote)
		}
		if got := ZoneOfRoute(tg.EngineRoute()); got != c.zone {
			t.Fatalf("model %q route %q parses back to zone %q, want %q", c.model, tg.EngineRoute(), got, c.zone)
		}
		seen[c.zone] = true
	}
	for _, z := range Zones() {
		if !seen[z] {
			t.Fatalf("the example roster must demonstrate zone %q — the whole point is that all three rungs are writable", z)
		}
	}
}
