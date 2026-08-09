package accounts

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// Third-party endpoint seats: BaseURL + ExtraEnv.
//
// The gap these close: a seat was describable by exactly one variable
// (CLAUDE_CONFIG_DIR), so a vendor gateway that needs its own endpoint, model id,
// headers, and client bootstrap toggles could not be registered as a seat at all.
//
// No fixture here carries a real credential or a real vendor host — the registry is a
// plaintext file, and these tests are the ones asserting that property.
// ---------------------------------------------------------------------------

// thirdPartySeat is a reserved vendor-gateway seat shaped exactly like the real use case:
// held out of rotation, reachable by name, endpoint in base_url, the non-secret client
// toggles in extra_env, and the credential NOWHERE in the registry (api_key_env names the
// variable the operator exports).
func thirdPartySeat() Home {
	return Home{
		Name:      "vendor",
		Dir:       "/tmp/.claude-vendor",
		Status:    StatusActive,
		Reserved:  true,
		CredKind:  CredKindAPIKey,
		APIKeyEnv: "VENDOR_AUTH_TOKEN",
		BaseURL:   "https://gateway.example.com/serving-endpoints/anthropic",
		ExtraEnv: map[string]string{
			"ANTHROPIC_MODEL":                "vendor-claude-sonnet-5",
			"ANTHROPIC_DEFAULT_SONNET_MODEL": "vendor-claude-sonnet-5",
			"ANTHROPIC_CUSTOM_HEADERS":       "x-vendor-mode: true",
			"CLAUDE_CODE_USE_GATEWAY":        "1",
			"ENABLE_PROMPT_CACHING_1H":       "1",
			"ENABLE_TOOL_SEARCH":             "1",
		},
	}
}

func TestValidateExtraEnvAcceptsTheRealToggles(t *testing.T) {
	if err := ValidateExtraEnv(thirdPartySeat().ExtraEnv); err != nil {
		t.Fatalf("the client bootstrap toggles a vendor gateway needs were refused: %v", err)
	}
	// Every seat written before the field existed.
	if err := ValidateExtraEnv(nil); err != nil {
		t.Fatalf("nil extra_env refused: %v", err)
	}
	if err := ValidateExtraEnv(map[string]string{}); err != nil {
		t.Fatalf("empty extra_env refused: %v", err)
	}
}

// TestValidateExtraEnvRefusesCredentialShapedNames is the leak guard. `fak accounts list
// --json` prints the registry, so a secret stored here reaches ordinary operator output.
func TestValidateExtraEnvRefusesCredentialShapedNames(t *testing.T) {
	for _, key := range []string{
		"ANTHROPIC_AUTH_TOKEN", // the exact variable a vendor PAT would be pasted into
		"ANTHROPIC_API_KEY",
		"OPENAI_API_KEY",
		"VENDOR_SECRET",
		"DB_PASSWORD",
		"MY_CREDENTIAL",
		"GH_TOKEN",
		"SIGNING_KEY",
		"SESSION_ID",
		"anthropic_auth_token", // lowercase must not slip past
		"KEY",
	} {
		err := ValidateExtraEnv(map[string]string{key: "value-would-be-published"})
		if err == nil {
			t.Errorf("extra_env accepted credential-shaped name %q; a secret there lands in `accounts list --json`", key)
			continue
		}
		if !strings.Contains(err.Error(), key) {
			t.Errorf("error for %q does not name the offending key: %v", key, err)
		}
	}
	// And the near-misses that are NOT secrets must still be usable, or the check is a
	// nuisance that pushes operators to bypass it.
	for _, key := range []string{"KEYRING_PATH", "KEYMAP", "MONKEY_MODE", "ENABLE_TOOL_SEARCH"} {
		if err := ValidateExtraEnv(map[string]string{key: "1"}); err != nil {
			t.Errorf("extra_env refused non-secret name %q: %v", key, err)
		}
	}
}

// TestValidateExtraEnvRefusesFakOwnedKeys pins that a seat cannot seize the variables that
// make a launch a SEAT launch.
func TestValidateExtraEnvRefusesFakOwnedKeys(t *testing.T) {
	t.Run("CLAUDE_CONFIG_DIR", func(t *testing.T) {
		// The worst case: the seat runs under ANOTHER seat's credentials and history while
		// every log line still names this one.
		err := ValidateExtraEnv(map[string]string{"CLAUDE_CONFIG_DIR": "/tmp/.claude-other"})
		if err == nil {
			t.Fatal("extra_env was allowed to redirect CLAUDE_CONFIG_DIR to a different seat")
		}
	})
	t.Run("ANTHROPIC_BASE_URL", func(t *testing.T) {
		// Hidden here it would bypass the guarded-launch refusal, which keys off base_url.
		err := ValidateExtraEnv(map[string]string{"ANTHROPIC_BASE_URL": "https://gateway.example.com"})
		if err == nil {
			t.Fatal("extra_env set ANTHROPIC_BASE_URL; the guarded-launch refusal would not see the endpoint")
		}
		if !strings.Contains(err.Error(), "base_url") {
			t.Fatalf("error does not point at the right field: %v", err)
		}
	})
}

func TestValidateExtraEnvRefusesUnpassableNames(t *testing.T) {
	if err := ValidateExtraEnv(map[string]string{"": "v"}); err == nil {
		t.Error("empty variable name accepted")
	}
	if err := ValidateExtraEnv(map[string]string{"A=B": "v"}); err == nil {
		t.Error("name containing '=' accepted; it cannot be passed to a process unambiguously")
	}
	if err := ValidateExtraEnv(map[string]string{"A\x00B": "v"}); err == nil {
		t.Error("name containing NUL accepted")
	}
}

// TestEnvOverlayIsDeterministic matters because the overlay is fed to the spawn broker,
// which records the launch env: an unstable order would make every audit line differ.
func TestEnvOverlayIsDeterministic(t *testing.T) {
	h := thirdPartySeat()
	want := []string{
		"ANTHROPIC_BASE_URL=https://gateway.example.com/serving-endpoints/anthropic",
		"ANTHROPIC_CUSTOM_HEADERS=x-vendor-mode: true",
		"ANTHROPIC_DEFAULT_SONNET_MODEL=vendor-claude-sonnet-5",
		"ANTHROPIC_MODEL=vendor-claude-sonnet-5",
		"CLAUDE_CODE_USE_GATEWAY=1",
		"ENABLE_PROMPT_CACHING_1H=1",
		"ENABLE_TOOL_SEARCH=1",
	}
	for i := range 8 {
		if got := h.EnvOverlay(); !reflect.DeepEqual(got, want) {
			t.Fatalf("overlay unstable or wrong on pass %d:\n got %q\nwant %q", i, got, want)
		}
	}
}

// TestEnvOverlayEmptyForFirstPartySeat is the compatibility half: every existing seat must
// produce the historical launch env (os.Environ + CLAUDE_CONFIG_DIR) exactly.
func TestEnvOverlayEmptyForFirstPartySeat(t *testing.T) {
	h := Home{Name: "sub", Dir: "/tmp/.claude-sub", Status: StatusActive}
	if got := h.EnvOverlay(); got != nil {
		t.Fatalf("a seat with no endpoint and no extra env contributed %q to the launch env", got)
	}
	if h.ThirdParty() {
		t.Fatal("a seat with no base_url reported itself third-party")
	}
	if got := h.EnvOverlayKeys(); len(got) != 0 {
		t.Fatalf("EnvOverlayKeys = %q, want none", got)
	}
}

func TestEnvOverlayCarriesEndpointAlone(t *testing.T) {
	// base_url with no extra_env is a legitimate minimal third-party seat.
	h := Home{Name: "v", BaseURL: "https://gateway.example.com"}
	if got, want := h.EnvOverlay(), []string{"ANTHROPIC_BASE_URL=https://gateway.example.com"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("overlay = %q, want %q", got, want)
	}
	if !h.ThirdParty() {
		t.Fatal("a seat with base_url did not report itself third-party")
	}
}

// TestEnvOverlayKeysNamesEveryVariableWithoutValues backs the launch plan's promise: an
// operator can see WHAT the seat changed and still paste the plan into an issue, because a
// value (ANTHROPIC_CUSTOM_HEADERS carries arbitrary header text) never appears.
func TestEnvOverlayKeysNamesEveryVariableWithoutValues(t *testing.T) {
	h := thirdPartySeat()
	keys := h.EnvOverlayKeys()
	if len(keys) != len(h.ExtraEnv)+1 {
		t.Fatalf("EnvOverlayKeys = %q, want one name per overlaid variable plus the endpoint", keys)
	}
	for i := 1; i < len(keys); i++ {
		if keys[i-1] > keys[i] {
			t.Fatalf("EnvOverlayKeys not sorted: %q", keys)
		}
	}
	joined := strings.Join(keys, ",")
	for _, secretish := range []string{"vendor-claude-sonnet-5", "x-vendor-mode", "gateway.example.com"} {
		if strings.Contains(joined, secretish) {
			t.Errorf("EnvOverlayKeys leaked a VALUE (%q): %q", secretish, joined)
		}
	}
}

// ---------------------------------------------------------------------------
// Registry integration
// ---------------------------------------------------------------------------

// TestThirdPartySeatRoundTripsThroughParseRegistry is load-bearing because ParseRegistry
// sets DisallowUnknownFields: until the Go fields exist, a registry naming them fails to
// parse at all, so this is what makes the JSON spelling usable by an operator.
func TestThirdPartySeatRoundTripsThroughParseRegistry(t *testing.T) {
	in := Registry{Version: RegistryVersion, Homes: []Home{thirdPartySeat()}}
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(b), `"base_url"`) || !strings.Contains(string(b), `"extra_env"`) {
		t.Fatalf("fields not serialized: %s", b)
	}
	out, err := ParseRegistry(b)
	if err != nil {
		t.Fatalf("ParseRegistry rejected a third-party seat: %v", err)
	}
	got := out.Homes[0]
	if got.BaseURL != in.Homes[0].BaseURL {
		t.Errorf("base_url = %q, want %q", got.BaseURL, in.Homes[0].BaseURL)
	}
	if !reflect.DeepEqual(got.ExtraEnv, in.Homes[0].ExtraEnv) {
		t.Errorf("extra_env = %v, want %v", got.ExtraEnv, in.Homes[0].ExtraEnv)
	}
	if !got.Reserved {
		t.Error("reserved lost in the round trip")
	}
}

// TestUnsetEndpointFieldsAreOmitted keeps every pre-existing registry byte-identical after a
// load/save cycle, so adding these fields does not rewrite unrelated operators' files.
func TestUnsetEndpointFieldsAreOmitted(t *testing.T) {
	b, err := json.Marshal(Home{Name: "sub", Dir: "/tmp/.claude-sub", Status: StatusActive})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, key := range []string{"base_url", "extra_env"} {
		if strings.Contains(string(b), key) {
			t.Errorf("unset field emitted %q: %s", key, b)
		}
	}
}

// TestMergeDiscoveredPreservesEndpointFields pins the claim the Home doc makes: these are
// AUTHORED fields, and a disk rescan (which owns identity, not policy) must not drop them.
// Without this, `fak accounts` would silently de-configure the seat on any refresh.
func TestMergeDiscoveredPreservesEndpointFields(t *testing.T) {
	home := t.TempDir()
	dir := filepath.Join(home, ".claude-vendor")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	// A config home the scan will recognize.
	if err := os.WriteFile(filepath.Join(dir, ".credentials.json"), []byte(`{}`), 0o600); err != nil {
		t.Fatalf("write credentials: %v", err)
	}

	seat := thirdPartySeat()
	seat.Dir = dir
	reg := Registry{Version: RegistryVersion, Homes: []Home{seat}}

	merged, err := reg.MergeDiscovered(home)
	if err != nil {
		t.Fatalf("MergeDiscovered: %v", err)
	}
	var found bool
	for _, h := range merged.Homes {
		if h.Name != "vendor" {
			continue
		}
		found = true
		if h.BaseURL != seat.BaseURL {
			t.Errorf("rescan dropped base_url: %q", h.BaseURL)
		}
		if !reflect.DeepEqual(h.ExtraEnv, seat.ExtraEnv) {
			t.Errorf("rescan dropped extra_env: %v", h.ExtraEnv)
		}
		if !h.Reserved {
			t.Error("rescan dropped reserved")
		}
	}
	if !found {
		t.Fatalf("seat vanished from the merged registry: %+v", merged.Homes)
	}
}

// TestReservedThirdPartySeatIsOutOfRotationYetResolvableByName is the user-facing
// requirement in one assertion: registered and callable ON PURPOSE, never picked FOR you.
//
// It also documents that this needed NO new field — Reserved plus the default
// avoid_reserved policy already meant exactly this — so the endpoint fields above are the
// only genuinely missing piece.
func TestReservedThirdPartySeatIsOutOfRotationYetResolvableByName(t *testing.T) {
	ordinary := Home{Name: "sub", Dir: "/tmp/.claude-sub", Status: StatusActive,
		Identity: Identity{Email: "sub@example.com", AccountUUID: "uuid-a", HasCreds: true, Exists: true}}
	seat := thirdPartySeat()
	seat.Identity = Identity{Email: "vendor@example.com", AccountUUID: "uuid-v", HasCreds: true, Exists: true}
	reg := Registry{Version: RegistryVersion, Homes: []Home{ordinary, seat}}

	// (1) Never volunteered: the rotation pool excludes it, with the reason recorded.
	plan := reg.RotationPlan()
	if !plan.Policy.AvoidReserved {
		t.Fatal("avoid_reserved is not the default; a reserved seat would be auto-rotated onto")
	}
	for _, s := range plan.Pool {
		if s.Name == "vendor" {
			t.Fatal("reserved vendor seat entered the rotation pool; automatic rotation could spend its credential unasked")
		}
	}
	var excludedAs RotationStatus
	for _, s := range plan.Excluded {
		if s.Name == "vendor" {
			excludedAs = s.Status
		}
	}
	if excludedAs != RotationReserved {
		t.Fatalf("vendor seat excluded as %q, want %q", excludedAs, RotationReserved)
	}

	// (2) Still reachable on purpose: resolution by name does not consult the pool.
	got, chain, err := reg.Resolve("vendor")
	if err != nil {
		t.Fatalf("Resolve(vendor) failed: a reserved seat must still be launchable by name: %v", err)
	}
	if got.Name != "vendor" || len(chain) != 0 {
		t.Fatalf("Resolve(vendor) = %q chain=%v", got.Name, chain)
	}
	if got.BaseURL != seat.BaseURL {
		t.Errorf("resolved seat lost its endpoint: %q", got.BaseURL)
	}
}
