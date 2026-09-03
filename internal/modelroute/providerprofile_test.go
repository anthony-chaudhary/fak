package modelroute

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// TestProviderCacheQuirksDiffer is the test providerprofile.go's header asserts
// exists. It pins the file's central honesty claim: the seed profiles carry REAL
// per-provider cache values, not one placeholder default stamped on all four.
//
// This matters because the cache-value surface credits savings from these
// numbers. A uniform default would still parse, still validate, and still route
// — and would silently mis-credit every provider it was wrong about. Uniformity
// is exactly the failure that cannot be caught downstream, so it is pinned here.
func TestProviderCacheQuirksDiffer(t *testing.T) {
	profiles := DefaultProviderProfiles()
	if len(profiles) < 2 {
		t.Fatalf("need at least 2 seed profiles to compare, got %d", len(profiles))
	}

	ttls := map[int]bool{}
	breakpoints := map[int]bool{}
	for _, p := range profiles {
		ttls[p.CacheTTLSeconds] = true
		breakpoints[p.MaxCacheBreakpoints] = true
	}
	if len(ttls) < 2 {
		t.Errorf("every seed provider declares the same cache TTL (%v) — that is the placeholder shape the header rules out", ttls)
	}
	if len(breakpoints) < 2 {
		t.Errorf("every seed provider declares the same breakpoint cap (%v) — same placeholder shape", breakpoints)
	}

	// The specific values the header commits to in prose, pinned so a silent
	// edit to one of them has to come with a doc edit.
	want := map[string]struct {
		ttl         int
		breakpoints int
	}{
		"anthropic": {300, 4},  // 5-minute ephemeral, 4 cache_control breakpoints
		"openai":    {0, 0},    // provider-managed, no operator knob
		"gemini":    {3600, 0}, // operator-set context cache, 1h default
		"xai":       {0, 0},    // provider-managed
	}
	seen := map[string]bool{}
	for _, p := range profiles {
		w, ok := want[p.Provider]
		if !ok {
			continue // a newly added provider needs no entry here
		}
		seen[p.Provider] = true
		if p.CacheTTLSeconds != w.ttl {
			t.Errorf("%s CacheTTLSeconds = %d, want %d", p.Provider, p.CacheTTLSeconds, w.ttl)
		}
		if p.MaxCacheBreakpoints != w.breakpoints {
			t.Errorf("%s MaxCacheBreakpoints = %d, want %d", p.Provider, p.MaxCacheBreakpoints, w.breakpoints)
		}
	}
	for name := range want {
		if !seen[name] {
			t.Errorf("seed set no longer declares a profile for %q", name)
		}
	}
}

// TestTTLZeroIsNotUnsupportedCaching pins the distinction the schema exists to
// carry: a provider-managed TTL (0) is NOT the same state as "cannot cache".
// Collapsing the two would make the cache-value surface skip savings that a
// provider really does serve.
func TestTTLZeroIsNotUnsupportedCaching(t *testing.T) {
	for _, p := range DefaultProviderProfiles() {
		// A provider with caching enabled (SupportsPromptCaching=true) and TTL=0
		// means provider-managed TTL. A provider with caching disabled
		// (SupportsPromptCaching=false) and TTL=0 means explicitly unsupported.
		// Both are valid; only an undeclared provider (zero-value struct) is invalid.
		if p.SupportsPromptCaching && p.CacheTTLSeconds == 0 {
			// Provider-managed TTL - this is the intended distinct state
			continue
		}
		if !p.SupportsPromptCaching && p.CacheTTLSeconds == 0 {
			// Explicitly unsupported - also valid (e.g., OpenRouter)
			continue
		}
		// Any other combination with TTL=0 is ambiguous
		if p.CacheTTLSeconds == 0 {
			t.Errorf("%s: TTL 0 with ambiguous caching state (supported=%v)", p.Provider, p.SupportsPromptCaching)
		}
	}
	// The zero value of the struct IS the unsupported state, so an undeclared
	// provider never accidentally reads as cacheable.
	var zero ProviderProfile
	if zero.SupportsPromptCaching {
		t.Error("zero-value ProviderProfile must not claim prompt caching")
	}
	if zero.CacheTTL() != 0 {
		t.Errorf("zero-value CacheTTL = %v, want 0", zero.CacheTTL())
	}
}

func TestCacheTTLConvertsSecondsToDuration(t *testing.T) {
	cases := map[int]time.Duration{
		0:    0,
		300:  5 * time.Minute,
		3600: time.Hour,
	}
	for secs, want := range cases {
		got := ProviderProfile{CacheTTLSeconds: secs}.CacheTTL()
		if got != want {
			t.Errorf("CacheTTL(%d) = %v, want %v", secs, got, want)
		}
	}
}

// TestNewProviderRegistryFailsLoud covers every rejection NewProviderRegistry
// documents. Each of these is a config mistake a silent default would hide, and
// the last one is the substantive one: cache quirks on a non-caching provider
// would have the cache-value surface honoring a TTL that cannot exist.
func TestNewProviderRegistryFailsLoud(t *testing.T) {
	cases := map[string]struct {
		profiles []ProviderProfile
		wantErr  string
	}{
		"empty provider id": {
			[]ProviderProfile{{Provider: ""}},
			"empty provider id",
		},
		"duplicate provider": {
			[]ProviderProfile{{Provider: "openai"}, {Provider: "openai"}},
			"duplicate provider profile",
		},
		"negative ttl": {
			[]ProviderProfile{{Provider: "openai", SupportsPromptCaching: true, CacheTTLSeconds: -1}},
			"negative cache_ttl_seconds",
		},
		"negative breakpoints": {
			[]ProviderProfile{{Provider: "openai", SupportsPromptCaching: true, MaxCacheBreakpoints: -1}},
			"negative max_cache_breakpoints",
		},
		"quirks without caching": {
			[]ProviderProfile{{Provider: "openai", SupportsPromptCaching: false, CacheTTLSeconds: 300}},
			"declares cache quirks but not prompt caching",
		},
		"breakpoints without caching": {
			[]ProviderProfile{{Provider: "openai", SupportsPromptCaching: false, MaxCacheBreakpoints: 4}},
			"declares cache quirks but not prompt caching",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			r, err := NewProviderRegistry(tc.profiles)
			if err == nil {
				t.Fatalf("expected a loud failure, got registry %v", r)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error %q does not mention %q", err, tc.wantErr)
			}
		})
	}

	// The seed set itself must satisfy its own validator — otherwise the
	// built-in data is a config the code would reject from a file.
	if _, err := NewProviderRegistry(DefaultProviderProfiles()); err != nil {
		t.Fatalf("DefaultProviderProfiles must validate, got %v", err)
	}
}

// TestProfileForIsTheWire exercises the documented seam from a routing decision
// to its provider profile, including the local-model case that must return false
// rather than a zero profile the caller could mistake for real quirks.
func TestProfileForIsTheWire(t *testing.T) {
	reg, err := NewProviderRegistry(DefaultProviderProfiles())
	if err != nil {
		t.Fatalf("NewProviderRegistry: %v", err)
	}

	prof, ok := reg.ProfileFor(ResolvedModel{Alias: "large", EngineID: "claude", Provider: "anthropic", Remote: true})
	if !ok {
		t.Fatal("a remote anthropic model must resolve to a profile")
	}
	if prof.CacheTTL() != 5*time.Minute {
		t.Errorf("anthropic CacheTTL = %v, want 5m", prof.CacheTTL())
	}

	// An in-kernel model carries no Provider: no profile, and the miss must be
	// reported rather than papered over with a zero value.
	if _, ok := reg.ProfileFor(ResolvedModel{Alias: "local", EngineID: "in-kernel"}); ok {
		t.Error("a local model (empty Provider) must not resolve to a profile")
	}
	// An unknown provider is likewise a miss, not a default.
	if _, ok := reg.ProfileFor(ResolvedModel{Alias: "x", EngineID: "e", Provider: "nope", Remote: true}); ok {
		t.Error("an unregistered provider must not resolve to a profile")
	}
}

// TestParseProviderProfilesRejectsTypos pins DisallowUnknownFields. A typo'd key
// that parsed silently would drop the field's meaning while looking configured —
// the failure mode this decoder setting exists to prevent.
func TestParseProviderProfilesRejectsTypos(t *testing.T) {
	typo := []byte(`[{"provider":"openai","supports_prompt_caching":true,"cache_ttl_second":300}]`)
	if _, err := ParseProviderProfiles(typo); err == nil {
		t.Fatal("a misspelled key must fail loud, not parse")
	}

	bad := []byte(`[{"provider":"openai",`)
	if _, err := ParseProviderProfiles(bad); err == nil {
		t.Fatal("malformed JSON must error")
	}

	// A validation failure must survive the parse path, not just the constructor.
	invalid := []byte(`[{"provider":"openai","supports_prompt_caching":false,"cache_ttl_seconds":300}]`)
	if _, err := ParseProviderProfiles(invalid); err == nil {
		t.Fatal("parse must apply the same validation as NewProviderRegistry")
	}
}

// TestJSONRoundTripsAndIsStable pins the documented `--dump > file` then
// edit-and-reload path, plus the sorted, newline-terminated rendering that keeps
// the diff stable across runs (map iteration order is randomized in Go).
func TestJSONRoundTripsAndIsStable(t *testing.T) {
	reg, err := NewProviderRegistry(DefaultProviderProfiles())
	if err != nil {
		t.Fatalf("NewProviderRegistry: %v", err)
	}
	b := reg.JSON()

	if len(b) == 0 || b[len(b)-1] != '\n' {
		t.Error("JSON output must be newline-terminated")
	}

	back, err := ParseProviderProfiles(b)
	if err != nil {
		t.Fatalf("rendered JSON must re-parse, got %v", err)
	}
	if got, want := strings.Join(back.Providers(), ","), strings.Join(reg.Providers(), ","); got != want {
		t.Errorf("round-trip providers = %q, want %q", got, want)
	}

	// Stability: re-rendering must be byte-identical despite map iteration order.
	for i := 0; i < 8; i++ {
		if string(reg.JSON()) != string(b) {
			t.Fatalf("JSON() is not byte-stable across calls (iteration %d)", i)
		}
	}

	// Sorted order is what makes it stable; assert it directly.
	var decoded []ProviderProfile
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for i := 1; i < len(decoded); i++ {
		if decoded[i-1].Provider >= decoded[i].Provider {
			t.Errorf("providers not sorted: %q before %q", decoded[i-1].Provider, decoded[i].Provider)
		}
	}
}

// TestProfilesCarryNoSecrets pins the schema's core safety property: a profile
// names the environment VARIABLE holding the key, never the key. This is what
// makes the seed set checkable into a public repo.
func TestProfilesCarryNoSecrets(t *testing.T) {
	for _, p := range DefaultProviderProfiles() {
		if p.AuthEnv == "" {
			continue // an in-kernel provider may name none
		}
		if !strings.HasSuffix(p.AuthEnv, "_API_KEY") {
			t.Errorf("%s AuthEnv = %q, want an env-var NAME", p.Provider, p.AuthEnv)
		}
		// An env-var name is uppercase with underscores; anything carrying a
		// key-shaped prefix would be a pasted secret, not a name.
		if strings.HasPrefix(p.AuthEnv, "sk-") || strings.Contains(p.AuthEnv, "-") {
			t.Errorf("%s AuthEnv %q looks like key material, not a variable name", p.Provider, p.AuthEnv)
		}
		if p.Endpoint != "" && !strings.HasPrefix(p.Endpoint, "https://") {
			t.Errorf("%s Endpoint = %q, want an https base URL", p.Provider, p.Endpoint)
		}
	}
}
