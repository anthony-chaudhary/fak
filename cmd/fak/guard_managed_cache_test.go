package main

import (
	"regexp"
	"strings"
	"testing"
)

// The managed-cache posture rule (epic #1844 C6): AUTO activates the 1h-TTL lever only on
// provable API-key billing on the Anthropic wire; every unknown/flat-rate posture stays
// passive; ON/OFF override; an unknown mode fails loud (usage, not a silent default).
func TestResolveGuardManagedCache(t *testing.T) {
	apiKeyBilled := guardManagedCacheInputs{provider: "anthropic", apiKey: "sk-ant-api-xyz"}
	cases := []struct {
		name       string
		mode       string
		in         guardManagedCacheInputs
		wantActive bool
		wantErr    bool
		wantReason string // substring the banner reason must carry
	}{
		{name: "auto api-key billing activates", mode: "auto", in: apiKeyBilled, wantActive: true, wantReason: "API-key billing"},
		{name: "empty mode is auto", mode: "", in: apiKeyBilled, wantActive: true, wantReason: "API-key billing"},
		{name: "auto subscription oauth stays passive", mode: "auto",
			in:         guardManagedCacheInputs{provider: "anthropic", apiKey: "sk-ant-oat-token", oauthSource: "CLAUDE_CODE_OAUTH_TOKEN"},
			wantReason: "subscription OAuth"},
		{name: "auto plain passthrough stays passive", mode: "auto",
			in:         guardManagedCacheInputs{provider: "anthropic"},
			wantReason: "billing unknown"},
		{name: "auto non-anthropic wire stays passive", mode: "auto",
			in:         guardManagedCacheInputs{provider: "openai", apiKey: "sk-openai"},
			wantReason: "no cache_control wire"},
		{name: "auto local model stays passive", mode: "auto",
			in:         guardManagedCacheInputs{localModel: true, provider: "anthropic"},
			wantReason: "local in-kernel model"},
		{name: "on forces active even on oauth", mode: "on",
			in:         guardManagedCacheInputs{provider: "anthropic", apiKey: "tok", oauthSource: "env"},
			wantActive: true, wantReason: "forced"},
		{name: "off disables even on api-key billing", mode: "off", in: apiKeyBilled, wantReason: "disabled"},
		{name: "unknown mode fails loud", mode: "sometimes", in: apiKeyBilled, wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := resolveGuardManagedCache(tc.mode, tc.in)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("resolveGuardManagedCache(%q) = %+v, want error", tc.mode, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("resolveGuardManagedCache(%q): %v", tc.mode, err)
			}
			if got.active != tc.wantActive {
				t.Fatalf("active = %v, want %v (%+v)", got.active, tc.wantActive, got)
			}
			if !strings.Contains(got.reason, tc.wantReason) {
				t.Fatalf("reason %q does not carry %q", got.reason, tc.wantReason)
			}
			// The banner is the operator's only view of the posture: it must state
			// ACTIVE vs passive and carry the reason verbatim.
			line := got.bannerLine()
			if got.active != strings.Contains(line, "ACTIVE") {
				t.Fatalf("bannerLine %q disagrees with active=%v", line, got.active)
			}
			if !strings.Contains(line, tc.wantReason) {
				t.Fatalf("bannerLine %q does not carry reason %q", line, tc.wantReason)
			}
		})
	}
}

// TestManagedCacheWiredIntoGuard is the regression sentinel for the --managed-cache WIRING
// (the token_defaults_test.go idiom). The posture shipped in 8b618eec and a later guard.go
// change silently dropped every cmdGuard hunk — the resolver and the unit test above survived,
// so nothing redded while the flag, the gateway lever, and the banner all vanished from the
// binary. These assertions read the REAL entrypoint source, so the concept's operator
// visibility (banner + fak_gateway_cache_ttl_upgrade_total witness) cannot detach from its
// declaration again without failing here.
func TestManagedCacheWiredIntoGuard(t *testing.T) {
	src := readEntrypoint(t, "guard.go")
	if !strings.Contains(src, `fs.String("managed-cache", guardManagedCacheAuto`) {
		t.Errorf("guard.go must register --managed-cache defaulted to guardManagedCacheAuto")
	}
	if !strings.Contains(src, "resolveGuardManagedCache(") {
		t.Errorf("guard.go must resolve the managed-cache posture from the resolved upstream")
	}
	if !regexp.MustCompile(`CacheTTL1H:\s+mcache\.active`).MatchString(src) {
		t.Errorf("guard.go must wire the resolved posture into gateway Config CacheTTL1H")
	}
	if !strings.Contains(src, "mcache.bannerLine()") {
		t.Errorf("guard.go must print the posture bannerLine into the startup report")
	}
}
