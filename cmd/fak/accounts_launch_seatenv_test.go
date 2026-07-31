package main

import (
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

func vendorSeat() accounts.Home {
	return accounts.Home{
		Name:    "vendor",
		Dir:     "/tmp/.claude-vendor",
		BaseURL: "https://gateway.example.com/serving-endpoints/anthropic",
		ExtraEnv: map[string]string{
			"ANTHROPIC_MODEL":         "vendor-claude-sonnet-5",
			"CLAUDE_CODE_USE_GATEWAY": "1",
		},
	}
}

// TestThirdPartySeatRefusesGuardedLaunch is the wrong-upstream guard. Under `fak guard` the
// child is repointed at guard's loopback and billed to guard's credential, so a guarded
// launch of a vendor-endpoint seat would look healthy while answering from the wrong account.
func TestThirdPartySeatRefusesGuardedLaunch(t *testing.T) {
	why, conflict := thirdPartyGuardConflict(vendorSeat(), true)
	if !conflict {
		t.Fatal("guarded launch of a base_url seat allowed; guard would replace the endpoint and bill another upstream")
	}
	// The message has to be actionable: name the seat, the endpoint, and the escape flag.
	for _, want := range []string{"vendor", "gateway.example.com", "--guard=false", "base_url"} {
		if !strings.Contains(why, want) {
			t.Errorf("refusal message omits %q: %s", want, why)
		}
	}
}

func TestThirdPartySeatAllowsUnguardedLaunch(t *testing.T) {
	if _, conflict := thirdPartyGuardConflict(vendorSeat(), false); conflict {
		t.Fatal("--guard=false is the sanctioned way to reach a seat's own endpoint; it must not be refused")
	}
}

// TestFirstPartySeatUnaffectedByTheConflictCheck pins that the refusal cannot touch any
// pre-existing seat: the guarded path is the DEFAULT and every ordinary seat uses it.
func TestFirstPartySeatUnaffectedByTheConflictCheck(t *testing.T) {
	ordinary := accounts.Home{Name: "sub", Dir: "/tmp/.claude-sub"}
	for _, guarded := range []bool{true, false} {
		if _, conflict := thirdPartyGuardConflict(ordinary, guarded); conflict {
			t.Fatalf("ordinary seat refused with useGuard=%v; the default launch path would break", guarded)
		}
	}
}

// TestLaunchSeatEnvPrecedence is the ordering contract. The env is collapsed through a
// name-keyed map, so LAST occurrence wins, and the two things that must win are the seat's
// overlay (over the shell it was launched from) and CLAUDE_CONFIG_DIR (over everything).
func TestLaunchSeatEnvPrecedence(t *testing.T) {
	base := []string{
		"PATH=/usr/bin",
		"ANTHROPIC_BASE_URL=http://127.0.0.1:4000", // an inherited guard proxy: must LOSE
		"ANTHROPIC_MODEL=claude-opus-5",            // an inherited model: must LOSE
		"CLAUDE_CONFIG_DIR=/tmp/.claude-someoneelse",
	}
	got, _ := launchSeatEnv(base, vendorSeat())

	last := map[string]string{}
	for _, kv := range got {
		if i := strings.IndexByte(kv, '='); i >= 0 {
			last[kv[:i]] = kv[i+1:]
		}
	}
	if last["ANTHROPIC_BASE_URL"] != "https://gateway.example.com/serving-endpoints/anthropic" {
		t.Errorf("inherited ANTHROPIC_BASE_URL won over the seat's endpoint: %q", last["ANTHROPIC_BASE_URL"])
	}
	if last["ANTHROPIC_MODEL"] != "vendor-claude-sonnet-5" {
		t.Errorf("inherited ANTHROPIC_MODEL won over the seat's: %q", last["ANTHROPIC_MODEL"])
	}
	if last["CLAUDE_CONFIG_DIR"] != "/tmp/.claude-vendor" {
		t.Errorf("CLAUDE_CONFIG_DIR = %q; the seat's own dir must win or the launch runs as another seat", last["CLAUDE_CONFIG_DIR"])
	}
	if last["PATH"] != "/usr/bin" {
		t.Errorf("ambient PATH lost: %q", last["PATH"])
	}
	// CLAUDE_CONFIG_DIR must be physically last, so it wins under any dedup rule — including
	// a first-wins consumer — not just the map-collapse this launch path happens to use.
	if final := got[len(got)-1]; final != "CLAUDE_CONFIG_DIR=/tmp/.claude-vendor" {
		t.Errorf("last entry = %q, want CLAUDE_CONFIG_DIR", final)
	}
}

// TestLaunchSeatEnvUnchangedForFirstPartySeat is the compatibility assertion: the historical
// env was exactly os.Environ()+CLAUDE_CONFIG_DIR, and for every existing seat it still is.
func TestLaunchSeatEnvUnchangedForFirstPartySeat(t *testing.T) {
	base := []string{"PATH=/usr/bin", "HOME=/home/x"}
	got, scrubbed := launchSeatEnv(base, accounts.Home{Name: "sub", Dir: "/tmp/.claude-sub"})
	want := []string{"PATH=/usr/bin", "HOME=/home/x", "CLAUDE_CONFIG_DIR=/tmp/.claude-sub"}
	if !slices.Equal(got, want) {
		t.Fatalf("launch env changed for an ordinary seat:\n got %q\nwant %q", got, want)
	}
	if len(scrubbed) != 0 {
		t.Fatalf("a first-party seat scrubbed %q; only a third-party seat has a credential collision", scrubbed)
	}
}

// TestLaunchSeatEnvScrubsInheritedFirstPartyCredentials is the wrong-token regression, taken
// from a live failure mode: a `fak guard` session exports ANTHROPIC_API_KEY and
// ANTHROPIC_BASE_URL into every child, so a third-party seat launched from one would inherit
// a first-party credential next to the vendor endpoint and could present the wrong token.
func TestLaunchSeatEnvScrubsInheritedFirstPartyCredentials(t *testing.T) {
	seat := vendorSeat()
	seat.APIKeyEnv = "ANTHROPIC_AUTH_TOKEN" // the seat's OWN credential variable
	base := []string{
		"PATH=/usr/bin",
		"ANTHROPIC_API_KEY=fak-guard-proxy-key",
		"CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-someone-else",
		"ANTHROPIC_AUTH_TOKEN=the-vendor-tenant-token",
	}
	got, scrubbed := launchSeatEnv(base, seat)

	last := map[string]string{}
	for _, kv := range got {
		if k, v, ok := strings.Cut(kv, "="); ok {
			last[k] = v
		}
	}
	for _, gone := range []string{"ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"} {
		if _, present := last[gone]; present {
			t.Errorf("%s survived into a third-party seat's launch env; the vendor could be sent another account's credential", gone)
		}
		if !slices.Contains(scrubbed, gone) {
			t.Errorf("%s dropped but not reported in scrubbed=%q; the operator would not know their key was ignored", gone, scrubbed)
		}
	}
	// The seat's OWN credential is the one thing that must survive.
	if last["ANTHROPIC_AUTH_TOKEN"] != "the-vendor-tenant-token" {
		t.Fatalf("the seat's own credential was scrubbed: %q — the launch would have no auth at all", last["ANTHROPIC_AUTH_TOKEN"])
	}
	if last["PATH"] != "/usr/bin" {
		t.Errorf("unrelated ambient variable dropped: PATH=%q", last["PATH"])
	}
}

// TestLaunchSeatEnvScrubKeepsSeatCredentialWhateverItsName pins that the keep-set follows
// APIKeyEnv rather than a hardcoded name: a seat whose credential is ANTHROPIC_API_KEY keeps
// that one and drops the bearer/OAuth variables instead.
func TestLaunchSeatEnvScrubKeepsSeatCredentialWhateverItsName(t *testing.T) {
	seat := vendorSeat()
	seat.APIKeyEnv = "ANTHROPIC_API_KEY"
	base := []string{"ANTHROPIC_API_KEY=this-seats-key", "ANTHROPIC_AUTH_TOKEN=stale-bearer"}
	got, scrubbed := launchSeatEnv(base, seat)
	joined := strings.Join(got, " ")
	if !strings.Contains(joined, "ANTHROPIC_API_KEY=this-seats-key") {
		t.Fatalf("the seat's named credential was scrubbed: %q", got)
	}
	if strings.Contains(joined, "stale-bearer") {
		t.Errorf("a non-seat credential survived: %q", got)
	}
	if !slices.Contains(scrubbed, "ANTHROPIC_AUTH_TOKEN") {
		t.Errorf("scrubbed = %q, want ANTHROPIC_AUTH_TOKEN", scrubbed)
	}
}

// ---------------------------------------------------------------------------
// --env KEY=VALUE parsing (fak accounts add)
// ---------------------------------------------------------------------------

func TestParseSeatExtraEnvHappyPath(t *testing.T) {
	got, err := parseSeatExtraEnv([]string{
		"ANTHROPIC_MODEL=vendor-claude-sonnet-5",
		"ANTHROPIC_CUSTOM_HEADERS=x-vendor-mode: true", // a value with '=' would also be fine
		"ENABLE_TOOL_SEARCH=1",
		"EMPTY=", // an explicitly-empty export is a real posture, distinct from unset
	})
	if err != nil {
		t.Fatalf("parseSeatExtraEnv: %v", err)
	}
	want := map[string]string{
		"ANTHROPIC_MODEL":          "vendor-claude-sonnet-5",
		"ANTHROPIC_CUSTOM_HEADERS": "x-vendor-mode: true",
		"ENABLE_TOOL_SEARCH":       "1",
		"EMPTY":                    "",
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("%s = %q, want %q", k, got[k], v)
		}
	}
	if len(got) != len(want) {
		t.Errorf("parsed %d vars, want %d: %v", len(got), len(want), got)
	}
	// No flags at all must stay nil, so an ordinary seat serializes without an extra_env key.
	if m, err := parseSeatExtraEnv(nil); err != nil || m != nil {
		t.Errorf("parseSeatExtraEnv(nil) = %v, %v; want nil, nil", m, err)
	}
}

// TestParseSeatExtraEnvKeepsFirstValueSplit pins that only the FIRST '=' splits, so a value
// containing '=' (a header, a query string) survives intact.
func TestParseSeatExtraEnvKeepsFirstValueSplit(t *testing.T) {
	got, err := parseSeatExtraEnv([]string{"OPTS=a=1,b=2"})
	if err != nil {
		t.Fatalf("parseSeatExtraEnv: %v", err)
	}
	if got["OPTS"] != "a=1,b=2" {
		t.Fatalf("OPTS = %q, want the whole value after the first '='", got["OPTS"])
	}
}

func TestParseSeatExtraEnvFailsLoud(t *testing.T) {
	cases := map[string][]string{
		"missing '='":          {"ANTHROPIC_MODEL"},
		"duplicate key":        {"ANTHROPIC_MODEL=a", "ANTHROPIC_MODEL=b"},
		"credential-shaped":    {"ANTHROPIC_AUTH_TOKEN=dapi-would-be-published"},
		"fak-owned config dir": {"CLAUDE_CONFIG_DIR=/tmp/.claude-other"},
		"endpoint via --env":   {"ANTHROPIC_BASE_URL=https://gateway.example.com"},
	}
	for name, argv := range cases {
		if _, err := parseSeatExtraEnv(argv); err == nil {
			t.Errorf("%s: parseSeatExtraEnv(%q) accepted; it must fail loud", name, argv)
		}
	}
}

// TestLaunchSeatEnvDoesNotAliasBase catches the append-in-place bug: os.Environ()'s slice
// has spare capacity, so appending to it directly can scribble into the caller's array.
func TestLaunchSeatEnvDoesNotAliasBase(t *testing.T) {
	base := make([]string, 2, 32)
	base[0], base[1] = "PATH=/usr/bin", "HOME=/home/x"
	_, _ = launchSeatEnv(base, vendorSeat())
	if len(base) != 2 || base[0] != "PATH=/usr/bin" || base[1] != "HOME=/home/x" {
		t.Fatalf("base environment mutated: %q", base)
	}
}
