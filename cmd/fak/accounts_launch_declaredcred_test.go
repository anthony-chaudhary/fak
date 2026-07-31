package main

import (
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// This file covers the seam that made a correctly-configured third-party seat fail auth: the
// spawn broker's always-on secret floor stripped the very credential the launch argv referenced
// with --api-key-env. The launch now DECLARES that variable to the floor. Two things need
// pinning — the declaration is derived from the seat (not hardcoded), and the wiring actually
// reaches the brokered SpawnAttempt the child is handed.

func apiKeySeat(name, credEnv string) accounts.Home {
	return accounts.Home{Name: name, Dir: "/tmp/.claude-" + name, CredKind: "api_key", APIKeyEnv: credEnv}
}

// TestSeatDeclaredCredentialEnvNamesTheSeatsOwnCredential is the regression that names the bug:
// a seat whose credential is $ANTHROPIC_AUTH_TOKEN must declare it, because "TOKEN" is exactly
// what policy.secretShapedName strips.
func TestSeatDeclaredCredentialEnvNamesTheSeatsOwnCredential(t *testing.T) {
	seat := apiKeySeat("vendor", "ANTHROPIC_AUTH_TOKEN")
	seat.BaseURL = "https://gateway.example.com/serving-endpoints/anthropic"
	if got := seatDeclaredCredentialEnv(seat); !slices.Equal(got, []string{"ANTHROPIC_AUTH_TOKEN"}) {
		t.Fatalf("seatDeclaredCredentialEnv = %v, want [ANTHROPIC_AUTH_TOKEN]", got)
	}
	// The same need exists WITHOUT a third-party endpoint: a first-party api-key seat whose
	// credential happens to be the bearer variable is fronted with `--api-key-env
	// ANTHROPIC_AUTH_TOKEN` too, and guard cannot read a variable the floor removed.
	if got := seatDeclaredCredentialEnv(apiKeySeat("firstparty", "ANTHROPIC_AUTH_TOKEN")); !slices.Equal(got, []string{"ANTHROPIC_AUTH_TOKEN"}) {
		t.Fatalf("first-party api-key seat declared %v; --api-key-env would reference a stripped variable", got)
	}
}

// TestSeatDeclaredCredentialEnvDeclaresNothingForAPlainSeat is the blast-radius bound: a seat
// that names no credential must not widen the floor at all. With $FAK_GUARD_API_KEY_ENV unset,
// an ordinary subscription seat declares nothing and its launch env is what it always was.
func TestSeatDeclaredCredentialEnvDeclaresNothingForAPlainSeat(t *testing.T) {
	t.Setenv(fleetGuardAPIKeyEnvEnv, "")
	if got := seatDeclaredCredentialEnv(accounts.Home{Name: "sub", Dir: "/tmp/.claude-sub"}); len(got) != 0 {
		t.Fatalf("a subscription seat declared %v; nothing should be exempted from the secret floor", got)
	}
	// An api_key seat with no reference has nothing to declare either.
	if got := seatDeclaredCredentialEnv(apiKeySeat("bare", "")); len(got) != 0 {
		t.Fatalf("an api-key seat with no --api-key-env declared %v", got)
	}
}

// TestSeatDeclaredCredentialEnvFollowsTheFleetKnob pins that the declaration tracks whatever
// the argv actually references: when the seat carries no reference, `--api-key-env` comes from
// the fleet-wide knob, and that is the variable guard is told to bill.
func TestSeatDeclaredCredentialEnvFollowsTheFleetKnob(t *testing.T) {
	t.Setenv(fleetGuardAPIKeyEnvEnv, "FLEET_BILLING_TOKEN")
	if got := seatDeclaredCredentialEnv(accounts.Home{Name: "sub", Dir: "/tmp/.claude-sub"}); !slices.Equal(got, []string{"FLEET_BILLING_TOKEN"}) {
		t.Fatalf("seatDeclaredCredentialEnv = %v, want the fleet knob's variable", got)
	}
	// A seat's own reference wins over the knob (launchSeatAPIKeyEnv's rule), and the two are
	// not both declared: only the one the argv names.
	seat := apiKeySeat("vendor", "ANTHROPIC_AUTH_TOKEN")
	seat.BaseURL = "https://gateway.example.com/serving-endpoints/anthropic"
	if got := seatDeclaredCredentialEnv(seat); !slices.Equal(got, []string{"ANTHROPIC_AUTH_TOKEN"}) {
		t.Fatalf("seatDeclaredCredentialEnv = %v; the seat's own reference must win and stand alone", got)
	}
}

// TestNewLaunchBrokerAttemptDeclaringPassesTheCredential is the wiring witness: it proves the
// declaration survives all the way to a.Spawn.Env — literally the environment the launcher
// hands the child — while every other ambient secret is still held out and recorded.
func TestNewLaunchBrokerAttemptDeclaringPassesTheCredential(t *testing.T) {
	env := map[string]string{
		"PATH":                    "/usr/bin:/bin",
		"CLAUDE_CONFIG_DIR":       "/tmp/.claude-vendor",
		"ANTHROPIC_BASE_URL":      "https://gateway.example.com/serving-endpoints/anthropic",
		"ANTHROPIC_AUTH_TOKEN":    "vendor-pat-0000",
		"GITHUB_TOKEN":            "ghp_redteam_canary",
		"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat-redteam-canary",
	}
	a := newLaunchBrokerAttemptDeclaring("accounts_launch", "claude",
		[]string{"claude", "-p", "hello"}, env, t.TempDir(), []string{"ANTHROPIC_AUTH_TOKEN"})

	if a.Env["ANTHROPIC_AUTH_TOKEN"] != "vendor-pat-0000" {
		t.Fatalf("declared credential missing from the brokered env (%q) — the child would report 'Not logged in'", a.Env["ANTHROPIC_AUTH_TOKEN"])
	}
	var spawned string
	for _, kv := range a.Spawn.Env {
		if kv.Name == "ANTHROPIC_AUTH_TOKEN" {
			spawned = kv.Value
		}
		if strings.Contains(kv.Value, "redteam") {
			t.Errorf("SpawnAttempt.Env leaked an undeclared credential via %s", kv.Name)
		}
	}
	if spawned != "vendor-pat-0000" {
		t.Fatalf("SpawnAttempt.Env (what the child is handed) lacks the declared credential: %q", spawned)
	}
	for _, gone := range []string{"GITHUB_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"} {
		if _, present := a.Env[gone]; present {
			t.Errorf("undeclared credential %s survived; declaring one variable must not open the floor", gone)
		}
		if !slices.Contains(a.Metadata.StrippedSecretEnv, gone) {
			t.Errorf("audit did not record stripped %s: %v", gone, a.Metadata.StrippedSecretEnv)
		}
	}
	if _, ok := a.Env["PATH"]; !ok {
		t.Error("non-secret config dropped")
	}
	// A declaring launch must be distinguishable from a strict one in the audit, and it is —
	// the surviving credential changes the env shape the digest covers.
	strict := newLaunchBrokerAttempt("accounts_launch", "claude", []string{"claude", "-p", "hello"}, env, t.TempDir())
	if strict.Metadata.EnvDigest == a.Metadata.EnvDigest {
		t.Error("declaring and strict launches share an env digest; the exemption is invisible to the audit")
	}
	if _, present := strict.Env["ANTHROPIC_AUTH_TOKEN"]; present {
		t.Error("the strict constructor kept a credential-shaped variable; the always-on floor regressed")
	}
}

// ---------------------------------------------------------------------------
// third-party model posture
// ---------------------------------------------------------------------------

// TestThirdPartySeatModelDefersToTheSeat pins that a vendor seat does not get the first-party
// default pinned into its argv: that id does not exist in the vendor's namespace.
func TestThirdPartySeatModelDefersToTheSeat(t *testing.T) {
	seat := vendorSeat()
	resolved, why, changed := thirdPartySeatModel(seat, defaultLaunchModel, false)
	if !changed {
		t.Fatalf("third-party seat kept the first-party default --model %q; the endpoint does not serve it", defaultLaunchModel)
	}
	if resolved != "" {
		t.Errorf("resolved = %q, want \"\" (defer to the seat's own $ANTHROPIC_MODEL)", resolved)
	}
	for _, want := range []string{"vendor", defaultLaunchModel, "--model"} {
		if !strings.Contains(why, want) {
			t.Errorf("notice omits %q: %s", want, why)
		}
	}
	// Emptying the primary is also what disables the first-party fallback chain, so one
	// substitution covers both and they cannot disagree.
	p := launchParams{model: resolved, fallbackModel: defaultLaunchFallbackModel}
	if chain, ok := modelFallbackChain("claude", p); ok {
		t.Errorf("fallback chain %v still applies to a vendor endpoint", chain)
	}
}

// TestThirdPartySeatModelRespectsExplicitModel pins the operator override: naming a model
// explicitly — a vendor id, or a probe of what the endpoint serves — must not be second-guessed.
func TestThirdPartySeatModelRespectsExplicitModel(t *testing.T) {
	resolved, _, changed := thirdPartySeatModel(vendorSeat(), "vendor-claude-sonnet-5", true)
	if changed {
		t.Fatal("an explicit --model was overridden for a third-party seat")
	}
	if resolved != "vendor-claude-sonnet-5" {
		t.Errorf("resolved = %q, want the explicit id", resolved)
	}
	// An already-empty --model is not a change either; there is nothing to defer.
	if _, _, changed := thirdPartySeatModel(vendorSeat(), "", false); changed {
		t.Error("an empty --model was reported as changed")
	}
}

// TestThirdPartySeatModelLeavesFirstPartySeatsAlone is the compatibility bound: the pinned
// first-party default is deliberate for every ordinary seat and must survive untouched.
func TestThirdPartySeatModelLeavesFirstPartySeatsAlone(t *testing.T) {
	ordinary := accounts.Home{Name: "sub", Dir: "/tmp/.claude-sub"}
	for _, explicit := range []bool{true, false} {
		resolved, why, changed := thirdPartySeatModel(ordinary, defaultLaunchModel, explicit)
		if changed || resolved != defaultLaunchModel || why != "" {
			t.Fatalf("ordinary seat (explicit=%v) model posture changed to %q (%s)", explicit, resolved, why)
		}
	}
}
