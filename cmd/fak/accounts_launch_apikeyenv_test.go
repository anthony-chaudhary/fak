package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The #5503 diagnosability contradiction, in one place.
//
// `fak accounts launch` splices the seat's own `--api-key-env NAME` into the child's argv
// (accounts_launch.go, guardCachePostureArgs(mcMode, launchSeatAPIKeyEnv(home))), and then
// hands that child an environment the always-on #2358 inherited-secret floor has already
// swept: newLaunchBrokerAttempt -> sanitizeLaunchEnv -> policy.StripInheritedSecrets drops
// every credential-shaped NAME (TOKEN / SECRET / PASSWORD / CREDENTIAL / ...) and every
// secret-shaped VALUE (sk-...) that is not one of the two spared provider names.
//
// So an argv can reference a variable the child provably cannot read. Before the refusal
// below existed, that launch SUCCEEDED: guard came up with no upstream key and the agent
// printed a bare "Not logged in" with nothing naming the floor that caused it.
//
// These tests pin the named refusal. They must permit nothing new: the floor still strips
// exactly what it stripped before, and a seat whose reference survives the floor still
// launches.

// launchAPIKeyEnvFakeSecret is an obviously fake placeholder. Its shape matters (the floor's
// value check keys off the "sk-" prefix); its content is not a credential.
const launchAPIKeyEnvFakeSecret = "sk-ant-api03-PLACEHOLDER-NOT-A-REAL-KEY"

// writeAPIKeySeatRegistry lands a one-seat api_key registry whose credential is the env-var
// NAME reference (never a value), and returns the registry path.
func writeAPIKeySeatRegistry(t *testing.T, home, seat, apiKeyEnv string) string {
	t.Helper()
	seatDir := filepath.Join(home, ".claude-"+seat)
	if err := os.MkdirAll(seatDir, 0o755); err != nil {
		t.Fatal(err)
	}
	reg := `{"version":"fak-config-homes/v1",` +
		`"homes":[{"name":"` + seat + `","dir":"` + jsonPath(seatDir) + `","cred_kind":"api_key","api_key_env":"` + apiKeyEnv + `"}]}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}
	return regPath
}

// TestLaunchAPIKeyEnvFloorContradictionPremise witnesses the premise directly, independent of
// the refusal: the exact environment `fak accounts launch` builds loses the seat's api-key
// variable to the #2358 floor, for BOTH strip rules, while the argv shaper keeps referencing
// it. This is the bug the refusal makes speakable; it stays true after the fix (the fix
// changes what the launcher DOES about it, not what the floor strips).
func TestLaunchAPIKeyEnvFloorContradictionPremise(t *testing.T) {
	for _, tc := range []struct {
		name    string
		envName string
		value   string
		why     string
	}{
		{
			name:    "credential-shaped NAME",
			envName: "FAK_TEST_5503_CORP_TOKEN",
			value:   "notsecretshaped",
			why:     "secretShapedName matches TOKEN by substring",
		},
		{
			name:    "secret-shaped VALUE under a non-provider NAME",
			envName: "FAK_TEST_5503_CORP_KEY",
			value:   launchAPIKeyEnvFakeSecret,
			why:     "secretShapedValue matches the sk- prefix and the NAME is not a spared provider name",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The launcher's own argv shaper still references the variable...
			args := guardCachePostureArgs("on", tc.envName)
			if !argvHasFlagValue(args, "--api-key-env", tc.envName) {
				t.Fatalf("guardCachePostureArgs dropped the reference: %q", args)
			}
			// ...but the launcher's own env sanitizer removes it from the child's environment.
			sanitized, stripped := sanitizeLaunchEnvExcept(map[string]string{
				"PATH":     "/usr/bin",
				tc.envName: tc.value,
			}, nil)
			if _, ok := sanitized[tc.envName]; ok {
				t.Fatalf("premise refuted: %s survived the #2358 floor (%s)", tc.envName, tc.why)
			}
			if !contains(stripped, tc.envName) {
				t.Fatalf("floor stripped %s without naming it in the audit list: %q", tc.envName, stripped)
			}
			if _, ok := sanitized["PATH"]; !ok {
				t.Fatalf("floor should spare PATH; sanitized=%v", sortedMapKeys(sanitized))
			}
		})
	}
}

// TestRunAccountsLaunchRefusesUnreachableAPIKeyEnv is the behavior gate. Launching an api-key
// seat whose reference the floor strips must REFUSE with a message that names the variable
// NAME and the floor, and must not start the child at all.
func TestRunAccountsLaunchRefusesUnreachableAPIKeyEnv(t *testing.T) {
	const envName = "FAK_TEST_5503_CORP_TOKEN"
	t.Setenv(envName, launchAPIKeyEnvFakeSecret)
	t.Setenv(fleetManagedCacheEnv, "")
	t.Setenv(fleetGuardAPIKeyEnvEnv, "")
	home := t.TempDir()
	regPath := writeAPIKeySeatRegistry(t, home, "corp", envName)

	launched := false
	orig := accountsLaunchRun
	accountsLaunchRun = func(_, _ io.Writer, _, _ []string) launchRunResult {
		launched = true
		return launchRunResult{Code: 0}
	}
	t.Cleanup(func() { accountsLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--name", "corp", "--registry", regPath, "--home", home})
	if rc == 0 {
		t.Fatalf("launch returned 0 while handing the child an argv referencing an unreachable $%s\nstderr:\n%s", envName, errb.String())
	}
	if launched {
		t.Fatalf("refused launch must not start the child")
	}
	got := errb.String()
	// The refusal names the variable NAME...
	if !strings.Contains(got, envName) {
		t.Fatalf("refusal must name the env-var NAME %q:\n%s", envName, got)
	}
	// ...and names the floor that caused it, so the operator is not left with "Not logged in".
	for _, want := range []string{"--api-key-env", "inherited-secret floor", "#2358"} {
		if !strings.Contains(got, want) {
			t.Fatalf("refusal must name %q as the cause:\n%s", want, got)
		}
	}
	// ...and never the value.
	if strings.Contains(got, launchAPIKeyEnvFakeSecret) || strings.Contains(out.String(), launchAPIKeyEnvFakeSecret) {
		t.Fatalf("refusal leaked the credential value")
	}
}

// TestRunAccountsLaunchNeverBlamesFloorForUnsetAPIKeyEnv is the mis-attribution gate. A
// variable the operator simply never exported is a DIFFERENT terminal with its own existing
// handling — the seat-servability walk refuses an api-key seat whose key is absent long before
// the launch is shaped — and the #5503 refusal must not reach in and blame the floor for a
// variable the floor never saw.
func TestRunAccountsLaunchNeverBlamesFloorForUnsetAPIKeyEnv(t *testing.T) {
	const envName = "FAK_TEST_5503_NEVER_EXPORTED"
	t.Setenv(fleetManagedCacheEnv, "")
	t.Setenv(fleetGuardAPIKeyEnvEnv, "")
	os.Unsetenv(envName)
	home := t.TempDir()
	regPath := writeAPIKeySeatRegistry(t, home, "corp", envName)

	launched := false
	orig := accountsLaunchRun
	accountsLaunchRun = func(_, _ io.Writer, _, _ []string) launchRunResult {
		launched = true
		return launchRunResult{Code: 0}
	}
	t.Cleanup(func() { accountsLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--name", "corp", "--registry", regPath, "--home", home})
	if rc == 0 || launched {
		t.Fatalf("launch rc=%d launched=%v while $%s is unset\nstderr:\n%s", rc, launched, envName, errb.String())
	}
	if strings.Contains(errb.String(), "inherited-secret floor") {
		t.Fatalf("an unset variable must not be blamed on the floor:\n%s", errb.String())
	}
	// The unit-level guarantee, independent of which gate fires first: a name the floor did
	// not strip is never refused as a floor casualty.
	grant := launchBrokerGrant{Env: map[string]string{"PATH": "/usr/bin"}}
	if got := launchStrippedAPIKeyEnvRefusal([]string{"fak", "guard", "--api-key-env", envName, "--", "claude"}, grant); got != "" {
		t.Fatalf("refusal fired for a variable the floor never stripped:\n%s", got)
	}
}

// TestLaunchStrippedAPIKeyEnvRefusalScope pins the refusal's edges directly: it reads only the
// GUARD half of the argv, it accepts both flag spellings, and a child that CAN read the
// variable is never refused.
func TestLaunchStrippedAPIKeyEnvRefusalScope(t *testing.T) {
	const envName = "CORP_LLM_TOKEN"
	strippedGrant := launchBrokerGrant{
		Env:      map[string]string{"PATH": "/usr/bin"},
		Metadata: launchBrokerMetadata{StrippedSecretEnv: []string{envName}},
	}
	for _, tc := range []struct {
		name  string
		argv  []string
		grant launchBrokerGrant
		want  bool
	}{
		{
			name:  "space spelling in the guard half refuses",
			argv:  []string{"fak", "guard", "--api-key-env", envName, "--", "claude"},
			grant: strippedGrant,
			want:  true,
		},
		{
			name:  "equals spelling in the guard half refuses",
			argv:  []string{"fak", "guard", "--api-key-env=" + envName, "--", "claude"},
			grant: strippedGrant,
			want:  true,
		},
		{
			name:  "the agent's own passthrough flag is not this launch's contradiction",
			argv:  []string{"fak", "guard", "--", "claude", "--api-key-env", envName},
			grant: strippedGrant,
			want:  false,
		},
		{
			name: "a readable variable launches",
			argv: []string{"fak", "guard", "--api-key-env", envName, "--", "claude"},
			grant: launchBrokerGrant{
				Env:      map[string]string{envName: launchAPIKeyEnvFakeSecret},
				Metadata: launchBrokerMetadata{StrippedSecretEnv: []string{envName}},
			},
			want: false,
		},
		{
			name:  "no reference at all",
			argv:  []string{"fak", "guard", "--managed-cache", "on", "--", "claude"},
			grant: strippedGrant,
			want:  false,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := launchStrippedAPIKeyEnvRefusal(tc.argv, tc.grant)
			if (got != "") != tc.want {
				t.Fatalf("refusal=%q, want refusal=%v", got, tc.want)
			}
			if got != "" && strings.Contains(got, launchAPIKeyEnvFakeSecret) {
				t.Fatalf("refusal leaked a value:\n%s", got)
			}
		})
	}
}

// TestRunAccountsLaunchAllowsSurvivingAPIKeyEnv is the permits-nothing-new gate: a reference
// the floor already spares (a provider API-key name) still launches, and the child really does
// receive the variable. If this ever fails, the refusal has become an over-refusal.
func TestRunAccountsLaunchAllowsSurvivingAPIKeyEnv(t *testing.T) {
	const envName = "ANTHROPIC_API_KEY" // spared by providerAPIKeyNames
	t.Setenv(envName, launchAPIKeyEnvFakeSecret)
	t.Setenv(fleetManagedCacheEnv, "")
	t.Setenv(fleetGuardAPIKeyEnvEnv, "")
	home := t.TempDir()
	regPath := writeAPIKeySeatRegistry(t, home, "corp", envName)

	var gotArgv, gotEnv []string
	orig := accountsLaunchRun
	accountsLaunchRun = func(_, _ io.Writer, argv, env []string) launchRunResult {
		gotArgv, gotEnv = argv, env
		return launchRunResult{Code: 0}
	}
	t.Cleanup(func() { accountsLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--name", "corp", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("a surviving api-key reference must still launch: rc=%d stderr=%s", rc, errb.String())
	}
	if !argvHasFlagValue(gotArgv, "--api-key-env", envName) {
		t.Fatalf("argv lost the seat's reference: %q", gotArgv)
	}
	if !envSliceHasName(gotEnv, envName) {
		t.Fatalf("child env should carry %s (the floor spares provider names)", envName)
	}
}

// argvHasFlagValue reports whether argv carries `flag value` as adjacent elements.
func argvHasFlagValue(argv []string, flag, value string) bool {
	for i := 0; i+1 < len(argv); i++ {
		if argv[i] == flag && argv[i+1] == value {
			return true
		}
	}
	return false
}

// envSliceHasName reports presence of NAME in a "NAME=VALUE" slice. Presence only — this
// never inspects or returns the value.
func envSliceHasName(env []string, name string) bool {
	for _, kv := range env {
		if n, _, ok := strings.Cut(kv, "="); ok && n == name {
			return true
		}
	}
	return false
}
