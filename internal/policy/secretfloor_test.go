package policy

import (
	"encoding/json"
	"os"
	"os/exec"
	"sort"
	"strings"
	"testing"
)

// secretFloorProbeSentinel gates the re-exec child probe below so a normal test
// run never trips it — only the red-team witness, which sets it in the child env.
const secretFloorProbeSentinel = "FAK_SECRET_FLOOR_PROBE"

// TestStripInheritedSecretsFloor is the unit witness for the always-on #2358
// secret floor: credential-bearing variables are removed by name OR by value
// shape, the non-secret config a worker needs survives, a provider API key is
// spared the value check, and the stripped report names variables without ever
// echoing a value.
func TestStripInheritedSecretsFloor(t *testing.T) {
	ambient := []string{
		"PATH=/usr/bin:/bin",
		"DISPATCH_LANE=core",
		"FLEET_RESOLVE_ISSUE=2358",
		"CLAUDE_CONFIG_DIR=/home/agent/.claude",
		"ANTHROPIC_API_KEY=sk-ant-live-should-survive-api-billing",
		"OPENAI_API_KEY=sk-proj-should-survive-api-billing",
		"GITHUB_TOKEN=ghp_should_be_stripped_by_name",
		"CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat-should-be-stripped",
		"AWS_SECRET_ACCESS_KEY=abc123shouldstrip",
		"DB_PASSWORD=hunter2",
		"HELPER_SESSION_COOKIE=deadbeef",
		"BENIGN_NAME_SECRET_VALUE=sk-live-leaked-in-benign-var",
		"MALFORMED_NO_EQUALS_ENTRY", // no '=' — preserved verbatim
	}
	kept, stripped := StripInheritedSecrets(ambient)

	keptSet := map[string]string{}
	for _, kv := range kept {
		if i := strings.IndexByte(kv, '='); i >= 0 {
			keptSet[kv[:i]] = kv[i+1:]
		}
	}
	mustKeep := []string{"PATH", "DISPATCH_LANE", "FLEET_RESOLVE_ISSUE", "CLAUDE_CONFIG_DIR", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"}
	for _, name := range mustKeep {
		if _, ok := keptSet[name]; !ok {
			t.Fatalf("StripInheritedSecrets dropped a var it must keep: %s\nkept=%v", name, kept)
		}
	}
	mustStrip := []string{"GITHUB_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "AWS_SECRET_ACCESS_KEY", "DB_PASSWORD", "HELPER_SESSION_COOKIE", "BENIGN_NAME_SECRET_VALUE"}
	for _, name := range mustStrip {
		if _, ok := keptSet[name]; ok {
			t.Fatalf("StripInheritedSecrets kept a credential it must strip: %s", name)
		}
	}

	// Malformed entries (no '=') are not name/value pairs this floor understands
	// and are preserved verbatim so a caller's non-pair data is never silently lost.
	foundMalformed := false
	for _, kv := range kept {
		if kv == "MALFORMED_NO_EQUALS_ENTRY" {
			foundMalformed = true
		}
	}
	if !foundMalformed {
		t.Fatalf("malformed no-'=' entry was dropped; want preserved: kept=%v", kept)
	}

	// The stripped report names the removed variables and NEVER a value.
	sort.Strings(stripped)
	wantStripped := append([]string(nil), mustStrip...)
	sort.Strings(wantStripped)
	if strings.Join(stripped, ",") != strings.Join(wantStripped, ",") {
		t.Fatalf("stripped names = %v, want %v", stripped, wantStripped)
	}
	for _, name := range stripped {
		if strings.ContainsAny(name, "=") {
			t.Fatalf("stripped entry %q looks like a NAME=VALUE pair; the report must carry names only", name)
		}
	}
	for _, secretValue := range []string{"ghp_should_be_stripped_by_name", "sk-ant-oat-should-be-stripped", "hunter2", "deadbeef", "sk-live-leaked-in-benign-var"} {
		if strings.Contains(strings.Join(stripped, "\n"), secretValue) {
			t.Fatalf("stripped report leaked a secret value: %q", secretValue)
		}
	}
}

// TestStripInheritedSecretsRedTeamCanary is the #2360 cross-process red-team
// witness for the inherited-secret channel: it plants secret canaries in an
// ambient environment, sanitizes it through the floor, launches a REAL child
// process with the sanitized env, and proves the child cannot observe either
// canary while the legitimate config (PATH, a dispatch var) still reaches it —
// so the sanitized child is provably both leak-free AND launchable.
func TestStripInheritedSecretsRedTeamCanary(t *testing.T) {
	const (
		canaryTokenName  = "FLEET_GITHUB_TOKEN"
		canaryTokenValue = "ghp_redteamcanary_must_not_reach_child"
		canarySecretName = "FAK_CANARY_SECRET"
		canarySecretVal  = "sk-proj-redteamcanary-must-not-reach-child"
		legitName        = "DISPATCH_LANE"
		legitValue       = "core"
	)
	// Base on the real environment so the child is definitely launchable (PATH,
	// and on Windows SystemRoot/TEMP for the Go runtime), then add the canaries.
	ambient := append([]string(nil), os.Environ()...)
	ambient = append(ambient,
		canaryTokenName+"="+canaryTokenValue,
		canarySecretName+"="+canarySecretVal,
		legitName+"="+legitValue,
	)

	kept, stripped := StripInheritedSecrets(ambient)

	// The sanitized slice itself must carry neither canary value.
	joined := strings.Join(kept, "\n")
	if strings.Contains(joined, canaryTokenValue) || strings.Contains(joined, canarySecretVal) {
		t.Fatalf("sanitized env still carries a canary secret value")
	}
	strippedSet := map[string]bool{}
	for _, name := range stripped {
		strippedSet[name] = true
	}
	if !strippedSet[canaryTokenName] || !strippedSet[canarySecretName] {
		t.Fatalf("stripped report missing a canary: %v", stripped)
	}

	// Launch a real child with the sanitized env and read back what it can see.
	seen := secretFloorChildProbe(t, kept)
	if seen[canaryTokenName] {
		t.Fatalf("child observed stripped token canary %s — inherited-secret leak", canaryTokenName)
	}
	if seen[canarySecretName] {
		t.Fatalf("child observed stripped secret canary %s — inherited-secret leak", canarySecretName)
	}
	if !seen["PATH"] {
		t.Fatalf("child did not inherit PATH — sanitized env is not launchable in practice")
	}
	if !seen[legitName] {
		t.Fatalf("child lost legitimate config %s — floor over-stripped", legitName)
	}
}

// secretFloorChildProbe re-execs the test binary as a child carrying exactly env,
// runs only TestSecretFloorChildProbe, and returns the set of environment
// variable NAMES the child observed (never values).
func secretFloorChildProbe(t *testing.T, env []string) map[string]bool {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=TestSecretFloorChildProbe", "--")
	cmd.Env = append(append([]string(nil), env...), secretFloorProbeSentinel+"=1")
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("secret-floor child probe failed: %v", err)
	}
	var names []string
	if err := json.Unmarshal(out, &names); err != nil {
		t.Fatalf("child probe returned invalid JSON (%d bytes): %v", len(out), err)
	}
	seen := make(map[string]bool, len(names))
	for _, n := range names {
		seen[n] = true
	}
	return seen
}

// TestSecretFloorChildProbe is the re-exec child body: when the sentinel is set
// it emits the NAMES of the environment it was launched with as JSON and exits,
// so the parent red-team test can assert canary absence without the probe ever
// echoing a secret value. It is inert in a normal test run (sentinel unset).
func TestSecretFloorChildProbe(t *testing.T) {
	if os.Getenv(secretFloorProbeSentinel) != "1" {
		return
	}
	names := make([]string, 0, len(os.Environ()))
	for _, kv := range os.Environ() {
		if i := strings.IndexByte(kv, '='); i >= 0 {
			names = append(names, kv[:i])
		}
	}
	_ = json.NewEncoder(os.Stdout).Encode(names)
	os.Exit(0)
}
