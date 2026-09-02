// Package main tests for the security-fleet-governance demo harness: the repo
// anchor the demo resolves at runtime, the copyFile primitive behind its
// active-policy workflow, the contains verdict helper, and the two policy
// manifests the demo feeds to `fak policy` and greps after landing. Resource
// and subprocess free: the demo's `go run ./cmd/fak` path is exercised by CI,
// never by these tests.
package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRepoRootAnchorsDemoInputs(t *testing.T) {
	root, err := repoRoot()
	if err != nil {
		t.Fatalf("repoRoot: %v", err)
	}
	for _, need := range []string{
		filepath.Join(root, "examples", "security-fleet-governance", "central-floor.json"),
		filepath.Join(root, "examples", "security-fleet-governance", "identity-narrowing.json"),
		filepath.Join(root, "cmd", "fak"),
	} {
		if _, err := os.Stat(need); err != nil {
			t.Fatalf("demo anchor missing: %s", need)
		}
	}
}

func TestCopyFile(t *testing.T) {
	t.Run("round trips content", func(t *testing.T) {
		dir := t.TempDir()
		src := filepath.Join(dir, "src.json")
		dst := filepath.Join(dir, "active-policy.json")
		if err := os.WriteFile(src, []byte(`{"version":"fak-policy/v1"}`), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := copyFile(src, dst); err != nil {
			t.Fatalf("copyFile: %v", err)
		}
		b, err := os.ReadFile(dst)
		if err != nil {
			t.Fatalf("read copy: %v", err)
		}
		if string(b) != `{"version":"fak-policy/v1"}` {
			t.Fatalf("copy content = %q", b)
		}
	})
	t.Run("missing source is an error", func(t *testing.T) {
		dir := t.TempDir()
		if err := copyFile(filepath.Join(dir, "absent.json"), filepath.Join(dir, "dst.json")); err == nil {
			t.Fatal("copyFile accepted a missing source")
		}
	})
	t.Run("unwritable destination is an error", func(t *testing.T) {
		dir := t.TempDir()
		src := filepath.Join(dir, "src.json")
		if err := os.WriteFile(src, []byte("{}"), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := copyFile(src, dir); err == nil {
			t.Fatal("copyFile accepted a directory as destination")
		}
	})
}

func TestContains(t *testing.T) {
	// The pass path must return without tripping must(); the failure path
	// exits the process, so it is covered by the demo run itself.
	contains("ALLOW read_corp_kb", "ALLOW")
}

func TestFloorManifestDeniesByDefaultAndGatesKBReads(t *testing.T) {
	b, err := os.ReadFile("central-floor.json")
	if err != nil {
		t.Fatalf("read floor manifest: %v", err)
	}
	var floor struct {
		Version   string            `json:"version"`
		Posture   string            `json:"posture"`
		Allow     []string          `json:"allow"`
		Deny      map[string]string `json:"deny"`
		SafeSinks []string          `json:"safe_sinks"`
		Sources   map[string]string `json:"sources"`
		ArgRules  []map[string]any  `json:"arg_rules"`
	}
	if err := json.Unmarshal(b, &floor); err != nil {
		t.Fatalf("floor manifest is not valid JSON: %v", err)
	}
	if floor.Posture != "fail_closed" {
		t.Fatalf("posture = %q, want fail_closed (the demo's default-deny floor)", floor.Posture)
	}
	if floor.Version != "fak-policy/v1" {
		t.Fatalf("version = %q", floor.Version)
	}
	if !containsString(floor.Allow, "read_corp_kb") {
		t.Fatal("floor does not allow read_corp_kb, so the demo's ALLOW preflight cannot pass")
	}
	for _, denied := range []string{"export_customer_data", "rotate_credentials", "transfer_funds"} {
		if floor.Deny[denied] != "POLICY_BLOCK" {
			t.Fatalf("deny[%s] = %q, want POLICY_BLOCK", denied, floor.Deny[denied])
		}
	}
	if !hasArgRule(floor.ArgRules, "read_corp_kb", "uri", "corp://security/**") {
		t.Fatal("floor lacks the corp://security/** allow_glob that makes the unapproved-URL preflight DENY")
	}
	if floor.Sources["read_corp_kb"] != "trusted_local" {
		t.Fatalf("read_corp_kb source = %q, want trusted_local", floor.Sources["read_corp_kb"])
	}
	if !containsString(floor.SafeSinks, "create_support_ticket") {
		t.Fatal("create_support_ticket is not a safe sink")
	}
}

func TestNarrowingCandidateTargetsIdentityScopeOnly(t *testing.T) {
	b, err := os.ReadFile("identity-narrowing.json")
	if err != nil {
		t.Fatalf("read narrowing manifest: %v", err)
	}
	var candidate struct {
		ArgRules []map[string]any `json:"arg_rules"`
	}
	if err := json.Unmarshal(b, &candidate); err != nil {
		t.Fatalf("narrowing manifest is not valid JSON: %v", err)
	}
	// The demo greps the landed preview for exactly this rendered glob.
	if !strings.Contains(string(b), `"allow_glob": "corp://security/identity/**"`) {
		t.Fatal("narrowing candidate does not render the identity glob the demo asserts")
	}
	if !hasArgRule(candidate.ArgRules, "read_corp_kb", "uri", "corp://security/identity/**") {
		t.Fatal("candidate does not narrow read_corp_kb to the identity scope")
	}
}

func containsString(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

func hasArgRule(rules []map[string]any, tool, arg, glob string) bool {
	for _, r := range rules {
		if r["tool"] == tool && r["arg"] == arg && r["allow_glob"] == glob {
			return true
		}
	}
	return false
}
