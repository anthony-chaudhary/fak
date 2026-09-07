package policy

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestLowerTierCannotWiden is the issue #2408 acceptance test: proves that a
// project-tier allow exceeding the managed floor is dropped or clamped at compile
// time, enforcing meet-only merge semantics.
func TestLowerTierCannotWiden(t *testing.T) {
	ctx := context.Background()

	// Managed tier (higher trust) defines the outer capability floor.
	managedManifest := Manifest{
		Version: Version,
		Allow:   []string{"read_file", "list_dir"},
	}

	// Project tier (lower trust) attempts to allow both an admitted tool and
	// unauthorized tools exceeding the managed floor.
	projectManifest := Manifest{
		Version: Version,
		Allow:   []string{"read_file", "delete_database", "bash"},
	}

	managedLayer := NewTierLayer(TierManaged, managedManifest)
	projectLayer := NewTierLayer(TierProject, projectManifest)

	// 1. Compile effective manifest: verify project-tier additions exceeding
	// the managed floor are dropped.
	effective, err := CompileTieredManifest(managedLayer, projectLayer)
	if err != nil {
		t.Fatalf("compile tiered manifest: %v", err)
	}

	// Only read_file should be allowed (delete_database and bash dropped)
	if len(effective.Allow) != 1 || effective.Allow[0] != "read_file" {
		t.Fatalf("effective Allow = %v; want only [read_file]", effective.Allow)
	}

	// 2. Resolver and adjudicator verification:
	resolver, err := NewTieredPolicyResolver(managedLayer, projectLayer)
	if err != nil {
		t.Fatalf("new tiered policy resolver: %v", err)
	}

	// read_file was in both tiers -> admitted and allowed
	vRead := resolver.Adjudicate(ctx, &abi.ToolCall{Tool: "read_file"})
	if vRead.Kind != abi.VerdictAllow {
		t.Fatalf("read_file: got %v (reason=%v), want VerdictAllow", vRead.Kind, vRead.Reason)
	}
	if epoch := vRead.Meta["policy_epoch"]; epoch != resolver.Epoch() {
		t.Fatalf("read_file verdict policy_epoch = %q, want %q", epoch, resolver.Epoch())
	}

	// delete_database exceeded the managed floor -> dropped -> DEFAULT_DENY
	vDelete := resolver.Adjudicate(ctx, &abi.ToolCall{Tool: "delete_database"})
	if vDelete.Kind != abi.VerdictDeny || vDelete.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("delete_database: got %v (reason=%s), want VerdictDeny / DEFAULT_DENY", vDelete.Kind, abi.ReasonName(vDelete.Reason))
	}

	// bash exceeded the managed floor -> dropped -> DEFAULT_DENY
	vBash := resolver.Adjudicate(ctx, &abi.ToolCall{Tool: "bash"})
	if vBash.Kind != abi.VerdictDeny || vBash.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("bash: got %v (reason=%s), want VerdictDeny / DEFAULT_DENY", vBash.Kind, abi.ReasonName(vBash.Reason))
	}
}

// TestLowerTierCannotWidenPrefix proves that lower-tier prefix allowances exceeding
// the higher-tier floor are dropped or clamped.
func TestLowerTierCannotWidenPrefix(t *testing.T) {
	ctx := context.Background()

	managedManifest := Manifest{
		Version:     Version,
		AllowPrefix: []string{"mcp_safe_"},
	}

	projectManifest := Manifest{
		Version:     Version,
		Allow:       []string{"mcp_safe_read", "dangerous_tool"},
		AllowPrefix: []string{"mcp_safe_sub_", "untrusted_"},
	}

	managedLayer := NewTierLayer(TierManaged, managedManifest)
	projectLayer := NewTierLayer(TierProject, projectManifest)

	effective, err := CompileTieredManifest(managedLayer, projectLayer)
	if err != nil {
		t.Fatalf("compile tiered manifest: %v", err)
	}

	// dangerous_tool does not match mcp_safe_ -> dropped
	for _, a := range effective.Allow {
		if a == "dangerous_tool" {
			t.Fatal("dangerous_tool should have been dropped")
		}
	}
	// untrusted_ does not start with mcp_safe_ -> dropped
	for _, p := range effective.AllowPrefix {
		if p == "untrusted_" {
			t.Fatal("untrusted_ prefix should have been dropped")
		}
	}

	resolver, err := NewTieredPolicyResolver(managedLayer, projectLayer)
	if err != nil {
		t.Fatalf("new resolver: %v", err)
	}

	// mcp_safe_read matches mcp_safe_ -> allowed
	vSafe := resolver.Adjudicate(ctx, &abi.ToolCall{Tool: "mcp_safe_read"})
	if vSafe.Kind != abi.VerdictAllow {
		t.Fatalf("mcp_safe_read: got %v, want VerdictAllow", vSafe.Kind)
	}

	// dangerous_tool was dropped -> denied
	vDang := resolver.Adjudicate(ctx, &abi.ToolCall{Tool: "dangerous_tool"})
	if vDang.Kind != abi.VerdictDeny {
		t.Fatalf("dangerous_tool: got %v, want VerdictDeny", vDang.Kind)
	}
}

// TestConfigDriftRefuses is the issue #2408 acceptance test: shows that mutating a layer
// file or input mid-session causes CheckConfigDrift and Adjudicate to refuse with
// typed reason CONFIG_DRIFT.
func TestConfigDriftRefuses(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()

	// Create initial layer files
	managedFile := filepath.Join(tmpDir, "managed.json")
	if err := os.WriteFile(managedFile, []byte(`{
		"version": "fak-policy/v1",
		"allow": ["read_file", "list_dir"]
	}`), 0644); err != nil {
		t.Fatalf("write managed: %v", err)
	}

	projectFile := filepath.Join(tmpDir, "project.json")
	if err := os.WriteFile(projectFile, []byte(`{
		"version": "fak-policy/v1",
		"allow": ["read_file"]
	}`), 0644); err != nil {
		t.Fatalf("write project: %v", err)
	}

	managedLayer, err := NewTierLayerFromFile(TierManaged, managedFile)
	if err != nil {
		t.Fatalf("new managed layer: %v", err)
	}
	projectLayer, err := NewTierLayerFromFile(TierProject, projectFile)
	if err != nil {
		t.Fatalf("new project layer: %v", err)
	}

	resolver, err := NewTieredPolicyResolver(managedLayer, projectLayer)
	if err != nil {
		t.Fatalf("new tiered resolver: %v", err)
	}

	// Pre-mutation check: no drift
	refusal, err := CheckConfigDrift(resolver)
	if err != nil || refusal != nil {
		t.Fatalf("expected no drift initially, got refusal=%v, err=%v", refusal, err)
	}

	// Pre-mutation adjudication: succeeds with policy epoch stamped
	v1 := resolver.Adjudicate(ctx, &abi.ToolCall{Tool: "read_file"})
	if v1.Kind != abi.VerdictAllow {
		t.Fatalf("pre-mutation call: got %v, want VerdictAllow", v1.Kind)
	}
	if v1.Meta["policy_epoch"] != resolver.Epoch() {
		t.Fatalf("epoch mismatch: got %q, want %q", v1.Meta["policy_epoch"], resolver.Epoch())
	}

	// Mid-session mutation: mutate the project file on disk
	if err := os.WriteFile(projectFile, []byte(`{
		"version": "fak-policy/v1",
		"allow": ["read_file", "delete_database"]
	}`), 0644); err != nil {
		t.Fatalf("mutate project file: %v", err)
	}

	// Post-mutation CheckConfigDrift: flags drift with typed refusal CONFIG_DRIFT
	driftRefusal, _ := CheckConfigDrift(resolver)
	if driftRefusal == nil {
		t.Fatal("expected config drift refusal after mutating project layer file, got nil")
	}
	if driftRefusal.Reason != ReasonConfigDriftName {
		t.Fatalf("drift refusal reason = %q, want %q", driftRefusal.Reason, ReasonConfigDriftName)
	}
	if driftRefusal.ReasonCode != ReasonConfigDrift {
		t.Fatalf("drift refusal reason code = %v, want %v", driftRefusal.ReasonCode, ReasonConfigDrift)
	}
	if driftRefusal.Tier != TierProject {
		t.Fatalf("drift refusal tier = %v, want %v", driftRefusal.Tier, TierProject)
	}

	// Post-mutation Adjudicate: immediately refuses with CONFIG_DRIFT
	v2 := resolver.Adjudicate(ctx, &abi.ToolCall{Tool: "read_file"})
	if v2.Kind != abi.VerdictDeny {
		t.Fatalf("post-mutation call: got Kind=%v, want VerdictDeny", v2.Kind)
	}
	if v2.Reason != ReasonConfigDrift {
		t.Fatalf("post-mutation call: got Reason=%v (%s), want %v (CONFIG_DRIFT)",
			v2.Reason, abi.ReasonName(v2.Reason), ReasonConfigDrift)
	}
	if v2.Meta["reason"] != ReasonConfigDriftName {
		t.Fatalf("post-mutation call Meta[reason] = %q, want %q", v2.Meta["reason"], ReasonConfigDriftName)
	}
	if v2.Meta["policy_epoch"] != resolver.Epoch() {
		t.Fatalf("post-mutation call Meta[policy_epoch] = %q, want %q", v2.Meta["policy_epoch"], resolver.Epoch())
	}
}

// TestConfigDriftInMemoryMutate verifies in-memory layer mutations are caught as drift.
func TestConfigDriftInMemoryMutate(t *testing.T) {
	ctx := context.Background()

	managedLayer := NewTierLayer(TierManaged, Manifest{
		Version: Version,
		Allow:   []string{"toolA"},
	})
	userLayer := NewTierLayer(TierUser, Manifest{
		Version: Version,
		Allow:   []string{"toolA"},
	})

	resolver, err := NewTieredPolicyResolver(managedLayer, userLayer)
	if err != nil {
		t.Fatalf("new resolver: %v", err)
	}

	// Initial adjudication works
	v1 := resolver.Adjudicate(ctx, &abi.ToolCall{Tool: "toolA"})
	if v1.Kind != abi.VerdictAllow {
		t.Fatalf("initial adjudication got %v, want VerdictAllow", v1.Kind)
	}

	// Mutate user layer in-memory
	resolver.MutateLayer(TierUser, []byte(`{"version":"fak-policy/v1","allow":["toolB"]}`))

	// Drift is detected
	refusal, _ := CheckConfigDrift(resolver)
	if refusal == nil {
		t.Fatal("expected drift refusal after in-memory mutation")
	}
	if refusal.Reason != ReasonConfigDriftName {
		t.Fatalf("refusal reason = %q, want %q", refusal.Reason, ReasonConfigDriftName)
	}

	// Subsequent adjudication refuses
	v2 := resolver.Adjudicate(ctx, &abi.ToolCall{Tool: "toolA"})
	if v2.Kind != abi.VerdictDeny || v2.Reason != ReasonConfigDrift {
		t.Fatalf("got %v/%s, want Deny/CONFIG_DRIFT", v2.Kind, abi.ReasonName(v2.Reason))
	}
}

// TestComputePolicyEpochDeterministic verifies that ComputePolicyEpoch produces
// identical hex hashes for identical inputs regardless of invocation order.
func TestComputePolicyEpochDeterministic(t *testing.T) {
	managed := NewTierLayer(TierManaged, Manifest{
		Version: Version,
		Allow:   []string{"read_file", "list_dir"},
	})
	user := NewTierLayer(TierUser, Manifest{
		Version: Version,
		Deny:    map[string]string{"list_dir": "POLICY_BLOCK"},
	})
	project := NewTierLayer(TierProject, Manifest{
		Version: Version,
		Allow:   []string{"read_file"},
	})

	effective, err := CompileTieredManifest(managed, user, project)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	epoch1 := ComputePolicyEpoch(effective, managed, user, project)
	epoch2 := ComputePolicyEpoch(effective, managed, user, project)

	if epoch1 == "" || epoch2 == "" {
		t.Fatal("empty epoch returned")
	}
	if epoch1 != epoch2 {
		t.Fatalf("epochs differ on identical inputs: %q vs %q", epoch1, epoch2)
	}

	// Reverse order of layer arguments: should produce identical epoch because
	// layers are stably sorted by tier rank internally.
	epochReversed := ComputePolicyEpoch(effective, project, user, managed)
	if epoch1 != epochReversed {
		t.Fatalf("epochs differ under layer argument reordering: %q vs %q", epoch1, epochReversed)
	}
}

// TestPolicyEpochChangesOnMutation verifies that altering any layer changes the epoch.
func TestPolicyEpochChangesOnMutation(t *testing.T) {
	m1 := Manifest{Version: Version, Allow: []string{"read_file"}}
	m2 := Manifest{Version: Version, Allow: []string{"read_file", "list_dir"}}

	layerA := NewTierLayer(TierManaged, m1)
	layerB := NewTierLayer(TierManaged, m2)

	epochA := ComputePolicyEpoch(m1, layerA)
	epochB := ComputePolicyEpoch(m2, layerB)

	if epochA == epochB {
		t.Fatalf("expected different epochs for different manifests, got same: %q", epochA)
	}
}

// TestHermeticPolicyEpoch verifies that hermetic mode ignores ambient tiers
// and produces a provably distinct epoch from non-hermetic execution.
func TestHermeticPolicyEpoch(t *testing.T) {
	managed := NewTierLayer(TierManaged, Manifest{
		Version: Version,
		Allow:   []string{"read_file", "list_dir"},
	})
	ambientUser := NewTierLayer(TierUser, Manifest{
		Version: Version,
		Deny:    map[string]string{"list_dir": "POLICY_BLOCK"},
	})

	// Non-hermetic resolver compiles user tier
	nonHermetic, err := NewTieredPolicyResolver(managed, ambientUser)
	if err != nil {
		t.Fatalf("non-hermetic: %v", err)
	}

	// Hermetic resolver ignores ambientUser
	hermetic, err := NewTieredPolicyResolverWithOptions(TieredResolverOptions{Hermetic: true}, managed, ambientUser)
	if err != nil {
		t.Fatalf("hermetic: %v", err)
	}

	if !hermetic.Hermetic() {
		t.Fatal("expected Hermetic() to be true")
	}
	if len(hermetic.Layers()) != 1 {
		t.Fatalf("hermetic layers count = %d, want 1 (only managed)", len(hermetic.Layers()))
	}
	if hermetic.Epoch() == nonHermetic.Epoch() {
		t.Fatalf("hermetic epoch must differ from non-hermetic epoch: got %q", hermetic.Epoch())
	}
}

// TestTieredDenyMerges verifies that deny rules from all tiers merge.
func TestTieredDenyMerges(t *testing.T) {
	managed := NewTierLayer(TierManaged, Manifest{
		Version: Version,
		Deny:    map[string]string{"rm_rf": "POLICY_BLOCK"},
	})
	project := NewTierLayer(TierProject, Manifest{
		Version: Version,
		Deny:    map[string]string{"git_push": "POLICY_BLOCK"},
	})

	effective, err := CompileTieredManifest(managed, project)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	if effective.Deny["rm_rf"] != "POLICY_BLOCK" {
		t.Fatalf("expected rm_rf in Deny, got %v", effective.Deny)
	}
	if effective.Deny["git_push"] != "POLICY_BLOCK" {
		t.Fatalf("expected git_push in Deny, got %v", effective.Deny)
	}
}

// TestTieredArgRulesMerge verifies that ArgRules from multiple tiers merge.
func TestTieredArgRulesMerge(t *testing.T) {
	managed := NewTierLayer(TierManaged, Manifest{
		Version: Version,
		ArgRules: []ArgRule{
			{Tool: "bash", Arg: "command", DenyRegex: "rm\\s+-rf.*", Reason: "POLICY_BLOCK"},
		},
	})
	project := NewTierLayer(TierProject, Manifest{
		Version: Version,
		ArgRules: []ArgRule{
			{Tool: "bash", Arg: "command", DenyRegex: "curl.*", Reason: "POLICY_BLOCK"},
		},
	})

	effective, err := CompileTieredManifest(managed, project)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	if len(effective.ArgRules) != 2 {
		t.Fatalf("expected 2 ArgRules, got %d", len(effective.ArgRules))
	}
}

// TestTieredRateLimitClamped verifies that lower tiers cannot widen rate limits.
func TestTieredRateLimitClamped(t *testing.T) {
	managed := NewTierLayer(TierManaged, Manifest{
		Version:   Version,
		RateLimit: &RateLimitRule{MaxCalls: 100},
	})
	projectWiden := NewTierLayer(TierProject, Manifest{
		Version:   Version,
		RateLimit: &RateLimitRule{MaxCalls: 200}, // Attempt to loosen
	})
	projectNarrow := NewTierLayer(TierProject, Manifest{
		Version:   Version,
		RateLimit: &RateLimitRule{MaxCalls: 50}, // Tighten
	})

	// Widen attempt clamped to managed ceiling
	eff1, err := CompileTieredManifest(managed, projectWiden)
	if err != nil {
		t.Fatalf("compile widen: %v", err)
	}
	if eff1.RateLimit.MaxCalls != 100 {
		t.Fatalf("expected MaxCalls clamped to 100, got %d", eff1.RateLimit.MaxCalls)
	}

	// Narrow attempt accepted
	eff2, err := CompileTieredManifest(managed, projectNarrow)
	if err != nil {
		t.Fatalf("compile narrow: %v", err)
	}
	if eff2.RateLimit.MaxCalls != 50 {
		t.Fatalf("expected MaxCalls narrowed to 50, got %d", eff2.RateLimit.MaxCalls)
	}
}

// TestTieredFailLoudValidation verifies that unknown fields or corrupt JSON fail loudly.
func TestTieredFailLoudValidation(t *testing.T) {
	// Unknown field in project manifest
	_, err := NewTierLayerFromBytes(TierProject, []byte(`{"version":"fak-policy/v1","allows":["read_file"]}`))
	if err == nil {
		t.Fatal("expected error on unknown field 'allows', got nil")
	}

	// Invalid refusal reason name
	_, err = NewTierLayerFromBytes(TierProject, []byte(`{"version":"fak-policy/v1","deny":{"tool":"NOT_A_REAL_REASON"}}`))
	if err == nil {
		t.Fatal("expected error on invalid deny reason, got nil")
	}
}

// TestFormatResolveProvenance verifies FormatResolve output contains per-rule provenance.
func TestFormatResolveProvenance(t *testing.T) {
	managed := NewTierLayer(TierManaged, Manifest{
		Version: Version,
		Allow:   []string{"read_file"},
	})
	project := NewTierLayer(TierProject, Manifest{
		Version: Version,
		Deny:    map[string]string{"write_file": "POLICY_BLOCK"},
	})

	resolver, err := NewTieredPolicyResolver(managed, project)
	if err != nil {
		t.Fatalf("new resolver: %v", err)
	}

	formatted := resolver.FormatResolve()
	if !strings.Contains(formatted, "read_file [managed]") {
		t.Fatalf("formatted output missing read_file [managed]:\n%s", formatted)
	}
	if !strings.Contains(formatted, "write_file: POLICY_BLOCK [project]") {
		t.Fatalf("formatted output missing write_file: POLICY_BLOCK [project]:\n%s", formatted)
	}
}
