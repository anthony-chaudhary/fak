package radixkv

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// safekv_epoch_test.go — the ADDITIVE namespace contract for the Context Epoch
// port. CacheIdentity.Epoch segments the private namespace so a prefix admitted
// under one system-context baseline is unreachable from its successor, while an
// empty epoch keeps the historical tenant/agent namespace byte-identical.

func TestScopeNamespaceEpochSegmentIsolatesPrefixes(t *testing.T) {
	cache := NewScoped(0)
	cfg := model.Config{NumLayers: 1, NumKVHeads: 1, HeadDim: 1}
	prefix := []int{91, 7, 42}
	old := CacheIdentity{Tenant: "t", Epoch: "sess@7"}
	if err := cache.AdmitPrivateSnapshot(old, prefix, &model.PrefixSnapshot{Cache: model.NewKVCache(cfg)}, nil); err != nil {
		t.Fatal(err)
	}

	// The successor epoch must not see the superseded prefix.
	newer := CacheIdentity{Tenant: "t", Epoch: "sess@8"}
	if got, err := cache.MatchLen(newer, prefix); err != nil || got != 0 {
		t.Fatalf("MatchLen under successor epoch = %d err=%v, want 0/nil", got, err)
	}
	// The owning epoch still sees it.
	if got, err := cache.MatchLen(old, prefix); err != nil || got != len(prefix) {
		t.Fatalf("MatchLen under owning epoch = %d err=%v, want %d/nil", got, err, len(prefix))
	}
	// Snapshot lookup agrees with the structural isolation.
	if snap, _, matched, _, err := cache.LookupSnapshot(newer, prefix); err != nil || snap != nil || matched != 0 {
		t.Fatalf("successor epoch snapshot = %v matched=%d err=%v, want nil/0/nil", snap, matched, err)
	}
	snap, _, matched, _, err := cache.LookupSnapshot(old, prefix)
	if err != nil {
		t.Fatal(err)
	}
	if snap == nil || matched != len(prefix) {
		t.Fatalf("owning epoch snapshot = %v matched=%d, want non-nil/%d", snap, matched, len(prefix))
	}
	snap.Close()
}

func TestScopeNamespaceWithoutEpochIsUnchanged(t *testing.T) {
	owner := CacheIdentity{Tenant: "t", Agent: "a"}
	cases := []struct {
		scope ShareScope
		want  string
	}{
		{ScopeTenant, "private/tenant/t"},
		{ScopeAgent, "private/tenant/t/agent/a"},
		// Namespace-string case only; TestScopeNamespaceFleetExemptionCrossesEpochs
		// adds the behavioural witness that a successor epoch still reaches it.
		{ScopeFleet, "shared/fleet"},
	}
	for _, tc := range cases {
		got, err := scopeNamespace(tc.scope, owner)
		if err != nil {
			t.Fatalf("scopeNamespace(%d): %v", tc.scope, err)
		}
		if got != tc.want {
			t.Fatalf("scopeNamespace(%d) = %q, want %q", tc.scope, got, tc.want)
		}
	}
}

// TestScopeNamespaceFleetExemptionCrossesEpochs PINS the documented ScopeFleet
// exemption: fleet visibility is an explicit Promote event, so scopeNamespace keeps
// the epoch segment OFF for ScopeFleet and a fleet-promoted prefix IS reachable by
// an owner in ANY epoch. A future change that segments fleet scope by epoch MUST
// break this test loudly.
func TestScopeNamespaceFleetExemptionCrossesEpochs(t *testing.T) {
	cache := NewScoped(0)
	cfg := model.Config{NumLayers: 1, NumKVHeads: 1, HeadDim: 1}
	prefix := []int{11, 22, 33}
	old := CacheIdentity{Tenant: "t", Epoch: "sess@7"}
	if err := cache.AdmitPrivate(old, prefix, model.NewKVCache(cfg), nil); err != nil {
		t.Fatal(err)
	}
	if err := cache.Promote(ScopeTenant, old, prefix); err != nil {
		t.Fatalf("Promote: %v", err)
	}

	// Documented exemption: a successor-epoch owner still sees the fleet-promoted
	// prefix because ScopeFleet drops the epoch segment on purpose.
	newer := CacheIdentity{Tenant: "t", Epoch: "sess@8"}
	if got, err := cache.MatchLen(newer, prefix); err != nil || got != len(prefix) {
		t.Fatalf("MatchLen under successor epoch for fleet-promoted prefix = %d err=%v, want %d/nil (ScopeFleet exemption)", got, err, len(prefix))
	}
	kv, _, matched, scope, err := cache.Lookup(newer, prefix)
	if err != nil {
		t.Fatal(err)
	}
	if kv == nil || matched != len(prefix) {
		t.Fatalf("successor-epoch lookup = %v matched=%d, want non-nil/%d (ScopeFleet exemption)", kv, matched, len(prefix))
	}
	if scope != ScopeFleet {
		t.Fatalf("successor-epoch lookup scope = %d, want ScopeFleet", scope)
	}
}

func TestScopeNamespaceEpochAppliesToAgentAndTenantScopes(t *testing.T) {
	owner := CacheIdentity{Tenant: "t", Agent: "a", Epoch: "sess@7"}
	cases := []struct {
		scope ShareScope
		want  string
	}{
		{ScopeTenant, "private/tenant/t/epoch/sess@7"},
		{ScopeAgent, "private/tenant/t/agent/a/epoch/sess@7"},
		{ScopeFleet, "shared/fleet"},
	}
	for _, tc := range cases {
		got, err := scopeNamespace(tc.scope, owner)
		if err != nil {
			t.Fatalf("scopeNamespace(%d): %v", tc.scope, err)
		}
		if got != tc.want {
			t.Fatalf("scopeNamespace(%d) = %q, want %q", tc.scope, got, tc.want)
		}
	}
}
