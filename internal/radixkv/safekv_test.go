package radixkv

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestPrivatePrefixInvisibleUntilFleetPromotion(t *testing.T) {
	cache := NewScoped(0)
	a := CacheIdentity{Tenant: "tenant-a", Agent: "worker-1"}
	b := CacheIdentity{Tenant: "tenant-b", Agent: "worker-2"}
	prefix := []int{91, 7, 42, 11}
	kv := model.NewKVCache(model.Config{NumLayers: 1, NumKVHeads: 1, HeadDim: 1})
	kv.K[0] = []float32{1, 2, 3, 4}
	kv.V[0] = []float32{5, 6, 7, 8}
	logits := []float32{.1, .9}
	if err := cache.AdmitPrivate(a, prefix, kv, logits); err != nil {
		t.Fatal(err)
	}
	_, _, matchedA, scopeA, err := cache.Lookup(a, prefix)
	if err != nil || matchedA != len(prefix) || scopeA != ScopeTenant {
		t.Fatalf("owner lookup matched=%d scope=%d err=%v", matchedA, scopeA, err)
	}
	_, _, matchedB, _, err := cache.Lookup(b, prefix)
	if err != nil || matchedB != 0 {
		t.Fatalf("cross-tenant private prefix exposed a hit-shaped match: matched=%d err=%v", matchedB, err)
	}
	if err := cache.Promote(ScopeTenant, a, prefix); err != nil {
		t.Fatal(err)
	}
	gotKV, gotLogits, matchedB, scopeB, err := cache.Lookup(b, prefix)
	if err != nil || matchedB != len(prefix) || scopeB != ScopeFleet {
		t.Fatalf("promoted lookup matched=%d scope=%d err=%v", matchedB, scopeB, err)
	}
	if gotKV == kv || !reflect.DeepEqual(gotLogits, logits) {
		t.Fatalf("promotion must copy payload: kv_alias=%v logits=%v", gotKV == kv, gotLogits)
	}
	if freed := cache.RevokeFleet(prefix); freed == 0 {
		t.Fatal("revocation freed no promoted tokens")
	}
	_, _, matchedB, _, _ = cache.Lookup(b, prefix)
	if matchedB != 0 {
		t.Fatalf("revoked fleet prefix remains visible: matched=%d", matchedB)
	}
	_, _, matchedA, scopeA, _ = cache.Lookup(a, prefix)
	if matchedA != len(prefix) || scopeA != ScopeTenant {
		t.Fatalf("fleet revocation damaged private source: matched=%d scope=%d", matchedA, scopeA)
	}
}

func TestScopedPrefixAdmissionFailsClosed(t *testing.T) {
	cache := NewScoped(0)
	prefix := []int{1, 2}
	if err := cache.AdmitPrivate(CacheIdentity{}, prefix, nil, nil); !errors.Is(err, ErrCacheIdentity) {
		t.Fatalf("missing tenant err=%v", err)
	}
	if err := cache.Admit(ScopeFleet, CacheIdentity{Tenant: "a"}, prefix, nil, nil); !errors.Is(err, ErrCacheScope) {
		t.Fatalf("direct fleet admission err=%v", err)
	}
	if err := cache.Promote(ScopeTenant, CacheIdentity{Tenant: "a"}, prefix); !errors.Is(err, ErrPrefixAbsent) {
		t.Fatalf("absent promotion err=%v", err)
	}
}

func TestScopedSnapshotLookupChoosesLongestCompleteSnapshotAcrossScopes(t *testing.T) {
	cache := NewScoped(0)
	owner := CacheIdentity{Tenant: "tenant-a", Agent: "worker-1"}
	cfg := model.Config{NumLayers: 1, NumKVHeads: 1, HeadDim: 1}
	tenantSnapshot := &model.PrefixSnapshot{Cache: model.NewKVCache(cfg)}
	if err := cache.AdmitPrivateSnapshot(owner, []int{1, 2}, tenantSnapshot, []float32{7}); err != nil {
		t.Fatal(err)
	}
	// This longer agent-private radix match has no complete device snapshot. It must
	// not mask the shorter tenant snapshot that is still visible to the same owner.
	if err := cache.Admit(ScopeAgent, owner, []int{1, 2, 3}, model.NewKVCache(cfg), nil); err != nil {
		t.Fatal(err)
	}
	got, logits, matched, scope, err := cache.LookupSnapshot(owner, []int{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	if got == nil {
		t.Fatal("longer snapshot-less agent path masked tenant device snapshot")
	}
	defer got.Close()
	if matched != 2 || scope != ScopeTenant {
		t.Fatalf("matched=%d scope=%d want matched=2 scope=%d", matched, scope, ScopeTenant)
	}
	if !reflect.DeepEqual(logits, []float32{7}) {
		t.Fatalf("logits=%v want [7]", logits)
	}
}
