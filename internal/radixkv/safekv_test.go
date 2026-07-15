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
