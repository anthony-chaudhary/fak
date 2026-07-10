package dispatchcache

import (
	"testing"
	"time"
)

func TestStoreTTLAndContentHashKey(t *testing.T) {
	now := time.Unix(100, 0)
	s := New[string](func() time.Time { return now })
	key := Key("repo", "current", 1000)
	if key == Key("repo", "other", 1000) || key == Key("repo", "current", 99) {
		t.Fatal("key did not separate inputs")
	}
	s.Put(key, "routed", time.Second)
	if got, ok := s.Get(key); !ok || got != "routed" {
		t.Fatalf("fresh = %q,%v", got, ok)
	}
	now = now.Add(time.Second)
	if _, ok := s.Get(key); ok {
		t.Fatal("entry remained fresh at TTL boundary")
	}
}
