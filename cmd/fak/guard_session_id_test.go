package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

func TestResolveGuardSessionIDFreshensImplicitDurableLaunches(t *testing.T) {
	meta := session.DescriptorMeta{CacheKey: "0123456789abcdefrest-of-cache-key"}
	first := resolveGuardSessionID("", true, meta, "launch-a")
	second := resolveGuardSessionID("", true, meta, "launch-b")
	if first != "guard-0123456789abcdef-launch-a" {
		t.Fatalf("first durable id = %q", first)
	}
	if first == second {
		t.Fatalf("two fresh durable launches reused trace %q", first)
	}
}

func TestResolveGuardSessionIDPreservesExplicitResumeIdentity(t *testing.T) {
	got := resolveGuardSessionID("  resume-me  ", true, session.DescriptorMeta{CacheKey: "ignored"}, "ignored")
	if got != "resume-me" {
		t.Fatalf("explicit session id = %q, want resume-me", got)
	}
}

func TestResolveGuardSessionIDKeepsOrdinaryLaunchProcessLocal(t *testing.T) {
	got := resolveGuardSessionID("", false, session.DescriptorMeta{CacheKey: "ignored"}, "ignored")
	if got != "guard" {
		t.Fatalf("ordinary implicit session id = %q, want guard", got)
	}
}

func TestNewGuardLaunchNonceIsNonEmptyAndChanges(t *testing.T) {
	a, b := newGuardLaunchNonce(), newGuardLaunchNonce()
	if a == "" || b == "" {
		t.Fatalf("empty launch nonce: %q %q", a, b)
	}
	if a == b {
		t.Fatalf("launch nonce repeated: %q", a)
	}
}
