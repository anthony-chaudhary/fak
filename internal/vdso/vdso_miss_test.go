package vdso

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestVDSOMissReasons is the witness that a fast-path MISS is no longer silent:
// each ok=false is attributed to the reason that produced it, so a low hit rate
// is explainable instead of opaque.
func TestVDSOMissReasons(t *testing.T) {
	ctx := context.Background()
	v := New(8)

	// (1) No readOnly/idempotent hint -> the soundness gate can't prove cacheable.
	if _, ok := v.Lookup(ctx, &abi.ToolCall{
		Tool: "read_thing",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte("{}")},
	}); ok {
		t.Fatal("an unhinted call must miss")
	}
	// (2) Write-shaped tool, even WITH hints, is never fast-path eligible.
	if _, ok := v.Lookup(ctx, roCall("delete_record", `{"id":1}`)); ok {
		t.Fatal("a write-shaped call must miss")
	}
	// (3) Cacheable read, but nothing has filled it -> genuine not-cached.
	if _, ok := v.Lookup(ctx, roCall("read_unregistered", `{"x":1}`)); ok {
		t.Fatal("an uncached read must miss")
	}

	m := v.MissReasons()
	if m[MissMissingHints] != 1 {
		t.Errorf("%s = %d, want 1", MissMissingHints, m[MissMissingHints])
	}
	if m[MissDestructive] != 1 {
		t.Errorf("%s = %d, want 1", MissDestructive, m[MissDestructive])
	}
	if m[MissNotCached] != 1 {
		t.Errorf("%s = %d, want 1 (got reasons %v)", MissNotCached, m[MissNotCached], m)
	}

	// The full family is always present as keys so the metric never has a hole.
	for _, r := range []string{MissDestructive, MissMissingHints, MissResourceMisnamed, MissWitnessRevoked, MissNotCached} {
		if _, ok := m[r]; !ok {
			t.Errorf("MissReasons missing key %q", r)
		}
	}

	// A hit must NOT bump any miss counter: register a static answer and serve it.
	v.RegisterStatic("ping", []byte("pong"))
	if _, ok := v.Lookup(ctx, roCall("ping", `{}`)); !ok {
		t.Fatal("static tool should hit")
	}
	if got := v.MissReasons(); got[MissNotCached] != 1 {
		t.Errorf("a hit must not record a miss; NOT_CACHED = %d, want 1", got[MissNotCached])
	}
}
