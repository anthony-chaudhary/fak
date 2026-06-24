package wirescreen

import (
	"context"
	"strings"
	"testing"
)

// TestHeuristicDigesterAuthorsLeadingLines proves the reference Digester authors a
// bounded digest from the leading non-blank lines of a body (the zero-model floor for
// rung 3): blank lines are skipped, the cap holds, and an empty body declines.
func TestHeuristicDigesterAuthorsLeadingLines(t *testing.T) {
	h := heuristicDigester{}
	ctx := context.Background()

	// Leading distinct lines are kept in order; blank lines skipped.
	body := []byte("\n\nfirst line\n\nsecond line\n   \nthird line\n")
	got, ok := h.Summarize(ctx, body, "read_file")
	if !ok {
		t.Fatalf("expected a digest for a non-empty body, got ok=false")
	}
	want := "first line\nsecond line\nthird line"
	if got != want {
		t.Errorf("digest = %q, want %q", got, want)
	}
	if Digests() < 1 {
		t.Errorf("Digests() = %d, want >= 1 (a digest was authored)", Digests())
	}

	// A huge body is truncated to the ~200-token cap.
	huge := strings.Repeat("x", digestMaxBytes*4)
	got, ok = h.Summarize(ctx, []byte(huge), "dump")
	if !ok {
		t.Fatalf("huge body: expected a truncated digest, got ok=false")
	}
	if len(got) > digestMaxBytes {
		t.Errorf("digest length %d exceeds the %d-byte cap", len(got), digestMaxBytes)
	}
	if len(got) != digestMaxBytes {
		t.Errorf("digest of one long line should fill the cap exactly: got %d, want %d", len(got), digestMaxBytes)
	}

	// An empty body declines (no digest to author).
	if _, ok := h.Summarize(ctx, []byte(""), "x"); ok {
		t.Errorf("empty body must decline (ok=false)")
	}
	// A whitespace-only body also declines.
	if _, ok := h.Summarize(ctx, []byte("  \n\n  "), "x"); ok {
		t.Errorf("whitespace-only body must decline (ok=false)")
	}
}

// TestActiveDigesterInertWithoutEnv proves the default-inert contract: with
// FAK_WIRE_SCREEN unset (the test process default), ActiveDigester() returns nil, so the
// adapter never authors a digest and the MMU's oversize path is the opaque v0.1 pointer.
func TestActiveDigesterInertWithoutEnv(t *testing.T) {
	// Force the resolved state to "no selection" for this test.
	dmu.Lock()
	dactive, dactiveResolved = nil, true
	dmu.Unlock()

	if d := ActiveDigester(); d != nil {
		t.Fatalf("default-inert violated: ActiveDigester() = %v, want nil with FAK_WIRE_SCREEN unset", d)
	}
}

// TestRegisterDigester proves the catalog holds the reference digester at init under the
// shared selection name, so FAK_WIRE_SCREEN=heuristic activates both rungs.
func TestRegisterDigester(t *testing.T) {
	dmu.RLock()
	_, ok := dregistry["heuristic"]
	dmu.RUnlock()
	if !ok {
		t.Fatalf("the heuristic reference Digester must be registered at init under \"heuristic\"")
	}
}
