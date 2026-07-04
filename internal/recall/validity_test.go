package recall

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

func TestBoundedValidityAsOfGate(t *testing.T) {
	ctx := context.Background()
	body := []byte("refund fee is 25 EUR until tick 100")
	d := Digest(body)
	s := &Session{
		Manifest: Manifest{Version: ManifestVersion, Pages: []Page{{
			Step:       0,
			Role:       "read_memory",
			Descriptor: "read_memory: refund fee is 25 EUR until tick 100",
			Digest:     d,
			Len:        int64(len(body)),
			Durability: durabilityBounded,
			ValidTo:    100,
		}}},
		cas:     map[string][]byte{d: body},
		cleared: map[string]bool{},
		gate:    ctxmmu.New(),
	}

	got, err := s.Resolve(ctx, 0, 50)
	if err != nil {
		t.Fatalf("bounded page should resolve inside validity interval: %v", err)
	}
	if string(got) != string(body) {
		t.Fatalf("bounded page not byte-identical: got %q want %q", got, body)
	}
	if set := s.Recall(ctx, "refund fee", 3, 50); len(set) != 1 || set[0].Step != 0 {
		t.Fatalf("bounded page should be recalled inside interval, got %+v", set)
	}

	delete(s.cas, d) // proves expiry is checked before the CAS fetch.
	if _, err := s.Resolve(ctx, 0, 150); !errors.Is(err, ErrExpired) {
		t.Fatalf("expired bounded page should return ErrExpired before CAS fetch, got %v", err)
	}
	if set := s.Recall(ctx, "refund fee", 3, 150); len(set) != 0 {
		t.Fatalf("expired bounded page leaked into recall working set: %+v", set)
	}
}

// boundedPage builds a single-page bounded Session with the given interval for the
// interval-boundary tests — the same construction TestBoundedValidityAsOfGate uses.
func boundedPage(validFrom, validTo int64) *Session {
	body := []byte("refund fee is 25 EUR")
	d := Digest(body)
	return &Session{
		Manifest: Manifest{Version: ManifestVersion, Pages: []Page{{
			Step:       0,
			Role:       "read_memory",
			Descriptor: "read_memory: refund fee is 25 EUR",
			Digest:     d,
			Len:        int64(len(body)),
			Durability: durabilityBounded,
			ValidFrom:  validFrom,
			ValidTo:    validTo,
		}}},
		cas:     map[string][]byte{d: body},
		cleared: map[string]bool{},
		gate:    ctxmmu.New(),
	}
}

// TestBoundedValidityHalfOpenInterval pins the issue's admissibility rule (#81):
// a bounded page is admissible only if as-of ∈ [ValidFrom, ValidTo) — a HALF-OPEN
// interval. The exclusive upper bound (as-of == ValidTo must expire, not the prior
// as-of > ValidTo) is the stale-as-current failure at the boundary tick; the lower
// bound refuses a page read before its validity opens.
func TestBoundedValidityHalfOpenInterval(t *testing.T) {
	ctx := context.Background()

	// Exclusive upper bound: at exactly ValidTo the window has closed (half-open).
	s := boundedPage(0, 100)
	if _, err := s.Resolve(ctx, 0, 99); err != nil {
		t.Fatalf("as-of just inside the window must resolve: %v", err)
	}
	if _, err := s.Resolve(ctx, 0, 100); !errors.Is(err, ErrExpired) {
		t.Fatalf("as-of == valid_to must expire (interval is half-open [from,to)), got %v", err)
	}

	// Lower bound: a page read before its validity opens is not admissible.
	s = boundedPage(50, 100)
	if _, err := s.Resolve(ctx, 0, 49); !errors.Is(err, ErrExpired) {
		t.Fatalf("as-of before valid_from must be refused (not yet valid), got %v", err)
	}
	if _, err := s.Resolve(ctx, 0, 50); err != nil {
		t.Fatalf("as-of == valid_from must resolve (interval is closed on the low end): %v", err)
	}
	if set := s.Recall(ctx, "refund fee", 3, 49); len(set) != 0 {
		t.Fatalf("not-yet-valid bounded page leaked into recall working set: %+v", set)
	}

	// A zero-value interval (old manifest, no bounds) stays unbounded — never gated.
	s = boundedPage(0, 0)
	if _, err := s.Resolve(ctx, 0, 1<<40); err != nil {
		t.Fatalf("unbounded (zero-value) bounded page must never expire: %v", err)
	}
}
