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
