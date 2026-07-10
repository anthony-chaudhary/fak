package durablestore

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// mustPut is a helper: Put and fail the test on error, returning the content-address.
func mustPut(t *testing.T, s *Store, in PutSpan) string {
	t.Helper()
	id, err := s.Put(in)
	if err != nil {
		t.Fatalf("Put(%q): %v", in.Descriptor, err)
	}
	return id
}

// TestPutSurvivesGatewayRestart is the core witness for issue #4194's first done-condition
// bullet — "after a gateway restart, fak_context_spans replays the same spans for a trace, and
// fak_context_restore still pages an unsealed span back". A gateway restart is simulated by
// DROPPING the first Store and opening a FRESH one over the same directory (a fresh process
// with an empty heap, rebuilding its index from disk). Spans() must replay the identical span
// set, and Get() must page byte-identical bytes back for every unsealed span. This is also the
// second bullet — a finished/closed session (the process that wrote the spans is gone) stays
// queryable and its bytes stay readable — because the reader shares nothing but the directory.
func TestPutSurvivesGatewayRestart(t *testing.T) {
	dir := t.TempDir()

	writer, err := Open(dir)
	if err != nil {
		t.Fatalf("Open(writer): %v", err)
	}
	want := map[string][]byte{}
	for _, body := range [][]byte{
		[]byte(`{"role":"user","text":"the originating task the compaction dropped"}`),
		[]byte(`{"role":"tool","text":"a second dropped span, distinct bytes"}`),
		[]byte("plain-bytes-third-span"),
	} {
		id := mustPut(t, writer, PutSpan{Descriptor: "excerpt: " + string(body[:min(12, len(body))]), Bytes: body})
		want[id] = body
	}

	// --- gateway restart: drop the writer, open a brand-new Store over the same dir ---
	reader, err := Open(dir)
	if err != nil {
		t.Fatalf("Open(reader after restart): %v", err)
	}

	// fak_context_spans replays the same spans for the trace.
	spans, err := reader.Spans(context.Background())
	if err != nil {
		t.Fatalf("Spans after restart: %v", err)
	}
	if len(spans) != len(want) {
		t.Fatalf("restart replayed %d spans, want %d", len(spans), len(want))
	}
	for _, sp := range spans {
		if _, ok := want[sp.ID]; !ok {
			t.Fatalf("restart replayed unexpected span id %q", sp.ID)
		}
		// The store address and content-address are one and the same (content-addressed store),
		// which is exactly what the gateway restoreFromStore seam matches on.
		if sp.Digest != sp.ID {
			t.Fatalf("span %q: Digest %q != ID (content-address must be the store address)", sp.ID, sp.Digest)
		}
	}

	// fak_context_restore still pages each unsealed span back, byte-identical.
	for id, body := range want {
		got, err := reader.Get(id)
		if err != nil {
			t.Fatalf("Get(%s) after restart: %v", id, err)
		}
		if string(got) != string(body) {
			t.Fatalf("Get(%s) after restart returned %q, want %q", id, got, body)
		}
	}
}

// TestContentAddressIntegrity witnesses the content-address integrity contract from two sides:
// Put refuses a caller-asserted id that disagrees with Digest(bytes), and a durable file whose
// on-disk bytes have been tampered so they no longer hash to their own filename is REFUSED at
// read (and at reopen). A content-address is a checkable function of the bytes — a record can
// never masquerade as a span it is not the hash of.
func TestContentAddressIntegrity(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	body := []byte("content-addressed payload")

	// (a) Put rejects an asserted id that is not the content hash.
	if _, err := s.Put(PutSpan{ID: "deadbeefnotarealhash", Bytes: body}); !errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("Put with wrong asserted id: err = %v, want ErrDigestMismatch", err)
	}

	// A clean Put (no/empty asserted id) computes the address itself and succeeds.
	id := mustPut(t, s, PutSpan{Descriptor: "excerpt", Bytes: body})

	// (b) Tamper the durable file so its bytes no longer hash to the filename, then witness
	// that both a direct Get and a fresh Open (restart) refuse it.
	tamperFile(t, dir, id)
	if _, err := s.Get(id); !errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("Get on tampered file: err = %v, want ErrDigestMismatch", err)
	}
	if _, err := Open(dir); !errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("Open over tampered dir: err = %v, want ErrDigestMismatch", err)
	}
}

// tamperFile rewrites the durable record for id with a Body that no longer hashes to id,
// leaving the filename and the record's own claimed id unchanged — the corruption the
// content-address check must catch.
func tamperFile(t *testing.T, dir, id string) {
	t.Helper()
	path := filepath.Join(dir, id+fileExt)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read for tamper: %v", err)
	}
	var rec persisted
	if err := json.Unmarshal(raw, &rec); err != nil {
		t.Fatalf("decode for tamper: %v", err)
	}
	rec.Body = append(rec.Body, []byte("-tampered")...) // bytes now hash to something else
	out, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("encode tampered: %v", err)
	}
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatalf("write tampered: %v", err)
	}
}

// TestGatePersistsAcrossRestart witnesses issue #4194's third done-condition bullet — "the
// durable store honors the taint stamps (a sealed span stays refused across restart)". A span
// is sealed (and, in the sub-case, tombstoned), the Store is dropped and reopened (restart),
// and Get must still refuse with the ctxplan sentinel — the exact error the gateway maps to
// its ErrRestoreRefused shape. A restart must never resurrect a suppressed span.
func TestGatePersistsAcrossRestart(t *testing.T) {
	cases := []struct {
		name    string
		flip    func(*Store, string) error
		wantErr error
	}{
		{"seal", (*Store).Seal, ctxplan.ErrSealed},
		{"tombstone", (*Store).Tombstone, ctxplan.ErrTombstoned},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			s, err := Open(dir)
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			id := mustPut(t, s, PutSpan{Descriptor: "quarantine me", Bytes: []byte("suppressed span bytes")})

			// Before the gate: the span pages back fine.
			if _, err := s.Get(id); err != nil {
				t.Fatalf("Get before gate: %v", err)
			}
			if err := tc.flip(s, id); err != nil {
				t.Fatalf("flip gate: %v", err)
			}

			// --- restart: fresh Store rebuilds the gate from disk ---
			reopened, err := Open(dir)
			if err != nil {
				t.Fatalf("Open after gate: %v", err)
			}
			if _, err := reopened.Get(id); !errors.Is(err, tc.wantErr) {
				t.Fatalf("Get after restart: err = %v, want %v (suppression must survive restart)", err, tc.wantErr)
			}
			// Spans() still enumerates it (audit visibility), byte-free, with the gate flag set.
			spans, err := reopened.Spans(context.Background())
			if err != nil || len(spans) != 1 {
				t.Fatalf("Spans after restart: spans=%d err=%v", len(spans), err)
			}
			if !spans[0].Sealed && !spans[0].Tombstoned {
				t.Fatalf("Spans after restart: gate flag not persisted on %q", id)
			}
		})
	}
}

// TestSpansAreByteFree witnesses that Spans() is a byte-free projection: the enumeration the
// gateway's fak_context_spans surfaces must carry descriptors and sizes but NEVER the payload
// bytes (a sealed span's bytes must not leak through the list). We assert structurally — the
// marshaled Spans() output does not contain the payload token — while Get() still returns it.
func TestSpansAreByteFree(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	const secret = "PAYLOAD-TOKEN-should-never-appear-in-a-descriptor-listing"
	body := []byte(`{"text":"` + secret + `"}`)
	id := mustPut(t, s, PutSpan{Descriptor: "safe excerpt only", Bytes: body})

	spans, err := s.Spans(context.Background())
	if err != nil {
		t.Fatalf("Spans: %v", err)
	}
	if len(spans) != 1 {
		t.Fatalf("Spans returned %d, want 1", len(spans))
	}
	sp := spans[0]
	if sp.Descriptor == "" {
		t.Fatalf("Spans projection dropped the descriptor")
	}
	if sp.Bytes != int64(len(body)) {
		t.Fatalf("Spans projection size = %d, want %d (the size proxy, not the bytes)", sp.Bytes, len(body))
	}
	// The whole enumeration, serialized, must not carry the payload — proving byte-freedom.
	enc, err := json.Marshal(spans)
	if err != nil {
		t.Fatalf("marshal spans: %v", err)
	}
	if strings.Contains(string(enc), secret) {
		t.Fatalf("Spans() leaked payload bytes into the byte-free projection: %s", enc)
	}
	// ...yet the bytes are still reachable through the gated read path.
	got, err := s.Get(id)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !strings.Contains(string(got), secret) {
		t.Fatalf("Get did not return the payload")
	}
}

// TestComposesWithCtxplanStoreSeam witnesses that the durable Store resolves exactly the way
// the gateway's restoreFromStore adapter drives a ctxplan.Store: match a span by its
// content-address Digest over Spans(), then Materialize by the span's ID. This is the seam the
// whole compose-don't-duplicate decision rests on, exercised without importing internal/gateway.
func TestComposesWithCtxplanStoreSeam(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	var store ctxplan.Store = s // must satisfy the interface the gateway resolves against

	body := []byte("bytes reachable only by content-address")
	digest := ctxplan.Digest(body)
	if _, err := s.Put(PutSpan{Descriptor: "excerpt", Bytes: body}); err != nil {
		t.Fatalf("Put: %v", err)
	}

	// Replicate restoreFromStore's resolution: find by Digest, materialize by ID.
	spans, err := store.Spans(context.Background())
	if err != nil {
		t.Fatalf("Spans: %v", err)
	}
	var got []byte
	for _, sp := range spans {
		if sp.Digest == digest {
			got, err = store.Materialize(context.Background(), sp.ID)
			if err != nil {
				t.Fatalf("Materialize: %v", err)
			}
			break
		}
	}
	if string(got) != string(body) {
		t.Fatalf("seam resolution returned %q, want %q", got, body)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
