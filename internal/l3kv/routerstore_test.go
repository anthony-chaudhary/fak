package l3kv

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
)

// digestB is a second valid span-digest-shaped key, distinct from digestA.
const digestB = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

// poolServer is a minimal path-style HTTP object store (PUT/GET/HEAD on
// /<digest>) — the same double blobhttp's own tests use, standing in for the
// remote pool. failPuts simulates an unreachable/refusing pool.
type poolServer struct {
	mu       sync.Mutex
	objects  map[string][]byte
	failPuts bool
}

func newPoolServer() *poolServer { return &poolServer{objects: map[string][]byte{}} }

func (o *poolServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	key := strings.TrimPrefix(r.URL.Path, "/")
	o.mu.Lock()
	defer o.mu.Unlock()
	switch r.Method {
	case http.MethodPut:
		if o.failPuts {
			http.Error(w, "pool refuses writes", http.StatusInternalServerError)
			return
		}
		var buf bytes.Buffer
		if _, err := buf.ReadFrom(r.Body); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		o.objects[key] = buf.Bytes()
		w.WriteHeader(http.StatusCreated)
	case http.MethodGet:
		b, ok := o.objects[key]
		if !ok {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write(b)
	case http.MethodHead:
		if _, ok := o.objects[key]; !ok {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(http.StatusOK)
	default:
		http.Error(w, "method", http.StatusMethodNotAllowed)
	}
}

// TestRouterStoreRoundTripAndReopen proves the local-only composition: a span
// put through the router-backed store round-trips bit-exact, an inline-sized
// span survives via the manifest alone, and BOTH survive a reopen over the same
// directory (manifest reload + blobfs restart survival).
func TestRouterStoreRoundTripAndReopen(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	big := spanBytes(8192) // > InlineMax: routed to blobfs
	tiny := spanBytes(16)  // <= InlineMax: rides in the manifest entry

	st1, err := newRouterStore(dir, "", "")
	if err != nil {
		t.Fatalf("newRouterStore: %v", err)
	}
	if err := st1.Put(ctx, digestA, big); err != nil {
		t.Fatalf("Put big: %v", err)
	}
	if err := st1.Put(ctx, digestB, tiny); err != nil {
		t.Fatalf("Put tiny: %v", err)
	}
	got, found, err := st1.Get(ctx, digestA)
	if err != nil || !found {
		t.Fatalf("Get big: found=%v err=%v", found, err)
	}
	if !bytes.Equal(got, big) {
		t.Fatal("big payload not bit-exact")
	}

	st2, err := newRouterStore(dir, "", "") // simulate a restart
	if err != nil {
		t.Fatalf("reopen newRouterStore: %v", err)
	}
	for _, tc := range []struct {
		key  string
		want []byte
	}{{digestA, big}, {digestB, tiny}} {
		got, found, err := st2.Get(ctx, tc.key)
		if err != nil || !found {
			t.Fatalf("after reopen Get(%s): found=%v err=%v", tc.key, found, err)
		}
		if !bytes.Equal(got, tc.want) {
			t.Fatalf("payload for %s not bit-exact after reopen", tc.key)
		}
	}
	// An unknown span stays a clean MISS, not a fault.
	if _, found, err := st2.Get(ctx, "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"); found || err != nil {
		t.Fatalf("unknown span: found=%v err=%v, want clean MISS", found, err)
	}
}

// TestRouterStoreStagesOffBox is the #1472 off-box witness: with a remote pool
// configured, a staged span's content physically lands in the pool (confirmed by
// the pool's own map, keyed by content digest), and a restore still succeeds
// AFTER the entire local content tier is deleted — the bytes come back from the
// pool, which the old flat-file diskStore could never do.
func TestRouterStoreStagesOffBox(t *testing.T) {
	ctx := context.Background()
	pool := newPoolServer()
	srv := httptest.NewServer(pool)
	defer srv.Close()

	dir := t.TempDir()
	payload := spanBytes(8192)

	st1, err := newRouterStore(dir, srv.URL, "")
	if err != nil {
		t.Fatalf("newRouterStore: %v", err)
	}
	if err := st1.Put(ctx, digestA, payload); err != nil {
		t.Fatalf("Put: %v", err)
	}

	// The pool must hold the content under its content digest (off-box residency
	// confirmed by the remote's own state, not by our return value).
	pool.mu.Lock()
	_, inPool := pool.objects[blob.Digest(payload)]
	pool.mu.Unlock()
	if !inPool {
		t.Fatal("staged span content did not land in the remote pool")
	}

	// Lose the box: delete the local content tier entirely, reopen, restore.
	if err := os.RemoveAll(filepath.Join(dir, contentDirName)); err != nil {
		t.Fatalf("remove local content tier: %v", err)
	}
	st2, err := newRouterStore(dir, srv.URL, "")
	if err != nil {
		t.Fatalf("reopen newRouterStore: %v", err)
	}
	got, found, err := st2.Get(ctx, digestA)
	if err != nil || !found {
		t.Fatalf("Get after local loss: found=%v err=%v", found, err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatal("payload restored from pool not bit-exact")
	}
}

// TestRouterStorePoolPutFailureIsError proves the fail-safe direction: when the
// pool is primary and refuses the write, Put returns an error (upstream: a typed
// FAULT, the live span is retained) — never a silent local-only OK masquerading
// as off-box residency.
func TestRouterStorePoolPutFailureIsError(t *testing.T) {
	ctx := context.Background()
	pool := newPoolServer()
	pool.failPuts = true
	srv := httptest.NewServer(pool)
	defer srv.Close()

	st, err := newRouterStore(t.TempDir(), srv.URL, "")
	if err != nil {
		t.Fatalf("newRouterStore: %v", err)
	}
	if err := st.Put(ctx, digestA, spanBytes(8192)); err == nil {
		t.Fatal("Put with a refusing pool returned nil, want error (span must be retained)")
	}
	// Nothing may be claimed after the failed stage: a clean MISS, not a hit.
	if _, found, err := st.Get(ctx, digestA); found || err != nil {
		t.Fatalf("after failed Put: found=%v err=%v, want clean MISS", found, err)
	}
}

// TestRouterStoreTamperedPoolIsFault proves the fail-closed integrity guard end
// to end: content tampered INSIDE the remote pool (the local mirror deleted) is
// a typed FAULT at restore, never a wrong hit.
func TestRouterStoreTamperedPoolIsFault(t *testing.T) {
	ctx := context.Background()
	pool := newPoolServer()
	srv := httptest.NewServer(pool)
	defer srv.Close()

	dir := t.TempDir()
	payload := spanBytes(4096)
	st, err := newRouterStore(dir, srv.URL, "")
	if err != nil {
		t.Fatalf("newRouterStore: %v", err)
	}
	if err := st.Put(ctx, digestA, payload); err != nil {
		t.Fatalf("Put: %v", err)
	}

	// Tamper the pool's copy and remove the local mirror so the pool must answer.
	d := blob.Digest(payload)
	pool.mu.Lock()
	tampered := append([]byte(nil), pool.objects[d]...)
	tampered[len(tampered)-1] ^= 0xFF
	pool.objects[d] = tampered
	pool.mu.Unlock()
	if err := os.RemoveAll(filepath.Join(dir, contentDirName)); err != nil {
		t.Fatalf("remove local content tier: %v", err)
	}
	st2, err := newRouterStore(dir, srv.URL, "")
	if err != nil {
		t.Fatalf("reopen newRouterStore: %v", err)
	}
	if _, found, err := st2.Get(ctx, digestA); err == nil {
		t.Fatalf("Get over tampered pool returned no error (found=%v) — integrity guard missed", found)
	}
}

// TestRouterStoreRefusesUnreadableManifest proves the fail-closed open: a
// corrupt manifest refuses the store (init then leaves the in-process default
// backend live) rather than silently serving an empty tier.
func TestRouterStoreRefusesUnreadableManifest(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, manifestName), []byte("not json{"), 0o644); err != nil {
		t.Fatalf("seed corrupt manifest: %v", err)
	}
	if _, err := newRouterStore(dir, "", ""); err == nil {
		t.Fatal("newRouterStore over a corrupt manifest returned nil error, want refusal")
	}
}

// TestRouterStoreBackendRoundTrip closes the loop at the abi.KVBackend layer:
// the l3kv backend composed over the router-backed store stages real bytes and
// restores them (Outcome OK both ways, BytesMoved = payload length).
func TestRouterStoreBackendRoundTrip(t *testing.T) {
	ctx := context.Background()
	payload := spanBytes(8192)
	mock := &stagerKV{spans: map[[2]int][]byte{{5, 40}: payload}, n: 100}
	st, err := newRouterStore(t.TempDir(), "", "")
	if err != nil {
		t.Fatalf("newRouterStore: %v", err)
	}
	b := New(mock, st)

	res, err := b.StageSpan(ctx, digestA, 5, 40)
	if err != nil {
		t.Fatalf("StageSpan err: %v", err)
	}
	if res.Outcome != abi.KVResidencyOK || res.BytesMoved != int64(len(payload)) {
		t.Fatalf("StageSpan = %+v, want OK with %d bytes moved (%s)", res, len(payload), res.Reason)
	}
	rr, err := b.RestoreSpan(ctx, digestA)
	if err != nil {
		t.Fatalf("RestoreSpan err: %v", err)
	}
	if rr.Outcome != abi.KVResidencyOK || rr.BytesMoved != int64(len(payload)) {
		t.Fatalf("RestoreSpan = %+v, want OK with %d bytes moved (%s)", rr, len(payload), rr.Reason)
	}
}
