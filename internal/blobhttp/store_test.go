package blobhttp

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
)

// objectServer is a minimal in-memory path-style object endpoint: PUT/GET/HEAD/
// DELETE at /<digest>. It stands in for S3/MinIO/a blob service so the driver is
// exercised end to end over real HTTP with no external dependency.
type objectServer struct {
	mu      sync.Mutex
	objects map[string][]byte
	puts    int
	authSaw string
}

func newObjectServer() *objectServer { return &objectServer{objects: map[string][]byte{}} }

func (o *objectServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	key := r.URL.Path[1:] // strip leading "/"
	o.mu.Lock()
	defer o.mu.Unlock()
	if a := r.Header.Get("Authorization"); a != "" {
		o.authSaw = a
	}
	switch r.Method {
	case http.MethodHead:
		if _, ok := o.objects[key]; ok {
			w.WriteHeader(http.StatusOK)
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	case http.MethodPut:
		b, _ := io.ReadAll(r.Body)
		o.objects[key] = b
		o.puts++
		w.WriteHeader(http.StatusCreated)
	case http.MethodGet:
		if b, ok := o.objects[key]; ok {
			w.Write(b)
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	case http.MethodDelete:
		delete(o.objects, key)
		w.WriteHeader(http.StatusNoContent)
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}

func payload(n int, fill byte) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = fill
	}
	return b
}

func TestPutResolveRoundTrip(t *testing.T) {
	srv := httptest.NewServer(newObjectServer())
	defer srv.Close()
	ctx := context.Background()
	s := New(srv.URL)

	want := payload(4096, 'h')
	r, err := s.Put(ctx, want)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if r.Kind != abi.RefBlob || r.Digest != blob.Digest(want) {
		t.Fatalf("unexpected Ref %+v", r)
	}
	got, err := s.Resolve(ctx, r)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("resolved bytes differ")
	}
}

func TestInlineSmallPayloadNoNetwork(t *testing.T) {
	os := newObjectServer()
	srv := httptest.NewServer(os)
	defer srv.Close()
	ctx := context.Background()
	s := New(srv.URL)

	small := payload(InlineMax, 's')
	r, err := s.Put(ctx, small)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if r.Kind != abi.RefInline {
		t.Fatalf("want RefInline, got kind %d", r.Kind)
	}
	if os.puts != 0 {
		t.Fatalf("inline payload should not PUT to the remote, saw %d puts", os.puts)
	}
}

func TestContentDedupViaHead(t *testing.T) {
	os := newObjectServer()
	srv := httptest.NewServer(os)
	defer srv.Close()
	ctx := context.Background()
	s := New(srv.URL)

	b := payload(2048, 'd')
	if _, err := s.Put(ctx, b); err != nil {
		t.Fatalf("Put 1: %v", err)
	}
	if _, err := s.Put(ctx, b); err != nil {
		t.Fatalf("Put 2: %v", err)
	}
	if os.puts != 1 {
		t.Fatalf("HEAD dedup failed: %d PUTs, want 1", os.puts)
	}
	if _, hits, _ := s.Stats(); hits != 1 {
		t.Fatalf("want 1 dedup hit, got %d", hits)
	}
}

func TestResolveMissingDigest(t *testing.T) {
	srv := httptest.NewServer(newObjectServer())
	defer srv.Close()
	ctx := context.Background()
	s := New(srv.URL)
	_, err := s.Resolve(ctx, abi.Ref{Kind: abi.RefBlob, Digest: "deadbeef"})
	if err == nil {
		t.Fatal("expected an error resolving an absent digest")
	}
}

func TestPageOutPageIn(t *testing.T) {
	srv := httptest.NewServer(newObjectServer())
	defer srv.Close()
	ctx := context.Background()
	s := New(srv.URL)

	body := payload(5000, 'q')
	handle, err := s.PageOut(ctx, abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))})
	if err != nil {
		t.Fatalf("PageOut: %v", err)
	}
	if handle.Kind != abi.RefBlob || len(handle.Inline) != 0 {
		t.Fatalf("page-out handle must be a bytes-absent RefBlob, got %+v", handle)
	}
	back, err := s.PageIn(ctx, handle)
	if err != nil {
		t.Fatalf("PageIn: %v", err)
	}
	if !bytes.Equal(back.Inline, body) {
		t.Fatalf("page-in bytes differ")
	}
}

func TestDelete(t *testing.T) {
	srv := httptest.NewServer(newObjectServer())
	defer srv.Close()
	ctx := context.Background()
	s := New(srv.URL)

	r, err := s.Put(ctx, payload(1024, 'x'))
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := s.Delete(ctx, r.Digest); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, err := s.Resolve(ctx, r); err == nil {
		t.Fatal("expected resolve to fail after delete")
	}
	// Deleting an already-gone object is a success (idempotent erasure).
	if err := s.Delete(ctx, r.Digest); err != nil {
		t.Fatalf("idempotent Delete: %v", err)
	}
}

func TestBearerSignerApplied(t *testing.T) {
	os := newObjectServer()
	srv := httptest.NewServer(os)
	defer srv.Close()
	ctx := context.Background()
	s := New(srv.URL, WithBearer("sekret"))

	if _, err := s.Put(ctx, payload(1024, 'a')); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if os.authSaw != "Bearer sekret" {
		t.Fatalf("signer not applied: server saw Authorization=%q", os.authSaw)
	}
}

func TestResolverInterface(t *testing.T) {
	var _ abi.Resolver = (*Store)(nil)
	var _ abi.PageOutBackend = pageOutBackend{}
}
