package l3kv

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/blob"
)

// fak-test:runtime fast est=100ms lane=default
// Estimate only: temporary local records and loopback HTTP, without model loads.
func TestRemoteStoreDoesNotResolveLocalPayload(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	payload := bytes.Repeat([]byte("remote-only-span"), 1024)
	for _, mirror := range []bool{false, true} {
		for _, response := range []string{"hit", "missing", "error", "corrupt"} {
			mode := "remote"
			if mirror {
				mode = "mirror"
			}
			t.Run(mode+"/"+response, func(t *testing.T) {
				dir := t.TempDir()
				local, err := newRouterStore(dir, "", "")
				if err != nil {
					t.Fatal(err)
				}
				if err := local.Put(ctx, digestA, payload); err != nil {
					t.Fatal(err)
				}
				var gets atomic.Int32
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					gets.Add(1)
					if r.Method != http.MethodGet || r.URL.Path != "/"+blob.Digest(payload) {
						t.Errorf("unexpected remote request: %s %s", r.Method, r.URL.Path)
						w.WriteHeader(http.StatusBadRequest)
						return
					}
					switch response {
					case "missing":
						w.WriteHeader(http.StatusNotFound)
					case "error":
						w.WriteHeader(http.StatusServiceUnavailable)
					case "corrupt":
						_, _ = w.Write(bytes.Repeat([]byte("x"), len(payload)))
					default:
						_, _ = w.Write(payload)
					}
				}))
				defer server.Close()
				var store Store
				if mirror {
					store, err = newRouterStore(dir, server.URL, "")
				} else {
					store, err = NewRemoteStore(dir, server.URL, "")
				}
				if err != nil {
					t.Fatal(err)
				}
				got, found, err := store.Get(ctx, digestA)
				wantGets := int32(1)
				if mirror {
					wantGets = 0
				}
				if gets.Load() != wantGets {
					t.Fatalf("remote requests = %d, want %d", gets.Load(), wantGets)
				}
				if mirror || response == "hit" {
					if err != nil || !found || !bytes.Equal(got, payload) {
						t.Fatalf("valid restore: found=%v err=%v bytes=%d", found, err, len(got))
					}
				} else if err == nil || found || got != nil {
					t.Fatalf("remote fault was hidden: found=%v err=%v bytes=%d", found, err, len(got))
				}
				// Remote-only means bypass, not destructive removal of prior local data.
				got, found, err = local.Get(ctx, digestA)
				if err != nil || !found || !bytes.Equal(got, payload) {
					t.Fatalf("prior local copy changed: found=%v err=%v", found, err)
				}
			})
		}
	}
}
