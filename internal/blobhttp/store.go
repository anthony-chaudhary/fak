// Package blobhttp is a content-addressed blob store backed by a REMOTE HTTP
// object endpoint — the "disaggregated / cloud" sibling of internal/blob
// (in-memory) and internal/blobfs (local disk). It is pure stdlib (net/http +
// crypto/sha256): a payload is addressed by its sha256 digest and stored at
// `<base>/<digest>` with PUT, fetched with GET, and probed with HEAD for dedup,
// so it works against any object service that speaks path-style HTTP — an S3 /
// MinIO bucket (path-style URL), a GCS/R2-compatible endpoint, a plain reverse
// proxy, or an in-process httptest server — WITHOUT importing a vendor SDK (which
// would create a go.sum and break the zero-dependency build).
//
// It attaches to the FROZEN ABI like its siblings: implements abi.Resolver
// (Put/Resolve) and abi.PageOutBackend (PageOut/PageIn), registered under id
// "blobhttp" so it coexists with the in-memory "blob" and on-disk "blobfs" codecs.
// It is the durable/shared tier the storedrv router composes for content that must
// outlive a single host (cross-fleet shared results, large cold payloads, audit
// segments). Small payloads stay inline on the Ref, avoiding a network round-trip
// on the hot path, exactly as the local stores do.
//
// AUTHENTICATION is a pluggable seam, not a vendor lock-in: a Signer hook runs on
// every request before it is sent, so an operator can drop in a bearer token
// (the FAK_BLOB_HTTP_TOKEN default), an AWS SigV4 signer (crypto/hmac, stdlib), or
// any header scheme without this package taking a dependency. The honest caveat
// (the storage-drivers design doc states it): a hand-rolled remote client is real
// code you own — reconnect, retries, and auth refresh are the operator's to wire
// via the http.Client and Signer; pure-stdlib means no go.sum, not zero effort.
//
// ENABLEMENT. blobhttp is OPT-IN and inert by default: set FAK_BLOB_HTTP_URL to
// the object-store base URL and the package's init registers the "blobhttp"
// page-out codec; unset, no codec is registered and the package is inert (the
// FAK_AUDIT_JOURNAL / FAK_BLOB_DIR env-toggle idiom).
package blobhttp

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
)

// InlineMax mirrors blob.InlineMax: a payload this small or smaller rides inline on
// the Ref (no network round-trip) instead of being PUT to the remote.
const InlineMax = blob.InlineMax

// Signer optionally mutates a request before it is sent — the auth seam. A bearer
// token, an AWS SigV4 signature (crypto/hmac, stdlib), or any header scheme plugs
// in here without this package depending on a vendor SDK. Nil = no signing.
type Signer func(*http.Request) error

// Store is a content-addressed blob store over a remote HTTP object endpoint. It is
// concurrency-safe (the http.Client is, and the only mutable state is atomic
// counters); a payload is immutable under its digest, so concurrent Puts of the
// same content are idempotent.
type Store struct {
	base   string // object-store base URL, no trailing slash
	client *http.Client
	sign   Signer

	puts    int64
	hits    int64 // HEAD found the digest already present (dedup)
	resolv  int64
	putErrs int64
}

// Option configures a Store.
type Option func(*Store)

// WithClient sets the http.Client (timeouts, transport, retries live here).
func WithClient(c *http.Client) Option { return func(s *Store) { s.client = c } }

// WithSigner sets the per-request auth hook.
func WithSigner(sign Signer) Option { return func(s *Store) { s.sign = sign } }

// WithBearer signs every request with an Authorization: Bearer <token> header.
func WithBearer(token string) Option {
	return WithSigner(func(r *http.Request) error {
		if token != "" {
			r.Header.Set("Authorization", "Bearer "+token)
		}
		return nil
	})
}

// New builds a Store addressing base (e.g. "https://s3.example.com/my-bucket").
func New(base string, opts ...Option) *Store {
	s := &Store{
		base:   strings.TrimRight(base, "/"),
		client: &http.Client{Timeout: 30 * time.Second},
	}
	for _, o := range opts {
		o(s)
	}
	return s
}

// urlFor is the content address as a fetchable URL: <base>/<digest>.
func (s *Store) urlFor(digest string) string { return s.base + "/" + digest }

// Put stores b and returns an addressable Ref. Small payloads ride inline; larger
// ones are PUT content-addressed to <base>/<digest>, with a HEAD dedup probe so a
// byte-identical payload already in the bucket is not re-uploaded.
func (s *Store) Put(ctx context.Context, b []byte) (abi.Ref, error) {
	d := blob.Digest(b)
	r := abi.Ref{Digest: d, Len: int64(len(b)), Taint: abi.TaintTainted, Scope: abi.ScopeAgent}
	if len(b) <= InlineMax {
		r.Kind = abi.RefInline
		r.Inline = append([]byte(nil), b...)
		return r, nil
	}
	r.Kind = abi.RefBlob
	if err := s.commit(ctx, d, b); err != nil {
		return abi.Ref{}, err
	}
	return r, nil
}

// commit uploads b to the remote under its digest unconditionally (HEAD-deduped,
// idempotent). It is the shared path behind Put's large branch AND PageOut —
// page-out must persist even a small body to a bytes-absent handle, so it cannot
// reuse Put's inline shortcut.
func (s *Store) commit(ctx context.Context, d string, b []byte) error {
	atomic.AddInt64(&s.puts, 1)
	if s.exists(ctx, d) {
		atomic.AddInt64(&s.hits, 1)
		return nil
	}
	req, err := s.newRequest(ctx, http.MethodPut, d, bytes.NewReader(b))
	if err != nil {
		atomic.AddInt64(&s.putErrs, 1)
		return err
	}
	req.ContentLength = int64(len(b))
	resp, err := s.client.Do(req)
	if err != nil {
		atomic.AddInt64(&s.putErrs, 1)
		return fmt.Errorf("blobhttp: PUT %s: %w", d, err)
	}
	defer drainClose(resp)
	if resp.StatusCode/100 != 2 {
		atomic.AddInt64(&s.putErrs, 1)
		return fmt.Errorf("blobhttp: PUT %s: status %s", d, resp.Status)
	}
	return nil
}

// Resolve materializes the bytes a Ref points at: inline Refs carry their own
// bytes; RefBlob/RefRegion GET <base>/<digest>.
func (s *Store) Resolve(ctx context.Context, r abi.Ref) ([]byte, error) {
	switch r.Kind {
	case abi.RefInline:
		return append([]byte(nil), r.Inline...), nil
	case abi.RefBlob, abi.RefRegion:
		atomic.AddInt64(&s.resolv, 1)
		req, err := s.newRequest(ctx, http.MethodGet, r.Digest, nil)
		if err != nil {
			return nil, err
		}
		resp, err := s.client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("blobhttp: GET %s: %w", r.Digest, err)
		}
		defer drainClose(resp)
		if resp.StatusCode == http.StatusNotFound {
			return nil, fmt.Errorf("blobhttp: unknown digest %s", r.Digest)
		}
		if resp.StatusCode/100 != 2 {
			return nil, fmt.Errorf("blobhttp: GET %s: status %s", r.Digest, resp.Status)
		}
		b, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("blobhttp: read %s: %w", r.Digest, err)
		}
		return b, nil
	default:
		return nil, fmt.Errorf("blobhttp: unknown RefKind %d", r.Kind)
	}
}

// exists probes the remote with HEAD so a byte-identical payload is not re-uploaded
// (content dedup). A transport error is treated as "absent" — the safe direction:
// at worst the payload is re-PUT (idempotent under its digest), never skipped.
func (s *Store) exists(ctx context.Context, digest string) bool {
	req, err := s.newRequest(ctx, http.MethodHead, digest, nil)
	if err != nil {
		return false
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return false
	}
	defer drainClose(resp)
	return resp.StatusCode/100 == 2
}

// PageOut moves a (possibly inline) Ref's bytes to the remote and returns a
// bytes-absent handle Ref — the durable/remote analogue of blob.PageOut.
func (s *Store) PageOut(ctx context.Context, r abi.Ref) (abi.Ref, error) {
	b, err := s.Resolve(ctx, r)
	if err != nil {
		return abi.Ref{}, err
	}
	stored, err := s.Put(ctx, b)
	if err != nil {
		return abi.Ref{}, err
	}
	return abi.Ref{Kind: abi.RefBlob, Digest: stored.Digest, Len: int64(len(b)), Taint: r.Taint, Scope: r.Scope}, nil
}

// PageIn re-materializes a paged-out handle Ref into an inline Ref.
func (s *Store) PageIn(ctx context.Context, handle abi.Ref) (abi.Ref, error) {
	b, err := s.Resolve(ctx, handle)
	if err != nil {
		return abi.Ref{}, err
	}
	return abi.Ref{Kind: abi.RefInline, Digest: handle.Digest, Inline: b, Len: int64(len(b)), Taint: handle.Taint, Scope: handle.Scope}, nil
}

// Delete removes the object for a digest (the provable-deletion / retention hook a
// disaggregated store needs). A 404 is treated as success (already gone).
func (s *Store) Delete(ctx context.Context, digest string) error {
	req, err := s.newRequest(ctx, http.MethodDelete, digest, nil)
	if err != nil {
		return err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return fmt.Errorf("blobhttp: DELETE %s: %w", digest, err)
	}
	defer drainClose(resp)
	if resp.StatusCode == http.StatusNotFound || resp.StatusCode/100 == 2 {
		return nil
	}
	return fmt.Errorf("blobhttp: DELETE %s: status %s", digest, resp.Status)
}

// newRequest builds a signed request for a digest's object URL.
func (s *Store) newRequest(ctx context.Context, method, digest string, body io.Reader) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, s.urlFor(digest), body)
	if err != nil {
		return nil, fmt.Errorf("blobhttp: build %s %s: %w", method, digest, err)
	}
	if s.sign != nil {
		if err := s.sign(req); err != nil {
			return nil, fmt.Errorf("blobhttp: sign %s %s: %w", method, digest, err)
		}
	}
	return req, nil
}

// drainClose drains and closes a response body so the connection can be reused.
func drainClose(resp *http.Response) {
	if resp != nil && resp.Body != nil {
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
	}
}

// Stats reports store activity (puts, dedup hits, resolves) for KPI taps.
func (s *Store) Stats() (puts, dedupHits, resolves int64) {
	return atomic.LoadInt64(&s.puts), atomic.LoadInt64(&s.hits), atomic.LoadInt64(&s.resolv)
}

// ID is the driver id used by the storedrv router and diagnostics.
func (s *Store) ID() string { return "blobhttp" }

// ----------------------------------------------------------------------------
// ABI registration: an OPT-IN remote page-out codec under id "blobhttp".
// ----------------------------------------------------------------------------

var active *Store

// Active returns the registered remote store, or nil if FAK_BLOB_HTTP_URL was unset
// at boot (the package is inert).
func Active() *Store { return active }

func init() {
	base := os.Getenv("FAK_BLOB_HTTP_URL")
	if base == "" {
		return // off by default: no codec registered, package inert
	}
	s := New(base, WithBearer(os.Getenv("FAK_BLOB_HTTP_TOKEN")))
	active = s
	abi.RegisterPageOutBackend("blobhttp", pageOutBackend{s})
	fmt.Fprintf(os.Stderr, "fak: remote blob store -> %s (content-addressed, id=blobhttp)\n", base)
}

// pageOutBackend adapts *Store to abi.PageOutBackend for the keyed registry.
type pageOutBackend struct{ s *Store }

// PageOut satisfies abi.PageOutBackend by delegating to the wrapped Store's PageOut.
func (b pageOutBackend) PageOut(ctx context.Context, r abi.Ref) (abi.Ref, error) {
	return b.s.PageOut(ctx, r)
}

// PageIn satisfies abi.PageOutBackend by delegating to the wrapped Store's PageIn.
func (b pageOutBackend) PageIn(ctx context.Context, h abi.Ref) (abi.Ref, error) {
	return b.s.PageIn(ctx, h)
}
