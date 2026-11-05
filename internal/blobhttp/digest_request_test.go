package blobhttp

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
)

type digestRequestTransport func(*http.Request) (*http.Response, error)

func (f digestRequestTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

// fak-test:runtime fast est=5ms lane=default
func TestObjectDigestCannotChangeSignedRequestTarget(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	const base = "https://objects.invalid/bucket"
	for _, tc := range []struct{ name, digest string }{
		{"empty", ""},
		{"short", "deadbeef"},
		{"traversal", "../other-object"},
		{"query", strings.Repeat("a", 64) + "?delete=other"},
		{"fragment", strings.Repeat("a", 64) + "#other"},
		{"slash-at-width", strings.Repeat("a", 63) + "/"},
		{"non-hex-at-width", strings.Repeat("g", 64)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			signed, sent := 0, 0
			s := New(base,
				WithSigner(func(*http.Request) error { signed++; return nil }),
				WithClient(&http.Client{Transport: digestRequestTransport(func(r *http.Request) (*http.Response, error) {
					sent++
					return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("")), Header: make(http.Header), Request: r}, nil
				})}),
			)
			for _, kind := range []abi.RefKind{abi.RefBlob, abi.RefRegion} {
				if _, err := s.Resolve(ctx, abi.Ref{Kind: kind, Digest: tc.digest}); err == nil {
					t.Errorf("Resolve kind %d accepted invalid digest", kind)
				}
			}
			if err := s.Delete(ctx, tc.digest); err == nil {
				t.Error("Delete accepted invalid digest")
			}
			if signed != 0 || sent != 0 {
				t.Fatalf("invalid digest reached signer %d or transport %d times", signed, sent)
			}
		})
	}

	t.Run("valid-requests", func(t *testing.T) {
		body := []byte("valid content-addressed payload")
		digest := blob.Digest(body)
		signed, sent := 0, 0
		s := New(base,
			WithSigner(func(r *http.Request) error {
				signed++
				r.Header.Set("X-Test-Signed", "yes")
				return nil
			}),
			WithClient(&http.Client{Transport: digestRequestTransport(func(r *http.Request) (*http.Response, error) {
				sent++
				if r.URL.String() != base+"/"+digest || r.Header.Get("X-Test-Signed") != "yes" {
					t.Fatalf("unexpected signed target: %s", r.URL)
				}
				return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(bytes.NewReader(body)), Header: make(http.Header), Request: r}, nil
			})}),
		)
		got, err := s.Resolve(ctx, abi.Ref{Kind: abi.RefBlob, Digest: digest})
		if err != nil || !bytes.Equal(got, body) {
			t.Fatalf("valid Resolve = %q, %v", got, err)
		}
		if err := s.Delete(ctx, digest); err != nil {
			t.Fatalf("valid Delete: %v", err)
		}
		if signed != 2 || sent != 2 {
			t.Fatalf("valid public requests signed=%d sent=%d, want two each", signed, sent)
		}
		for _, address := range []string{digest, strings.ToUpper(digest)} {
			for _, method := range []string{http.MethodGet, http.MethodHead, http.MethodPut, http.MethodDelete} {
				r, err := s.newRequest(ctx, method, address, nil)
				if err != nil || r.Method != method || r.URL.String() != base+"/"+address || r.Header.Get("X-Test-Signed") != "yes" {
					t.Fatalf("valid %s request = %v, %v", method, r, err)
				}
			}
		}
	})
}
