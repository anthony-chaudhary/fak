package enginecache

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

type engineCacheRedirectTransport func(*http.Request) (*http.Response, error)

func (f engineCacheRedirectTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

// fak-test:runtime fast est=5ms lane=default
func TestInvalidateRequiresDirectControlResponse(t *testing.T) {
	t.Parallel()
	for _, exact := range []bool{false, true} {
		for _, status := range []int{http.StatusOK, http.StatusFound, http.StatusTemporaryRedirect} {
			t.Run(fmt.Sprintf("exact=%v/status=%d", exact, status), func(t *testing.T) {
				path := "/reset_prefix_cache"
				if exact {
					path = "/exact"
				}
				var requests, redirectCalls int
				httpClient := &http.Client{
					Timeout: 3 * time.Second,
					CheckRedirect: func(*http.Request, []*http.Request) error {
						redirectCalls++
						return nil
					},
					Transport: engineCacheRedirectTransport(func(r *http.Request) (*http.Response, error) {
						requests++
						if r.Body != nil {
							defer r.Body.Close()
						}
						code := status
						body := "control response"
						header := make(http.Header)
						if r.URL.Path == "/login" {
							// This successful landing page must not count as an invalidation.
							code, body = http.StatusOK, "login page"
						} else {
							if r.URL.Path != path || r.Method != http.MethodPost {
								return nil, fmt.Errorf("unexpected control request: %s %s", r.Method, r.URL.Path)
							}
							if r.Header.Get("Authorization") != "Bearer synthetic-test-key" {
								return nil, fmt.Errorf("missing control authorization")
							}
							if status != http.StatusOK {
								header.Set("Location", "/login")
							}
						}
						return &http.Response{StatusCode: code, Header: header, Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
					}),
				}
				client := Client{Engine: EngineVLLM, BaseURL: "http://control.invalid/v1", AdminAPIKey: "synthetic-test-key", HTTPClient: httpClient}
				if exact {
					client.RequiredScope = ScopeExactSpan
					client.ExactSpanEndpoint = "/exact"
				}
				res, err := client.Invalidate(context.Background(), []cachemeta.ExternalInvalidationDirective{sampleDirective("vllm")})
				if (err == nil) != (status == http.StatusOK) {
					t.Fatalf("status=%d: result=%+v err=%v", status, res, err)
				}
				if res.StatusCode != status || res.Endpoint != "http://control.invalid"+path || requests != 1 || redirectCalls != 0 {
					t.Fatalf("result=%+v requests=%d redirectCalls=%d", res, requests, redirectCalls)
				}
				// The local policy must not mutate the caller's reusable client.
				if httpClient.Timeout != 3*time.Second || httpClient.CheckRedirect == nil || httpClient.CheckRedirect(nil, nil) != nil || redirectCalls != 1 {
					t.Fatal("caller HTTP client was changed")
				}
			})
		}
	}
}
