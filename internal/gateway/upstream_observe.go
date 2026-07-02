package gateway

import "net/http"

// upstream_observe.go is the host's read-only window onto UPSTREAM provider
// responses. The proxy planners speak to the provider through their own
// http.Client; wrapping that client's transport here lets a host observe each
// response's status + headers — the provider's anthropic-ratelimit-* /
// x-ratelimit-* ACCOUNT-usage headers `fak guard` folds into its account view
// (internal/accountobs) — on every upstream hop (buffered, streaming, and every
// retry) without touching the request path or the body. The gateway itself stays
// account-blind: it only carries the seam.

// upstreamObserveTransport wraps an http.RoundTripper and reports each response's
// status + headers to the observer. Errors and bodies pass through untouched; the
// observer is called after headers arrive (before the body is read), so a
// streaming response is observed at stream open.
type upstreamObserveTransport struct {
	base    http.RoundTripper
	observe func(status int, header http.Header)
}

func (t *upstreamObserveTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.base.RoundTrip(req)
	if resp != nil && t.observe != nil {
		t.observe(resp.StatusCode, resp.Header)
	}
	return resp, err
}

// wrapUpstreamObserver installs observe onto client's transport. A nil observer or
// client leaves everything untouched, so the default (no Config.
// UpstreamResponseObserver) path keeps its transport byte-for-byte.
func wrapUpstreamObserver(client *http.Client, observe func(status int, header http.Header)) {
	if client == nil || observe == nil {
		return
	}
	base := client.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	client.Transport = &upstreamObserveTransport{base: base, observe: observe}
}
