package agent

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/httptrust"
)

// UpstreamStatusError is returned by Complete when the upstream provider answered
// with a non-2xx HTTP status that was not retried away — a 4xx request error (e.g.
// a 404 for an unknown model), or a 5xx that survived every retry. It carries the
// upstream's own status code so the gateway can SURFACE it to the client: a model
// the upstream 404s must reach the caller as a non-200, not be silently swallowed
// into a misleading 200/502 (#82). Body is a short, truncated copy of the
// provider's error text for the OPERATOR LOG only — it is not meant to cross the
// trust boundary verbatim to a (possibly unauthenticated) downstream caller.
type UpstreamStatusError struct {
	Status int
	Body   string
	// RetryAfter is the upstream's Retry-After response header VERBATIM ("" when
	// absent). It is the one piece of upstream-supplied error metadata fak
	// propagates downstream — a rate-limited (429) or overloaded (503) upstream
	// names when to retry, and a wrapped agent that backs off correctly instead of
	// hammering is the whole point of surfacing it. fak NEVER parses or interprets
	// the value (RFC 7231 allows delta-seconds OR an HTTP-date); it is echoed only
	// as the downstream Retry-After header, so a malformed upstream value can never
	// reach fak's control flow. Unlike Body it is safe to forward: it carries no
	// provider error text, only timing. Empty for every non-rate-limit/overload
	// status (the header is not set on those), so it is a clean no-op there.
	RetryAfter string
	// LimitReason and LimitResetHint are sanitized provider-limit metadata for
	// HTTP 429 responses. They are operator-readable category/reset hints, not the
	// raw upstream body; the gateway may use them in downstream-safe messages.
	LimitReason    string
	LimitResetHint string
}

// Error formats the upstream's HTTP status and truncated error body as
// "planner: HTTP <status>: <body>". RetryAfter is deliberately NOT embedded — a
// downstream caller that logs err.Error() must not pick up an echoed header
// (the value is surfaced only as the response header, never the message body).
func (e *UpstreamStatusError) Error() string {
	return fmt.Sprintf("planner: HTTP %d: %s", e.Status, e.Body)
}

// UpstreamUnreachableError is returned by Complete when the upstream could not be
// reached AT ALL — a deterministic dial-time transport failure (connection
// refused, DNS NXDOMAIN, TLS handshake) that a retry cannot fix. Unlike a
// transient timeout it is returned IMMEDIATELY, skipping the 4-attempt backoff
// loop that otherwise stalls a misconfigured --base-url for ~8s (#346). The
// gateway maps it to a distinct, actionable client signal (code
// "upstream_unreachable") instead of the generic "upstream model error". Err
// carries the underlying dial cause for the OPERATOR LOG; it is not forwarded
// verbatim across the trust boundary.
type UpstreamUnreachableError struct {
	Err error
}

// Error formats the underlying dial-time cause as "planner: upstream unreachable: <err>".
//
// A certificate-verification failure additionally carries the trust hint (#8172):
// this classifier already knew the chain was untrusted rather than the endpoint
// unreachable, and dropping that on the floor is what made an intercepted TLS
// connection read to every operator as a firewall. Other causes are formatted
// exactly as before.
func (e *UpstreamUnreachableError) Error() string {
	if hint := httptrust.TLSTrustHint(e.Err); hint != "" {
		return fmt.Sprintf("planner: upstream unreachable: %v — %s", e.Err, hint)
	}
	return fmt.Sprintf("planner: upstream unreachable: %v", e.Err)
}

// Unwrap returns the underlying dial-time transport error for errors.Is/As.
func (e *UpstreamUnreachableError) Unwrap() error { return e.Err }
