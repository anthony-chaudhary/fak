package gateway

import (
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
)

// UpstreamFailureReceipt is bounded, sanitized evidence for one failed upstream attempt.
type UpstreamFailureReceipt struct {
	EmittingLayer      string            `json:"emitting_layer"`
	HTTPStatus         int               `json:"http_status,omitempty"`
	Method             string            `json:"method"`
	PathClass          string            `json:"path_class"`
	TargetID           string            `json:"target_id,omitempty"`
	Diagnostic         map[string]string `json:"diagnostic_headers,omitempty"`
	ProviderRequestID  string            `json:"provider_request_id,omitempty"`
	ProxyRequestID     string            `json:"proxy_request_id,omitempty"`
	Attempt            int               `json:"attempt"`
	RetryBudget        int               `json:"retry_budget"`
	RetryDisposition   string            `json:"retry_disposition"`
	RetryReason        string            `json:"retry_reason,omitempty"`
	CallerBytesEmitted bool              `json:"caller_bytes_emitted"`
	Cause              string            `json:"cause,omitempty"`
	SessionID          string            `json:"session_id,omitempty"`
	TraceID            string            `json:"trace_id,omitempty"`
	CallID             string            `json:"call_id,omitempty"`
	Outcome            string            `json:"outcome"`
	Confidence         string            `json:"confidence"`
	Evidence           string            `json:"evidence"`
}

var receiptHeaderAllowlist = []string{"Cf-Ray", "Request-Id", "X-Request-Id", "X-Openai-Request-Id", "Anthropic-Request-Id", "X-Proxy-Request-Id", "Via", "X-Envoy-Upstream-Service-Time", "Retry-After"}

func upstreamFailureReceipt(req *http.Request, status int, header http.Header, attempt int, cause error) UpstreamFailureReceipt {
	r := UpstreamFailureReceipt{HTTPStatus: status, Attempt: attempt, RetryBudget: 3, RetryDisposition: "eligible", Outcome: "terminal", Confidence: "high", Evidence: "transport error", EmittingLayer: "transport"}
	if req != nil {
		r.Method, r.PathClass, r.TargetID = req.Method, sanitizePathClass(req.URL), sanitizeTarget(req.URL)
		r.SessionID, r.CallID, r.TraceID = boundedHeader(req.Header, "X-Fak-Session-Id"), boundedHeader(req.Header, "X-Fak-Call-Id"), boundedHeader(req.Header, "Traceparent")
	}
	if cause != nil {
		r.Cause, r.RetryReason = boundedCause(cause.Error()), "transient_transport"
		return r
	}
	r.Evidence, r.RetryReason, r.Diagnostic = "http response headers", "http_status", allowlistedDiagnostics(header)
	if v := firstHeader(header, "X-Openai-Request-Id", "Anthropic-Request-Id", "Request-Id"); v != "" {
		r.ProviderRequestID, r.EmittingLayer = v, "provider"
	} else if hasLocalProxyEvidence(header) {
		r.ProxyRequestID, r.EmittingLayer = firstHeader(header, "X-Proxy-Request-Id", "X-Request-Id"), "local_proxy"
	} else {
		r.EmittingLayer, r.Confidence, r.Evidence = "unknown", "low", "status without origin header"
	}
	return r
}

func sanitizeTarget(u *url.URL) string {
	if u == nil {
		return ""
	}
	h := strings.ToLower(strings.TrimSpace(u.Hostname()))
	if len(h) > 128 {
		h = h[:128]
	}
	return h
}
func sanitizePathClass(u *url.URL) string {
	if u == nil {
		return ""
	}
	p := u.EscapedPath()
	if p == "" {
		return "/"
	}
	parts := strings.Split(strings.Trim(p, "/"), "/")
	if len(parts) > 3 {
		parts = parts[:3]
	}
	for i, v := range parts {
		if len(v) > 48 {
			parts[i] = "{id}"
		}
	}
	return "/" + strings.Join(parts, "/")
}
func boundedHeader(h http.Header, key string) string {
	v := strings.TrimSpace(h.Get(key))
	if len(v) > 160 {
		v = v[:160]
	}
	return v
}
func firstHeader(h http.Header, keys ...string) string {
	for _, k := range keys {
		if v := boundedHeader(h, k); v != "" {
			return v
		}
	}
	return ""
}
func hasLocalProxyEvidence(h http.Header) bool {
	return h.Get("Via") != "" || h.Get("X-Envoy-Upstream-Service-Time") != "" || h.Get("X-Proxy-Request-Id") != ""
}
func allowlistedDiagnostics(h http.Header) map[string]string {
	out := map[string]string{}
	for _, k := range receiptHeaderAllowlist {
		if v := boundedHeader(h, k); v != "" {
			out[http.CanonicalHeaderKey(k)] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
func boundedCause(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 200 {
		s = s[:200]
	}
	return s
}

type receiptBody struct {
	io.ReadCloser
	once    sync.Once
	req     *http.Request
	status  int
	header  http.Header
	attempt int
	emit    func(UpstreamFailureReceipt)
}

func (b *receiptBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if err != nil && err != io.EOF {
		b.once.Do(func() { b.emit(upstreamFailureReceipt(b.req, b.status, b.header, b.attempt, err)) })
	}
	return n, err
}
