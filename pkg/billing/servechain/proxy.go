package servechain

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"strings"

	billingledger "github.com/anthony-chaudhary/fak/pkg/billing/ledger"
	billingpricebook "github.com/anthony-chaudhary/fak/pkg/billing/pricebook"
)

// serveUpstream proxies one fully admitted request to its bound upstream:
// the served model name is rewritten to the upstream's model id, the /v1
// prefix is stripped so the upstream's base path merges cleanly (a base of
// http://host:8000/v1 plus /v1/chat/completions must not double the prefix),
// and usage is captured on the way back for settlement.
func (c *Chain) serveUpstream(w http.ResponseWriter, r *http.Request, st *reqState, payload map[string]any) {
	payload["model"] = st.upstreamModel
	body, err := json.Marshal(payload)
	if err != nil {
		// Nothing was served and nothing is owed: release the hold.
		c.releaseOnce(st)
		writeError(w, http.StatusInternalServerError, "encoding_error",
			"the request could not be re-encoded for the upstream")
		return
	}

	// Re-arm the inbound body with the rewritten payload; the proxy clones
	// it for the outbound request.
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	r.Header.Set("Content-Length", strconv.Itoa(len(body)))

	// Every response this chain writes — success, upstream error, or 502 —
	// carries the billing request id, so operators can join a wire request
	// to its ledger events.
	w.Header().Set("X-Fak-Billing-Request-Id", st.reqID)

	rt := c.routes[st.servedModel]
	proxy := &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.SetURL(rt.base)
			pr.Out.URL.Path = joinBasePath(rt.base, r.URL.Path)
			pr.Out.URL.RawPath = ""
			pr.SetXForwarded()
			// The gateway consumes the upstream's response itself; a
			// compressed body would be unreadable for usage capture.
			pr.Out.Header.Set("Accept-Encoding", "identity")
		},
		ModifyResponse: func(resp *http.Response) error {
			if resp.StatusCode >= http.StatusBadRequest {
				// The upstream refused the request: nothing was served,
				// nothing is owed. Release the hold and pass the upstream's
				// own error through.
				c.releaseOnce(st)
				return nil
			}
			counts, known := captureUsage(resp)
			c.foldUsage(st.tenantID, st.sessionID, counts, known, c.now())
			c.settle.enqueue(c, settleJob{
				reqID:         st.reqID,
				reservationID: st.reservationID,
				servedModel:   st.servedModel,
				counts:        counts,
				known:         known,
			})
			return nil
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, perr error) {
			// Transport-level upstream failure: release the hold, surface 502.
			c.releaseOnce(st)
			c.stats.ProxyErrors.Add(1)
			writeError(w, http.StatusBadGateway, "upstream_unavailable",
				"the upstream engine could not be reached; the reservation was released")
		},
	}
	proxy.ServeHTTP(w, r)
}

// releaseOnce releases the request's reservation exactly once, whichever
// path gets there first. An already-applied release (a retried event id) is
// success. A release the ledger refuses leaves the reservation PENDING for
// operator reconciliation — the safe direction for money.
func (c *Chain) releaseOnce(st *reqState) {
	if st.reservationID == "" || st.released {
		return
	}
	st.released = true
	err := c.cfg.Ledger.Release(st.reqID+"-release", st.reservationID)
	if err == nil || errors.Is(err, billingledger.ErrAlreadyApplied) {
		c.stats.Released.Add(1)
	}
}

// captureUsage reads the upstream's JSON response body for its usage object,
// replacing the body so the client still receives it. A non-JSON body (a
// streaming SSE response, an opaque error) or an unparseable one yields
// unknown usage: the caller leaves the reservation PENDING. This is the
// chain's documented streaming limitation — streamed completions are served
// and proxied, but their usage is not captured.
func captureUsage(resp *http.Response) (billingpricebook.Counts, bool) {
	if !strings.Contains(resp.Header.Get("Content-Type"), "application/json") {
		return billingpricebook.Counts{}, false
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		resp.Body.Close()
		return billingpricebook.Counts{}, false
	}
	resp.Body.Close()
	resp.Body = io.NopCloser(bytes.NewReader(body))
	var envelope struct {
		Usage *upstreamUsage `json:"usage"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil || envelope.Usage == nil {
		return billingpricebook.Counts{}, false
	}
	return envelope.Usage.counts(), true
}

// joinBasePath merges the upstream's base path with the inbound path after
// the /v1 prefix is stripped. The inbound path is always one of the closed
// served paths, so the strip is total.
func joinBasePath(base *url.URL, inbound string) string {
	in := strings.TrimPrefix(inbound, "/v1")
	b := strings.TrimSuffix(base.Path, "/")
	if b == "" {
		return in
	}
	return b + in
}

// reject counts and writes one chain refusal. Every rejection happens
// before reservation, so nothing needs releasing here.
func (c *Chain) reject(w http.ResponseWriter, st *reqState, code int, codeTag, msg string) {
	c.stats.Rejected.Add(1)
	writeError(w, code, codeTag, msg)
}

// writeError writes the OpenAI-style error envelope.
func writeError(w http.ResponseWriter, code int, codeTag, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    "billing_error",
			"code":    codeTag,
		},
	})
}
