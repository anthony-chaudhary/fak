package servechain

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	billing "github.com/anthony-chaudhary/fak/pkg/billing"
	billingkeystore "github.com/anthony-chaudhary/fak/pkg/billing/keystore"
	billingledger "github.com/anthony-chaudhary/fak/pkg/billing/ledger"
)

// servedPaths is the closed set of routes the chain serves, exactly as
// clients send them (with the /v1 prefix). Anything else is 404.
var servedPaths = map[string]bool{
	"/v1/chat/completions": true,
	"/v1/completions":      true,
	"/v1/embeddings":       true,
}

// scrubHeaders is the closed set of inbound credential headers the chain
// strips before proxying. The chain's keys authenticate the caller to the
// gateway; they are not upstream credentials and must not leak forward.
var scrubHeaders = []string{
	"Authorization",
	"X-Api-Key",
	"Cookie",
	"X-Fak-Session",
}

// reqState is one request's billing state, carried through the proxy's
// callbacks.
type reqState struct {
	chain         *Chain
	reqID         string
	tenantID      string
	sessionID     string
	servedModel   string
	upstreamModel string
	reservationID string
	// released is the release-once guard: whichever path releases first
	// (the proxy error handler or the ModifyResponse error path) sets it;
	// both run on the serving goroutine, so a plain bool is the CAS.
	released bool
}

// ServeHTTP admits one request through the closed pipeline and, when it
// passes, proxies it to the bound upstream. Every refusal writes an
// OpenAI-style error envelope; no refusal ever reaches the upstream.
func (c *Chain) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	c.stats.Requests.Add(1)
	now := c.now()
	st := &reqState{chain: c, reqID: newRequestID()}

	// Route stage: the closed served-path set, POST only.
	if !servedPaths[r.URL.Path] {
		c.reject(w, st, http.StatusNotFound, "not_found",
			fmt.Sprintf("unknown path %q; the billing chain serves /v1/chat/completions, /v1/completions, /v1/embeddings", r.URL.Path))
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		c.reject(w, st, http.StatusMethodNotAllowed, "method_not_allowed",
			fmt.Sprintf("method %s is not served; use POST", r.Method))
		return
	}

	// Auth stage: bearer (or x-api-key) against the key store. The tenant
	// the key names owns every downstream decision.
	key := bearerKey(r)
	if key == "" {
		c.reject(w, st, http.StatusUnauthorized, "missing_key",
			"no bearer key presented; send Authorization: Bearer <key> or X-Api-Key")
		return
	}
	rec, err := c.cfg.Keys.Validate(key)
	if errors.Is(err, billingkeystore.ErrUnknownKey) {
		c.reject(w, st, http.StatusUnauthorized, "unknown_key", "the presented key is not recognized")
		return
	}
	if errors.Is(err, billingkeystore.ErrKeyRevoked) {
		c.reject(w, st, http.StatusForbidden, "key_revoked", "the presented key has been revoked")
		return
	}
	if err != nil {
		c.reject(w, st, http.StatusInternalServerError, "key_store_error", "the key store refused lookup")
		return
	}
	st.tenantID = rec.TenantID

	// Session attribution is read before the scrub: the session header is
	// caller-supplied identity for the session cap, never an upstream
	// credential.
	st.sessionID = r.Header.Get("X-Fak-Session")

	// Scrub stage: strip credential headers so they never reach upstream.
	for _, h := range scrubHeaders {
		r.Header.Del(h)
	}

	// Bounded-body stage: read the body under the configured cap.
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, c.cfg.MaxBodyBytes))
	if err != nil {
		var over *http.MaxBytesError
		if errors.As(err, &over) {
			c.reject(w, st, http.StatusRequestEntityTooLarge, "body_too_large",
				fmt.Sprintf("request body exceeds the %d-byte billing limit", over.Limit))
			return
		}
		c.reject(w, st, http.StatusBadRequest, "unreadable_body", "the request body could not be read")
		return
	}

	// Shape stage: a JSON object naming a model.
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil || payload == nil {
		c.reject(w, st, http.StatusBadRequest, "invalid_json", "the request body must be a JSON object")
		return
	}
	name, _ := payload["model"].(string)
	if name == "" {
		c.reject(w, st, http.StatusBadRequest, "missing_model", "the request body must name a model")
		return
	}

	// Allowlist stage: the model must be a served name in the manifest.
	rt, ok := c.routes[name]
	if !ok {
		c.reject(w, st, http.StatusNotFound, "unknown_model",
			fmt.Sprintf("model %q is not served by this deployment", name))
		return
	}
	st.servedModel = rt.ref.Name
	st.upstreamModel = rt.ref.UpstreamModel

	// Rate stage: a saturated fixed one-minute window refuses before any
	// money moves.
	if c.window.saturated(now) {
		c.reject(w, st, http.StatusTooManyRequests, "rate_limited",
			"the deployment's per-minute rate window is saturated; retry after the minute rolls")
		return
	}

	// Policy stage: the billing authority's verdict, then the chain's
	// own spend caps (folded synchronously at capture, so a breach here is
	// deterministic).
	if c.cfg.Policy != nil {
		switch v := c.cfg.Policy.Admit(billing.AdmissionContext{
			Principal:   billing.Principal{TenantID: st.tenantID, SessionID: st.sessionID},
			RequestedAt: now,
		}); v {
		case billing.VerdictAdmit:
			// proceed
		case billing.VerdictRejectRate:
			c.reject(w, st, http.StatusTooManyRequests, "rate_limited", "the billing policy refused the attempt at rate")
			return
		default: // VerdictRejectBudget and any foreign verdict: refuse at money
			c.reject(w, st, http.StatusPaymentRequired, "policy_refused", "the billing policy refused the attempt")
			return
		}
	}
	if c.caps.Breached(billing.Principal{TenantID: st.tenantID, SessionID: st.sessionID}, now) {
		c.reject(w, st, http.StatusPaymentRequired, "cap_reached",
			"a configured spend cap for this principal is exhausted")
		return
	}

	// Reserve stage: hold the configured maximum before the upstream runs.
	res, err := c.cfg.Ledger.Reserve(st.reqID+"-reserve", st.tenantID, st.reqID, c.cfg.ReserveUnits)
	if err != nil {
		if errors.Is(err, billingledger.ErrInsufficientBalance) {
			c.reject(w, st, http.StatusPaymentRequired, "insufficient_balance",
				fmt.Sprintf("tenant %q has insufficient available balance for the %d-unit hold", st.tenantID, c.cfg.ReserveUnits))
			return
		}
		c.reject(w, st, http.StatusInternalServerError, "ledger_error", "the ledger refused reservation")
		return
	}
	st.reservationID = res.ID

	// Full admission: only now does the request consume rate.
	c.window.commitRequest(now)
	c.stats.Admitted.Add(1)
	c.serveUpstream(w, r, st, payload)
}

// now is the configured clock.
func (c *Chain) now() time.Time { return c.cfg.Now() }

// bearerKey extracts the presented key: the Authorization bearer token
// (case-insensitive scheme) or the x-api-key header, whichever carries one.
func bearerKey(r *http.Request) string {
	auth := r.Header.Get("Authorization")
	if len(auth) > 7 && equalFold(auth[:6], "bearer") && auth[6] == ' ' {
		return auth[7:]
	}
	return r.Header.Get("X-Api-Key")
}

// equalFold is a tiny ASCII case-insensitive compare for the bearer scheme.
func equalFold(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		ca, cb := a[i], b[i]
		if 'A' <= ca && ca <= 'Z' {
			ca += 'a' - 'A'
		}
		if 'A' <= cb && cb <= 'Z' {
			cb += 'a' - 'A'
		}
		if ca != cb {
			return false
		}
	}
	return true
}

// newRequestID mints one hex request id; the ledger event ids derive from it
// (<id>-reserve / <id>-settle / <id>-release), and the ledger dedups events
// globally by id, so one minted id names exactly one billing event family.
func newRequestID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		// Crypto/rand failing is a process fault, not a request fault; a
		// time-derived id still keeps billing coherent for this request.
		return fmt.Sprintf("fak-%x", time.Now().UnixNano())
	}
	return "fak-" + hex.EncodeToString(b[:])
}
