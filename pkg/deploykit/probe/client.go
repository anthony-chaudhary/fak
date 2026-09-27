package probe

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/deploykit/stamp"
)

// Check names one step of the probe sequence. Checks run in declaration order.
type Check string

const (
	CheckLiveness  Check = "liveness"
	CheckReadiness Check = "readiness"
	CheckCatalog   Check = "catalog"
	CheckIdentity  Check = "identity"
)

// FailureKind is the closed vocabulary a failed probe reports. Only
// FailureNotListening and FailureNotReady are retried by Wait.
type FailureKind string

const (
	// FailureInvalidTarget: the Target or Expect is unusable; no request was sent.
	FailureInvalidTarget FailureKind = "invalid_target"
	// FailureNotListening: the request could not complete (refused, reset, timed out).
	FailureNotListening FailureKind = "not_listening"
	// FailureTransport: a server answered but the connection cannot work as
	// configured (TLS verification failed, or http/https scheme mismatch).
	FailureTransport FailureKind = "transport_error"
	// FailureNotReady: the server answered 502, 503 or 504, or its catalog reported is_ready:false.
	FailureNotReady FailureKind = "not_ready"
	// FailureUnauthorized: the server answered 401 or 403.
	FailureUnauthorized FailureKind = "unauthorized"
	// FailureBadStatus: the server answered a status the contract does not allow.
	FailureBadStatus FailureKind = "bad_status"
	// FailureMalformedResponse: the body is not the document the contract requires.
	FailureMalformedResponse FailureKind = "malformed_response"
	// FailureEmptyCatalog: /v1/models answered an empty data list.
	FailureEmptyCatalog FailureKind = "empty_catalog"
	// FailureIdentityDrift: /version reports a clean commit other than the expected one.
	FailureIdentityDrift FailureKind = "identity_drift"
	// FailureIdentityUnproven: /version reports a dirty or unstamped build, or a
	// commit that is not a full object ID, so the identity cannot be proven.
	FailureIdentityUnproven FailureKind = "identity_unproven"
)

// Retryable reports whether Wait samples again after this kind. Everything
// except not-listening and not-ready fails fast.
func (k FailureKind) Retryable() bool {
	return k == FailureNotListening || k == FailureNotReady
}

// ProbeError reports the failing check and the failure kind. Cause never
// carries a response body.
type ProbeError struct {
	Kind       FailureKind
	Check      Check
	StatusCode int
	Cause      error
}

func (e *ProbeError) Error() string {
	var b strings.Builder
	b.WriteString("probe")
	if e.Check != "" {
		b.WriteString(" " + string(e.Check))
	}
	b.WriteString(": " + string(e.Kind))
	if e.StatusCode != 0 {
		fmt.Fprintf(&b, " (HTTP %d)", e.StatusCode)
	}
	if e.Cause != nil {
		b.WriteString(": " + e.Cause.Error())
	}
	return b.String()
}

func (e *ProbeError) Unwrap() error { return e.Cause }

// Target is the deployable to probe.
type Target struct {
	// BaseURL is the server origin, e.g. "http://127.0.0.1:8080". It must be an
	// absolute http or https URL with no user info, path, query or fragment.
	BaseURL string `json:"base_url"`
	// Bearer is sent as "Authorization: Bearer <token>" on the catalog and
	// identity checks only. Liveness and readiness are unauthenticated.
	Bearer string `json:"-"`
}

// Expect is what the probe must observe beyond liveness, readiness and a
// non-empty catalog.
type Expect struct {
	// Commit is the full 40-hex commit the running binary must report on
	// /version. It is required unless SkipIdentity is set.
	Commit string `json:"commit,omitempty"`
	// SkipIdentity skips the /version check entirely. Updaters must not set it:
	// it exists for observers that do not know which build should be running.
	SkipIdentity bool `json:"skip_identity,omitempty"`
}

// ProbeResult is the typed outcome of Probe or Wait, shaped for a deploy receipt.
type ProbeResult struct {
	BaseURL string `json:"base_url"`
	Ready   bool   `json:"ready"`
	// Failure, Check and StatusCode describe the last failed sample. They are
	// empty when Ready is true.
	Failure    FailureKind `json:"failure,omitempty"`
	Check      Check       `json:"check,omitempty"`
	StatusCode int         `json:"status_code,omitempty"`
	// Attempts is the number of samples taken.
	Attempts int `json:"attempts"`
	// TimedOut is true when Wait gave up at its deadline while still retrying.
	TimedOut  bool  `json:"timed_out,omitempty"`
	ElapsedMS int64 `json:"elapsed_ms"`
	// Models is the catalog size the last sample observed.
	Models int `json:"models,omitempty"`
	// Identity is the build identity the last sample observed on /version.
	Identity *stamp.Identity `json:"identity,omitempty"`
}

// HTTPDoer sends one HTTP request. *http.Client satisfies it.
type HTTPDoer interface {
	Do(*http.Request) (*http.Response, error)
}

// Defaults for a zero Client.
const (
	DefaultTimeout        = 60 * time.Second
	DefaultInterval       = 250 * time.Millisecond
	DefaultRequestTimeout = 2 * time.Second
)

// Response-size bounds. A body over its bound is a malformed response.
const (
	maxStatusBody  = 64 << 10
	maxCatalogBody = 1 << 20
	maxVersionBody = 64 << 10
)

// Client probes a deployable. The zero value is ready to use.
type Client struct {
	// HTTP sends the requests. Nil uses a client that does not follow redirects.
	HTTP HTTPDoer
	// Timeout bounds one Wait. Zero means DefaultTimeout. A ctx deadline that is
	// earlier still wins.
	Timeout time.Duration
	// Interval is the pause between samples. Zero means DefaultInterval.
	Interval time.Duration
	// RequestTimeout bounds each HTTP request. Zero means DefaultRequestTimeout.
	RequestTimeout time.Duration
}

var defaultHTTP HTTPDoer = &http.Client{
	CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
}

func (c Client) doer() HTTPDoer {
	if c.HTTP != nil {
		return c.HTTP
	}
	return defaultHTTP
}

func orDefault(d, def time.Duration) time.Duration {
	if d > 0 {
		return d
	}
	return def
}

// Wait samples the target until one sample passes every check, a sample fails
// with a kind that is not retryable, or the deadline passes. On a deadline or
// ctx cancellation the returned *ProbeError keeps the last complete sample's
// kind and check, and wraps the context error.
func (c Client) Wait(ctx context.Context, t Target, e Expect) (ProbeResult, error) {
	start := time.Now()
	base, commit, perr := prepare(t, e)
	if perr != nil {
		return ProbeResult{Failure: perr.Kind}, perr
	}
	ctx, cancel := context.WithTimeout(ctx, orDefault(c.Timeout, DefaultTimeout))
	defer cancel()
	interval := orDefault(c.Interval, DefaultInterval)

	var last ProbeResult
	var lastErr *ProbeError
	for attempts := 1; ; attempts++ {
		if err := ctx.Err(); err != nil {
			return giveUp(start, attempts-1, last, lastErr, base, err)
		}
		res, err := c.sample(ctx, base, t.Bearer, commit, e.SkipIdentity)
		if err != nil && err.Kind == FailureNotListening && lastErr != nil && ctx.Err() != nil {
			// Wait's own deadline or cancel cut this sample short. Report the
			// last complete sample, not the aborted request.
			return giveUp(start, attempts-1, last, lastErr, base, ctx.Err())
		}
		res.Attempts, res.ElapsedMS = attempts, time.Since(start).Milliseconds()
		if err == nil {
			return res, nil
		}
		if !err.Kind.Retryable() {
			return res, err
		}
		last, lastErr = res, err
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return giveUp(start, attempts, last, lastErr, base, ctx.Err())
		case <-timer.C:
		}
	}
}

// Probe takes exactly one sample: every check once, in order, with no retry.
func (c Client) Probe(ctx context.Context, t Target, e Expect) (ProbeResult, error) {
	start := time.Now()
	base, commit, perr := prepare(t, e)
	if perr != nil {
		return ProbeResult{Failure: perr.Kind}, perr
	}
	res, err := c.sample(ctx, base, t.Bearer, commit, e.SkipIdentity)
	res.Attempts, res.ElapsedMS = 1, time.Since(start).Milliseconds()
	if err != nil {
		return res, err
	}
	return res, nil
}

// giveUp builds the result for a Wait that ran out of time or was cancelled
// while the target was still not listening or not ready.
func giveUp(start time.Time, attempts int, last ProbeResult, lastErr *ProbeError, base string, ctxErr error) (ProbeResult, error) {
	cause := fmt.Errorf("gave up before the first sample: %w", ctxErr)
	if lastErr == nil {
		// No sample ran, so nothing was observed: no check, no status.
		lastErr = &ProbeError{Kind: FailureNotListening}
		last = ProbeResult{BaseURL: base, Failure: lastErr.Kind}
	} else {
		cause = fmt.Errorf("gave up after %d attempt(s): %w (last: %s)", attempts, ctxErr, lastErr.Error())
	}
	last.Ready = false
	last.Attempts = attempts
	last.TimedOut = errors.Is(ctxErr, context.DeadlineExceeded)
	last.ElapsedMS = time.Since(start).Milliseconds()
	return last, &ProbeError{
		Kind:       lastErr.Kind,
		Check:      lastErr.Check,
		StatusCode: lastErr.StatusCode,
		Cause:      cause,
	}
}

// prepare validates the inputs before any request, so a bad target or
// expectation fails fast instead of looking like a server that never listens.
func prepare(t Target, e Expect) (base, commit string, perr *ProbeError) {
	invalid := func(format string, args ...any) (string, string, *ProbeError) {
		return "", "", &ProbeError{Kind: FailureInvalidTarget, Cause: fmt.Errorf(format, args...)}
	}
	u, err := url.Parse(strings.TrimSpace(t.BaseURL))
	if err != nil {
		// Quote only the parse reason: *url.Error embeds the raw URL, which may
		// carry credentials.
		var ue *url.Error
		if errors.As(err, &ue) {
			err = ue.Err
		}
		return invalid("base URL: %v", err)
	}
	if (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.User != nil ||
		(u.Path != "" && u.Path != "/") || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" {
		return invalid("base URL must be an absolute http(s) origin with no user info, path, query or fragment")
	}
	for _, r := range t.Bearer {
		if r < 0x20 || r == 0x7f {
			return invalid("bearer contains a control character")
		}
	}
	if !e.SkipIdentity {
		commit = strings.ToLower(strings.TrimSpace(e.Commit))
		if !fullCommit(commit) {
			return invalid("expected commit must be a full 40-hex object ID (or set SkipIdentity)")
		}
	}
	return u.Scheme + "://" + u.Host, commit, nil
}

// sample runs liveness, readiness, catalog and identity once, in that order,
// and stops at the first failure.
func (c Client) sample(ctx context.Context, base, bearer, commit string, skipIdentity bool) (ProbeResult, *ProbeError) {
	res := ProbeResult{BaseURL: base}
	fail := func(err *ProbeError) (ProbeResult, *ProbeError) {
		res.Ready, res.Failure, res.Check, res.StatusCode = false, err.Kind, err.Check, err.StatusCode
		return res, err
	}

	if _, err := c.fetch(ctx, CheckLiveness, base+PathHealth, "", maxStatusBody); err != nil {
		return fail(err)
	}
	if _, err := c.fetch(ctx, CheckReadiness, base+PathReady, "", maxStatusBody); err != nil {
		return fail(err)
	}

	body, err := c.fetch(ctx, CheckCatalog, base+PathModels, bearer, maxCatalogBody)
	if err != nil {
		return fail(err)
	}
	n, err := checkCatalog(body)
	res.Models = n
	if err != nil {
		return fail(err)
	}

	if !skipIdentity {
		body, err := c.fetch(ctx, CheckIdentity, base+PathVersion, bearer, maxVersionBody)
		if err != nil {
			return fail(err)
		}
		id, err := checkIdentity(body, commit)
		res.Identity = id
		if err != nil {
			return fail(err)
		}
	}
	res.Ready = true
	return res, nil
}

var errBodyTooLarge = errors.New("response body exceeds the probe limit")

// fetch GETs one route and maps transport and status failures onto the closed
// failure vocabulary. Only a 200 passes.
func (c Client) fetch(ctx context.Context, check Check, endpoint, bearer string, limit int64) ([]byte, *ProbeError) {
	rctx, cancel := context.WithTimeout(ctx, orDefault(c.RequestTimeout, DefaultRequestTimeout))
	defer cancel()
	req, err := http.NewRequestWithContext(rctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, &ProbeError{Kind: FailureInvalidTarget, Check: check, Cause: err}
	}
	req.Header.Set("Accept", "application/json")
	if bearer != "" {
		req.Header.Set("Authorization", "Bearer "+bearer)
	}
	resp, err := c.doer().Do(req)
	if err != nil {
		return nil, &ProbeError{Kind: transportFailure(err), Check: check, Cause: err}
	}
	if resp == nil || resp.Body == nil {
		return nil, &ProbeError{Kind: FailureNotListening, Check: check, Cause: errors.New("HTTP client returned an empty response")}
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return nil, &ProbeError{Kind: FailureNotListening, Check: check, StatusCode: resp.StatusCode, Cause: err}
	}
	switch code := resp.StatusCode; {
	case code == http.StatusOK:
	case code == http.StatusBadGateway || code == http.StatusServiceUnavailable || code == http.StatusGatewayTimeout:
		return nil, &ProbeError{Kind: FailureNotReady, Check: check, StatusCode: code}
	case code == http.StatusUnauthorized || code == http.StatusForbidden:
		return nil, &ProbeError{Kind: FailureUnauthorized, Check: check, StatusCode: code}
	default:
		return nil, &ProbeError{Kind: FailureBadStatus, Check: check, StatusCode: code}
	}
	if int64(len(raw)) > limit {
		return nil, &ProbeError{Kind: FailureMalformedResponse, Check: check, StatusCode: resp.StatusCode, Cause: errBodyTooLarge}
	}
	return raw, nil
}

// transportFailure classifies a request that got no response. A TLS
// verification failure or an http/https scheme mismatch means a server did
// answer, and retrying cannot fix it, so it fails fast. Everything else (refused,
// reset, timed out) is not-listening.
func transportFailure(err error) FailureKind {
	var (
		verifyErr  *tls.CertificateVerificationError
		authority  x509.UnknownAuthorityError
		hostname   x509.HostnameError
		invalid    x509.CertificateInvalidError
		recordHead tls.RecordHeaderError
	)
	if errors.Is(err, http.ErrSchemeMismatch) || errors.As(err, &verifyErr) || errors.As(err, &authority) ||
		errors.As(err, &hostname) || errors.As(err, &invalid) || errors.As(err, &recordHead) {
		return FailureTransport
	}
	return FailureNotListening
}

// checkCatalog requires an OpenAI-compatible model list: object "list" and a
// non-empty data list whose items are object "model" with a trimmed, non-empty
// id. These are the same shape rules the private router updater enforces;
// unknown fields are tolerated. An explicit is_ready:false is not-ready, so
// Wait samples again.
func checkCatalog(body []byte) (int, *ProbeError) {
	malformed := func(format string, args ...any) (int, *ProbeError) {
		return 0, &ProbeError{Kind: FailureMalformedResponse, Check: CheckCatalog, StatusCode: http.StatusOK, Cause: fmt.Errorf(format, args...)}
	}
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return malformed("models response is not a JSON object")
	}
	var catalog struct {
		Object  *string `json:"object"`
		IsReady *bool   `json:"is_ready"`
		Data    *[]struct {
			ID     *string `json:"id"`
			Object *string `json:"object"`
		} `json:"data"`
	}
	if err := json.Unmarshal(trimmed, &catalog); err != nil {
		return malformed("decode models response: %v", err)
	}
	if catalog.IsReady != nil && !*catalog.IsReady {
		return 0, &ProbeError{Kind: FailureNotReady, Check: CheckCatalog, StatusCode: http.StatusOK, Cause: errors.New("models response reports is_ready false")}
	}
	if catalog.Object == nil || *catalog.Object != "list" {
		return malformed("models response object is not list")
	}
	if catalog.Data == nil {
		return malformed("models response has no data list")
	}
	models := *catalog.Data
	if len(models) == 0 {
		return 0, &ProbeError{Kind: FailureEmptyCatalog, Check: CheckCatalog, StatusCode: http.StatusOK, Cause: errors.New("models response has no models")}
	}
	for i, m := range models {
		if m.ID == nil || strings.TrimSpace(*m.ID) == "" {
			return malformed("model %d has an empty id", i)
		}
		if strings.TrimSpace(*m.ID) != *m.ID {
			return malformed("model %d has an untrimmed id", i)
		}
		if m.Object == nil || *m.Object != "model" {
			return malformed("model %d object is not model", i)
		}
	}
	return len(models), nil
}

// checkIdentity parses /version with the one stamp parser and compares the
// observed build to the expected commit. Only a clean, stamped build of
// exactly that commit passes; no identity failure is retried.
func checkIdentity(body []byte, commit string) (*stamp.Identity, *ProbeError) {
	id, err := stamp.ParseVersionJSON(body)
	switch {
	case errors.Is(err, stamp.ErrInvalidCommit):
		return &id, &ProbeError{Kind: FailureIdentityUnproven, Check: CheckIdentity, StatusCode: http.StatusOK, Cause: err}
	case err != nil:
		return nil, &ProbeError{Kind: FailureMalformedResponse, Check: CheckIdentity, StatusCode: http.StatusOK, Cause: err}
	}
	_, cause := stamp.Explain(id.Stamp(), commit)
	switch cause {
	case stamp.CauseMatched:
		return &id, nil
	case stamp.CauseDiverged:
		return &id, &ProbeError{Kind: FailureIdentityDrift, Check: CheckIdentity, StatusCode: http.StatusOK,
			Cause: fmt.Errorf("running %s, expected %s", short(id.Commit), short(commit))}
	default:
		return &id, &ProbeError{Kind: FailureIdentityUnproven, Check: CheckIdentity, StatusCode: http.StatusOK,
			Cause: fmt.Errorf("running build cannot prove its commit (%s)", cause)}
	}
}

// fullCommit reports whether s is a full 40-character lower-case hex object ID.
func fullCommit(s string) bool {
	if len(s) != 40 {
		return false
	}
	for _, r := range s {
		if !((r >= '0' && r <= '9') || (r >= 'a' && r <= 'f')) {
			return false
		}
	}
	return true
}

func short(commit string) string {
	if len(commit) > 12 {
		return commit[:12]
	}
	return commit
}

// clip bounds a server-supplied value quoted into an error.
func clip(s string) string {
	const max = 80
	if len(s) > max {
		return s[:max] + "..."
	}
	return s
}
