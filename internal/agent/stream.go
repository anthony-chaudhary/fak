package agent

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

// StreamSink receives incremental assistant CONTENT fragments as they arrive from
// the upstream model. It is the live half of a streamed turn: each call carries the
// next chunk of natural-language output the moment the provider emits it, so a
// downstream client sees a real time-to-first-token instead of waiting for the whole
// turn to finish.
//
// Tool-call deltas are deliberately NOT delivered here — they are buffered inside
// CompleteStream and returned in the final Completion, so the caller (the gateway)
// can route every proposed call through the kernel's adjudication BEFORE the client
// ever sees it. Streaming a tool call live would bypass that gate; streaming content
// does not, because content is the model's own prose, which the buffered path
// forwards verbatim too. A non-nil error returned by the sink aborts the stream
// (e.g. the client disconnected) and surfaces from CompleteStream.
type StreamSink func(contentDelta string) error

// StreamingPlanner is the optional capability a Planner advertises when it can
// stream the upstream completion token-by-token. It is a strict superset of Planner:
// CompleteStream behaves exactly like Complete (same sampling, same quarantine, same
// adjudication-relevant return shape) but invokes sink for each content fragment as
// it arrives, then returns the fully-accumulated Completion (content + buffered tool
// calls + usage + finish reason). A Planner that cannot stream simply does not
// implement this interface, and the gateway falls back to its buffered path.
type StreamingPlanner interface {
	Planner
	// StreamingSupported reports whether a live token stream is available for the
	// planner's CURRENT configuration (e.g. the OpenAI-compatible chat wire), so the
	// gateway can decide to take the streaming path WITHOUT committing to a request
	// it would have to unwind. False means callers must use Complete.
	StreamingSupported() bool
	// CompleteStream is Complete with a live content sink. On a planner whose wire
	// does not support streaming it returns ErrStreamingUnsupported without touching
	// the network, so the caller can fall back having written nothing.
	CompleteStream(ctx context.Context, sink StreamSink, messages []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error)
}

// ErrStreamingUnsupported is returned by CompleteStream when the planner's wire
// cannot stream (every non-OpenAI-compatible provider, for now). It is a sentinel so
// the gateway can distinguish "this wire can't stream, fall back cleanly" from a real
// upstream failure.
var ErrStreamingUnsupported = errors.New("agent: streaming not supported for this provider wire")

// upstreamCall is the fully-resolved input to one upstream round-trip — the shared
// product of the buffered Complete and the streaming CompleteStream. Extracting it
// guarantees both paths apply the SAME pre-send quarantine, coherence shaping,
// sampling resolution, request-model + credential pass-through, and (Anthropic)
// raw-body passthrough, so a streamed request differs from a buffered one ONLY by the
// `stream` flag in the body.
type upstreamCall struct {
	adapter      TranscriptAdapter
	url          string
	body         []byte
	apiKey       string
	upstreamBeta string
	extraHeaders map[string]string
	quarantined  int
	redacted     int                   // rung 5 (#572): messages whose content was span-redacted pre-send
	redactions   []TranscriptRedaction // the full reversible records (CAS Original) behind that count (#882)
	// authRefreshable marks the pinned/rotating-credential path — no per-request
	// UpstreamAPIKey (the transparent passthrough hop authenticates with the client's
	// OWN key, which we must not second-guess) AND a live APIKeyFunc on the planner. On
	// this path a single upstream 401 is recoverable: the on-disk subscription token may
	// have rotated mid-flight (Claude Code rewrites .credentials.json ~hourly) or been
	// briefly torn by that rewrite, so re-resolving the credential FRESH and retrying once
	// turns an intermittent 401 into a self-healed turn. False everywhere else, so the
	// static-key and passthrough paths are byte-for-byte unchanged.
	authRefreshable bool
}

// Auth-refresh outcomes reported to HTTPPlanner.AuthRefreshNotify. A 401 on the rotating-
// subscription path either RECOVERED (a fresh token was adopted and the call re-sent in place,
// so the live session healed across a re-login) or was EXHAUSTED (no fresher token appeared
// within the grace window, so the 401 is about to surface and the agent drops into its own
// /login). The two are counted apart so an operator can tell a self-healed blip from a session
// about to die.
const (
	AuthRefreshRecovered = "recovered"
	AuthRefreshExhausted = "exhausted"
)

// notifyAuthRefresh fires the planner's AuthRefreshNotify hook if set. Centralized so the three
// 401 self-heal sites (Complete, CompleteStream, StreamAnthropicRaw) report identically.
func notifyAuthRefresh(p *HTTPPlanner, outcome string, attempt int) {
	if p != nil && p.AuthRefreshNotify != nil {
		p.AuthRefreshNotify(outcome, attempt)
	}
}

// Forbidden-retry outcomes reported to HTTPPlanner.ForbiddenRetryNotify. A 403's bounded
// recovery arm either RECOVERED (a retry within the short window returned 200, so a transient
// abuse/capacity gate cleared and the live session healed in place instead of dropping into a
// spurious /login) or was EXHAUSTED (the window/attempts elapsed still 403ing, so the denial
// is the permanent entitlement kind and now surfaces with the actionable answer). Counted
// apart so an operator can tell a self-healed 403 flap from a session dying on a real
// permission denial — the same recovered/exhausted split the 401 self-heal already draws.
const (
	ForbiddenRetryRecovered = "recovered"
	ForbiddenRetryExhausted = "exhausted"
)

// notifyForbiddenRetry fires the planner's ForbiddenRetryNotify hook if set. Centralized so the
// three 403 recovery sites (Complete, CompleteStream, StreamAnthropicRaw) report identically,
// mirroring notifyAuthRefresh.
func notifyForbiddenRetry(p *HTTPPlanner, outcome string, attempt int) {
	if p != nil && p.ForbiddenRetryNotify != nil {
		p.ForbiddenRetryNotify(outcome, attempt)
	}
}

// Account-failover outcomes reported to HTTPPlanner.AccountFailoverNotify. When a 403 names an
// ACCOUNT-SCOPED wall (org/region/billing — classifyUpstream -> RemedyFailoverAccount), the arm
// either RECOVERED (a permitted sibling account's credential was adopted and the call re-sent in
// place, so a walled session healed onto a working account instead of dropping into a futile
// /login) or was EXHAUSTED (no failover target existed — every sibling walled/absent — so the
// account-scoped 403 surfaces terminally). Counted apart from the transient-403 flap because the
// cause is different (a permanent per-credential wall, not a clearing capacity gate) and so is the
// fix (swap accounts, not wait).
const (
	AccountFailoverRecovered = "recovered"
	AccountFailoverExhausted = "exhausted"
)

// Seat-rehome outcomes reported to HTTPPlanner.AccountFailoverNotify when the arm fired for a
// 429 ACCOUNT CAP (session/weekly/usage — isAccountCap429) rather than a 403 org wall. A 429
// account cap can hold for the full 5h/7d reset window (and is frequently a multi-account or
// billing condition longer than it looks), so instead of sleeping on the capped seat toward its
// named reset, the arm rehomes to a permitted sibling seat that can serve the turn now. It shares
// AccountFailoverNotify (same swap mechanism, same telemetry family) but reports a DISTINCT
// outcome so a cap-driven rehome is never conflated with an org-wall failover: RehomedSeat when a
// sibling seat was adopted and the call re-sent in place, or RehomeSeatUnavailable when no sibling
// seat was free (every one walled/capped/absent), leaving the cap-aware backoff to ride it out.
const (
	RehomedSeat           = "rehomed_seat"
	RehomeSeatUnavailable = "rehome_seat_unavailable"
)

// notifyAccountFailover fires the planner's AccountFailoverNotify hook if set. Centralized so the
// three account-failover sites (Complete, CompleteStream, StreamAnthropicRaw) report identically,
// mirroring notifyForbiddenRetry.
func notifyAccountFailover(p *HTTPPlanner, outcome string, attempt int) {
	if p != nil && p.AccountFailoverNotify != nil {
		p.AccountFailoverNotify(outcome, attempt)
	}
}

// failoverAccountCred asks the planner's AccountFailoverFunc for a REPLACEMENT credential from a
// permitted sibling account, adopting it onto this call in place. Unlike refreshAPIKeyWait — which
// polls for a FRESHER token of the SAME (walled) account and so can never escape an org-scoped
// wall — this rotates to a DIFFERENT account whose org still permits the request. reason is the
// classified remedy label (never the raw upstream body). It returns true and mutates c.apiKey when
// a failover target was found (the caller re-sends the same attempt with the new credential), or
// false when there is none — leaving the 403 to surface terminally. A nil AccountFailoverFunc (no
// sibling roster wired) makes this a no-op false, so the historical terminal-on-org-403 behavior is
// exactly preserved when failover is not configured.
func (c *upstreamCall) failoverAccountCred(p *HTTPPlanner, reason string) bool {
	if p == nil || p.AccountFailoverFunc == nil {
		return false
	}
	if newCred, ok := p.AccountFailoverFunc(reason); ok && newCred != "" && newCred != c.apiKey {
		c.apiKey = newCred
		c.extraHeaders = p.effectiveExtraHeaders()
		return true
	}
	return false
}

// refreshAPIKeyWait is refreshAPIKey with a bounded grace window for the RE-LOGIN race.
// A 401 on the rotating-subscription path means the token fak just sent is dead; the fix
// is to send the rotated-in replacement. But when the token expired, the upstream 401s the
// instant it dies while the credential file is only rewritten a beat later — by Claude Code
// refreshing it, or by a user running `claude` / `claude /login` in another terminal to log
// back in. A single read at the 401 instant therefore usually still sees the SAME stale
// token, so refreshAPIKey gives up and the 401 surfaces to the wrapped agent, which then
// drops into its own /login and the live guarded session is lost — the exact "logging in
// again does not revive the session" failure. So here we POLL effectiveAPIKey across a
// short window (authRefreshWindow, default 3s) until a different non-empty token appears,
// adopt it, and report success so the caller re-sends in place. The common case — the token
// already rotated on disk — is caught by the first read with zero added latency. A
// genuinely-dead credential with no re-login coming polls out the window and returns false
// (fail fast, as before, just after the grace period). The context cancels the wait
// promptly, so a disconnected client or a cancelled turn never blocks here. A zero window
// (FAK_AUTH_REFRESH_WINDOW=0) collapses this to the historical single read.
func (c *upstreamCall) refreshAPIKeyWait(ctx context.Context, p *HTTPPlanner) bool {
	if !c.authRefreshable {
		return false
	}
	if fresh, ok := waitForFreshAPIKey(ctx, p, c.apiKey); ok {
		c.apiKey = fresh
		c.extraHeaders = p.effectiveExtraHeaders()
		return true
	}
	return false
}

// waitForFreshAPIKey polls the planner's live APIKeyFunc for a non-empty token DIFFERENT
// from current, across the bounded re-login grace window (authRefreshWindow). It returns
// the fresh token the first time one appears, or ("", false) if the window elapses, the
// context is cancelled, or the planner has no APIKeyFunc. The first read is free (zero
// latency when the rotated token is already on disk); a zero window collapses to that
// single read. Shared by the buffered (refreshAPIKeyWait) and streaming (CompleteStream /
// StreamAnthropicRaw) 401 self-heal paths so all recover an in-session re-login identically.
func waitForFreshAPIKey(ctx context.Context, p *HTTPPlanner, current string) (string, bool) {
	if p == nil || p.APIKeyFunc == nil {
		return "", false
	}
	if fresh := p.effectiveAPIKey(); fresh != "" && fresh != current {
		return fresh, true
	}
	window := authRefreshWindow()
	if window <= 0 {
		return "", false
	}
	deadline := time.Now().Add(window)
	for {
		wait := authRefreshPollInterval
		if rem := time.Until(deadline); rem < wait {
			wait = rem
		}
		if wait <= 0 {
			return "", false
		}
		if err := sleepCtx(ctx, wait); err != nil {
			// Context cancelled/expired: do not keep an agent waiting on a re-login the
			// caller has already abandoned. The 401 will surface, correct for an aborted turn.
			return "", false
		}
		if fresh := p.effectiveAPIKey(); fresh != "" && fresh != current {
			return fresh, true
		}
	}
}

// headers builds the per-request header set, applying the Anthropic-wire beta union
// (the inbound client's negotiated betas merged with any the auth scheme required).
// It mirrors the header logic Complete ran inline before the extraction.
func (c *upstreamCall) headers() map[string]string {
	h := c.adapter.Headers(c.apiKey)
	for k, v := range c.extraHeaders {
		if strings.TrimSpace(k) != "" {
			h[k] = v
		}
	}
	if c.upstreamBeta != "" && c.adapter.Provider() == ProviderAnthropic {
		h["anthropic-beta"] = mergeBeta(h["anthropic-beta"], c.upstreamBeta)
	}
	return h
}

// applyHeaders sets every non-empty resolved header onto req, the shared pre-send step
// for both the buffered (Complete) and streaming (CompleteStream) round-trips.
func (c *upstreamCall) applyHeaders(req *http.Request) {
	for k, v := range c.headers() {
		if v != "" {
			req.Header.Set(k, v)
		}
	}
}

// prepareUpstream resolves messages+tools+opts into a single upstreamCall. stream
// selects whether the marshaled body asks the provider to deliver an SSE token
// stream (honored only by the OpenAI-compatible chat wire; other adapters ignore the
// flag, so an unsupported-wire body is byte-for-byte the non-stream one).
func (p *HTTPPlanner) prepareUpstream(messages []Message, tools []ToolDef, stream bool, opts ...SampleOpt) (*upstreamCall, error) {
	adapter, err := p.transcriptAdapter()
	if err != nil {
		return nil, err
	}
	safeMessages := messages
	var quarantines []TranscriptQuarantine
	if p.QuarantineTranscript {
		safeMessages, quarantines = QuarantineOutboundMessages(messages)
	}
	// §A4 coherence shaping: after the safety quarantine, give the coherence layer a
	// chance to break the provider prefix when a world witness has been refuted. nil
	// hook = unchanged path; applied copy-on-shape so the caller's slice is untouched.
	if p.CoherenceShaper != nil {
		safeMessages = p.CoherenceShaper(safeMessages)
	}
	sp := applySampleOpts(opts...)
	// Request-model pass-through (#82): a client-supplied model wins for THIS turn.
	modelID := p.ModelID
	if sp.Model != "" {
		modelID = sp.Model
	}
	maxTokens := p.MaxTokens
	if sp.MaxTokens != nil && *sp.MaxTokens > 0 {
		maxTokens = *sp.MaxTokens
	}
	temperature := p.Temperature
	if sp.Temperature != nil {
		temperature = *sp.Temperature
	}
	// Passthrough: forward the client's ORIGINAL bytes so its prompt-cache prefix
	// survives (a real upstream cache hit). The provider gate is load-bearing:
	// RawRequestBody is Anthropic-shaped, so it must never reach an OpenAI/Gemini/xAI
	// endpoint — only the anthropic→anthropic proxy forwards it. The upstream hop is
	// non-streaming (the gateway re-synthesizes the SSE), so stream:true in the raw
	// body is forced off or the buffered ParseResponse chokes on the SSE reply — the
	// same fix Complete applied inline before the extraction.
	var reqBody []byte
	redactedN := 0
	var redactions []TranscriptRedaction
	if len(sp.RawRequestBody) > 0 && adapter.Provider() == ProviderAnthropic {
		reqBody = forceAnthropicNonStreaming(sp.RawRequestBody)
	} else {
		// Rung 5 (#572): span-level PII/secret redaction on the non-passthrough
		// re-marshal path. The Anthropic passthrough above forwards req.Raw verbatim
		// and never serializes these messages, so redaction runs ONLY here, where the
		// re-marshal can carry it to the wire. Default-inert: with FAK_WIRE_REDACT
		// unset, RedactOutboundMessages returns safeMessages unchanged at zero cost.
		safeMessages, redactions = RedactOutboundMessages(safeMessages)
		redactedN = len(redactions)
		extraBody, err := mergeGuidedDecodeExtraBody(p.ExtraBody, sp.GuidedDecode)
		if err != nil {
			return nil, err
		}
		reqBody, err = adapter.MarshalRequest(adapterRequest{
			Model:          modelID,
			Messages:       safeMessages,
			Tools:          tools,
			Temperature:    temperature,
			MaxTokens:      maxTokens,
			TopP:           sp.TopP,
			TopK:           sp.TopK,
			Stop:           sp.Stop,
			ResponseFormat: sp.ResponseFormat,
			LogitBias:      sp.LogitBias,
			ExtraBody:      extraBody,
			Stream:         stream,
		})
		if err != nil {
			return nil, err
		}
	}
	// Transparent hop: when the inbound client supplied its own upstream credential
	// (passthrough), authenticate with THAT key rather than the planner's. Otherwise use
	// the planner's EFFECTIVE key, which re-resolves a rotating subscription token per
	// request (effectiveAPIKey) instead of a frozen boot-time string.
	apiKey := p.effectiveAPIKey()
	extraHeaders := p.effectiveExtraHeaders()
	authRefreshable := sp.UpstreamAPIKey == "" && p.APIKeyFunc != nil
	if sp.UpstreamAPIKey != "" {
		apiKey = sp.UpstreamAPIKey
	}
	return &upstreamCall{
		adapter:         adapter,
		url:             adapter.Endpoint(p.BaseURL, modelID),
		body:            reqBody,
		apiKey:          apiKey,
		upstreamBeta:    sp.UpstreamBeta,
		extraHeaders:    extraHeaders,
		quarantined:     len(quarantines),
		redacted:        redactedN,
		redactions:      redactions,
		authRefreshable: authRefreshable,
	}, nil
}

// StreamingSupported reports whether the planner's configured wire can stream. Only
// the OpenAI-compatible chat wire (OpenAI and the xAI/vLLM/SGLang-compatible servers
// that share its SSE delta format) is wired today; every other provider returns false
// so the gateway keeps its buffered path for them.
func (p *HTTPPlanner) StreamingSupported() bool {
	switch p.Provider {
	case ProviderOpenAI, ProviderXAI, "":
		return true
	default:
		return false
	}
}

// openAIStreamChunk is one OpenAI-compatible `chat.completion.chunk` SSE event. Each
// `data:` line carries one of these; the terminal `data: [DONE]` carries none. The
// delta fields are all optional and additive across chunks: content fragments
// concatenate into the final text, and tool-call fragments accumulate by index (id +
// name on first sight, arguments concatenated). Usage rides the final chunk only when
// the request asked for it (stream_options.include_usage).
type openAIStreamChunk struct {
	Model   string `json:"model"`
	Choices []struct {
		Delta struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *Usage `json:"usage"`
}

// CompleteStream performs one streamed chat-completions round-trip on the
// OpenAI-compatible wire: it forwards each content fragment to sink as it arrives and
// returns the fully-assembled Completion (content + buffered tool calls + usage +
// finish) once the upstream closes the stream. Tool calls are accumulated, NEVER
// streamed, so the caller can adjudicate them before exposing them.
//
// Like Complete it retries a transient transport error or a retryable status (429/503/
// 529/…) with backoff — but ONLY before the first byte reaches the sink, where a retry is
// invisible to the client and the caller can still choose an HTTP status. It NEVER retries
// mid-stream: once bytes have flowed a read error is returned as-is. A non-OpenAI wire
// returns ErrStreamingUnsupported without a network call.
// streamConnect dials the upstream with the same retry/backoff/Retry-After policy as
// Complete, but ONLY before the first byte is streamed: a pre-stream failure has emitted
// nothing to the sink, so the retry is safe and invisible to the client. A deterministic
// dial failure fails fast; a 401 on the rotating-credential path self-heals once via a
// fresh-token re-send. It returns a live 200 response (body still open — the caller closes
// it) or, after exhausting attempts, the true upstream status error (never a later glitch).
func (p *HTTPPlanner) streamConnect(ctx context.Context, call *upstreamCall) (*http.Response, error) {
	maxAttempts, deadline, budgetOn := retryBounds(time.Now())
	var rs retryState // shared between-attempt truth (#1358, #1362) — see retry_state.go
	triedAuthRefresh := false
	var fbState forbiddenRetryState // bounded transient-403 recovery arm (see retry.go), mirrors Complete
	// 429-account-cap seat rehome, one swap max (see Complete). Safe on this pre-stream retry loop:
	// a retryable status here has emitted nothing to the sink yet, so swapping seats and re-sending
	// is invisible to the client. rehomePending defers the confirmed-rehome notify to the 200.
	triedRehome := false
	rehomePending := false
	var resp *http.Response
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 {
			stop, err := p.retryBackoffWait(ctx, attempt, rs.lastStatus, rs.lastRetryAfter, rs.lastCapWait, rs.lastStatusErr, deadline, budgetOn)
			if err != nil {
				return nil, err
			}
			if stop {
				break
			}
		}
		req, err := http.NewRequestWithContext(ctx, "POST", call.url, bytes.NewReader(call.body))
		if err != nil {
			return nil, err
		}
		call.applyHeaders(req)
		req.Header.Set("Accept", "text/event-stream")
		r, err := p.Client.Do(req)
		if err != nil {
			if deterministicTransportError(err) {
				return nil, &UpstreamUnreachableError{Err: err}
			}
			rs.noteTransportGlitch(err)
			continue
		}
		if r.StatusCode == http.StatusOK {
			// A 200 after the 403 arm fired is a CONFIRMED transient-403 self-heal (see Complete).
			if fbState.attempted() {
				notifyForbiddenRetry(p, ForbiddenRetryRecovered, attempt)
				fbState.fired = false
			}
			// A 200 after a 429-account-cap seat rehome is a CONFIRMED rehome (see Complete).
			if rehomePending {
				notifyAccountFailover(p, RehomedSeat, attempt)
				rehomePending = false
			}
			resp = r
			break
		}
		raw, _ := io.ReadAll(io.LimitReader(r.Body, 4096))
		r.Body.Close()
		if retryableStatus(r.StatusCode) {
			rs.noteRetryableStatus(r.StatusCode, raw, r.Header, 400)
			// A 429 ACCOUNT CAP (session/weekly/usage) can hold for a hours-away reset; rehome the
			// session ONCE to a permitted sibling seat rather than sleep on the capped one. Same swap
			// mechanism as the 403 failover; on no free seat, fall through to the cap-aware backoff.
			// A transient rate_limited throttle keeps its seat (isAccountCap429 false).
			if !triedRehome && p.AccountFailoverFunc != nil &&
				isAccountCap429(r.StatusCode, raw, r.Header, time.Now()) {
				if call.failoverAccountCred(p, RehomedSeat) {
					triedRehome = true
					rehomePending = true
					rs.lastRetryAfter = ""
					rs.lastCapWait = ""
					attempt--
					continue
				}
				notifyAccountFailover(p, RehomeSeatUnavailable, attempt)
				triedRehome = true
			}
			continue
		}
		// A 401 on the rotating-subscription path: re-resolve the credential fresh and retry
		// ONCE (attempt-- so the refresh re-send is immediate and uncounted), mirroring Complete.
		// refreshAPIKeyWait polls the on-disk token across the re-login grace window, so a user
		// logging back in mid-stream is adopted and the live session self-heals in place.
		if r.StatusCode == http.StatusUnauthorized && !triedAuthRefresh && call.authRefreshable {
			if call.refreshAPIKeyWait(ctx, p) {
				triedAuthRefresh = true
				notifyAuthRefresh(p, AuthRefreshRecovered, attempt)
				rs.lastErr = &UpstreamStatusError{Status: r.StatusCode, Body: truncate(raw, 400), RetryAfter: r.Header.Get("Retry-After")}
				rs.lastStatus = r.StatusCode
				rs.lastRetryAfter = ""
				attempt--
				continue
			}
			notifyAuthRefresh(p, AuthRefreshExhausted, attempt)
		}
		// A 403's bounded transient-recovery arm (self-contained short paced wait; see Complete):
		// retry a transient abuse/capacity denial a few times before surfacing it terminally.
		if r.StatusCode == http.StatusForbidden {
			if fbState.step(ctx, raw) == forbiddenRetryGo {
				attempt--
				continue
			}
			if fbState.attempted() {
				notifyForbiddenRetry(p, ForbiddenRetryExhausted, attempt)
			}
		}
		return nil, &UpstreamStatusError{Status: r.StatusCode, Body: truncate(raw, 400), RetryAfter: r.Header.Get("Retry-After")}
	}
	if resp == nil {
		return nil, rs.exhausted("planner: streaming failed after retries")
	}
	return resp, nil
}

func (p *HTTPPlanner) CompleteStream(ctx context.Context, sink StreamSink, messages []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error) {
	if !p.StreamingSupported() {
		return nil, ErrStreamingUnsupported
	}
	call, err := p.prepareUpstream(messages, tools, true, opts...)
	if err != nil {
		return nil, err
	}
	// Retry a transient transport error OR a retryable status (429/503/529/…) with the
	// SAME backoff+jitter+Retry-After policy as Complete — but ONLY here, before the first
	// byte is streamed. A pre-stream failure has emitted nothing to the sink, so the retry
	// is safe and invisible to the client. A deterministic dial failure fails fast (no
	// backoff). A 401 on the rotating-credential path self-heals once via a fresh-token
	// re-send (a no-op on the static-key/passthrough paths). Each non-200 response body is
	// drained+closed in the loop; only the successful 200 escapes to the streaming reader.
	resp, err := p.streamConnect(ctx, call)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Some "OpenAI-compatible" servers ignore stream:true and answer with a single
	// buffered JSON body. Detect that by content-type and fall back to the buffered
	// parser — deliver the whole content as one fragment — so the client gets the
	// correct (if not incremental) turn instead of an empty stream.
	if ct := resp.Header.Get("Content-Type"); ct != "" && !strings.Contains(ct, "event-stream") {
		raw, rerr := io.ReadAll(resp.Body)
		if rerr != nil {
			return nil, fmt.Errorf("planner: %s: read body: %w", call.adapter.Provider(), rerr)
		}
		comp, perr := call.adapter.ParseResponse(raw)
		if perr != nil {
			return nil, fmt.Errorf("planner: %s: %w", call.adapter.Provider(), perr)
		}
		comp = normalizeCompletionToolCalls(comp)
		if sink != nil && comp.Message.Content != "" {
			if serr := sink(comp.Message.Content); serr != nil {
				return nil, serr
			}
		}
		p.attachProviderCacheTelemetry(comp, call.body, call.adapter.Provider())
		comp.Raw = raw
		comp.PreSendQuarantines = call.quarantined
		comp.PreSendRedactions = call.redacted
		comp.PreSendRedactionRecords = call.redactions
		return comp, nil
	}

	var (
		content strings.Builder
		rawBuf  bytes.Buffer // reconstructs the wire transcript for Completion.Raw
		toolAcc = map[int]*ToolCall{}
		usage   Usage
		model   string
		finish  string
	)
	// Idle-read deadline: an upstream that opens the stream then goes silent (a transient
	// overload / "API issue") fails in ≤streamStallTimeout() rather than blocking the
	// scanner on resp.Body.Read until the whole-request Client.Timeout (600s under guard)
	// fires. A healthy stream's steady deltas reset the window; only true silence trips it.
	sr := newStallReader(resp.Body, streamStallTimeout())
	defer sr.Close()
	sc := bufio.NewScanner(sr)
	// A single SSE data line can carry a large tool-call argument fragment; raise the
	// scanner ceiling well past the 64 KiB default so a big chunk is never truncated.
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		// Keep the raw frames so the streamed Completion carries the same Raw transcript
		// witness the buffered path does (the bytes are otherwise consumed line-by-line).
		rawBuf.Write(sc.Bytes())
		rawBuf.WriteByte('\n')
		data, ok := strings.CutPrefix(sc.Text(), "data:")
		if !ok {
			continue // SSE comments / event: lines / blank separators
		}
		data = strings.TrimSpace(data)
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			break
		}
		var chunk openAIStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue // tolerate a keep-alive or non-JSON heartbeat line
		}
		if chunk.Model != "" {
			model = chunk.Model
		}
		if chunk.Usage != nil {
			usage = *chunk.Usage
		}
		for _, ch := range chunk.Choices {
			if ch.FinishReason != "" {
				finish = ch.FinishReason
			}
			if ch.Delta.Content != "" {
				content.WriteString(ch.Delta.Content)
				if sink != nil {
					if err := sink(ch.Delta.Content); err != nil {
						return nil, err
					}
				}
			}
			for _, tcd := range ch.Delta.ToolCalls {
				acc := toolAcc[tcd.Index]
				if acc == nil {
					acc = &ToolCall{Type: "function"}
					toolAcc[tcd.Index] = acc
				}
				if tcd.ID != "" {
					acc.ID = tcd.ID
				}
				if tcd.Type != "" {
					acc.Type = tcd.Type
				}
				if tcd.Function.Name != "" {
					acc.Function.Name = tcd.Function.Name
				}
				acc.Function.Arguments += tcd.Function.Arguments
			}
		}
	}
	if err := sc.Err(); err != nil {
		if errors.Is(err, ErrUpstreamStalled) {
			return nil, &UpstreamStalledError{Idle: streamStallTimeout(), Err: err}
		}
		return nil, fmt.Errorf("planner: %s: stream read: %w", call.adapter.Provider(), err)
	}

	calls := make([]ToolCall, 0, len(toolAcc))
	for _, idx := range sortedIndices(toolAcc) {
		calls = append(calls, *toolAcc[idx])
	}
	comp := normalizeCompletionToolCalls(&Completion{
		Message:      Message{Role: RoleAssistant, Content: content.String(), ToolCalls: calls},
		FinishReason: finish,
		Usage:        usage,
		Model:        model,
	})
	p.attachProviderCacheTelemetry(comp, call.body, call.adapter.Provider())
	comp.Raw = rawBuf.Bytes()
	comp.PreSendQuarantines = call.quarantined
	comp.PreSendRedactions = call.redacted
	comp.PreSendRedactionRecords = call.redactions
	return comp, nil
}

func sortedIndices(m map[int]*ToolCall) []int {
	idx := make([]int, 0, len(m))
	for k := range m {
		idx = append(idx, k)
	}
	sort.Ints(idx)
	return idx
}
