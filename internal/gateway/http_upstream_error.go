package gateway

import (
	"fmt"
	"net/http"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

// upstream4xxStatus classifies an upstream 4xx (a REQUEST error the client can act on)
// into a distinct, actionable OpenAI-style code + message. The upstream status passes
// straight through in every arm (no remap — remapping 429 would break fak's own backoff
// and any downstream client's). Every message is built from se.Status + fixed literals
// ONLY — the upstream's raw Body never crosses the trust boundary (#82/#346 invariant).
func upstream4xxStatus(se *agent.UpstreamStatusError) (status int, code, msg string) {
	// A 403/402 that fak already classified as a self-recovering USAGE/OVERAGE cap
	// (se.LimitReason set to a session/weekly/usage window by the header-aware
	// classifyLimit429 on the retry path) is NOT a permanent permission wall — it clears
	// at its named reset. It only reaches here after the cap-aware wait exhausted the retry
	// budget without the window resetting. Give it an honest "capped, recovers at reset —
	// do NOT re-login" message keyed on LimitReason, so the wrapped agent/operator is not
	// pushed toward a futile /login for a condition login cannot fix. The reset is echoed as
	// Retry-After by writeUpstreamErr. This precedes the per-status arms so a capped 403
	// never falls into the generic "re-login or check your plan" message below.
	if (se.Status == http.StatusForbidden || se.Status == http.StatusPaymentRequired) && isUsageCapReason(se.LimitReason) {
		return se.Status, "upstream_usage_cap",
			fmt.Sprintf("upstream is at a usage/overage cap (HTTP %d): a rolling 5h/7d window hit its limit with overage disabled. "+
				"The credential is VALID and RECOVERS on its own at the window reset — do NOT re-login (a fresh token hits the same cap). "+
				"Wait for the reset (see the Retry-After response header), reduce usage, or ask the org admin to enable overage.", se.Status)
	}
	switch se.Status {
	case http.StatusBadRequest: // 400
		return se.Status, "upstream_invalid_request",
			"upstream rejected the request as malformed (HTTP 400) — check the model name, message roles, and parameter ranges"
	case http.StatusUnauthorized: // 401
		return se.Status, "upstream_unauthorized",
			"upstream rejected the credential (HTTP 401) — the upstream API key/token is missing, wrong, or expired (re-login or check --api-key-env)"
	case http.StatusForbidden: // 403
		// This 403 already SURVIVED fak's bounded transient-recovery arm (a few paced
		// retries across a short window; see agent.forbiddenRetryState) — so it is the
		// PERSISTENT kind, not a server-side abuse/capacity flap that would have cleared.
		// The message says so and names the real fixes, deliberately NOT leading with
		// "run /login": on the 2026-07-03 gem8 storm the harness turned this into a
		// spurious /login for a denial that was transient and login could not fix.
		//
		// The ORGANIZATION-scoped OAuth disable is the canonical deceiving case: the
		// credential is valid, but this org has subscription/OAuth inference turned off
		// upstream, so EVERY re-login mints another token for the same walled org and is
		// futile. When fak also has an account roster it will already have tried to fail
		// over to a permitted sibling (see account-failover); reaching this terminal message
		// means that failover found no permitted account. So this arm names the real cause
		// and the real fixes (a different account, or a plain API key) and does NOT tell the
		// operator to re-login — the one instruction that cannot possibly help here.
		// Reaching here means the 403 carried the org-disable body AND was NOT a
		// usage/overage rejection (agent.classifyUpstream routes an overage-rejected 403 —
		// `overage-status: rejected` with the account otherwise allowed — to a cap-aware
		// backoff toward its reset, never to this terminal message). So this is a genuine
		// standing org wall, not a self-recovering usage cap: re-login is futile and the
		// fix is a different account or a plain API key.
		if agent.IsOrgOAuthDisabled([]byte(se.Body)) {
			return se.Status, "upstream_org_oauth_disabled",
				"upstream denied access (HTTP 403): this organization has OAuth/subscription inference disabled upstream (and this was not a usage-cap rejection, which would have self-recovered at its reset). The credential is valid but the ORG is walled, so re-login cannot fix it — every login mints another token for the same org. Fix: switch to an account whose organization permits access, or use a plain API key (ANTHROPIC_API_KEY / --api-key-env) for API billing, or ask the org admin to re-enable subscription access."
		}
		return se.Status, "upstream_forbidden",
			"upstream denied access (HTTP 403), persisting past fak's retry window — the credential is valid but lacks permission for this model, org, or region. If this entitlement should exist, re-login or check the subscription/plan; if you meant a different model, switch to a permitted one. (A transient 403 would have self-healed; this one did not.)"
	case http.StatusNotFound: // 404
		return se.Status, "upstream_model_not_found",
			"upstream does not know this model or endpoint (HTTP 404) — verify the model id and that --base-url targets the right API"
	case http.StatusRequestTimeout: // 408
		return se.Status, "upstream_request_timeout",
			"upstream timed out receiving the request (HTTP 408) — retry; if it persists, reduce the request size"
	case http.StatusRequestEntityTooLarge: // 413
		return se.Status, "upstream_payload_too_large",
			"upstream rejected the request as too large (HTTP 413) — reduce the prompt/context size or max_tokens"
	case http.StatusTooManyRequests: // 429
		return se.Status, "upstream_rate_limited",
			"upstream rate-limited the request (HTTP 429) — back off and retry (see the Retry-After response header when the provider supplied one)"
	default:
		// Any other 4xx (402, 405, 409, 422, 451, …): the historical generic text,
		// now reached ONLY by un-enumerated statuses rather than every 4xx.
		return se.Status, "upstream_request_rejected",
			fmt.Sprintf("upstream rejected the request (HTTP %d)", se.Status)
	}
}

// surfaceUpstreamStatus writes an upstream failure straight through to a client that has
// not yet seen a byte: the distinct status + code the classifier maps the error to, plus
// any Retry-After the provider supplied, so the client learns WHAT failed (a persistent
// 429 stays a 429 carrying its reset) instead of reading a generic 502. Surfacing only —
// the caller observes the metric, so a failure stays counted exactly once. note is the
// caller's log phrase, kept per-surface because the same status means something different
// on a relay that DECLINED to re-issue than on a planner turn that simply failed.
func (s *Server) surfaceUpstreamStatus(w http.ResponseWriter, err error, note string) {
	status, code, msg := upstreamErrorStatus(err)
	if ra := upstreamRetryAfter(err); ra != "" {
		w.Header().Set("Retry-After", ra)
	}
	s.logf("gateway: %s: %v", note, err)
	writeErrCode(w, status, code, msg)
}

// failClosedOnUnparsedToolCalls is the streamed tool-call CONFORMANCE rule, written once
// for every streamed surface. The upstream announced tool_calls but none survived parsing
// plus the text-lift fallback; proceeding would skip adjudication on a call the model
// intended to make — the exact silent no-op the buffered paths refuse. So the turn ends
// here, and how it ends depends only on whether we still own the status line: nothing on
// the wire yet means a clean 502, otherwise the headers are spent and the client gets a
// terminal error frame instead, so its parser closes rather than reading a benign empty
// stop on a skipped call. Reports whether it took the turn; false means the completion is
// conformant and the caller proceeds.
//
// terminal writes that frame in the caller's own wire dialect (Anthropic SSE events vs
// OpenAI data + [DONE]) and is only called in the mid-stream arm. preTag/midTag are the
// caller's log parentheticals, kept per-surface so an operator grepping the log can still
// tell which stream refused.
func (s *Server) failClosedOnUnparsedToolCalls(w http.ResponseWriter, comp *agent.Completion, started bool, preTag, midTag string, terminal func()) bool {
	if comp == nil || !comp.ToolCallsDropped || len(comp.Message.ToolCalls) > 0 {
		return false
	}
	if !started {
		s.logf("gateway: upstream announced tool_calls but none parsed (%s); model=%s", preTag, s.model)
		writeErr(w, http.StatusBadGateway, "upstream tool-call format not recognized; refusing to skip adjudication")
		return true
	}
	s.logf("gateway: upstream announced tool_calls but none parsed mid-stream (%s); model=%s", midTag, s.model)
	terminal()
	return true
}

// isUsageCapReason reports whether a LimitReason names a self-recovering ACCOUNT CAP — a
// session/weekly/usage window that clears at its reset — as opposed to a plain rate throttle or
// no limit. It is the gateway-message counterpart of the retry loop's cap classification: a 403/402
// carrying one of these means "capped, recovers at reset," not "permanent wall," so the client is
// told to wait rather than re-login. The reason strings are the resume package's stable vocabulary.
func isUsageCapReason(reason string) bool {
	switch reason {
	case resume.LimitSession, resume.LimitWeekly, resume.LimitUsage:
		return true
	default:
		return false
	}
}

// isAccountBlockCode reports whether an upstreamErrorStatus code names a block that is
// SCOPED TO THE ACTIVE ACCOUNT/SEAT — a rate-limit ceiling, a usage/overage cap, or a
// credential/permission/org wall — as opposed to a request-shaped error (bad model, too
// large, malformed) that no seat switch could fix. Only these are worth naming the active
// seat on: they are exactly the conditions where the operator's next move is "which of my
// seats hit this, and should I switch off it or wait for its reset." The roster (fak info)
// already shows the active seat, but the message that actually STOPS a turn never did — so
// when one of these fires, appendActiveAccount folds the live seat name in (trusted path
// only). The strings are the stable codes minted by upstreamErrorStatus / upstream4xxStatus.
func isAccountBlockCode(code string) bool {
	switch code {
	case "upstream_retry_ceiling", "upstream_usage_cap", "upstream_unauthorized",
		"upstream_forbidden", "upstream_org_oauth_disabled", "upstream_rate_limited":
		return true
	default:
		return false
	}
}

// activeAccountLabel renders the live "which seat" suffix for an account-scoped upstream
// block: the seat currently serving turns (with its advisory email), plus how many sibling
// seats a failover already walled this session — so an operator/wrapped agent reading a 403
// or a 429 ceiling knows WHICH of several accounts hit the wall without cross-referencing
// the /debug/vars roster by hand. It reads the SAME live pull provider /debug/vars uses
// (s.sessionEndpoints), so a mid-run failover onto a sibling seat is reflected, and returns
// "" when no roster is wired or no seat is active (a plain fak serve, or a single-key
// session) — leaving the message untouched. Display metadata only (seat name + email),
// never a credential, so it honors the payload-free contract the roster already lives under.
func (s *Server) activeAccountLabel() string {
	ep, ok := s.sessionEndpoints()
	if !ok {
		return ""
	}
	var active *SessionAccount
	walled := 0
	for i := range ep.Accounts {
		if ep.Accounts[i].Active {
			active = &ep.Accounts[i]
		}
		if ep.Accounts[i].Walled {
			walled++
		}
	}
	if active == nil {
		return ""
	}
	label := "Active account: " + active.Name
	if active.Email != "" {
		label += " <" + active.Email + ">"
	}
	if walled > 0 {
		label += fmt.Sprintf(" (%d sibling seat(s) already walled this session)", walled)
	}
	return label + "."
}
