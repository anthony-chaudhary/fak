package routing

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/deadlineadmit"
)

// FailureClass is the single closed classification of an inference failure; each
// class maps to exactly one Action. Add a needed distinction here, never as a
// private taxonomy elsewhere; credential eligibility belongs to platform/keyring.
type FailureClass string

const (
	// FailureRateLimited is a provider-side rate limit.
	FailureRateLimited FailureClass = "RATE_LIMITED"
	// FailureTimeout is a transport or server deadline exceeded.
	FailureTimeout FailureClass = "UPSTREAM_TIMEOUT"
	// FailureUnavailable is a server-side overload or outage.
	FailureUnavailable FailureClass = "UPSTREAM_UNAVAILABLE"
	// FailureEngineOOM is the serving engine out of memory; it acts like
	// UPSTREAM_UNAVAILABLE but is a separate class so journals can name the cause.
	FailureEngineOOM FailureClass = "ENGINE_OOM"
	// FailureAuthRejected is a rejected credential or forbidden caller.
	FailureAuthRejected FailureClass = "AUTH_REJECTED"
	// FailureContextOverflow is a prompt larger than the model's context window;
	// the same engine can never fit it.
	FailureContextOverflow FailureClass = "CONTEXT_OVERFLOW"
	// FailureModelMissing is a model handle the provider does not serve.
	FailureModelMissing FailureClass = "MODEL_UNAVAILABLE"
	// FailureBadPayload is a malformed request, a caller bug.
	FailureBadPayload FailureClass = "PAYLOAD_REJECTED"
	// FailureCanceled is a caller-intent cancellation.
	FailureCanceled FailureClass = "CANCELED"
	// FailureCreditExhausted is a billing/credit wall: the credential cannot succeed
	// until funds are added, so fail over and never retry it.
	FailureCreditExhausted FailureClass = "CREDIT_EXHAUSTED"
	// FailureConflict is an HTTP 409 state conflict that a peer engine does not
	// share, so it fails over rather than reading as a caller bug.
	FailureConflict FailureClass = "CONFLICT"
	// FailureDeadlineInfeasible is a healthy node's typed 503 + Retry-After refusal
	// (pkg/deadlineadmit); it counts against neither backend nor credential.
	FailureDeadlineInfeasible FailureClass = "DEADLINE_INFEASIBLE"
	// FailureStreamRequired is a stream-only upstream refusing a buffered request;
	// the remedy is another backend or a stream:true reissue.
	FailureStreamRequired FailureClass = "STREAM_REQUIRED"
	// FailureCertInvalid is a TLS chain verification verdict, server-side trust
	// state that fails over. A DNS "no such host" stays UPSTREAM_UNAVAILABLE.
	FailureCertInvalid FailureClass = "CERT_INVALID"
	// FailureUnknown is the fallback for an unplaceable failure; it moves to the
	// next engine rather than guess.
	FailureUnknown FailureClass = "UNKNOWN"
)

// failureClasses is the closed failure-class vocabulary in declaration order.
var failureClasses = []FailureClass{
	FailureRateLimited,
	FailureTimeout,
	FailureUnavailable,
	FailureEngineOOM,
	FailureAuthRejected,
	FailureContextOverflow,
	FailureModelMissing,
	FailureBadPayload,
	FailureCanceled,
	FailureCreditExhausted,
	FailureConflict,
	FailureDeadlineInfeasible,
	FailureStreamRequired,
	FailureCertInvalid,
	FailureUnknown,
}

// FailureClasses returns a copy of every representable failure class.
func FailureClasses() []FailureClass {
	out := make([]FailureClass, len(failureClasses))
	copy(out, failureClasses)
	return out
}

// IsKnown reports whether f is a member of the closed failure-class vocabulary.
func (f FailureClass) IsKnown() bool {
	for _, known := range failureClasses {
		if f == known {
			return true
		}
	}
	return false
}

// Action is the single closed response to a failure; a closed vocabulary keeps
// Failover deterministic and auditable.
type Action string

const (
	// ActionRetrySame retries the same engine after backoff.
	ActionRetrySame Action = "RETRY_SAME"
	// ActionFailoverNext moves to the next engine.
	ActionFailoverNext Action = "FAILOVER_NEXT"
	// ActionFatal means neither retry nor fail over: a caller bug or caller intent.
	ActionFatal Action = "FATAL"
)

var failureActions = []Action{
	ActionRetrySame,
	ActionFailoverNext,
	ActionFatal,
}

// Actions returns a copy of every representable failure action.
func Actions() []Action {
	out := make([]Action, len(failureActions))
	copy(out, failureActions)
	return out
}

// IsKnown reports whether a is a member of the closed action vocabulary.
func (a Action) IsKnown() bool {
	for _, known := range failureActions {
		if a == known {
			return true
		}
	}
	return false
}

// Failure is the unified understanding of one inference failure: class, action,
// retry budget and evidence. Detail records only the deciding token, never the
// response body, which may carry prompt text or credentials.
type Failure struct {
	// Class is the closed failure class.
	Class FailureClass `json:"class"`
	// Action is the closed response action.
	Action Action `json:"action"`
	// Retries is the recommended retry budget; zero means fail over immediately.
	Retries int `json:"retries"`
	// Detail is the evidence the classification rested on.
	Detail string `json:"detail,omitempty"`
	// Refusal is the closed client-facing token for a refusal no caller-side retry
	// can cure (see ClientRefusal); empty otherwise.
	Refusal ClientRefusal `json:"refusal,omitempty"`
	// OrgWide marks a wall on the whole credential org: rolling sibling keys is
	// futile and tombstones each live one, so RetryBudget returns zero. Set on a bare 405.
	OrgWide bool `json:"org_wide,omitempty"`
}

// DefaultAction maps each failure class to its deterministic response: transient
// classes and DEADLINE_INFEASIBLE retry the same engine, caller bugs and
// cancellation are fatal, everything else fails over.
func (f FailureClass) DefaultAction() Action {
	switch f {
	case FailureRateLimited, FailureTimeout, FailureDeadlineInfeasible:
		return ActionRetrySame
	case FailureUnavailable, FailureEngineOOM, FailureAuthRejected, FailureContextOverflow, FailureModelMissing, FailureCreditExhausted, FailureConflict, FailureStreamRequired, FailureCertInvalid:
		return ActionFailoverNext
	case FailureBadPayload, FailureCanceled:
		return ActionFatal
	default:
		return ActionFailoverNext
	}
}

// Transient reports whether the same engine may answer the same input once time
// passes (OOM included: concurrent requests may release the memory it lacked).
// DEADLINE_INFEASIBLE is deliberately excluded: the node names the wait in
// its Retry-After, and a fixed re-walk backoff would re-dial too early.
func (f FailureClass) Transient() bool {
	switch f {
	case FailureRateLimited, FailureTimeout, FailureUnavailable, FailureEngineOOM:
		return true
	default:
		return false
	}
}

// Terminal reports whether the failure retires the credential that drew it. Only
// CREDIT_EXHAUSTED is: a 403 may be a transient abuse gate or an entitlement
// refusal, so retiring on AUTH_REJECTED would take a serving backend out.
func (f FailureClass) Terminal() bool {
	return f == FailureCreditExhausted
}

// DefaultRetries is the recommended retry budget per class: only rate limits and
// timeouts retry; everything else fails over immediately.
func (f FailureClass) DefaultRetries() int {
	switch f {
	case FailureRateLimited:
		return 2
	case FailureTimeout:
		return 1
	// Neither a stream-only refusal nor a certificate wall clears on the same target.
	case FailureStreamRequired:
		return 0
	case FailureCertInvalid:
		return 0
	default:
		return 0
	}
}

// RetryBudget is the one retry-budget seam a routing walk consults: how many
// further attempts at the same target are authorized, capped by credentialHeadroom.
// Fatal, unclassified, payload-rejected, admission-refused and OrgWide failures get
// zero; a class with a Retries budget is capped at it; otherwise the full headroom.
func (f Failure) RetryBudget(credentialHeadroom int) int {
	if credentialHeadroom <= 0 {
		return 0
	}
	if f.OrgWide {
		return 0
	}
	if f.Class == "" || f.Action == ActionFatal || f.Class == FailureBadPayload || f.Class.AdmissionRefused() {
		return 0
	}
	if f.Retries > 0 && f.Retries < credentialHeadroom {
		return f.Retries
	}
	return credentialHeadroom
}

// contextOverflowSignatures are checked before any other body signature, since a
// 400 overflow would otherwise read as a bad payload. "request too large for" is
// deliberately absent: it is a TPM rate limit and would roll the whole key pool.
// Snake_case codes are explicit entries, never a global "_" normalisation, which
// would match JSON field names such as "context_length".
var contextOverflowSignatures = []string{
	"context length",
	"maximum context",
	"too many tokens",
	"context_length_exceeded",
	"exceed_context_size_error",
	"exceeds the available context size",
	"leaves no generation position",
	"max_seq_len",
}

// creditExhaustedSignatures are the billing-wall body signatures, checked before the
// rate-limit and auth signatures. No bare "billing" or "usage limit" entry: rate-limit
// bodies mention both, so a bare form would make a retryable limit a terminal wall.
var creditExhaustedSignatures = []string{
	"payment required",
	"exhausted your budget",
	"add funds",
	"insufficient_quota",
	"insufficient quota",
	"credit balance is too low",
	"out of credit",
	"credit_balance_exhausted",
	"billing_hard_limit_reached",
	"billing_not_active",
	"usage_limit_reached",
	"usage limit reached",
	"usage_not_included",
}

// orgPausedSignatures are kept out of creditExhaustedSignatures because an org
// pause counts only on a 405 or in-band error; on a 401/403 it is an auth verdict.
var orgPausedSignatures = []string{
	"organization is currently paused",
	"organisation is currently paused",
}

// rateLimitedSignatures are shared by Classify and ClassifyError so they cannot
// drift; the snake_case entries are in-band codes of a 200-stream error event.
var rateLimitedSignatures = []string{
	"rate limit",
	"too many requests",
	"rate_limit_exceeded",
	"rate_limit_error",
	"too_many_requests",
}

// modelMissingSignatures are shared by Classify and ClassifyError. The
// unavailable/invalid-name wordings are a provider withdrawing a model behind a
// bare 400: a model wall that fails over, not a caller bug.
var modelMissingSignatures = []string{
	"model not found",
	"no such model",
	"unknown model",
	"model_not_found",
	"model is unavailable",
	"model unavailable",
	"model is not available",
	"invalid model name",
}

// freeTierSignatures mark a 403 free-tier refusal. The key serves other models, so
// it is a per-model wall that outranks the auth rung and never tombstones the key.
var freeTierSignatures = []string{
	"freetiererror",
	"free tier can only be used",
}

var engineOOMSignatures = []string{
	"in_kernel_oom",
}

// streamRequiredSignatures mirror the agent plane's streamRequired matcher verbatim
// so there is one taxonomy, not two.
var streamRequiredSignatures = []string{
	"stream required",
	"streaming required",
	"streaming is required",
	"must stream",
	"stream must be",
	"streaming must be",
	"non-streaming is not",
	"nonstreaming is not",
	"only supports streaming",
	"only support streaming",
	"requires stream",
	"requires streaming",
	"enable streaming",
}

// certInvalidSignatures are used by ClassifyError only, since a certificate verdict
// arrives as a handshake error value. The generic "x509: certificate" tail stays
// last so the specific wordings win the Detail token.
var certInvalidSignatures = []string{
	"certificate signed by unknown authority",
	"certificate has expired",
	"cert_has_expired",
	"certificate is not valid for",
	"tls: failed to verify certificate",
	"x509: certificate",
}

// classSignatures maps body substrings to classes; first match wins, so the order is
// the precedence (Classify checks context overflow before this table).
var classSignatures = []struct {
	class      FailureClass
	signatures []string
	refusal    ClientRefusal
}{
	// The exact engine code outranks prose rungs: the OOM message discusses budgets
	// a looser signature could misread.
	{FailureEngineOOM, engineOOMSignatures, ""},
	{FailureCreditExhausted, creditExhaustedSignatures, ""},
	{FailureRateLimited, rateLimitedSignatures, ""},
	// Outranks timeout/availability, whose 500 would otherwise read UNAVAILABLE.
	{FailureStreamRequired, streamRequiredSignatures, ""},
	{FailureTimeout, []string{"timed out", "timeout", "deadline exceeded"}, ""},
	// "slow_down" is an overload code, not a rate limit; "tokenizer cache full" is a
	// server-side transient answered as a 400, which would otherwise end the walk.
	{FailureUnavailable, []string{"overloaded", "temporarily unavailable", "slow_down", "tokenizer cache full"}, ""},
	{FailureModelMissing, freeTierSignatures, RefusalFreeTierClientOnly},
	{FailureAuthRejected, []string{"invalid api key", "unauthorized", "forbidden", "authentication", "invalid_api_key", "permission_error"}, ""},
	{FailureModelMissing, modelMissingSignatures, ""},
}

// standardStatusClasses is the status map Classify(status, "") must agree with. A
// status outside it resolves by range. 405 is the gateway's org-pause signal, so
// it fails over rather than being fatal.
var standardStatusClasses = map[int]FailureClass{
	400: FailureBadPayload,
	401: FailureAuthRejected,
	402: FailureCreditExhausted,
	403: FailureAuthRejected,
	404: FailureModelMissing,
	405: FailureUnavailable,
	406: FailureBadPayload,
	407: FailureAuthRejected,
	408: FailureTimeout,
	409: FailureConflict,
	410: FailureBadPayload,
	411: FailureBadPayload,
	412: FailureBadPayload,
	413: FailureContextOverflow,
	414: FailureBadPayload,
	415: FailureBadPayload,
	416: FailureBadPayload,
	417: FailureBadPayload,
	418: FailureBadPayload,
	421: FailureBadPayload,
	422: FailureBadPayload,
	423: FailureBadPayload,
	424: FailureBadPayload,
	425: FailureBadPayload,
	426: FailureBadPayload,
	428: FailureRateLimited,
	429: FailureRateLimited,
	431: FailureBadPayload,
	451: FailureBadPayload,
	500: FailureUnavailable,
	501: FailureUnavailable,
	502: FailureUnavailable,
	503: FailureUnavailable,
	504: FailureTimeout,
	505: FailureUnavailable,
	506: FailureUnavailable,
	507: FailureUnavailable,
	508: FailureUnavailable,
	510: FailureUnavailable,
	511: FailureUnavailable,
}

// normalizeSignatureText lowercases and collapses whitespace so a signature matches
// a re-cased or line-wrapped spelling; it never removes a separator. It feeds
// matching only: Detail keeps the original bytes.
func normalizeSignatureText(s string) string {
	return strings.Join(strings.Fields(strings.ToLower(s)), " ")
}

// Classify is the HTTP-shaped failure classifier, a pure function of status and
// body. Body signatures run first because status alone is lossy; then the status
// map, then the range. An in-band 2xx caller must pass only the extracted error
// event, never relayed model text.
func Classify(status int, body string) Failure {
	lower := normalizeSignatureText(body)

	// Precedes the auth rung, since a paused-org body also reads as auth failure.
	if status == 405 || (status >= 200 && status < 400) {
		for _, sig := range orgPausedSignatures {
			if strings.Contains(lower, sig) {
				return newFailure(FailureCreditExhausted, fmt.Sprintf("body signature %q", sig))
			}
		}
	}
	// Precedes the shared signatures: its body type "server_overloaded" would read as
	// an outage. Keyed on the code plus 503 so a stray mention is not claimed.
	if status == 503 && strings.Contains(lower, deadlineadmit.CodeDeadlineInfeasible) {
		return newFailure(FailureDeadlineInfeasible, fmt.Sprintf("body signature %q", deadlineadmit.CodeDeadlineInfeasible))
	}
	for _, sig := range contextOverflowSignatures {
		if strings.Contains(lower, sig) {
			return newFailure(FailureContextOverflow, fmt.Sprintf("body signature %q", sig))
		}
	}
	for _, entry := range classSignatures {
		for _, sig := range entry.signatures {
			if strings.Contains(lower, sig) {
				f := newFailure(entry.class, fmt.Sprintf("body signature %q", sig))
				f.Refusal = entry.refusal
				return f
			}
		}
	}

	if class, ok := standardStatusClasses[status]; ok {
		f := newFailure(class, fmt.Sprintf("status=%d", status))
		// A bare 405 is the org-pause signal; OrgWide stops the walk rolling siblings.
		if status == 405 {
			f.OrgWide = true
		}
		return f
	}
	var class FailureClass
	switch {
	case status >= 400 && status <= 499:
		class = FailureBadPayload
	case status >= 500 && status <= 599:
		class = FailureUnavailable
	default:
		class = FailureUnknown
	}
	return newFailure(class, fmt.Sprintf("status=%d", status))
}

// newFailure fills action and retries from the class defaults so they cannot
// disagree with the class.
func newFailure(class FailureClass, detail string) Failure {
	return Failure{
		Class:   class,
		Action:  class.DefaultAction(),
		Retries: class.DefaultRetries(),
		Detail:  detail,
	}
}

// errorSignatures maps a lowercased error string to a class; first match wins, so
// the order is the precedence.
var errorSignatures = []struct {
	class      FailureClass
	signatures []string
}{
	{FailureCanceled, []string{"context canceled", "canceled", "cancelled"}},
	// Ahead of the timeout rung, whose bare "deadline" would claim this sentinel too.
	{FailureDeadlineInfeasible, []string{deadlineadmit.CodeDeadlineInfeasible}},
	{FailureContextOverflow, contextOverflowSignatures},
	{FailureEngineOOM, engineOOMSignatures},
	// An error value has no status, so an org pause ranks with the credit walls.
	{FailureCreditExhausted, append(append([]string{}, creditExhaustedSignatures...), orgPausedSignatures...)},
	{FailureRateLimited, rateLimitedSignatures},
	{FailureTimeout, []string{"timed out", "timeout", "deadline"}},
	// A failed TLS handshake is a trust verdict, not an outage, so it outranks availability.
	{FailureCertInvalid, certInvalidSignatures},
	// "actively refused" and "forcibly closed by the remote host" are the Windows
	// wordings, which lack the POSIX phrases.
	{FailureUnavailable, []string{"connection reset", "connection refused", "actively refused", "forcibly closed by the remote host", "broken pipe", "no such host", "eof", "overloaded", "slow_down", "tokenizer cache full"}},
	{FailureAuthRejected, []string{"invalid api key", "unauthorized", "forbidden", "invalid_api_key", "permission_error"}},
	{FailureModelMissing, modelMissingSignatures},
}

// failureDetailLimit bounds the evidence string so a long error cannot flood the audit trail.
const failureDetailLimit = 120

// ClassifyError is the error-value classifier: case-insensitive, first match wins.
// A nil error is a caller bug and fails closed as UNKNOWN + FATAL, the one
// exception to the class-to-action mapping.
func ClassifyError(err error) Failure {
	if err == nil {
		return Failure{Class: FailureUnknown, Action: ActionFatal, Detail: "nil error"}
	}
	// lower stays the faithful original because it becomes the UNKNOWN audit detail.
	lower := strings.ToLower(err.Error())
	matchable := normalizeSignatureText(err.Error())
	for _, entry := range errorSignatures {
		for _, sig := range entry.signatures {
			if strings.Contains(matchable, sig) {
				return newFailure(entry.class, truncateDetail(sig))
			}
		}
	}
	return newFailure(FailureUnknown, truncateDetail(lower))
}

func truncateDetail(s string) string {
	if len(s) <= failureDetailLimit {
		return s
	}
	return s[:failureDetailLimit]
}
