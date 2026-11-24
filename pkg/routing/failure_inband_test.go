package routing

import (
	"errors"
	"strings"
	"testing"
)

// In-band failure witnesses: a streaming provider that already answered 200 reports a
// mid-stream failure as a snake_case code in an SSE error event, so the body signature is
// the only evidence. Each positive row carries exactly one signature wrapped in neutral
// text, so deleting that signature fails exactly that row.
//
// On the wire, Classify sees only upstream statuses >= 400; these tests witness Classify's
// contract for a 200 body, not that the router acts on an in-band 200 error. A future
// adapter must pass only the extracted error event, never the relayed content stream,
// which could mention a signature phrase in model text.

// inBandCodeCase is one snake_case provider code and the verdict it must earn
// on its own.
type inBandCodeCase struct {
	code        string
	wantClass   FailureClass
	wantAction  Action
	wantRetries int
}

// inBandCodes is the full in-band snake_case vocabulary the classifier must
// recognise. Retries/actions are the documented class defaults: a rate limit
// retries the same engine twice; every other class here fails over at once.
var inBandCodes = []inBandCodeCase{
	{code: "rate_limit_exceeded", wantClass: FailureRateLimited, wantAction: ActionRetrySame, wantRetries: 2},
	{code: "rate_limit_error", wantClass: FailureRateLimited, wantAction: ActionRetrySame, wantRetries: 2},
	// too_many_requests is NOT in the upstream taxonomy: it is fak's own snake
	// twin of the spaced "too many requests" phrase, and stays a rate limit.
	{code: "too_many_requests", wantClass: FailureRateLimited, wantAction: ActionRetrySame, wantRetries: 2},
	// slow_down is an overload code, not a rate limit: it fails over as
	// UPSTREAM_UNAVAILABLE instead of retrying the same engine.
	{code: "slow_down", wantClass: FailureUnavailable, wantAction: ActionFailoverNext, wantRetries: 0},
	// Credit codes: credit_balance_exhausted, billing_hard_limit_reached,
	// usage_limit_reached, usage_not_included and insufficient_quota are the
	// upstream taxonomy's; billing_not_active is OpenAI's inactive-account code, which fak
	// also reads as a credit wall.
	{code: "credit_balance_exhausted", wantClass: FailureCreditExhausted, wantAction: ActionFailoverNext, wantRetries: 0},
	{code: "billing_hard_limit_reached", wantClass: FailureCreditExhausted, wantAction: ActionFailoverNext, wantRetries: 0},
	{code: "billing_not_active", wantClass: FailureCreditExhausted, wantAction: ActionFailoverNext, wantRetries: 0},
	{code: "usage_limit_reached", wantClass: FailureCreditExhausted, wantAction: ActionFailoverNext, wantRetries: 0},
	{code: "usage_not_included", wantClass: FailureCreditExhausted, wantAction: ActionFailoverNext, wantRetries: 0},
	{code: "insufficient_quota", wantClass: FailureCreditExhausted, wantAction: ActionFailoverNext, wantRetries: 0},
	{code: "context_length_exceeded", wantClass: FailureContextOverflow, wantAction: ActionFailoverNext, wantRetries: 0},
	{code: "invalid_api_key", wantClass: FailureAuthRejected, wantAction: ActionFailoverNext, wantRetries: 0},
	{code: "permission_error", wantClass: FailureAuthRejected, wantAction: ActionFailoverNext, wantRetries: 0},
	{code: "model_not_found", wantClass: FailureModelMissing, wantAction: ActionFailoverNext, wantRetries: 0},
}

// inBandShapes are the envelope shapes a provider uses to carry the code. Each
// wraps the code with neutral text only; none adds a second signature. The SSE
// shape is the extracted error event with no preceding content delta, matching what an
// adapter would hand Classify.
var inBandShapes = []struct {
	name string
	wrap func(code string) string
}{
	{
		name: "nested error.code",
		wrap: func(code string) string {
			return `{"error":{"code":"` + code + `","message":"see docs"}}`
		},
	},
	{
		name: "nested error.type",
		wrap: func(code string) string {
			return `{"type":"error","error":{"type":"` + code + `","message":"see docs"}}`
		},
	},
	{
		name: "top-level code",
		wrap: func(code string) string {
			return `{"type":"error","code":"` + code + `","message":"see docs"}`
		},
	},
	{
		name: "extracted SSE error event",
		wrap: func(code string) string {
			return "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"" + code + "\",\"message\":\"see docs\"}}\n\n"
		},
	},
	{
		name: "uppercase code is case-insensitive",
		wrap: func(code string) string {
			return `{"error":{"code":"` + strings.ToUpper(code) + `","message":"see docs"}}`
		},
	},
}

// assertInBand checks class, action, retries, and closed-vocabulary membership
// of one Classify verdict.
func assertInBand(t *testing.T, status int, body string, wantClass FailureClass, wantAction Action, wantRetries int) {
	t.Helper()
	got := Classify(status, body)
	if got.Class != wantClass {
		t.Errorf("Classify(%d, %q).Class = %q, want %q", status, body, got.Class, wantClass)
	}
	if got.Action != wantAction {
		t.Errorf("Classify(%d, %q).Action = %q, want %q", status, body, got.Action, wantAction)
	}
	if got.Retries != wantRetries {
		t.Errorf("Classify(%d, %q).Retries = %d, want %d", status, body, got.Retries, wantRetries)
	}
	if !got.Class.IsKnown() {
		t.Errorf("class %q is outside the closed vocabulary", got.Class)
	}
	if !got.Action.IsKnown() {
		t.Errorf("action %q is outside the closed vocabulary", got.Action)
	}
}

// TestClassifyInBandSnakeCaseCodes is the LOAD-BEARING witness for the in-band
// 200 gap: a stream that answered 200 and then carried a snake_case provider
// code must earn that code's named class, action, and retry budget in every envelope
// shape instead of falling through to UNKNOWN.
func TestClassifyInBandSnakeCaseCodes(t *testing.T) {
	for _, tc := range inBandCodes {
		for _, shape := range inBandShapes {
			t.Run(tc.code+"/"+shape.name, func(t *testing.T) {
				assertInBand(t, 200, shape.wrap(tc.code), tc.wantClass, tc.wantAction, tc.wantRetries)
			})
		}
	}
}

// TestClassifyInBandUnknownStaysUnknown witnesses the conservative floor: a 200
// with no body, or with an unrecognised snake code, is still UNKNOWN and fails
// over, so recognising snake codes does not turn every in-band error into a named class.
func TestClassifyInBandUnknownStaysUnknown(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{name: "empty body", body: ""},
		{name: "unlisted nested code", body: `{"error":{"code":"some_unlisted_code","message":"see docs"}}`},
		{name: "unlisted code in SSE error frame", body: "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"some_unlisted_code\"}}\n\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assertInBand(t, 200, tc.body, FailureUnknown, ActionFailoverNext, 0)
		})
	}
}

// TestClassifyInBandOutOfScopeCodesStayUnknown pins the deliberate omissions:
// policy codes (no policy class exists), "invalid_token" (would substring-collide with
// tokenizer errors such as invalid_token_id), and "server_error" (would outrank auth/model
// codes in the same body) are not signatures, so a 200 carrying only one stays UNKNOWN.
func TestClassifyInBandOutOfScopeCodesStayUnknown(t *testing.T) {
	for _, code := range []string{
		"cyber_policy",
		"bio_policy",
		"invalid_prompt",
		"misalignment_policy_violation",
		"invalid_token_id",
		"server_error",
	} {
		t.Run(code, func(t *testing.T) {
			assertInBand(t, 200, `{"error":{"code":"`+code+`","message":"see docs"}}`, FailureUnknown, ActionFailoverNext, 0)
		})
	}
}

// TestClassifyInBandSnakeCasePrecedence witnesses that snake codes obey the
// documented Classify precedence (context overflow > credit > rate limit >
// timeout > availability > auth > model missing) when two appear in one body, whatever
// their order in the text. Every adjacent rung pair is pinned; "timed out" stands in for
// the timeout rung, which has no snake code, and slow_down pins the availability rung.
func TestClassifyInBandSnakeCasePrecedence(t *testing.T) {
	cases := []struct {
		name        string
		winner      string
		loser       string
		wantClass   FailureClass
		wantAction  Action
		wantRetries int
	}{
		{name: "context overflow beats credit", winner: "context_length_exceeded", loser: "insufficient_quota", wantClass: FailureContextOverflow, wantAction: ActionFailoverNext},
		{name: "context overflow beats rate limit", winner: "context_length_exceeded", loser: "rate_limit_exceeded", wantClass: FailureContextOverflow, wantAction: ActionFailoverNext},
		{name: "credit beats rate limit", winner: "credit_balance_exhausted", loser: "rate_limit_error", wantClass: FailureCreditExhausted, wantAction: ActionFailoverNext},
		{name: "rate limit beats timeout", winner: "rate_limit_error", loser: "timed out", wantClass: FailureRateLimited, wantAction: ActionRetrySame, wantRetries: 2},
		{name: "timeout beats availability", winner: "timed out", loser: "slow_down", wantClass: FailureTimeout, wantAction: ActionRetrySame, wantRetries: 1},
		{name: "availability beats auth", winner: "slow_down", loser: "invalid_api_key", wantClass: FailureUnavailable, wantAction: ActionFailoverNext},
		{name: "rate limit beats auth", winner: "rate_limit_exceeded", loser: "invalid_api_key", wantClass: FailureRateLimited, wantAction: ActionRetrySame, wantRetries: 2},
		{name: "auth beats model missing", winner: "invalid_api_key", loser: "model_not_found", wantClass: FailureAuthRejected, wantAction: ActionFailoverNext},
	}
	for _, tc := range cases {
		t.Run(tc.name+"/loser first", func(t *testing.T) {
			body := `{"error":{"code":"` + tc.loser + `","type":"` + tc.winner + `"}}`
			assertInBand(t, 200, body, tc.wantClass, tc.wantAction, tc.wantRetries)
		})
		t.Run(tc.name+"/winner first", func(t *testing.T) {
			body := `{"error":{"code":"` + tc.winner + `","type":"` + tc.loser + `"}}`
			assertInBand(t, 200, body, tc.wantClass, tc.wantAction, tc.wantRetries)
		})
	}
}

// TestClassifyInBandBodyOutranksStatus witnesses that a snake-code body
// signature outranks the status map exactly as a spaced signature does: a 429
// naming a hard billing wall is CREDIT_EXHAUSTED, a 500 naming a rate limit is
// RATE_LIMITED, a 429 or 503 naming slow_down fails over, and a 400 naming a missing
// model fails over instead of going FATAL.
func TestClassifyInBandBodyOutranksStatus(t *testing.T) {
	cases := []struct {
		name        string
		status      int
		code        string
		wantClass   FailureClass
		wantAction  Action
		wantRetries int
	}{
		{name: "429 with billing wall is credit", status: 429, code: "billing_hard_limit_reached", wantClass: FailureCreditExhausted, wantAction: ActionFailoverNext},
		{name: "429 with slow_down is unavailable not rate limited", status: 429, code: "slow_down", wantClass: FailureUnavailable, wantAction: ActionFailoverNext},
		{name: "503 with slow_down is unavailable", status: 503, code: "slow_down", wantClass: FailureUnavailable, wantAction: ActionFailoverNext},
		{name: "500 with rate limit code is rate limited", status: 500, code: "rate_limit_error", wantClass: FailureRateLimited, wantAction: ActionRetrySame, wantRetries: 2},
		{name: "503 with credit code is credit", status: 503, code: "credit_balance_exhausted", wantClass: FailureCreditExhausted, wantAction: ActionFailoverNext},
		{name: "404 with auth code is auth", status: 404, code: "invalid_api_key", wantClass: FailureAuthRejected, wantAction: ActionFailoverNext},
		{name: "400 with model code fails over not fatal", status: 400, code: "model_not_found", wantClass: FailureModelMissing, wantAction: ActionFailoverNext},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := `{"error":{"code":"` + tc.code + `","message":"see docs"}}`
			assertInBand(t, tc.status, body, tc.wantClass, tc.wantAction, tc.wantRetries)
		})
	}
}

// TestClassify400ModelIsUnavailableFailsOver pins the OpenCode Go wording for a
// model id it does not serve. Read as PAYLOAD_REJECTED it would end the walk and mask
// earlier upstream 503s with a fatal turn instead of a retry.
func TestClassify400ModelIsUnavailableFailsOver(t *testing.T) {
	body := `{"type":"server_error","message":"Upstream request failed: Model is unavailable."}`
	assertInBand(t, 400, body, FailureModelMissing, ActionFailoverNext, 0)
}

// TestClassifyInBand429CreditCodeSpendsHeadroom pins the 429 row of
// docs/ROUTER-FAILOVER-AUTHORITY.md §1: a 429 whose body carries a credit code
// is CREDIT_EXHAUSTED / FAILOVER_NEXT with the full credential headroom as RetryBudget
// (a walled key's remedy is the next key), not the RATE_LIMITED cap of 2 a bare 429 earns.
func TestClassifyInBand429CreditCodeSpendsHeadroom(t *testing.T) {
	const headroom = 9
	t.Run("bare 429 contrast is rate limited with capped budget", func(t *testing.T) {
		assertInBand(t, 429, "", FailureRateLimited, ActionRetrySame, 2)
		if got := Classify(429, "").RetryBudget(headroom); got != 2 {
			t.Errorf("Classify(429, \"\").RetryBudget(%d) = %d, want 2", headroom, got)
		}
	})
	for _, code := range []string{"insufficient_quota", "billing_not_active", "usage_limit_reached"} {
		t.Run(code, func(t *testing.T) {
			body := `{"error":{"code":"` + code + `","message":"see docs"}}`
			assertInBand(t, 429, body, FailureCreditExhausted, ActionFailoverNext, 0)
			if got := Classify(429, body).RetryBudget(headroom); got != headroom {
				t.Errorf("Classify(429, %q).RetryBudget(%d) = %d, want the full headroom %d", body, headroom, got, headroom)
			}
		})
	}
}

// TestClassifyInBandFalsePositivePins guards the two misreadings the explicit
// snake-signature design exists to avoid: OpenAI's TPM rate-limit message
// mentions ".../account/billing", and credit is checked before rate limit, so a bare
// "billing" signature would mislabel a transient rate limit as a terminal credit wall;
// and a "context_length" metadata field in a model-not-found body must not read as
// CONTEXT_OVERFLOW, which a global "_"->space normalisation would cause.
func TestClassifyInBandFalsePositivePins(t *testing.T) {
	const tpmMessage = "Rate limit reached for gpt-4 in organization org-x on tokens per min. " +
		"Limit: 10000, Used: 9000, Requested: 2000. " +
		"You can increase your rate limit by adding a payment method to your account at https://platform.openai.com/account/billing."
	tpmEnvelope := `{"error":{"message":"` + tpmMessage + `","type":"tokens","param":null,"code":"rate_limit_exceeded"}}`

	cases := []struct {
		name        string
		status      int
		body        string
		wantClass   FailureClass
		wantAction  Action
		wantRetries int
	}{
		{name: "openai TPM message on 200 is rate limited not credit", status: 200, body: tpmMessage, wantClass: FailureRateLimited, wantAction: ActionRetrySame, wantRetries: 2},
		{name: "openai TPM message on 429 is rate limited not credit", status: 429, body: tpmMessage, wantClass: FailureRateLimited, wantAction: ActionRetrySame, wantRetries: 2},
		{name: "openai TPM envelope on 200 is rate limited not credit", status: 200, body: tpmEnvelope, wantClass: FailureRateLimited, wantAction: ActionRetrySame, wantRetries: 2},
		{name: "openai TPM envelope on 429 is rate limited not credit", status: 429, body: tpmEnvelope, wantClass: FailureRateLimited, wantAction: ActionRetrySame, wantRetries: 2},
		{
			name:       "context_length metadata in model-not-found body is not overflow",
			status:     404,
			body:       `{"error":{"message":"model not found"},"context_length":8192}`,
			wantClass:  FailureModelMissing,
			wantAction: ActionFailoverNext,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assertInBand(t, tc.status, tc.body, tc.wantClass, tc.wantAction, tc.wantRetries)
		})
	}
}

// groqTPMBody and openAITPMBody are the single-request-over-TPM-budget
// envelopes Groq (413) and OpenAI (429) return. Both open with "Request too
// large for <model>" and carry rate_limit_exceeded; Groq's also links a billing URL that
// must not read as a credit wall. Groq quotes identifiers in backticks, hence the
// interpreted string literal.
const (
	groqTPMBody = "{\"error\":{\"message\":\"Request too large for model `llama-3.1-70b-versatile` in organization `org_01abc` service tier `on_demand` on tokens per minute (TPM): Limit 6000, Requested 12411, please reduce your message size and try again. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing\",\"type\":\"tokens\",\"code\":\"rate_limit_exceeded\"}}"

	openAITPMMessage = "Request too large for gpt-4o in organization org-abc on tokens per min (TPM): Limit 30000, Requested 51234. " +
		"The input or output tokens must be reduced in order to run successfully. " +
		"Visit https://platform.openai.com/account/rate-limits to learn more."
	openAITPMBody = `{"error":{"message":"` + openAITPMMessage + `","type":"tokens","param":null,"code":"rate_limit_exceeded"}}`
)

// TestClassifyRequestTooLargeForTPMStaysRateLimited pins the
// single-request-over-TPM-budget case as a rate limit on every status, including Groq's
// 413 that the status map alone calls CONTEXT_OVERFLOW.
//
// The label is imperfect but the less damaging retry budget: a TPM limit is org-wide, so
// rolling keys is futile, and RATE_LIMITED caps that waste at RetryBudget 2 while
// CONTEXT_OVERFLOW would roll and rest the whole key pool. Re-adding "request too large
// for" as an overflow signature must fail these rows.
func TestClassifyRequestTooLargeForTPMStaysRateLimited(t *testing.T) {
	const headroom = 9
	cases := []struct {
		name   string
		status int
		body   string
	}{
		{name: "groq 413 TPM request too large", status: 413, body: groqTPMBody},
		{name: "openai 429 TPM request too large", status: 429, body: openAITPMBody},
		{name: "openai TPM request too large in-band on 200", status: 200, body: openAITPMBody},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assertInBand(t, tc.status, tc.body, FailureRateLimited, ActionRetrySame, 2)
			if got := Classify(tc.status, tc.body).RetryBudget(headroom); got != 2 {
				t.Errorf("Classify(%d, %q).RetryBudget(%d) = %d, want 2 (RATE_LIMITED caps futile same-backend key rolls)", tc.status, tc.body, headroom, got)
			}
		})
	}

	t.Run("sdk-wrapped openai TPM request too large error", func(t *testing.T) {
		err := errors.New("error, status code: 429, status: 429 Too Many Requests, message: " + openAITPMMessage + ", code: rate_limit_exceeded")
		assertErrorVerdict(t, err, FailureRateLimited, ActionRetrySame, 2)
		if got := ClassifyError(err).RetryBudget(headroom); got != 2 {
			t.Errorf("ClassifyError(%q).RetryBudget(%d) = %d, want 2", err, headroom, got)
		}
	})

	// Contrast: a 413 with no body signature keeps the status-map
	// CONTEXT_OVERFLOW verdict and its full-headroom budget — exactly the budget
	// the TPM rows above must not earn.
	t.Run("bare 413 contrast is overflow with full headroom", func(t *testing.T) {
		assertInBand(t, 413, "", FailureContextOverflow, ActionFailoverNext, 0)
		if got := Classify(413, "").RetryBudget(headroom); got != headroom {
			t.Errorf("Classify(413, \"\").RetryBudget(%d) = %d, want the full headroom %d", headroom, got, headroom)
		}
	})
}

// assertErrorVerdict checks class, action, retries, and closed-vocabulary
// membership of one ClassifyError verdict.
func assertErrorVerdict(t *testing.T, err error, wantClass FailureClass, wantAction Action, wantRetries int) {
	t.Helper()
	got := ClassifyError(err)
	if got.Class != wantClass {
		t.Errorf("ClassifyError(%q).Class = %q, want %q", err, got.Class, wantClass)
	}
	if got.Action != wantAction {
		t.Errorf("ClassifyError(%q).Action = %q, want %q", err, got.Action, wantAction)
	}
	if got.Retries != wantRetries {
		t.Errorf("ClassifyError(%q).Retries = %d, want %d", err, got.Retries, wantRetries)
	}
	if !got.Class.IsKnown() {
		t.Errorf("class %q is outside the closed vocabulary", got.Class)
	}
	if !got.Action.IsKnown() {
		t.Errorf("action %q is outside the closed vocabulary", got.Action)
	}
}

// TestClassifyErrorSnakeCaseParity witnesses that the error-string arm
// recognises every in-band snake code with the same verdict as the HTTP arm,
// so a transport- or SDK-wrapped code classifies the same as a body. This is per-code
// parity: the arms' unavailable and auth lists are separate and not identical.
func TestClassifyErrorSnakeCaseParity(t *testing.T) {
	for _, tc := range inBandCodes {
		t.Run(tc.code, func(t *testing.T) {
			assertErrorVerdict(t, errors.New("provider error: code="+tc.code), tc.wantClass, tc.wantAction, tc.wantRetries)
		})
	}
}

// TestClassifyErrorOverloadedIsUnavailable pins the error arm's unavailable
// rung on "overloaded": server_is_overloaded and overloaded_error are the
// overload codes grouped with slow_down, so an SDK-wrapped copy must fail over as
// UPSTREAM_UNAVAILABLE. No Go transport error contains "overloaded", so a local failure
// cannot be misread.
func TestClassifyErrorOverloadedIsUnavailable(t *testing.T) {
	for _, msg := range []string{
		"provider error: code=server_is_overloaded",
		"provider error: type=overloaded_error",
	} {
		t.Run(msg, func(t *testing.T) {
			assertErrorVerdict(t, errors.New(msg), FailureUnavailable, ActionFailoverNext, 0)
		})
	}
}

// TestClassifyErrorSnakeCasePrecedence mirrors TestClassifyInBandSnakeCasePrecedence
// for the error-string arm: when two signatures appear in one error, the
// documented ClassifyError precedence (canceled > context overflow > credit > rate limit >
// timeout > unavailable > auth > model missing) decides, whatever their order in the text.
// "timed out" and "context canceled" stand in for rungs with no snake code.
func TestClassifyErrorSnakeCasePrecedence(t *testing.T) {
	cases := []struct {
		name        string
		winner      string
		loser       string
		wantClass   FailureClass
		wantAction  Action
		wantRetries int
	}{
		{name: "canceled beats context overflow", winner: "context canceled", loser: "context_length_exceeded", wantClass: FailureCanceled, wantAction: ActionFatal},
		{name: "context overflow beats credit", winner: "context_length_exceeded", loser: "insufficient_quota", wantClass: FailureContextOverflow, wantAction: ActionFailoverNext},
		{name: "credit beats rate limit", winner: "credit_balance_exhausted", loser: "rate_limit_exceeded", wantClass: FailureCreditExhausted, wantAction: ActionFailoverNext},
		{name: "rate limit beats timeout", winner: "rate_limit_error", loser: "timed out", wantClass: FailureRateLimited, wantAction: ActionRetrySame, wantRetries: 2},
		{name: "timeout beats unavailable", winner: "timed out", loser: "slow_down", wantClass: FailureTimeout, wantAction: ActionRetrySame, wantRetries: 1},
		{name: "unavailable beats auth", winner: "slow_down", loser: "invalid_api_key", wantClass: FailureUnavailable, wantAction: ActionFailoverNext},
		{name: "auth beats model missing", winner: "invalid_api_key", loser: "model_not_found", wantClass: FailureAuthRejected, wantAction: ActionFailoverNext},
	}
	for _, tc := range cases {
		t.Run(tc.name+"/loser first", func(t *testing.T) {
			err := errors.New("provider error: code=" + tc.loser + " code=" + tc.winner)
			assertErrorVerdict(t, err, tc.wantClass, tc.wantAction, tc.wantRetries)
		})
		t.Run(tc.name+"/winner first", func(t *testing.T) {
			err := errors.New("provider error: code=" + tc.winner + " code=" + tc.loser)
			assertErrorVerdict(t, err, tc.wantClass, tc.wantAction, tc.wantRetries)
		})
	}
}

// TestClassifyErrorRealisticSDKStrings pins two error strings shaped the way
// Go SDKs actually wrap provider failures. A 429 "Too Many Requests" that names
// insufficient_quota is a billing wall (credit outranks rate limit), and a 403 carrying
// model_not_found is an auth rejection (auth outranks model missing). Both fail over.
func TestClassifyErrorRealisticSDKStrings(t *testing.T) {
	cases := []struct {
		name      string
		msg       string
		wantClass FailureClass
	}{
		{
			name:      "429 too many requests with insufficient_quota is credit",
			msg:       "error, status code: 429, status: 429 Too Many Requests, message: You exceeded your current quota, code: insufficient_quota",
			wantClass: FailureCreditExhausted,
		},
		{
			name:      "403 no access to model with model_not_found is auth",
			msg:       "403 Forbidden: Project does not have access to model gpt-4o (code: model_not_found)",
			wantClass: FailureAuthRejected,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assertErrorVerdict(t, errors.New(tc.msg), tc.wantClass, ActionFailoverNext, 0)
		})
	}
}
