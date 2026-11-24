package routing

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

// TestFailureVocabularyIsClosed is the registry witness: every failure class and
// every action the package publishes is a member of its own closed vocabulary,
// and an invented token is rejected. It witnesses that a caller can assert on a
// class/action without the set silently growing or drifting.
func TestFailureVocabularyIsClosed(t *testing.T) {
	for _, class := range FailureClasses() {
		if !class.IsKnown() {
			t.Errorf("FailureClasses() contains %q but IsKnown is false", class)
		}
		if act := class.DefaultAction(); act == "" {
			t.Errorf("class %q has an empty DefaultAction", class)
		} else if !act.IsKnown() {
			t.Errorf("class %q DefaultAction %q is outside the closed action vocabulary", class, act)
		}
	}
	for _, act := range Actions() {
		if !act.IsKnown() {
			t.Errorf("Actions() contains %q but IsKnown is false", act)
		}
	}
	// The stream-required rung (D5): named in the closed vocabulary, mapped to
	// FAILOVER_NEXT with a zero same-target budget.
	if !FailureStreamRequired.IsKnown() {
		t.Error("FailureStreamRequired is outside the closed vocabulary")
	}
	if act := FailureStreamRequired.DefaultAction(); act != ActionFailoverNext {
		t.Errorf("FailureStreamRequired.DefaultAction = %q, want %q (another backend may accept buffered)", act, ActionFailoverNext)
	}
	if n := FailureStreamRequired.DefaultRetries(); n != 0 {
		t.Errorf("FailureStreamRequired.DefaultRetries = %d, want 0 (re-sending the same shape cannot succeed)", n)
	}
	// The certificate-failure rung: named in the closed vocabulary, mapped to
	// FAILOVER_NEXT with a zero same-target budget (the peer's chain cannot
	// cure itself; another engine may hold a valid one).
	if !FailureCertInvalid.IsKnown() {
		t.Error("FailureCertInvalid is outside the closed vocabulary")
	}
	if act := FailureCertInvalid.DefaultAction(); act != ActionFailoverNext {
		t.Errorf("FailureCertInvalid.DefaultAction = %q, want %q (another engine may hold a valid chain)", act, ActionFailoverNext)
	}
	if n := FailureCertInvalid.DefaultRetries(); n != 0 {
		t.Errorf("FailureCertInvalid.DefaultRetries = %d, want 0 (the same target cannot cure its own chain)", n)
	}

	if FailureClass("MADE_UP").IsKnown() {
		t.Error("unknown failure class reported as known")
	}
	if Action("TELEPORT").IsKnown() {
		t.Error("unknown action reported as known")
	}
}

// TestClassifyStatusTable is the HTTP classifier witness: each status maps to
// the documented class, action, and retry budget, and an unmapped status falls
// conservatively to UNKNOWN/FAILOVER_NEXT rather than guessing.
func TestClassifyStatusTable(t *testing.T) {
	cases := []struct {
		name        string
		status      int
		body        string
		wantClass   FailureClass
		wantAction  Action
		wantRetries int
	}{
		{name: "429 rate limited retries same", status: 429, wantClass: FailureRateLimited, wantAction: ActionRetrySame, wantRetries: 2},
		{name: "503 unavailable fails over", status: 503, wantClass: FailureUnavailable, wantAction: ActionFailoverNext, wantRetries: 0},
		{name: "401 auth rejected fails over", status: 401, wantClass: FailureAuthRejected, wantAction: ActionFailoverNext},
		{name: "404 model missing fails over", status: 404, wantClass: FailureModelMissing, wantAction: ActionFailoverNext},
		{name: "504 timeout retries same", status: 504, wantClass: FailureTimeout, wantAction: ActionRetrySame, wantRetries: 1},
		{name: "400 bad payload fatal", status: 400, wantClass: FailureBadPayload, wantAction: ActionFatal},
		{name: "unmapped 5xx fails over as unavailable", status: 599, wantClass: FailureUnavailable, wantAction: ActionFailoverNext},
		{name: "200 with no signature is unknown", status: 200, wantClass: FailureUnknown, wantAction: ActionFailoverNext},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Classify(tc.status, tc.body)
			if got.Class != tc.wantClass {
				t.Errorf("Classify(%d, %q).Class = %q, want %q", tc.status, tc.body, got.Class, tc.wantClass)
			}
			if got.Action != tc.wantAction {
				t.Errorf("Classify(%d, %q).Action = %q, want %q", tc.status, tc.body, got.Action, tc.wantAction)
			}
			if got.Retries != tc.wantRetries {
				t.Errorf("Classify(%d, %q).Retries = %d, want %d", tc.status, tc.body, got.Retries, tc.wantRetries)
			}
			if !got.Class.IsKnown() {
				t.Errorf("class %q is outside the closed vocabulary", got.Class)
			}
			if !got.Action.IsKnown() {
				t.Errorf("action %q is outside the closed vocabulary", got.Action)
			}
		})
	}
}

// TestClassifyBodySignaturePrecedence is the LOAD-BEARING witness that body
// signatures outrank the status map: a 400 whose body says the prompt overflowed
// the context window is a CONTEXT_OVERFLOW (which must never retry the same
// engine), NOT a PAYLOAD_REJECTED, and a benign 429 body stays RATE_LIMITED.
func TestClassifyBodySignaturePrecedence(t *testing.T) {
	overflow := Classify(400, "request too large: maximum context length is 8192 tokens")
	if overflow.Class != FailureContextOverflow {
		t.Errorf("Classify(400, context body).Class = %q, want %q (body signature must outrank status)", overflow.Class, FailureContextOverflow)
	}
	if overflow.Action != ActionFailoverNext {
		t.Errorf("context overflow action = %q, want %q (same input can never fit twice)", overflow.Action, ActionFailoverNext)
	}

	rate := Classify(429, "please slow down")
	if rate.Class != FailureRateLimited {
		t.Errorf("Classify(429, benign).Class = %q, want %q", rate.Class, FailureRateLimited)
	}
	if rate.Action != ActionRetrySame {
		t.Errorf("rate limited action = %q, want %q", rate.Action, ActionRetrySame)
	}
}

// TestClassifyStreamRequired witnesses the D5 fix: a stream-only upstream's
// refusal of a buffered request now names STREAM_REQUIRED / FAILOVER_NEXT / 0
// instead of reading PAYLOAD_REJECTED/FATAL on a 400 or UPSTREAM_UNAVAILABLE on
// a 500. The signatures mirror the agent plane's streamRequired matcher
// verbatim (fak internal/agent/upstream_remedy.go) — one taxonomy, not two.
func TestClassifyStreamRequired(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
	}{
		{name: "400 json envelope", status: 400, body: `{"error":"stream must be true"}`},
		{name: "500 bare phrase", status: 500, body: "streaming is required"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Classify(tc.status, tc.body)
			if got.Class != FailureStreamRequired {
				t.Errorf("Classify(%d, %q).Class = %q, want %q", tc.status, tc.body, got.Class, FailureStreamRequired)
			}
			if got.Action != ActionFailoverNext {
				t.Errorf("Classify(%d, %q).Action = %q, want %q", tc.status, tc.body, got.Action, ActionFailoverNext)
			}
			if got.Retries != 0 {
				t.Errorf("Classify(%d, %q).Retries = %d, want 0 (re-sending the same shape cannot succeed)", tc.status, tc.body, got.Retries)
			}
		})
	}

	// Precedence guard: the rate-limit rung sits ahead of the stream-required
	// rung, so a body naming both resolves RATE_LIMITED (earlier rung wins).
	if got := Classify(400, `{"error":"rate limit: stream must be true"}`); got.Class != FailureRateLimited {
		t.Errorf("Classify(400, both phrases).Class = %q, want %q (earlier rung wins)", got.Class, FailureRateLimited)
	}

	// Regression guard: a bare 400 without the signature stays a fatal caller
	// bug; standardStatusClasses is unchanged.
	if got := Classify(400, "invalid json payload"); got.Class != FailureBadPayload || got.Action != ActionFatal {
		t.Errorf("Classify(400, invalid json payload) = %q/%q, want PAYLOAD_REJECTED/FATAL", got.Class, got.Action)
	}
}

// TestClassifyErrorCertInvalid witnesses the certificate rung: a TLS handshake
// verdict names CERT_INVALID / FAILOVER_NEXT / 0 instead of falling to UNKNOWN,
// and a DNS miss ("no such host") keeps reading UPSTREAM_UNAVAILABLE — there
// is no certificate to verify at all.
// fak-test:runtime fast est=5ms lane=default
func TestClassifyErrorCertInvalid(t *testing.T) {
	cases := []struct {
		name string
		msg  string
	}{
		{name: "unknown authority", msg: `Post "https://api.example.com/v1": tls: failed to verify certificate: x509: certificate signed by unknown authority`},
		{name: "expired certificate", msg: "tls: failed to verify certificate: x509: certificate has expired or is not yet valid"},
		{name: "provider snake code", msg: "upstream handshake: cert_has_expired"},
		{name: "hostname mismatch", msg: "x509: certificate is not valid for api.example.com"},
		{name: "generic x509 tail", msg: "tls: handshake failed: x509: certificate unknown reason"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyError(errors.New(tc.msg))
			if got.Class != FailureCertInvalid {
				t.Errorf("ClassifyError(%q).Class = %q, want %q", tc.msg, got.Class, FailureCertInvalid)
			}
			if got.Action != ActionFailoverNext {
				t.Errorf("ClassifyError(%q).Action = %q, want %q (another engine may hold a valid chain)", tc.msg, got.Action, ActionFailoverNext)
			}
			if got.Retries != 0 {
				t.Errorf("ClassifyError(%q).Retries = %d, want 0 (the same target cannot cure its own chain)", tc.msg, got.Retries)
			}
			if !got.Class.IsKnown() {
				t.Errorf("class %q is outside the closed vocabulary", got.Class)
			}
		})
	}

	// Precedence guard: the certificate rung sits ahead of the availability
	// rung, so a handshake error that also mentions "connection reset" reads
	// CERT_INVALID (earlier rung wins).
	if got := ClassifyError(errors.New("connection reset during tls: failed to verify certificate")); got.Class != FailureCertInvalid {
		t.Errorf("ClassifyError(reset+cert).Class = %q, want %q (earlier rung wins)", got.Class, FailureCertInvalid)
	}

	// Regression guard: a DNS miss has no certificate to verify and stays
	// UPSTREAM_UNAVAILABLE.
	if got := ClassifyError(errors.New("dial tcp: lookup api.example.com: no such host")); got.Class != FailureUnavailable {
		t.Errorf("ClassifyError(no such host).Class = %q, want %q", got.Class, FailureUnavailable)
	}
}

// TestClassifyCreditExhausted402 is the load-bearing witness for the observed
// router defect: a 402 Payment Required was falling through to UNKNOWN/status=402
// even though it is a structurally terminal account wall. It must classify
// CREDIT_EXHAUSTED with FAILOVER_NEXT (the same credential can never recover;
// a differently-funded engine might), both by the bare status and by the body
// signatures nebius/OpenAI/Anthropic actually emit.
func TestClassifyCreditExhausted402(t *testing.T) {
	cases := []struct {
		name  string
		body  string
		class FailureClass
	}{
		{
			name:  "bare 402 has no body signature but the status is decisive",
			body:  ``,
			class: FailureCreditExhausted,
		},
		{
			name:  "nebius exhausted-budget body",
			body:  `{"detail":"Payment Required: You have exhausted your budget. Please add funds to continue using the API."}`,
			class: FailureCreditExhausted,
		},
		{
			name:  "openai insufficient_quota body",
			body:  `{"error":{"type":"insufficient_quota","message":"You exceeded your current quota"}}`,
			class: FailureCreditExhausted,
		},
		{
			name:  "anthropic credit balance body",
			body:  `{"error":{"message":"Your credit balance is too low"}}`,
			class: FailureCreditExhausted,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Classify(402, tc.body)
			if got.Class != tc.class {
				t.Errorf("Classify(402, %q).Class = %q, want %q", tc.body, got.Class, tc.class)
			}
			if got.Action != ActionFailoverNext {
				t.Errorf("Classify(402, %q).Action = %q, want %q (a walled account cannot be retried; a funded engine can)", tc.body, got.Action, ActionFailoverNext)
			}
			if got.Retries != 0 {
				t.Errorf("Classify(402, %q).Retries = %d, want 0 (a payment wall is not transient)", tc.body, got.Retries)
			}
			if !got.Class.IsKnown() {
				t.Errorf("class %q is outside the closed vocabulary", got.Class)
			}
		})
	}

	// A payment wall must outrank the generic 4xx status map order: a 403 whose
	// body names a billing wall is CREDIT_EXHAUSTED, not AUTH_REJECTED.
	if got := Classify(403, `{"error":{"message":"insufficient_quota"}}`); got.Class != FailureCreditExhausted {
		t.Errorf("Classify(403, billing body).Class = %q, want %q (body signature must outrank the status map)", got.Class, FailureCreditExhausted)
	}
}

// TestClassifyErrorCreditExhausted witnesses the error-string arm of the same
// wall: a transport-wrapped "insufficient_quota" or "Payment Required" must
// name the class rather than fall through to UNKNOWN.
func TestClassifyErrorCreditExhausted(t *testing.T) {
	for _, msg := range []string{
		"402 Payment Required: insufficient_quota",
		"provider error: Your credit balance is too low",
		"upstream: exhausted your budget",
	} {
		got := ClassifyError(errors.New(msg))
		if got.Class != FailureCreditExhausted {
			t.Errorf("ClassifyError(%q).Class = %q, want %q", msg, got.Class, FailureCreditExhausted)
		}
		if got.Action != ActionFailoverNext {
			t.Errorf("ClassifyError(%q).Action = %q, want %q", msg, got.Action, ActionFailoverNext)
		}
	}
}

// TestClassifyLlamaContextOverflow400 pins llama.cpp's exact HTTP 400 error
// envelope. Either its stable type or its human-readable message is sufficient
// evidence of context overflow; an unrelated 400 remains a fatal caller error.
func TestClassifyLlamaContextOverflow400(t *testing.T) {
	const overflowType = `"type":"exceed_context_size_error"`
	const overflowMessage = `"message":"request (271334 tokens) exceeds the available context size (32768 tokens), try increasing it"`
	cases := []struct {
		name       string
		body       string
		wantClass  FailureClass
		wantAction Action
	}{
		{
			name:       "exact provider envelope",
			body:       `{"code":400,` + overflowMessage + `,` + overflowType + `}`,
			wantClass:  FailureContextOverflow,
			wantAction: ActionFailoverNext,
		},
		{
			name:       "type only",
			body:       `{` + overflowType + `}`,
			wantClass:  FailureContextOverflow,
			wantAction: ActionFailoverNext,
		},
		{
			name:       "message only",
			body:       `{` + overflowMessage + `}`,
			wantClass:  FailureContextOverflow,
			wantAction: ActionFailoverNext,
		},
		{
			name:       "unrelated bare 400",
			body:       `{"error":"invalid request"}`,
			wantClass:  FailureBadPayload,
			wantAction: ActionFatal,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Classify(400, tc.body)
			if got.Class != tc.wantClass || got.Action != tc.wantAction {
				t.Fatalf("Classify(400, %q) = class %q action %q, want class %q action %q",
					tc.body, got.Class, got.Action, tc.wantClass, tc.wantAction)
			}
		})
	}
}

// TestClassifyErrorTable is the error-value classifier witness: nil fails closed
// FATAL (there is nothing to classify), each documented signature maps to its
// class, and an unrecognised error fails over conservatively.
func TestClassifyErrorTable(t *testing.T) {
	cases := []struct {
		name       string
		err        error
		wantClass  FailureClass
		wantAction Action
	}{
		{name: "nil error fails closed fatal", err: nil, wantClass: FailureUnknown, wantAction: ActionFatal},
		{name: "context canceled", err: errors.New("context canceled"), wantClass: FailureCanceled, wantAction: ActionFatal},
		{name: "connection refused", err: errors.New("connection refused"), wantClass: FailureUnavailable, wantAction: ActionFailoverNext},
		{name: "windows refused dial", err: errors.New(`dial tcp 127.0.0.1:1: connectex: No connection could be made because the target machine actively refused it.`), wantClass: FailureUnavailable, wantAction: ActionFailoverNext},
		{name: "windows connection reset", err: errors.New(`read tcp 127.0.0.1:5000->127.0.0.1:8080: wsarecv: An existing connection was forcibly closed by the remote host.`), wantClass: FailureUnavailable, wantAction: ActionFailoverNext},
		{name: "context deadline exceeded", err: errors.New("context deadline exceeded"), wantClass: FailureTimeout, wantAction: ActionRetrySame},
		{name: "rate limit exceeded", err: errors.New("rate limit exceeded"), wantClass: FailureRateLimited, wantAction: ActionRetrySame},
		{name: "context length", err: errors.New("maximum context length is 4096 tokens"), wantClass: FailureContextOverflow, wantAction: ActionFailoverNext},
		{name: "unrecognised error fails over", err: errors.New("something inexplicable"), wantClass: FailureUnknown, wantAction: ActionFailoverNext},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyError(tc.err)
			if got.Class != tc.wantClass {
				t.Errorf("ClassifyError(%v).Class = %q, want %q", tc.err, got.Class, tc.wantClass)
			}
			if got.Action != tc.wantAction {
				t.Errorf("ClassifyError(%v).Action = %q, want %q", tc.err, got.Action, tc.wantAction)
			}
		})
	}
}

// TestClassifyIsDeterministic proves both classifiers are pure: the same input
// yields the identical Failure across repeated calls, so an audit can replay a
// decision.
func TestClassifyIsDeterministic(t *testing.T) {
	first := Classify(400, "maximum context length exceeded")
	for i := 0; i < 10; i++ {
		if got := Classify(400, "maximum context length exceeded"); !reflect.DeepEqual(got, first) {
			t.Fatalf("Classify iteration %d differs:\n got %+v\nwant %+v", i, got, first)
		}
	}

	firstErr := ClassifyError(errors.New("connection refused"))
	for i := 0; i < 10; i++ {
		if got := ClassifyError(errors.New("connection refused")); !reflect.DeepEqual(got, firstErr) {
			t.Fatalf("ClassifyError iteration %d differs:\n got %+v\nwant %+v", i, got, firstErr)
		}
	}
}

// TestClassifySignaturesAreCaseInsensitive witnesses that both classifiers
// lowercase their input before matching, so "RATE LIMIT" and "Deadline
// Exceeded" classify identically to their lowercase forms. Without this, a
// provider that shouts its error text would fall through to UNKNOWN.
func TestClassifySignaturesAreCaseInsensitive(t *testing.T) {
	cases := []struct {
		name      string
		body      string
		errText   string
		wantClass FailureClass
	}{
		{name: "rate limit", body: "RATE LIMIT EXCEEDED", errText: "RATE LIMIT EXCEEDED", wantClass: FailureRateLimited},
		{name: "context length", body: "Maximum Context Length is 4096", errText: "Maximum Context Length is 4096", wantClass: FailureContextOverflow},
		{name: "deadline exceeded", body: "Deadline Exceeded", errText: "Deadline Exceeded", wantClass: FailureTimeout},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Classify(500, tc.body); got.Class != tc.wantClass {
				t.Errorf("Classify(500, %q).Class = %q, want %q (body must be lowercased before matching)", tc.body, got.Class, tc.wantClass)
			}
			if got := ClassifyError(errors.New(tc.errText)); got.Class != tc.wantClass {
				t.Errorf("ClassifyError(%q).Class = %q, want %q (error must be lowercased before matching)", tc.errText, got.Class, tc.wantClass)
			}
		})
	}
}

// TestClassifyErrorDetailIsTruncated witnesses the audit budget: an
// unrecognised error's Detail is clipped to the 120-char cap, so a long,
// possibly sensitive error message cannot flood the audit trail. The matched-
// signature paths carry short tokens and are not the truncation surface.
func TestClassifyErrorDetailIsTruncated(t *testing.T) {
	long := strings.Repeat("x", 500)
	got := ClassifyError(errors.New(long))
	if got.Class != FailureUnknown {
		t.Fatalf("ClassifyError(long unrecognised).Class = %q, want %q", got.Class, FailureUnknown)
	}
	if len(got.Detail) != failureDetailLimit {
		t.Errorf("Detail length = %d, want %d (capped)", len(got.Detail), failureDetailLimit)
	}
	if got.Detail != long[:failureDetailLimit] {
		t.Errorf("Detail = %q, want the first %d characters of the input", got.Detail, failureDetailLimit)
	}

	// At exactly the limit the detail is preserved, not clipped.
	exact := strings.Repeat("y", failureDetailLimit)
	if got := ClassifyError(errors.New(exact)); len(got.Detail) != failureDetailLimit || got.Detail != exact {
		t.Errorf("Detail at the boundary = %d chars %q, want the %d-char input preserved", len(got.Detail), got.Detail, failureDetailLimit)
	}
}

func TestClassifyStandardStatusClasses(t *testing.T) {
	for status, want := range standardStatusClasses {
		got := Classify(status, "")
		if got.Class != want {
			t.Errorf("Classify(%d, empty).Class = %q, standardStatusClasses says %q", status, got.Class, want)
		}
		if !got.Class.IsKnown() {
			t.Errorf("Classify(%d).Class %q is outside the closed vocabulary", status, got.Class)
		}
		if got.Action != want.DefaultAction() {
			t.Errorf("Classify(%d).Action = %q, want the class default %q", status, got.Action, want.DefaultAction())
		}
	}
}

func TestClassify409IsNamedAndFailsOver(t *testing.T) {
	got := Classify(409, "")
	if got.Class != FailureConflict {
		t.Fatalf("Classify(409, empty).Class = %q, want %q", got.Class, FailureConflict)
	}
	if got.Action != ActionFailoverNext {
		t.Errorf("Classify(409).Action = %q, want %q", got.Action, ActionFailoverNext)
	}
	if got := Classify(409, "conflict: version mismatch"); got.Class != FailureConflict {
		t.Errorf("Classify(409, body).Class = %q, want %q", got.Class, FailureConflict)
	}
}

func TestClassifyStatusTotalAndCatchAll(t *testing.T) {
	for status := 100; status <= 599; status++ {
		got := Classify(status, "")
		if !got.Class.IsKnown() {
			t.Errorf("Classify(%d).Class = %q, not in the closed vocabulary", status, got.Class)
		}
		if !got.Action.IsKnown() {
			t.Errorf("Classify(%d).Action = %q, not in the closed vocabulary", status, got.Action)
		}
		if status >= 400 && got.Class == FailureUnknown {
			t.Errorf("Classify(%d).Class = UNKNOWN; a standard failure range status must be named", status)
		}
	}
	cases := []struct {
		name   string
		status int
		want   FailureClass
		action Action
	}{
		{name: "unassigned 4xx is a fatal caller class", status: 499, want: FailureBadPayload, action: ActionFatal},
		{name: "unassigned 5xx is a failover class", status: 599, want: FailureUnavailable, action: ActionFailoverNext},
		{name: "3xx is not a named failure", status: 302, want: FailureUnknown, action: ActionFailoverNext},
		{name: "2xx is not a named failure", status: 200, want: FailureUnknown, action: ActionFailoverNext},
		{name: "no status observed is not a named failure", status: 0, want: FailureUnknown, action: ActionFailoverNext},
		{name: "negative status is not a named failure", status: -1, want: FailureUnknown, action: ActionFailoverNext},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Classify(tc.status, "")
			if got.Class != tc.want {
				t.Errorf("Classify(%d).Class = %q, want %q", tc.status, got.Class, tc.want)
			}
			if got.Action != tc.action {
				t.Errorf("Classify(%d).Action = %q, want %q", tc.status, got.Action, tc.action)
			}
		})
	}
}

// The tests below witness the SPACED usage-limit wall: nebius serves a budget wall
// whose HTTP body is the spaced string "usage limit reached", while the credit
// signature list carried only the snake_case provider code "usage_limit_reached"
// (the in-band 200-stream code). The two are different byte sequences, so
// matching one never implies the other.
//
// The gap this file closes is NARROWER than "a 402 was misclassified", and the
// tests are written to keep that honest:
//
//   - A bare 402 was ALREADY CREDIT_EXHAUSTED by the status map (failure.go
//     402: FailureCreditExhausted), so the 402 path never depended on the body.
//     The rows below pin that as a contrast, not as the defect.
//   - The real gap is a status that does NOT itself mean a budget wall. A 429
//     carrying the spaced wall body classified RATE_LIMITED / RETRY_SAME / 2
//     retries — a walled account recorded as a transient blip, the exact
//     misreading the file's class->action contract forbids. 403 read
//     AUTH_REJECTED and 413 read CONTEXT_OVERFLOW for the same body.
//
// TestCreditExhaustedSpacedUsageLimitBodyBeatsRateLimitStatus is the reproducing
// test: it fails on the base commit (429/403/413 rows) and passes once the
// spaced signature is added. Removing the "usage limit reached" entry reproduces
// the reported symptom exactly: RATE_LIMITED / RETRY_SAME / 2 retries /
// Transient()=true for a walled account.

// nebiusUsageLimitBody is the witnessed nebius body: the wall named in spaced
// English, with and without the JSON envelope a real gateway relays.
const nebiusUsageLimitBody = "usage limit reached"

func TestCreditExhaustedSpacedUsageLimitBodyBeatsRateLimitStatus(t *testing.T) {
	bodies := []struct {
		name string
		body string
	}{
		{name: "bare spaced body", body: nebiusUsageLimitBody},
		{name: "flat JSON envelope", body: `{"error":"usage limit reached"}`},
		{name: "nested message envelope", body: `{"error":{"message":"usage limit reached"}}`},
		{name: "detail envelope", body: `{"detail":"usage limit reached"}`},
		{name: "sentence form", body: "Payment quota: usage limit reached for this organization"},
	}

	// The statuses whose OWN meaning is not a budget wall. For these the body is
	// the only evidence, so before the fix each one read as its generic class
	// and the wall was misfiled.
	for _, status := range []int{429, 403, 413} {
		for _, b := range bodies {
			t.Run(b.name+"/status "+itoa(status), func(t *testing.T) {
				got := Classify(status, b.body)
				if got.Class != FailureCreditExhausted {
					t.Errorf("Classify(%d, %q).Class = %q, want %q (the spaced wall body is the only evidence at this status, and a walled account is not retryable)",
						status, b.body, got.Class, FailureCreditExhausted)
				}
				if got.Action != ActionFailoverNext {
					t.Errorf("Classify(%d, %q).Action = %q, want %q (funding the same engine is the only cure)",
						status, b.body, got.Action, ActionFailoverNext)
				}
				if got.Retries != 0 {
					t.Errorf("Classify(%d, %q).Retries = %d, want 0 (a budget wall does not clear on its own)",
						status, b.body, got.Retries)
				}
				if got.Class.Transient() {
					t.Errorf("Classify(%d, %q).Class.Transient() = true, want false", status, b.body)
				}
			})
		}
	}
}

// TestCreditExhaustedSpacedBodySpendsFullCredentialHeadroom pins the budget
// consequence, not just the label. The defect's harm was never the string
// "RATE_LIMITED"; it was that RATE_LIMITED caps a keyring-backed walk at 2
// further attempts (RATE_LIMITED is org-wide, so every same-backend key roll is
// futile), which is how a dead source kept being dialled. A wall must instead
// authorize the full credential headroom, so the walk rolls to a DIFFERENT key
// rather than exhausting the futile same-backend budget.
func TestCreditExhaustedSpacedBodySpendsFullCredentialHeadroom(t *testing.T) {
	const headroom = 9

	bare := Classify(429, "")
	if got := bare.RetryBudget(headroom); got != 2 {
		t.Fatalf("Classify(429, \"\").RetryBudget(%d) = %d, want 2 (bare 429 contrast must stay RATE_LIMITED)", headroom, got)
	}

	body := `{"error":"usage limit reached"}`
	got := Classify(429, body)
	if got.Class != FailureCreditExhausted {
		t.Fatalf("Classify(429, %q).Class = %q, want %q", body, got.Class, FailureCreditExhausted)
	}
	if budget := got.RetryBudget(headroom); budget != headroom {
		t.Errorf("Classify(429, %q).RetryBudget(%d) = %d, want the full headroom %d (a walled key's remedy is the next key, not 2 futile retries of this one)",
			body, headroom, budget, headroom)
	}
}

// TestCreditExhausted402AlreadyClassifiedByStatusNotBody is the honest
// CONTRAST row, kept deliberately so this file cannot be misread as claiming the
// 402 path was broken. A 402 has always been CREDIT_EXHAUSTED by the status map
// alone (failure.go maps 402 to FailureCreditExhausted), body or no body. The
// spaced signature therefore does not change the 402 verdict; it makes the
// verdict INDEPENDENT of the status code, which is what closes the 429/403/413
// gap. If someone later removes the spaced signature, this row still passes
// (the 402 map covers it) while the tests above fail: the two together are what
// distinguish "credit because the status says so" from "credit because the body
// says so".
func TestCreditExhausted402AlreadyClassifiedByStatusNotBody(t *testing.T) {
	for _, body := range []string{"", nebiusUsageLimitBody, `{"error":"usage limit reached"}`} {
		got := Classify(402, body)
		if got.Class != FailureCreditExhausted {
			t.Errorf("Classify(402, %q).Class = %q, want %q (402 is the class by status, body or no body)", body, got.Class, FailureCreditExhausted)
		}
	}
}

// TestCreditExhaustedSignatureDoesNotShadowRetryableRateLimits is the
// precedence GUARD. It is the exact hazard the credit list's own comment warns
// about, restated as an executable row, and it must fail if this signature is
// ever widened to a bare "usage limit" or "billing".
//
// The hazard is structural, not incidental: CREDIT_EXHAUSTED is checked BEFORE
// rateLimitedSignatures, so any credit signature that also occurs inside a
// genuine retryable rate-limit body silently converts a transient limit into a
// terminal wall. Three witnessed shapes must therefore survive the new
// signature unchanged:
//
//   - OpenAI's TPM body: "Rate limit reached ... adding a payment method ...
//     /account/billing." It says "rate limit", never "usage limit", so it is
//     untouched by the new signature (and remains protected from "billing" by
//     the pre-existing no-bare-billing rule).
//   - A body naming BOTH rungs, which is the real test of precedence: the
//     snake_case in-band code row already covers "credit beats rate limit"; this
//     adds the SPACED pair, so "usage limit reached" alongside "rate limit" must
//     resolve to the wall.
//   - A rate-limit body that merely mentions a usage quota without the wall
//     wording (e.g. a 429 that says "usage limits are per-minute here"): it
//     must stay RATE_LIMITED, which is what a NARROW signature buys over a bare
//     "usage limit".
func TestCreditExhaustedSignatureDoesNotShadowRetryableRateLimits(t *testing.T) {
	cases := []struct {
		name        string
		status      int
		body        string
		wantClass   FailureClass
		wantAction  Action
		wantRetries int
	}{
		{
			name:   "openai TPM rate limit mentioning /account/billing stays retryable",
			status: 429,
			body: `{"error":{"message":"Rate limit reached for gpt-4 in organization org-x on tokens per min. ` +
				`Limit: 10000, Used: 9000, Requested: 2000. You can increase your rate limit by adding a ` +
				`payment method to your account at https://platform.openai.com/account/billing.",` +
				`"type":"tokens","param":null,"code":"rate_limit_exceeded"}}`,
			wantClass: FailureRateLimited, wantAction: ActionRetrySame, wantRetries: 2,
		},
		{
			name:       "rate limit body that mentions a usage quota without the wall wording stays retryable",
			status:     429,
			body:       "rate limit exceeded: usage limits are enforced per minute on this endpoint",
			wantClass:  FailureRateLimited,
			wantAction: ActionRetrySame, wantRetries: 2,
		},
		{
			name:       "bare 429 with no body stays retryable",
			status:     429,
			body:       "",
			wantClass:  FailureRateLimited,
			wantAction: ActionRetrySame, wantRetries: 2,
		},
		{
			name:       "too many requests stays retryable",
			status:     429,
			body:       "too many requests",
			wantClass:  FailureRateLimited,
			wantAction: ActionRetrySame, wantRetries: 2,
		},
		{
			name:       "the spaced wall beats a co-occurring rate limit wording (credit outranks rate limit)",
			status:     429,
			body:       `{"error":"usage limit reached","hint":"rate limit resets are not applicable"}`,
			wantClass:  FailureCreditExhausted,
			wantAction: ActionFailoverNext, wantRetries: 0,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Classify(tc.status, tc.body)
			if got.Class != tc.wantClass {
				t.Errorf("Classify(%d, %q).Class = %q, want %q", tc.status, tc.body, got.Class, tc.wantClass)
			}
			if got.Action != tc.wantAction {
				t.Errorf("Classify(%d, %q).Action = %q, want %q", tc.status, tc.body, got.Action, tc.wantAction)
			}
			if got.Retries != tc.wantRetries {
				t.Errorf("Classify(%d, %q).Retries = %d, want %d", tc.status, tc.body, got.Retries, tc.wantRetries)
			}
		})
	}
}

// TestCreditExhaustedErrorArmMatchesHTTPBodyForm covers the
// error-string arm (ClassifyError shares the same credit signature list), so the
// two classifiers cannot drift on this body the way they could if only the HTTP
// rung had learned it. A transport-wrapped wall must still name the class
// rather than fall through to UNKNOWN.
func TestCreditExhaustedErrorArmMatchesHTTPBodyForm(t *testing.T) {
	for _, msg := range []string{
		"provider error: usage limit reached",
		`upstream 429 body: {"error":"usage limit reached"}`,
	} {
		got := ClassifyError(errString(msg))
		if got.Class != FailureCreditExhausted {
			t.Errorf("ClassifyError(%q).Class = %q, want %q", msg, got.Class, FailureCreditExhausted)
		}
		if got.Action != ActionFailoverNext {
			t.Errorf("ClassifyError(%q).Action = %q, want %q", msg, got.Action, ActionFailoverNext)
		}
	}
}

// errString avoids importing errors just for two literals; ClassifyError takes
// any error and only reads Error().
type errString string

func (e errString) Error() string { return string(e) }

// TestCreditExhaustedUsageLimitReachedFormsClassifyIdentically is the ACCEPTANCE table: every
// spelling of the usage-limit wall — the snake_case in-band code, the spaced HTTP
// phrase, and the case/whitespace variants of the phrase — must land on ONE
// verdict with ONE action and ONE budget. It states the fix's intent as an
// equivalence: the verdict is a property of the CONDITION, not of how the
// provider happened to format it.
//
// It pins the normalisation boundary from both sides. The right-hand rows are
// covered by normalizeSignatureText (case + whitespace collapse). The LEFT-hand
// row is the boundary normalisation must NOT cross: collapsing whitespace never
// removes the underscore, so the snake code stays a SEPARATE table entry. If
// someone ever "simplifies" the table by deleting the snake entry on the theory
// that the phrase subsumes it, the code rows below stop matching.
func TestCreditExhaustedUsageLimitReachedFormsClassifyIdentically(t *testing.T) {
	// forms that must all classify as the terminal credit wall.
	wallForms := []struct {
		name string
		body string
	}{
		{"snake in-band code", `{"error":{"code":"usage_limit_reached"}}`},
		{"spaced phrase", "usage limit reached"},
		{"title case", "Usage Limit Reached"},
		{"upper case", "USAGE LIMIT REACHED"},
		{"doubled inner space", "usage  limit  reached"},
		{"line wrapped", "usage\nlimit reached"},
		{"tab separated", "usage\tlimit reached"},
		{"crlf wrapped", "usage\r\nlimit reached"},
		{"spaced inside JSON", `{"error":{"message":"Usage Limit Reached"}}`},
	}

	// The witness status is 429: a status that does not itself mean a budget wall,
	// so the body is the ONLY evidence and a miss is visible as RATE_LIMITED.
	const status = 429

	for _, form := range wallForms {
		t.Run(form.name, func(t *testing.T) {
			got := Classify(status, form.body)
			if got.Class != FailureCreditExhausted {
				t.Fatalf("Classify(%d, %q).Class = %q, want %q", status, form.body, got.Class, FailureCreditExhausted)
			}
			if got.Action != ActionFailoverNext {
				t.Errorf("Classify(%d, %q).Action = %q, want %q", status, form.body, got.Action, ActionFailoverNext)
			}
			if got.Retries != 0 {
				t.Errorf("Classify(%d, %q).Retries = %d, want 0", status, form.body, got.Retries)
			}
			if got.Class.Transient() {
				t.Errorf("Classify(%d, %q).Class.Transient() = true, want false", status, form.body)
			}
		})
	}

	// The equivalence itself: every form yields the identical Failure, so no
	// downstream consumer (failover, keyring feedback, the audit trail) can
	// observe a difference between the spellings.
	want := Classify(status, "usage limit reached")
	for _, form := range wallForms {
		if got := Classify(status, form.body); got.Class != want.Class || got.Action != want.Action || got.Retries != want.Retries {
			t.Errorf("form %q classified %+v, want the identical verdict %+v", form.name, got, want)
		}
	}

	// The same equivalence across the error-string arm, so ClassifyError cannot
	// drift from Classify on formatting.
	for _, form := range wallForms {
		if got := ClassifyError(errString(form.body)); got.Class != FailureCreditExhausted {
			t.Errorf("ClassifyError(%q).Class = %q, want %q", form.body, got.Class, FailureCreditExhausted)
		}
	}

	// Boundary: the snake code is matched by its OWN entry. Nothing else in the
	// credit or rate-limit tables covers it, so if the entry were deleted this
	// row would fall through to the 429 status default.
	if got := Classify(status, "usage_limit_reached"); got.Class != FailureCreditExhausted {
		t.Errorf("Classify(%d, %q).Class = %q, want %q — the snake code needs its own signature entry", status, "usage_limit_reached", got.Class, FailureCreditExhausted)
	}
}

// TestCreditExhaustedRetiresWalledEngineFromCandidateChain is the DOWNSTREAM
// consequence witness, and it is deliberately scoped to what this package can
// actually decide. Failover is the pure decision "where does the request go
// next"; given the spaced wall body on a 429 it must EXCLUDE the walled engine
// and assign the fallback, exactly as it does for a 402. Before the fix the
// class was RATE_LIMITED, whose action is RETRY_SAME, so Failover keeps the
// failed engine at the FRONT of the reordered set (failover.go step 3): the walk
// would re-pick the walled source first and only reach the funded engine after
// the rate-limit budget was spent. That is the mechanism by which a dead key
// source kept being dialled.
//
// SCOPE HONESTY, because it bounds what this test proves. This witnesses the
// routing FABRIC's contribution only: the engine is excluded and the request
// moves. It does NOT witness that a key is tombstoned in the keyring, because
// that decision is owned by platform/keyring eligibility and is fed from the
// observed HTTP STATUS (KeyringRotation.recordOutcome), not from this class.
// The key-level consequence is witnessed where it lives, in
// platform/gateway TestKeyringRotationPaymentRequiredFeedback and
// platform/keyring TestPaymentRequiredIsTombstoned. The two halves meet in
// routing_proxy.go, which calls lease.Observe(resp.StatusCode) and
// routing.Classify(resp.StatusCode, body) from the same response.
func TestCreditExhaustedRetiresWalledEngineFromCandidateChain(t *testing.T) {
	req := TaskRequest{Candidate: candidate("credit-wall", "docs"), Role: "worker"}
	engines := failoverEngines()
	body := `{"error":"usage limit reached"}`

	// Contrast: the bare 429 keeps the failed engine at the front (RETRY_SAME).
	// Pinned so this test cannot pass for the trivial reason that Failover
	// always excludes.
	retrySame, _, _, err := Failover(req, "primary", Classify(429, ""), engines, DefaultDegradationChain(), DefaultPolicy())
	if err != nil {
		t.Fatalf("Failover(bare 429) error = %v", err)
	}
	if retrySame.EngineID != "primary" {
		t.Errorf("bare 429 assigned %q, want the failed engine kept at the front (RETRY_SAME keeps it)", retrySame.EngineID)
	}

	failure := Classify(429, body)
	if failure.Class != FailureCreditExhausted {
		t.Fatalf("precondition: Classify(429, %q) = %q, want %q", body, failure.Class, FailureCreditExhausted)
	}
	a, _, _, err := Failover(req, "primary", failure, engines, DefaultDegradationChain(), DefaultPolicy())
	if err != nil {
		t.Fatalf("Failover(credit wall) error = %v", err)
	}
	if a.EngineID == "primary" {
		t.Error("Failover re-assigned the walled primary; a CREDIT_EXHAUSTED engine must be excluded (FAILOVER_NEXT), not retried first")
	}
	if a.EngineID != "fallback" {
		t.Errorf("assigned engine = %q, want %q", a.EngineID, "fallback")
	}
	if !auditContains(a.AssignmentAudit, "CREDIT_EXHAUSTED", "FAILOVER_NEXT") {
		t.Errorf("audit does not name CREDIT_EXHAUSTED/FAILOVER_NEXT: %v", a.AssignmentAudit)
	}
}
