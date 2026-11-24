package routing

import "testing"

const witnessedOpencodeGoModelUnavailable = `{"error":{"type":"server_error","message":"Upstream request failed: Model is unavailable."}}`

// fak-test:runtime fast est=5ms lane=default
func TestClassifyModelUnavailableWordingsFailOver(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
	}{
		{name: "witnessed opencode-go 400", status: 400, body: witnessedOpencodeGoModelUnavailable},
		{name: "witnessed body in-band on 200", status: 200, body: witnessedOpencodeGoModelUnavailable},
		{name: "invalid model name", status: 400, body: "Invalid Model Name"},
		{name: "model unavailable", status: 400, body: `{"error":{"message":"model unavailable"}}`},
		{name: "model is not available", status: 400, body: `{"error":{"message":"The model is not available"}}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assertInBand(t, tc.status, tc.body, FailureModelMissing, ActionFailoverNext, 0)
		})
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestClassifyModelUnavailableDoesNotWidenPayloadRejected(t *testing.T) {
	for _, body := range []string{"", "missing required field messages"} {
		t.Run(body, func(t *testing.T) {
			assertInBand(t, 400, body, FailureBadPayload, ActionFatal, 0)
		})
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestClassifyModelUnavailableKeepsPrecedence(t *testing.T) {
	cases := []struct {
		name      string
		body      string
		wantClass FailureClass
	}{
		{name: "context overflow outranks", body: "model is unavailable: context length exceeded", wantClass: FailureContextOverflow},
		{name: "auth outranks", body: "invalid api key; model is unavailable", wantClass: FailureAuthRejected},
		{name: "temporarily unavailable stays availability", body: "model is temporarily unavailable", wantClass: FailureUnavailable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Classify(400, tc.body); got.Class != tc.wantClass {
				t.Fatalf("Classify(400, %q).Class = %q, want %q", tc.body, got.Class, tc.wantClass)
			}
		})
	}
}
