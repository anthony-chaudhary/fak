package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestClassifyDecode pins the #4247 coherence policy: the three degenerate
// shapes the issue names are rejected, and benign small-model output — including
// short replies — stays coherent. Pure classifier, no live model needed.
func TestClassifyDecode(t *testing.T) {
	cases := []struct {
		name string
		text string
		want degenerateKind
	}{
		{"empty", "", decodeEmpty},
		{"whitespace only", "  \n\t ", decodeEmpty},
		{"single bang", "!", decodePunctuation},
		{"repeated bang", "!!!!!!!!", decodePunctuation},
		{"mixed punctuation", " ?!?! ... ", decodePunctuation},
		{"repeated word", "the the the the the", decodeRepeatedToken},
		{"repeated rune", "aaaaaaaa", decodeRepeatedToken},
		{"benign ok", "OK", decodeCoherent},
		{"benign hi", "Hi there", decodeCoherent},
		{"benign number", "42", decodeCoherent},
		{"benign hmm", "hmm", decodeCoherent},
		{"benign sentence", "The answer is 4.", decodeCoherent},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := classifyDecode(tc.text); got != tc.want {
				t.Fatalf("classifyDecode(%q) = %q, want %q", tc.text, got, tc.want)
			}
		})
	}
}

// TestHealthzRejectsDegenerateStartupDecode captures the SERVED /healthz response
// (the issue's proof bar): a degenerate boot decode flips ok:false with a reason,
// benign output stays ready, and a never-probed serve is unaffected. This test
// fails before the handleHealth wiring (ok stays true, no degenerate_decode block)
// and passes after.
func TestHealthzRejectsDegenerateStartupDecode(t *testing.T) {
	// benign decode stays ready.
	benign := &Server{}
	benign.SetStartupDecodeProbe("The answer is 4.")
	if body := healthzBody(t, benign); body["ok"] != true {
		t.Fatalf("benign startup decode: /healthz ok = %v, want true", body["ok"])
	}

	// degenerate decode flips ready -> not ready with a 503 status, and the reason
	// captured: a proxy client must not route on this body.
	bad := &Server{}
	bad.SetStartupDecodeProbe("!!!!!!!!")
	code, body := healthzStatus(t, bad)
	if code != http.StatusServiceUnavailable {
		t.Fatalf("degenerate startup decode: /healthz status = %d, want 503", code)
	}
	if body["ok"] != false {
		t.Fatalf("degenerate startup decode: /healthz ok = %v, want false", body["ok"])
	}
	dd, ok := body["degenerate_decode"].(map[string]any)
	if !ok {
		t.Fatalf("degenerate startup decode: missing degenerate_decode block, got %v", body)
	}
	if dd["kind"] != string(decodePunctuation) {
		t.Fatalf("degenerate_decode.kind = %v, want %q", dd["kind"], decodePunctuation)
	}

	// a serve that never ran the probe (proxy/mock) is unaffected.
	unprobed := &Server{}
	if body := healthzBody(t, unprobed); body["ok"] != true {
		t.Fatalf("unprobed serve: /healthz ok = %v, want true", body["ok"])
	}

	// a repeated-single-token decode is rejected under its own kind.
	repeat := &Server{}
	repeat.SetStartupDecodeProbe("the the the the the")
	rcode, rb := healthzStatus(t, repeat)
	if rcode != http.StatusServiceUnavailable {
		t.Fatalf("repeated-token decode: /healthz status = %d, want 503", rcode)
	}
	if rb["ok"] != false {
		t.Fatalf("repeated-token decode: /healthz ok = %v, want false", rb["ok"])
	}
	if dd, _ := rb["degenerate_decode"].(map[string]any); dd["kind"] != string(decodeRepeatedToken) {
		t.Fatalf("repeated-token decode: kind = %v, want %q", dd["kind"], decodeRepeatedToken)
	}
}

// healthzBody serves GET /healthz through the real handler and returns the
// decoded JSON body, requiring the READY status (200) — the served readiness
// response the issue asks the test to capture. healthzStatus is the not-ready
// counterpart that also lets a test pin the HTTP status of an ok:false body.
func healthzBody(t *testing.T, s *Server) map[string]any {
	t.Helper()
	code, body := healthzStatus(t, s)
	if code != http.StatusOK {
		t.Fatalf("/healthz status = %d, want %d (ready body)", code, http.StatusOK)
	}
	return body
}

// healthzStatus serves GET /healthz through the real handler and returns the
// status code and decoded body without asserting readiness.
func healthzStatus(t *testing.T, s *Server) (int, map[string]any) {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec := httptest.NewRecorder()
	s.handleHealth(rec, req)
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode /healthz body %q: %v", rec.Body.String(), err)
	}
	return rec.Code, body
}
