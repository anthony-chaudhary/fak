package gatewayauth

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"testing"
)

const testKey = "gateway-test-key"

func testNonce(fill byte, n int) []byte {
	return bytes.Repeat([]byte{fill}, n)
}

func wantProof(key string, nonce []byte) string {
	mac := hmac.New(sha256.New, []byte(key))
	mac.Write([]byte("fak-health-v1\x00"))
	mac.Write(nonce)
	return base64.StdEncoding.EncodeToString(mac.Sum(nil))
}

func challengeRequest(method, path, challenge string) *http.Request {
	r := httptest.NewRequest(method, path, nil)
	if challenge != "" {
		r.Header.Set("X-Fak-Auth-Challenge", challenge)
	}
	return r
}

type headerOnlyWriter struct {
	header      http.Header
	wroteHeader bool
	wroteBody   bool
}

func (w *headerOnlyWriter) Header() http.Header         { return w.header }
func (w *headerOnlyWriter) WriteHeader(int)             { w.wroteHeader = true }
func (w *headerOnlyWriter) Write(p []byte) (int, error) { w.wroteBody = true; return len(p), nil }

// fak-test:runtime fast est=5ms
func TestWriteHealthProofSetsKeyedHMACForValidChallenge(t *testing.T) {
	nonce := testNonce(0x5a, 32)
	w := &headerOnlyWriter{header: http.Header{}}
	WriteHealthProof(w, challengeRequest(http.MethodGet, "/health", base64.StdEncoding.EncodeToString(nonce)), testKey)

	if got, want := w.header.Get("X-Fak-Auth-Proof"), wantProof(testKey, nonce); got != want {
		t.Fatalf("proof = %q, want %q", got, want)
	}
	if w.wroteHeader || w.wroteBody {
		t.Fatalf("WriteHealthProof wrote status=%v body=%v, want header-only", w.wroteHeader, w.wroteBody)
	}
}

// fak-test:runtime fast est=5ms
func TestWriteHealthProofBindsKeyAndNonce(t *testing.T) {
	nonce := testNonce(0x01, 32)
	proof := func(key string, n []byte) string {
		w := &headerOnlyWriter{header: http.Header{}}
		WriteHealthProof(w, challengeRequest(http.MethodGet, "/health", base64.StdEncoding.EncodeToString(n)), key)
		return w.header.Get("X-Fak-Auth-Proof")
	}
	base := proof(testKey, nonce)
	if base == "" {
		t.Fatal("valid challenge produced no proof")
	}
	if proof("other-key", nonce) == base {
		t.Fatal("proof did not change with the key")
	}
	if proof(testKey, testNonce(0x02, 32)) == base {
		t.Fatal("proof did not change with the nonce")
	}
	undomained := hmac.New(sha256.New, []byte(testKey))
	undomained.Write(nonce)
	if base == base64.StdEncoding.EncodeToString(undomained.Sum(nil)) {
		t.Fatal("proof is not domain-separated by the fak-health-v1 prefix")
	}
}

// fak-test:runtime fast est=5ms
func TestWriteHealthProofLeavesResponseUnchangedOnRejectedInput(t *testing.T) {
	valid := base64.StdEncoding.EncodeToString(testNonce(0x7f, 32))
	cases := map[string]struct {
		key, challenge string
	}{
		"empty key":            {key: "", challenge: valid},
		"missing challenge":    {key: testKey, challenge: ""},
		"invalid base64":       {key: testKey, challenge: "not*base64!"},
		"raw url base64":       {key: testKey, challenge: base64.RawURLEncoding.EncodeToString(testNonce(0xfb, 32))},
		"short nonce 31 bytes": {key: testKey, challenge: base64.StdEncoding.EncodeToString(testNonce(0x7f, 31))},
		"long nonce 33 bytes":  {key: testKey, challenge: base64.StdEncoding.EncodeToString(testNonce(0x7f, 33))},
		"empty decoded nonce":  {key: testKey, challenge: "===="},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			w := &headerOnlyWriter{header: http.Header{"X-Existing": {"kept"}}}
			WriteHealthProof(w, challengeRequest(http.MethodGet, "/health", tc.challenge), tc.key)
			if _, ok := w.header["X-Fak-Auth-Proof"]; ok {
				t.Fatalf("proof header set for rejected input")
			}
			if len(w.header) != 1 || w.header.Get("X-Existing") != "kept" {
				t.Fatalf("headers changed: %v", w.header)
			}
			if w.wroteHeader || w.wroteBody {
				t.Fatalf("wrote status=%v body=%v, want untouched response", w.wroteHeader, w.wroteBody)
			}
		})
	}
}

// fak-test:runtime fast est=5ms
func TestServeKeyProofIssuesProofOnValidGet(t *testing.T) {
	nonce := testNonce(0x33, 32)
	rec := httptest.NewRecorder()
	ServeKeyProof(rec, challengeRequest(http.MethodGet, KeyProofPath, base64.StdEncoding.EncodeToString(nonce)), testKey)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	if got, want := rec.Header().Get("X-Fak-Auth-Proof"), wantProof(testKey, nonce); got != want {
		t.Fatalf("proof = %q, want %q", got, want)
	}
	if got := rec.Header().Get("Cache-Control"); got != "no-store" {
		t.Fatalf("Cache-Control = %q, want no-store", got)
	}
	if rec.Body.Len() != 0 {
		t.Fatalf("body length = %d, want empty", rec.Body.Len())
	}
}

// fak-test:runtime fast est=10ms
func TestServeKeyProofRefusalBranches(t *testing.T) {
	valid := base64.StdEncoding.EncodeToString(testNonce(0x44, 32))
	cases := map[string]struct {
		method, path, challenge, key string
		wantStatus                   int
		wantAllow                    string
	}{
		"wrong path":                 {http.MethodGet, "/v1/fak/key-proof/", valid, testKey, http.StatusNotFound, ""},
		"wrong path beats method":    {http.MethodPost, "/health", valid, testKey, http.StatusNotFound, ""},
		"post":                       {http.MethodPost, KeyProofPath, valid, testKey, http.StatusMethodNotAllowed, http.MethodGet},
		"head":                       {http.MethodHead, KeyProofPath, valid, testKey, http.StatusMethodNotAllowed, http.MethodGet},
		"method beats missing key":   {http.MethodPut, KeyProofPath, valid, "", http.StatusMethodNotAllowed, http.MethodGet},
		"empty key":                  {http.MethodGet, KeyProofPath, valid, "", http.StatusServiceUnavailable, ""},
		"key beats no challenge":     {http.MethodGet, KeyProofPath, "", "", http.StatusServiceUnavailable, ""},
		"missing challenge":          {http.MethodGet, KeyProofPath, "", testKey, http.StatusBadRequest, ""},
		"invalid challenge encoding": {http.MethodGet, KeyProofPath, "%%%", testKey, http.StatusBadRequest, ""},
		"short challenge":            {http.MethodGet, KeyProofPath, base64.StdEncoding.EncodeToString(testNonce(0x44, 16)), testKey, http.StatusBadRequest, ""},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			rec.Header().Set("X-Fak-Auth-Proof", "stale-upstream-proof")
			ServeKeyProof(rec, challengeRequest(tc.method, tc.path, tc.challenge), tc.key)

			if rec.Code != tc.wantStatus {
				t.Fatalf("status = %d, want %d", rec.Code, tc.wantStatus)
			}
			if _, ok := rec.Header()["X-Fak-Auth-Proof"]; ok {
				t.Fatalf("proof header present on refusal: %q", rec.Header().Get("X-Fak-Auth-Proof"))
			}
			if got := rec.Header().Get("Cache-Control"); got != "no-store" {
				t.Fatalf("Cache-Control = %q, want no-store", got)
			}
			if got := rec.Header().Get("Allow"); got != tc.wantAllow {
				t.Fatalf("Allow = %q, want %q", got, tc.wantAllow)
			}
		})
	}
}

// fak-test:runtime fast est=5ms
func TestServeKeyProofReplacesStaleProofHeader(t *testing.T) {
	nonce := testNonce(0x55, 32)
	rec := httptest.NewRecorder()
	rec.Header().Set("X-Fak-Auth-Proof", "stale-upstream-proof")
	ServeKeyProof(rec, challengeRequest(http.MethodGet, KeyProofPath, base64.StdEncoding.EncodeToString(nonce)), testKey)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	if got := rec.Header().Values("X-Fak-Auth-Proof"); len(got) != 1 || got[0] != wantProof(testKey, nonce) {
		t.Fatalf("proof values = %q, want single fresh proof", got)
	}
}
