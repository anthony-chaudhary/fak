package gateway

// account_rehome_test.go — the POST /v1/fak/account/rehome contract: inert 404 with no
// swap function installed, POST-only, optional body, the wired function's result served
// verbatim on 200, and its refusal surfaced as the 409 account_rehome_unavailable code.

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAccountRehomeInertWithoutHostWiring(t *testing.T) {
	rr := httptest.NewRecorder()
	(&Server{}).handleFakAccountRehome(rr, httptest.NewRequest(http.MethodPost, "/v1/fak/account/rehome", nil))
	if rr.Code != http.StatusNotFound {
		t.Fatalf("unwired rehome answered %d, want 404 (inert)", rr.Code)
	}
}

func TestAccountRehomePostOnly(t *testing.T) {
	s := &Server{}
	s.SetAccountRehomeFunc(func(string) (AccountRehome, error) { return AccountRehome{}, nil })
	rr := httptest.NewRecorder()
	s.handleFakAccountRehome(rr, httptest.NewRequest(http.MethodGet, "/v1/fak/account/rehome", nil))
	if rr.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET answered %d, want 405", rr.Code)
	}
}

func TestAccountRehomeServesSwapResult(t *testing.T) {
	s := &Server{}
	var gotReason string
	s.SetAccountRehomeFunc(func(reason string) (AccountRehome, error) {
		gotReason = reason
		return AccountRehome{From: "day26", FromEmail: "a@x.test", To: "july4", ToEmail: "b@x.test", Reason: reason}, nil
	})

	rr := httptest.NewRecorder()
	body := strings.NewReader(`{"reason":"operator_rehome"}`)
	s.handleFakAccountRehome(rr, httptest.NewRequest(http.MethodPost, "/v1/fak/account/rehome", body))
	if rr.Code != http.StatusOK {
		t.Fatalf("wired rehome answered %d, want 200; body=%s", rr.Code, rr.Body.String())
	}
	if gotReason != "operator_rehome" {
		t.Fatalf("swap function saw reason %q, want the posted operator_rehome", gotReason)
	}
	var res AccountRehome
	if err := json.Unmarshal(rr.Body.Bytes(), &res); err != nil {
		t.Fatalf("response not JSON: %v", err)
	}
	if res.From != "day26" || res.To != "july4" {
		t.Fatalf("response = %+v, want the swap function's from/to verbatim", res)
	}
}

func TestAccountRehomeEmptyBodyIsFine(t *testing.T) {
	s := &Server{}
	called := false
	s.SetAccountRehomeFunc(func(reason string) (AccountRehome, error) {
		called = true
		if reason != "" {
			t.Errorf("empty body produced reason %q, want \"\" (host defaults it)", reason)
		}
		return AccountRehome{From: "a", To: "b"}, nil
	})
	rr := httptest.NewRecorder()
	s.handleFakAccountRehome(rr, httptest.NewRequest(http.MethodPost, "/v1/fak/account/rehome", nil))
	if rr.Code != http.StatusOK || !called {
		t.Fatalf("empty-body POST answered %d (called=%v), want 200 with the swap invoked", rr.Code, called)
	}
}

func TestAccountRehomeMalformedBodyIs400(t *testing.T) {
	s := &Server{}
	s.SetAccountRehomeFunc(func(string) (AccountRehome, error) {
		t.Error("swap function must not run on a malformed body")
		return AccountRehome{}, nil
	})
	rr := httptest.NewRecorder()
	s.handleFakAccountRehome(rr, httptest.NewRequest(http.MethodPost, "/v1/fak/account/rehome", strings.NewReader("{not json")))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("malformed body answered %d, want 400", rr.Code)
	}
}

func TestAccountRehomeRefusalIs409(t *testing.T) {
	s := &Server{}
	s.SetAccountRehomeFunc(func(string) (AccountRehome, error) {
		return AccountRehome{}, errors.New("no available sibling seat")
	})
	rr := httptest.NewRecorder()
	s.handleFakAccountRehome(rr, httptest.NewRequest(http.MethodPost, "/v1/fak/account/rehome", nil))
	if rr.Code != http.StatusConflict {
		t.Fatalf("refused swap answered %d, want 409", rr.Code)
	}
	var env struct {
		Error struct {
			Message string `json:"message"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &env); err != nil {
		t.Fatalf("409 body not the error envelope: %v", err)
	}
	if env.Error.Code != "account_rehome_unavailable" || !strings.Contains(env.Error.Message, "no available sibling seat") {
		t.Fatalf("409 envelope = %+v, want code account_rehome_unavailable carrying the swap error", env.Error)
	}
}

// SetAccountRehomeFunc must be a safe no-op on a nil Server (the seam contract every
// host-injected provider on Server follows).
func TestAccountRehomeNilServerSafe(t *testing.T) {
	var s *Server
	s.SetAccountRehomeFunc(func(string) (AccountRehome, error) { return AccountRehome{}, nil })
	if fn := s.accountRehomeFunc(); fn != nil {
		t.Fatal("nil Server must report no swap function")
	}
}
