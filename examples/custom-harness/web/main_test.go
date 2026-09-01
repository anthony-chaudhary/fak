package main

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func TestTurnRendersDeterministicGovernedEvents(t *testing.T) {
	a, err := newApp()
	if err != nil {
		t.Fatal(err)
	}

	form := url.Values{"prompt": {"teach me the seam"}}
	req := httptest.NewRequest(http.MethodPost, "/turn", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	recorder := httptest.NewRecorder()

	a.routes().ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	body := recorder.Body.String()
	for _, want := range []string{
		"teach me the seam",
		"turn.started",
		"model.response",
		"tool.requested",
		"tool.completed",
		"turn.completed",
		"example/custom-harness-web received",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("response missing %q:\n%s", want, body)
		}
	}
}

func TestTurnRejectsBlankPrompt(t *testing.T) {
	a, err := newApp()
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/turn", strings.NewReader("prompt=+"))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	recorder := httptest.NewRecorder()

	a.routes().ServeHTTP(recorder, req)

	if recorder.Code != http.StatusBadRequest || !strings.Contains(recorder.Body.String(), "Write a prompt first.") {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
}
