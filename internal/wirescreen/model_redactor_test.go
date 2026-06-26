//go:build fakwiremodel

package wirescreen

import (
	"bytes"
	"context"
	"strings"
	"testing"

	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

type stubRedactionBackend struct {
	spans []Span
	calls int
}

func (s *stubRedactionBackend) ProposeRedactions(_ context.Context, _ []byte, _ string) []Span {
	s.calls++
	return append([]Span(nil), s.spans...)
}

func setModelRedactionBackendForTest(b modelRedactionBackend) func() {
	modelRedactionMu.Lock()
	prev := modelRedactionCurrent
	modelRedactionCurrent = b
	modelRedactionMu.Unlock()
	return func() {
		modelRedactionMu.Lock()
		modelRedactionCurrent = prev
		modelRedactionMu.Unlock()
	}
}

func TestModelRedactorRegisteredUnderModel(t *testing.T) {
	rmu.RLock()
	r, ok := rregistry["model"]
	rmu.RUnlock()
	if !ok {
		t.Fatal(`init() did not register a "model" redactor in the tagged build`)
	}
	if r.Name() != "model" {
		t.Fatalf(`registered redactor Name() = %q, want "model"`, r.Name())
	}
}

func TestModelRedactorDefaultInertDoesNotCallBackend(t *testing.T) {
	stub := &stubRedactionBackend{spans: []Span{{Start: 0, End: 4, Kind: "model_secret"}}}
	restoreBackend := setModelRedactionBackendForTest(stub)
	defer restoreBackend()
	t.Setenv("FAK_WIRE_REDACT", "")

	rmu.Lock()
	ractive, ractiveResolved = nil, false
	rmu.Unlock()

	if r := ActiveRedactor(); r != nil {
		t.Fatalf("ActiveRedactor() with FAK_WIRE_REDACT unset = %v, want nil", r)
	}
	if stub.calls != 0 {
		t.Fatalf("model backend was called on the inert path: %d calls", stub.calls)
	}

	floor := PIIRedactor()
	if spans := floor.Propose(context.Background(), []byte("reach alice@example.com"), "user"); !hasKind(spans, "email") {
		t.Fatalf("deterministic redaction floor stopped working in tagged build: %v", spans)
	}
}

func TestModelRedactorStubBackendAppliesAndRestores(t *testing.T) {
	body := []byte("normal framing; opaque session token is hf_token_alpha_beta; continue")
	secret := []byte("hf_token_alpha_beta")
	start := bytes.Index(body, secret)
	if start < 0 {
		t.Fatal("test fixture missing secret")
	}
	stub := &stubRedactionBackend{
		spans: []Span{{Start: start, End: start + len(secret), Kind: "model_secret"}},
	}
	restoreBackend := setModelRedactionBackendForTest(stub)
	defer restoreBackend()
	restoreActive := SetActiveRedactorForTest("model")
	defer restoreActive()

	r := ActiveRedactor()
	if r == nil || r.Name() != "model" {
		t.Fatalf("ActiveRedactor() = %v, want model", r)
	}
	red, ok := Apply(context.Background(), r, body, "assistant")
	if !ok {
		t.Fatal("Apply(model redactor) returned ok=false")
	}
	if stub.calls != 1 {
		t.Fatalf("stub backend calls = %d, want 1", stub.calls)
	}
	if red.By != "model" {
		t.Fatalf("redaction By = %q, want model", red.By)
	}
	if bytes.Contains(red.Redacted, secret) {
		t.Fatalf("redacted output still contains stub-proposed secret: %s", red.Redacted)
	}
	if !strings.Contains(string(red.Redacted), "[REDACTED:model_secret]") {
		t.Fatalf("redacted output missing model placeholder: %s", red.Redacted)
	}
	got, err := Restore(context.Background(), red.Original)
	if err != nil {
		t.Fatalf("Restore failed: %v", err)
	}
	if !bytes.Equal(got, body) {
		t.Fatalf("Restore mismatch:\n got=%q\nwant=%q", got, body)
	}
}

func TestModelRedactorPreservesPIIFloorWhenBackendDeclines(t *testing.T) {
	stub := &stubRedactionBackend{}
	restoreBackend := setModelRedactionBackendForTest(stub)
	defer restoreBackend()

	spans := modelRedactor{}.Propose(context.Background(), []byte("send alice@example.com a receipt"), "user")
	if !hasKind(spans, "email") {
		t.Fatalf("model redactor must include pii floor spans, got %v", spans)
	}
}
