package routing

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

const engineOOMBody = `{"error":{"code":"in_kernel_oom","message":"kernel allocator exhausted device memory"}}`

func TestEngineOOMIsAKnownClass(t *testing.T) {
	if FailureEngineOOM != FailureClass("ENGINE_OOM") {
		t.Fatalf("FailureEngineOOM = %q, want ENGINE_OOM", FailureEngineOOM)
	}
	if !FailureEngineOOM.IsKnown() {
		t.Fatal("FailureEngineOOM is not in the closed vocabulary")
	}
	found := false
	for _, c := range FailureClasses() {
		if c == FailureEngineOOM {
			found = true
		}
	}
	if !found {
		t.Fatal("FailureClasses() does not list FailureEngineOOM")
	}
}

func TestClassifyEngineOOMPrecedence(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
		want   FailureClass
	}{
		{"engine json", 503, engineOOMBody, FailureEngineOOM},
		{"bare code", 503, "in_kernel_oom", FailureEngineOOM},
		{"with rate limit phrase", 503, `{"error":{"code":"in_kernel_oom","message":"rate limit exceeded"}}`, FailureEngineOOM},
		{"with unavailable phrase", 503, `{"error":{"code":"in_kernel_oom","message":"service unavailable"}}`, FailureEngineOOM},
		{"with timeout phrase", 503, `{"error":{"code":"in_kernel_oom","message":"timed out"}}`, FailureEngineOOM},
		{"with context overflow phrase", 503, `{"error":{"code":"in_kernel_oom","message":"context_length_exceeded"}}`, FailureContextOverflow},
		{"bare 503", 503, "", FailureUnavailable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Classify(tc.status, tc.body); got.Class != tc.want {
				t.Fatalf("Classify(%d, %q).Class = %q, want %q", tc.status, tc.body, got.Class, tc.want)
			}
		})
	}
}

func TestClassifyErrorEngineOOM(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want FailureClass
	}{
		{"plain", errors.New("upstream 503: in_kernel_oom"), FailureEngineOOM},
		{"wrapped", fmt.Errorf("relay: %w", errors.New(engineOOMBody)), FailureEngineOOM},
		{"canceled outranks", fmt.Errorf("in_kernel_oom: %w", context.Canceled), FailureCanceled},
		{"context overflow outranks", errors.New("in_kernel_oom after context_length_exceeded"), FailureContextOverflow},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ClassifyError(tc.err); got.Class != tc.want {
				t.Fatalf("ClassifyError(%v).Class = %q, want %q", tc.err, got.Class, tc.want)
			}
		})
	}
}

func TestEngineOOMSemanticsMatchUnavailable(t *testing.T) {
	if got := FailureEngineOOM.DefaultAction(); got != ActionFailoverNext {
		t.Errorf("DefaultAction() = %q, want %q", got, ActionFailoverNext)
	}
	if got, want := FailureEngineOOM.DefaultAction(), FailureUnavailable.DefaultAction(); got != want {
		t.Errorf("DefaultAction() = %q, UNAVAILABLE has %q", got, want)
	}
	if got := FailureEngineOOM.DefaultRetries(); got != 0 {
		t.Errorf("DefaultRetries() = %d, want 0", got)
	}
	if got, want := FailureEngineOOM.Transient(), FailureUnavailable.Transient(); got != want || !got {
		t.Errorf("Transient() = %v, want true (UNAVAILABLE has %v)", got, want)
	}
	if FailureEngineOOM.RequestScoped() {
		t.Error("RequestScoped() = true, want false")
	}
	f := Classify(503, engineOOMBody)
	if f.Action != ActionFailoverNext {
		t.Errorf("Classify action = %q, want %q", f.Action, ActionFailoverNext)
	}
	for _, h := range []int{0, 1, 3, 7} {
		u := Classify(503, "")
		if got, want := f.RetryBudget(h), u.RetryBudget(h); got != want || (h > 0 && got != h) {
			t.Errorf("RetryBudget(%d) = %d, want %d (UNAVAILABLE %d)", h, got, h, want)
		}
	}
}
