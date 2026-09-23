package agent

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestClassifyTerminationClosedCauses(t *testing.T) {
	tests := []struct {
		err   error
		cause string
	}{{errors.New("provider status 429: secret payload"), TerminationRateLimited}, {errors.New("maximum context length exceeded"), TerminationContextLimit}, {errors.New("POLICY_BLOCK: secret rule"), TerminationRefused}, {context.Canceled, TerminationCanceled}, {errors.New("socket exploded: secret"), TerminationUnknown}}
	for _, tt := range tests {
		got := ClassifyTermination(tt.err)
		if got.Cause != tt.cause {
			t.Fatalf("%v: %+v", tt.err, got)
		}
		if strings.Contains(got.Evidence, "secret") {
			t.Fatalf("unsafe evidence: %+v", got)
		}
	}
}

func TestClassifyTerminationAuthenticationFailure(t *testing.T) {
	for _, raw := range []string{
		"provider status 401: missing_credentials secret payload",
		"http 401 invalid_credentials bearer secret-token",
		"provider status 403: forbidden secret payload",
		"http 403 permission denied bearer secret-token",
		"403 forbidden secret payload",
		"status code 403: access rejected secret-token",
	} {
		t.Run(raw[:8], func(t *testing.T) {
			got := ClassifyTermination(errors.New(raw))
			if got.Cause != TerminationAuth {
				t.Fatalf("cause = %q, want %q; termination=%+v", got.Cause, TerminationAuth, got)
			}
			if got.Evidence == "" || len(got.Evidence) > 128 {
				t.Fatalf("evidence must be non-empty and bounded: %q", got.Evidence)
			}
			for _, secret := range []string{"secret payload", "secret-token", "missing_credentials", "invalid_credentials"} {
				if strings.Contains(got.Evidence, secret) {
					t.Fatalf("evidence leaked raw provider detail %q: %q", secret, got.Evidence)
				}
			}
		})
	}
}
