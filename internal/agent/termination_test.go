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
