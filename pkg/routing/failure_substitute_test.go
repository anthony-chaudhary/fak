package routing

import (
	"testing"
)

func TestPayloadRejectedAtSubstituteRungFailsOver(t *testing.T) {
	t.Parallel()
	bare := Classify(400, `{"error":{"code":"upstream_invalid_request"}}`)
	if bare.Class != FailureBadPayload || bare.Action != ActionFatal {
		t.Fatalf("bare 400 = %s/%s, want PAYLOAD_REJECTED/FATAL", bare.Class, bare.Action)
	}
	sub := bare.AtSubstituteRung()
	if sub.Class != FailureBadPayload || sub.Action != ActionFailoverNext || sub.Retries != 0 {
		t.Fatalf("substitute = %s/%s retries=%d, want PAYLOAD_REJECTED/FAILOVER_NEXT/0", sub.Class, sub.Action, sub.Retries)
	}
	if got := sub.RetryBudget(5); got != 0 {
		t.Fatalf("substitute RetryBudget(5) = %d, want 0: another key rejects the same payload", got)
	}
	if !legalNonDefaultPair(sub) {
		t.Fatalf("substitute pair must be a documented legal exception")
	}
}

func TestPayloadRejectedSubstituteLeavesOtherClassesAlone(t *testing.T) {
	t.Parallel()
	for _, f := range []Failure{
		Classify(429, ""),
		Classify(503, ""),
		Classify(400, "context length exceeded"),
		ClassifyError(nil),
	} {
		if got := f.AtSubstituteRung(); got != f {
			t.Fatalf("AtSubstituteRung changed %s/%s to %s/%s", f.Class, f.Action, got.Class, got.Action)
		}
	}
	if legalNonDefaultPair(Failure{Class: FailureRateLimited, Action: ActionFatal}) {
		t.Fatalf("an arbitrary mismatched pair must stay illegal")
	}
}

func TestPayloadRejectedRequestScopedClasses(t *testing.T) {
	t.Parallel()
	want := map[FailureClass]bool{
		FailureBadPayload:         true,
		FailureContextOverflow:    true,
		FailureRateLimited:        false,
		FailureUnavailable:        false,
		FailureEngineOOM:          false,
		FailureTimeout:            false,
		FailureAuthRejected:       false,
		FailureCreditExhausted:    false,
		FailureModelMissing:       false,
		FailureConflict:           false,
		FailureCanceled:           false,
		FailureDeadlineInfeasible: true,
		FailureUnknown:            false,
	}
	for class, scoped := range want {
		if got := class.RequestScoped(); got != scoped {
			t.Fatalf("%s.RequestScoped() = %v, want %v", class, got, scoped)
		}
	}
}
