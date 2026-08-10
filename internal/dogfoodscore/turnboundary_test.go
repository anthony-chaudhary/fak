package dogfoodscore

import "testing"

func TestTurnBoundaryRefusesFreshStopHookFailure(t *testing.T) {
	transcript := []byte(asstLine("Implementation is ready for final verification.") + "\n" +
		`{"type":"user","isMeta":true,"message":{"role":"user","content":"Stop hook error: verification failed"}}`)

	got := CheckTurnBoundary(transcript)
	if got.AllowFinal || !got.FreshFailure {
		t.Fatalf("fresh failure must refuse final narration: %+v", got)
	}
	if got.HarnessLine == "" {
		t.Fatalf("refusal must carry the harness evidence: %+v", got)
	}
}

func TestTurnBoundaryClearsAfterFailureIsHandled(t *testing.T) {
	transcript := []byte(asstLine("Ready.") + "\n" +
		`{"type":"user","isMeta":true,"message":{"role":"user","content":"Stop hook error: verification failed"}}` + "\n" +
		asstLine("I handled the hook failure and reran verification."))

	got := CheckTurnBoundary(transcript)
	if !got.AllowFinal || got.FreshFailure {
		t.Fatalf("a subsequent assistant turn must clear the pre-final refusal: %+v", got)
	}
}

func TestTurnBoundaryIgnoresAssistantQuote(t *testing.T) {
	got := CheckTurnBoundary([]byte(asstLine("The log example says Stop hook error: verification failed.")))
	if !got.AllowFinal || got.FreshFailure {
		t.Fatalf("assistant prose is not harness evidence: %+v", got)
	}
}
