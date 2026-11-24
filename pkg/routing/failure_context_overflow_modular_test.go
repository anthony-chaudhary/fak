package routing

import (
	"errors"
	"testing"
)

// failure_context_overflow_modular_test.go — regression for ticket
// router-modular-overflow-misclass-20261108. Modular answers an oversized
// prompt with a 400 whose body names "input length N leaves no generation
// position (max_seq_len M)" (witnessed live 2026-11-08; see
// docs/notes/AUDIT-API-ERROR-400-MALFORMED-UPSTREAM-2026-11-08.md §3). Before
// the fix the body named no signature, so the 400 fell to the bare-400 default
// and read PAYLOAD_REJECTED/FATAL: the walk ended instead of failing over to
// the one longer-window rung that could still serve.

// modularOverflowBody is the verbatim upstream 400 body opencode saw on
// 2026-11-08 (the audit's §7 evidence line, without its "stream error:" prefix).
const modularOverflowBody = "rpc error: code = InvalidArgument desc = input length 270892 leaves no generation position (max_seq_len 262144)"

func TestClassifyModularContextOverflow400(t *testing.T) {
	got := Classify(400, modularOverflowBody)
	if got.Class != FailureContextOverflow {
		t.Fatalf("Classify(400, %q).Class = %q, want %q", modularOverflowBody, got.Class, FailureContextOverflow)
	}
	if want := FailureContextOverflow.DefaultAction(); got.Action != want {
		t.Fatalf("Classify(400, %q).Action = %q, want %q", modularOverflowBody, got.Action, want)
	}
	if got.Retries != FailureContextOverflow.DefaultRetries() {
		t.Fatalf("Classify(400, %q).Retries = %d, want %d", modularOverflowBody, got.Retries, FailureContextOverflow.DefaultRetries())
	}
	// CONTEXT_OVERFLOW carries no same-target budget, so the fabric authorizes
	// the full credential headroom: rolling keys is the remedy, not retrying a
	// doomed key (docs/ROUTER-FAILOVER-AUTHORITY.md §4).
	if got.RetryBudget(3) != 3 {
		t.Fatalf("RetryBudget(headroom 3) = %d, want 3", got.RetryBudget(3))
	}
	if got.RetryBudget(0) != 0 {
		t.Fatalf("RetryBudget(headroom 0) = %d, want 0", got.RetryBudget(0))
	}
}

// Each new signature must claim the body on its own, so neither entry is dead
// weight if the other wording ever changes upstream.
func TestClassifyModularOverflowEachSignatureClaimsAlone(t *testing.T) {
	for _, sig := range []string{"leaves no generation position", "max_seq_len"} {
		body := "rpc error: code = InvalidArgument desc = input length 319483 " + sig
		got := Classify(400, body)
		if got.Class != FailureContextOverflow || got.Action != ActionFailoverNext {
			t.Fatalf("Classify(400, %q) = class %q action %q, want %q/%q", body, got.Class, got.Action, FailureContextOverflow, ActionFailoverNext)
		}
	}
}

// The stream-error string form opencode saw verbatim (audit §7): the shared
// list must claim it through ClassifyError too, not just the HTTP body arm.
func TestClassifyErrorModularStreamContextOverflow(t *testing.T) {
	got := ClassifyError(errors.New("stream error: " + modularOverflowBody))
	if got.Class != FailureContextOverflow || got.Action != ActionFailoverNext {
		t.Fatalf("ClassifyError(stream error).Class/Action = %q/%q, want %q/%q", got.Class, got.Action, FailureContextOverflow, ActionFailoverNext)
	}
}

// Guard against over-widening: a bare 400 that names no overflow signature
// stays the fatal caller-bug verdict.
func TestClassifyBare400WithoutOverflowSignatureStaysPayloadRejected(t *testing.T) {
	for _, body := range []string{
		`{"error":{"message":"upstream rejected the request as malformed (HTTP 400)"}}`,
		`{"error":"invalid request"}`,
	} {
		got := Classify(400, body)
		if got.Class != FailureBadPayload || got.Action != ActionFatal {
			t.Fatalf("Classify(400, %q) = class %q action %q, want %q/%q (no over-widening)", body, got.Class, got.Action, FailureBadPayload, ActionFatal)
		}
	}
}
