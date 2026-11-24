package routing

import "testing"

// witnessedZenFreeTierRefusal is the body opencode.ai/zen answered with a 403
// for mimo-v2.6-flash-free on 2026-10-09, with both the anonymous "public"
// key and a paid key: the same key served other models.
const witnessedZenFreeTierRefusal = `{"type":"FreeTierError","message":"OpenCode's free tier can only be used from within OpenCode"}`

// fak-test:runtime fast est=5ms lane=default
func TestClassifyFreeTierRefusalIsModelWallNotCredential(t *testing.T) {
	got := Classify(403, witnessedZenFreeTierRefusal)
	if got.Class != FailureModelMissing || got.Action != ActionFailoverNext {
		t.Fatalf("Classify(403, FreeTierError) = (%s, %s), want (%s, %s)", got.Class, got.Action, FailureModelMissing, ActionFailoverNext)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestClassifyBare403StaysAuthRejected(t *testing.T) {
	for _, body := range []string{"", `{"error":{"message":"forbidden"}}`} {
		if got := Classify(403, body); got.Class != FailureAuthRejected {
			t.Fatalf("Classify(403, %q).Class = %s, want %s", body, got.Class, FailureAuthRejected)
		}
	}
}
