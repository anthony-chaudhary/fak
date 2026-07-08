package modelroute

import "testing"

// TestRetentionKnobsTripSensitiveRemoteFloor proves the OpenRouter-style zero-data-
// retention knobs (zdr / data_collection / retention) make a subject sensitive, so it
// cannot resolve to a REMOTE member — exactly like the explicit sensitive label —
// while the retention-allowed values leave a remote resolution permitted. The same
// contract is mirrored at the engine adjudication floor by TestRetentionKnobMirror.
func TestRetentionKnobsTripSensitiveRemoteFloor(t *testing.T) {
	r := smallLargeRegistry(t) // "small" is remote (openai), "large" is local (in-kernel)

	blocks := []map[string]string{
		{ZDRLabel: "true"},
		{ZDRLabel: "1"},
		{DataCollectionLabel: "deny"},
		{DataCollectionLabel: "none"},
		{RetentionLabel: "none"},
		{RetentionLabel: "zero"},
		{RetentionLabel: "zdr"},
	}
	for _, labels := range blocks {
		s := Subject{Labels: labels}
		_, err := r.Resolve(s, "small") // small is remote -> must refuse
		var re *ResolveError
		if !asResolveError(err, &re) || re.Reason != "sensitive_remote" {
			t.Fatalf("labels %v: want sensitive_remote refusal on a remote member, got %v", labels, err)
		}
		// The floor blocks the REMOTE route, not all routes: a local member still resolves.
		if _, err := r.Resolve(s, "large"); err != nil {
			t.Fatalf("labels %v: local member should still resolve, got %v", labels, err)
		}
	}

	allows := []map[string]string{
		{DataCollectionLabel: "allow"},
		{RetentionLabel: "allow"},
		{RetentionLabel: "any"},
		{ZDRLabel: "false"},
		{ZDRLabel: "0"},
	}
	for _, labels := range allows {
		if _, err := r.Resolve(Subject{Labels: labels}, "small"); err != nil {
			t.Fatalf("labels %v: a retention-allowed subject should resolve remote, got %v", labels, err)
		}
	}
}
