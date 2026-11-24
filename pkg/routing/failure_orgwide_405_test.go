package routing

import "testing"

// TestBare405IsOrgWideAndSpendsNoHeadroom pins the fix for the documented
// divergence between the authority-doc budget table (bare 405 -> Budget 0) and
// Failure.RetryBudget, which used to grant the FULL credential headroom for any
// non-fatal class. An org pause is org-wide: rolling sibling keys of the same
// provider each roll the same wall and tombstone a live key, so the walk must
// move to the next BACKEND, not the next key.
func TestBare405IsOrgWideAndSpendsNoHeadroom(t *testing.T) {
	const headroom = 5

	bare := Classify(405, "")
	if bare.Class != FailureUnavailable {
		t.Fatalf("Classify(405, \"\").Class = %q, want %q", bare.Class, FailureUnavailable)
	}
	if bare.Action != ActionFailoverNext {
		t.Errorf("Classify(405, \"\").Action = %q, want %q", bare.Action, ActionFailoverNext)
	}
	if !bare.OrgWide {
		t.Error("a bare 405 must be marked OrgWide: the wall is on the credential org, not the key")
	}
	if got := bare.RetryBudget(headroom); got != 0 {
		t.Errorf("Classify(405, \"\").RetryBudget(%d) = %d, want 0 (rolling sibling keys burns the pool)", headroom, got)
	}

	// A 405 whose BODY names the pause is claimed earlier as CREDIT_EXHAUSTED,
	// which already spends full headroom on the keyring plane (its remedy is a
	// differently-funded backend). That row must not regress to OrgWide.
	paused := Classify(405, `{"error":"organization is currently paused"}`)
	if paused.Class != FailureCreditExhausted {
		t.Fatalf("Classify(405, org-pause body).Class = %q, want %q", paused.Class, FailureCreditExhausted)
	}

	// The contrast: other UPSTREAM_UNAVAILABLE statuses still spend headroom,
	// because a provider outage is not a key fault and rolling is the remedy.
	for _, status := range []int{500, 502, 503} {
		f := Classify(status, "")
		if f.OrgWide {
			t.Errorf("Classify(%d, \"\") must not be OrgWide", status)
		}
		if got := f.RetryBudget(headroom); got != headroom {
			t.Errorf("Classify(%d, \"\").RetryBudget(%d) = %d, want full headroom", status, headroom, got)
		}
	}
}

// TestClassifyErrorOrgPauseBodyStillCreditWall keeps the error-value arm honest:
// the org-pause wording on an error VALUE has no 405 status to conflict with, so
// it keeps classifying as CREDIT_EXHAUSTED (the wall), not a bare-405 outage.
func TestClassifyErrorOrgPauseBodyStillCreditWall(t *testing.T) {
	got := ClassifyError(errText("organization is currently paused"))
	if got.Class != FailureCreditExhausted {
		t.Fatalf("ClassifyError(org-pause) = %q, want %q", got.Class, FailureCreditExhausted)
	}
}

type errText string

func (e errText) Error() string { return string(e) }
