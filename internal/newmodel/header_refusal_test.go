package newmodel

import "testing"

// fak-test:runtime fast est=1ms
func TestHeaderRefusalPreflightVocabulary(t *testing.T) {
	for _, reason := range []RefusalReason{RefusalHeaderConfigInvalid, RefusalKeyUnread, RefusalHFMismatch, RefusalTensorMissing, RefusalTensorOrphan, RefusalTensorShape, RefusalUnsupportedDtype} {
		r := HeaderRefusal(reason, "axis", "detail")
		if r.Schema != RefusalSchema || r.Phase != PhaseHeaderPreflight || r.Reason != reason {
			t.Fatalf("HeaderRefusal(%s) = %+v", reason, r)
		}
	}
}
