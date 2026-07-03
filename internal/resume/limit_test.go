package resume

import "testing"

func TestClassifyLimitTextBareFableLimit(t *testing.T) {
	reason, ok := ClassifyLimitText("You've reached your Fable 5 limit. Run /usage-credits to continue.")
	if !ok || reason != LimitUsage {
		t.Fatalf("ClassifyLimitText = (%q,%v), want (%q,true)", reason, ok, LimitUsage)
	}
}
