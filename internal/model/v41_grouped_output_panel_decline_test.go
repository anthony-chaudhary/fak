package model

import "testing"

func TestV41GroupedPanelDeclineCountsReasons(t *testing.T) {
	v41GroupedPanelDeclineReset()
	t.Cleanup(v41GroupedPanelDeclineReset)
	t.Setenv("FAK_VULKAN_DISPATCH_PROFILE", "")
	if got := v41GroupedPanelDecline(0, 8, "publish_count"); got != v41ProjectionDeclined {
		t.Fatalf("outcome = %v, want declined", got)
	}
	v41GroupedPanelDecline(1, 8, "publish_count")
	v41GroupedPanelDecline(1, 8, "backend_caps")
	if got, want := v41GroupedPanelDeclineSummary(), "backend_caps=1,publish_count=2"; got != want {
		t.Fatalf("summary = %q, want %q", got, want)
	}
}

func TestV41GroupedPanelCallerReason(t *testing.T) {
	for _, c := range []struct {
		full, state, owner, same bool
		want                     string
	}{
		{false, true, true, true, "caller_not_full_profile"},
		{true, false, false, false, "caller_no_state"},
		{true, true, false, false, "caller_no_owner"},
		{true, true, true, false, "caller_owner_model_mismatch"},
	} {
		if got := v41GroupedPanelCallerReason(c.full, c.state, c.owner, c.same); got != c.want {
			t.Fatalf("reason(%v,%v,%v,%v) = %q, want %q", c.full, c.state, c.owner, c.same, got, c.want)
		}
	}
}

// A shape refusal is attributed rather than silent.
func TestV41GroupedPanelShapeDeclineIsCounted(t *testing.T) {
	v41GroupedPanelDeclineReset()
	t.Cleanup(v41GroupedPanelDeclineReset)
	var s *Session
	if _, outcome, err := s.v41GroupedOutputPanel(0, nil, 4, 2, 2, 1, 1, 1); outcome != v41ProjectionDeclined || err != nil {
		t.Fatalf("outcome=%v err=%v", outcome, err)
	}
	if got := v41GroupedPanelDeclineSummary(); got != "shape=1" {
		t.Fatalf("summary = %q", got)
	}
}
