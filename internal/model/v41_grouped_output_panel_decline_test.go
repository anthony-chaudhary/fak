package model

import (
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

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

// Runtime environment edits must not disagree with the native startup gate.
// fak-test:runtime fast est=10ms lane=default
func TestV41GroupedPanelDeclineUsesNativeStartupProfile(t *testing.T) {
	v41GroupedPanelDeclineReset()
	t.Cleanup(v41GroupedPanelDeclineReset)
	enabled := compute.VulkanDispatchProfileEnabled()
	capture, err := os.CreateTemp(t.TempDir(), "decline-stderr-")
	if err != nil {
		t.Fatal(err)
	}
	stderr := os.Stderr
	os.Stderr = capture
	defer func() {
		os.Stderr = stderr
		if err := capture.Close(); err != nil {
			t.Error(err)
		}
	}()
	for _, setting := range []string{"", "1"} {
		t.Setenv("FAK_VULKAN_DISPATCH_PROFILE", setting)
		if compute.VulkanDispatchProfileEnabled() != enabled {
			t.Fatal("native startup profile changed after an environment edit")
		}
		if got := v41GroupedPanelDecline(2, 8, "publish_count"); got != v41ProjectionDeclined {
			t.Fatalf("outcome = %v, want declined", got)
		}
	}
	if got := v41GroupedPanelDeclineSummary(); got != "publish_count=2" {
		t.Fatalf("summary = %q, want both declines counted", got)
	}
	data, err := os.ReadFile(capture.Name())
	if err != nil {
		t.Fatal(err)
	}
	want := ""
	if enabled {
		want = "v41 grouped_output panel declined reason=publish_count layer=2 rows=8 count=1\n" +
			"v41 grouped_output panel declined reason=publish_count layer=2 rows=8 count=2\n"
	}
	if string(data) != want {
		t.Fatalf("decline log = %q, want %q for native startup profile=%v", data, want, enabled)
	}
}
