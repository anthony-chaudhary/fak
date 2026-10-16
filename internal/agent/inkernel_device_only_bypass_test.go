package agent

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=1ms lane=default
func TestDeviceOnlyBypassGrantsOneSessionAtATime(t *testing.T) {
	p := &InKernelPlanner{deviceOnlyBypass: "device-only-session"}
	first, second := &model.Session{M: &model.Model{}}, &model.Session{M: &model.Model{}}
	p.applyDeviceOnlyBypass(first)
	p.applyDeviceOnlyBypass(second)
	if first.HostPathBypass() != "device-only-session" || second.HostPathBypass() != "" {
		t.Fatalf("holders: first=%q second=%q, want only the first", first.HostPathBypass(), second.HostPathBypass())
	}
	p.applyDeviceOnlyBypass(first)
	first.Close()
	third := &model.Session{M: &model.Model{}}
	p.applyDeviceOnlyBypass(third)
	if third.HostPathBypass() != "device-only-session" {
		t.Fatalf("grant not released on Close: third=%q", third.HostPathBypass())
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDeviceOnlyBypassPerfLineCarriesDegradedToken(t *testing.T) {
	if got := (&InKernelPlanner{}).hostPathDegradedClause(); got != "" {
		t.Fatalf("no bypass clause=%q, want empty", got)
	}
	got := (&InKernelPlanner{deviceOnlyBypass: "device-only-session"}).hostPathDegradedClause()
	if !strings.Contains(got, HostPathDegradedToken+"=device-only-session") {
		t.Fatalf("bypass clause=%q, want %s token", got, HostPathDegradedToken)
	}
}
