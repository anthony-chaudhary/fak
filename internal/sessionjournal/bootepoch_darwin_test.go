//go:build darwin

package sessionjournal

import (
	"testing"
	"time"
)

func TestBootTimeDarwin(t *testing.T) {
	now := time.Now()
	boot, source := BootTime(now)
	if source != "sysctl-kern-boottime" {
		t.Fatalf("BootTime() source = %q, want %q", source, "sysctl-kern-boottime")
	}
	if boot.IsZero() {
		t.Fatalf("BootTime() returned zero time")
	}
	baseline := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)
	if !boot.After(baseline) {
		t.Fatalf("BootTime() = %v, want after baseline %v", boot, baseline)
	}
	if !boot.Before(now) {
		t.Fatalf("BootTime() = %v, want before now %v", boot, now)
	}
}
