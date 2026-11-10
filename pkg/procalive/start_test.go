package procalive

import (
	"os"
	"runtime"
	"testing"
	"time"
)

// A live process reports a start time at or before now, so a recorded start
// can tell the owner apart from a later process that reuses its pid.
//
// fak-test:runtime fast est=10ms
func TestStartTimeOfSelf(t *testing.T) {
	if runtime.GOOS != "windows" && runtime.GOOS != "linux" {
		t.Skip("start time is only resolved on windows and linux")
	}
	start, ok := StartTime(os.Getpid())
	if !ok {
		t.Fatalf("StartTime(self) ok=false, want a resolved start")
	}
	if start.After(time.Now().Add(time.Second)) {
		t.Fatalf("StartTime(self)=%s is in the future", start)
	}
}

// fak-test:runtime fast est=10ms
func TestStartTimeUnknownPid(t *testing.T) {
	if _, ok := StartTime(0); ok {
		t.Fatalf("StartTime(0) ok=true, want false")
	}
}
